using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace NNtrain;

internal static class QwenLoraCommand
{
    internal sealed record Example(string Prompt, string Response);
    internal sealed record TrainingExample(int[] Tokens, int ResponseStartIndex);

    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        try
        {
            string? modelPath = null, configPath = null;
            bool resume = false;
            bool dryRun = false;
            var seen = new HashSet<string>(StringComparer.Ordinal);
            for (int i = 1; i < args.Length; ++i)
            {
                string option = args[i];
                if (!seen.Add(option)) throw new ArgumentException($"Duplicate qwen-lora option: {option}");
                if (option == "--resume") { resume = true; continue; }
                if (option == "--dry-run") { dryRun = true; continue; }
                if (option is not ("--model" or "--config"))
                    throw new ArgumentException($"Unknown qwen-lora option: {option}");
                if (++i == args.Length) throw new ArgumentException($"Missing value for {option}.");
                if (option == "--model") modelPath = Path.GetFullPath(args[i]);
                else configPath = Path.GetFullPath(args[i]);
            }
            if (modelPath is null || configPath is null)
                throw new ArgumentException("Usage: lora (or qwen-lora) --model <qwen35.gguf> --config <qwen-lora.json> [--resume] [--dry-run]");
            QwenLoraTrainingConfiguration config = QwenLoraTrainingConfiguration.Load(configPath);
            config.Validate();
            string directory = Path.GetDirectoryName(configPath)!;
            string dataPath = Path.GetFullPath(config.DataPath, directory);
            string adapterPath = Path.GetFullPath(config.AdapterPath, directory);
            string graphPath = QwenLoraMetricReporter.ResolveHtmlPath(configPath, config);
            string metricPath = TrainingMetricReporter.GetSidecarPath(graphPath);
            ValidateDistinctPaths(modelPath, configPath, dataPath, adapterPath, graphPath, metricPath);
            if (Directory.Exists(graphPath) || Directory.Exists(metricPath))
                throw new ArgumentException("Loss graph and metrics paths must name files.");
            if (!File.Exists(modelPath)) throw new FileNotFoundException("GGUF model not found.", modelPath);
            if (!File.Exists(dataPath)) throw new FileNotFoundException("LoRA JSONL dataset not found.", dataPath);
            if (Directory.Exists(adapterPath)) throw new ArgumentException("adapterPath must name a file.");
            if (resume && !File.Exists(adapterPath))
                throw new FileNotFoundException("--resume requires an existing adapter checkpoint.", adapterPath);
            if (!resume && File.Exists(adapterPath))
                throw new InvalidOperationException("Adapter already exists. Use --resume or choose a new adapterPath.");

            // All metadata, data and sequence-length validation precedes GPU allocation.
            Qwen35GgufDescriptor descriptor = Qwen35Gguf.Inspect(modelPath);
            if (config.ContextLength > descriptor.ContextLength)
                throw new ArgumentException("Configured contextLength exceeds the GGUF context length.");
            if (config.Layers is not null && config.Layers.Any(x => x >= descriptor.LayerCount))
                throw new ArgumentException("Configured layers exceed the GGUF layer count.");
            Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(modelPath);
            byte[] dataset = File.ReadAllBytes(dataPath);
            TrainingExample[] examples = ReadExamples(dataset, tokenizer, config);
            bool shuffleExamples = config.ShuffleExamples ?? true;
            string identity = TrainingIdentity(dataset, tokenizer.EosTokenId!.Value, config, shuffleExamples);
            if (resume && config.ShuffleExamples is null)
            {
                // Configs written before shuffling existed must continue their
                // original sequence. A new shuffled checkpoint keeps its order.
                string legacyIdentity = TrainingIdentity(dataset, tokenizer.EosTokenId.Value, config, false);
                if (ReadCheckpointTrainingIdentity(adapterPath) == legacyIdentity)
                {
                    shuffleExamples = false;
                    identity = legacyIdentity;
                }
            }

            Qwen35ExecutionOptions executionOptions = config.ExecutionOptions(
                examples.Select(example => example.Tokens.Length).ToArray(), resume);
            output.WriteLine($"Resolved LoRA speed flags: row fusion={executionOptions.TrainingFusedAttentionRows}, "
                + $"direct handoff={executionOptions.TrainingHostCheckpointBufferHandoff}, "
                + $"streamed tile rows={executionOptions.TrainingStreamedAttentionTileRows}, "
                + $"rolling transpose={executionOptions.TrainingIQ2RollingTranspose}. "
                + "Measured speed evidence uses FP16/XMX and one synthetic update; epoch quality is unverified.");
            if (dryRun)
            {
                if (resume && ReadCheckpointTrainingIdentity(adapterPath) != identity)
                    throw new InvalidDataException("Resume checkpoint header does not match this dataset/training configuration.");
                int[] counts = examples.Select(example => example.Tokens.Length).ToArray();
                output.WriteLine($"CPU dry-run valid: examples={examples.Length}, total tokens including EOS={counts.Min()}..{counts.Max()}, context upper bound={config.ContextLength}, maxSteps={config.MaxSteps}, precision={config.Iq2ForwardPrecision}, resume={resume}.");
                output.WriteLine("No padding, packing, truncation, GPU model load, optimizer update or checkpoint write.");
                return 0;
            }
            using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(modelPath, config.Devices,
                output.WriteLine, options: executionOptions);
            long[] encodedBeforeTraining = model.ResidentWeightBytes.ToArray();
            if (resume) model.LoadLora(adapterPath, identity);
            else model.AttachLora(config.AdapterOptions());
            output.WriteLine($"Qwen3.5 LoRA: rank={config.Rank}, trainable={model.LoraParameterCount}, "
                + $"examples={examples.Length}, context={config.ContextLength}, step={model.LoraStep}/{config.MaxSteps}");
            output.WriteLine("Frozen quantized GPU base; response-only next-token loss including EOS; one example per step.");
            output.WriteLine(shuffleExamples
                ? $"Example order: deterministic per-epoch shuffle (seed={config.Seed})."
                : "Example order: dataset order (legacy/explicit).");
            output.WriteLine($"IQ2 forward precision={config.Iq2ForwardPrecision}, "
                + $"host projection cache={config.Iq2ProjectionCacheMiB} MiB"
                + (config.Iq2ProjectionCacheMiB > 0 && config.Iq2ProjectionCachePrioritize ? " (prioritized)" : "")
                + ", "
                + $"GPU projection cache={config.Iq2GpuProjectionCacheMiB} MiB/Arc, "
                + $"training buffer pool={config.TrainingBufferPoolMiB} MiB/Arc");
            using var metrics = new QwenLoraMetricReporter(config, graphPath, examples.Length,
                model.LoraStep, resume, output, error);
            long savedStep = resume ? model.LoraStep : -1;
            long currentEpoch = -1;
            int[]? epochOrder = null;
            while (model.LoraStep < config.MaxSteps)
            {
                long epoch = model.LoraStep / examples.Length;
                if (shuffleExamples && epoch != currentEpoch)
                {
                    epochOrder = ShuffledExampleOrder(examples.Length, config.Seed, epoch);
                    currentEpoch = epoch;
                }
                int position = (int)(model.LoraStep % examples.Length);
                int index = shuffleExamples ? epochOrder![position] : position;
                TrainingExample example = examples[index];
                var timer = Stopwatch.StartNew();
                Qwen35LoraStepResult result = model.TrainLora(example.Tokens, example.ResponseStartIndex);
                metrics.Append(result.Step, result.Loss);
                output.WriteLine(FormattableString.Invariant(
                    $"LoRA step={result.Step}, example={index + 1}/{examples.Length}, loss={result.Loss:F6}, gradient-norm={result.GradientNorm:G6}, supervised-tokens={result.SupervisedTokens}, elapsed={timer.Elapsed.TotalSeconds:F3} s"));
                output.Flush();
                if (model.LoraStep % config.SaveEverySteps == 0) Save();
            }
            if (!encodedBeforeTraining.SequenceEqual(model.ResidentWeightBytes))
                throw new InvalidOperationException("Encoded GGUF weight residency changed during LoRA training.");
            if (savedStep != model.LoraStep) Save();
            metrics.Flush();
            long[] live = model.LiveDeviceBytes.ToArray(), peak = model.PeakDeviceBytes.ToArray();
            for (int slot = 0; slot < config.Devices.Length; ++slot)
                output.WriteLine($"Arc {config.Devices[slot]} GPU bytes: encoded={encodedBeforeTraining[slot]}, "
                    + $"live={live[slot]}, peak={peak[slot]}; encoded weight byte count unchanged.");
            output.Flush();
            return 0;

            void Save()
            {
                Directory.CreateDirectory(Path.GetDirectoryName(adapterPath)!);
                // The core writer atomically replaces the checkpoint, including
                // adapter/optimizer state, base identity and this training contract.
                model.SaveLora(adapterPath, identity);
                savedStep = model.LoraStep;
                output.WriteLine($"LoRA checkpoint = {adapterPath}, step {savedStep}");
                output.Flush();
            }
        }
        catch (Exception exception) when (exception is not OutOfMemoryException and not StackOverflowException)
        {
            error.WriteLine($"Error: {exception.Message}");
            error.WriteLine(exception.StackTrace);
            return 2;
        }
    }

    internal static void ValidateDistinctPaths(params string[] paths)
    {
        if (paths.Select(Path.GetFullPath).Distinct(StringComparer.OrdinalIgnoreCase).Count() != paths.Length)
            throw new ArgumentException("Model, configuration, dataset, adapter, loss graph and metrics paths must all be distinct.");
    }

    internal static TrainingExample[] ReadExamples(byte[] dataset, Qwen2GgufTokenizer tokenizer,
        QwenLoraTrainingConfiguration config)
    {
        if (tokenizer.EosTokenId is not int eos || eos < 0 || eos >= tokenizer.VocabularySize)
            throw new InvalidDataException("A valid GGUF EOS token is required for LoRA supervision.");
        string text = new UTF8Encoding(false, true).GetString(dataset).TrimStart('\uFEFF');
        using var reader = new StringReader(text);
        var examples = new List<TrainingExample>();
        int lineNumber = 0;
        while (reader.ReadLine() is { } line)
        {
            ++lineNumber;
            if (string.IsNullOrWhiteSpace(line)) continue;
            try
            {
                Example sample = JsonSerializer.Deserialize<Example>(line, LoraConfiguration.Json)
                    ?? throw new InvalidDataException("Empty example.");
                examples.Add(BuildExample(sample, tokenizer, config));
            }
            catch (Exception exception) when (exception is JsonException or ArgumentException or InvalidDataException)
            {
                throw new InvalidDataException($"LoRA dataset line {lineNumber}: {exception.Message}", exception);
            }
        }
        if (examples.Count == 0) throw new InvalidDataException("LoRA dataset has no examples.");
        return examples.ToArray();
    }

    internal static TrainingExample BuildExample(Example sample, Qwen2GgufTokenizer tokenizer,
        QwenLoraTrainingConfiguration config)
    {
        if (sample.Prompt is null || string.IsNullOrWhiteSpace(sample.Response))
            throw new InvalidDataException("Examples require prompt and non-empty response strings.");
        int eos = tokenizer.EosTokenId
            ?? throw new InvalidDataException("A GGUF EOS token is required for LoRA supervision.");
        int[] prompt = tokenizer.Encode(config.PromptPrefix + sample.Prompt + config.ResponsePrefix);
        int[] response = tokenizer.Encode(sample.Response);
        if (prompt.Length == 0 || response.Length == 0)
            throw new InvalidDataException("The formatted prompt and response must each encode to at least one token.");
        int count = checked(prompt.Length + response.Length + 1);
        if (count > config.ContextLength)
            throw new InvalidDataException($"Example has {count} tokens including EOS, exceeding contextLength={config.ContextLength}. Shorten it explicitly; no truncation is performed.");
        return new TrainingExample([.. prompt, .. response, eos], prompt.Length);
    }

    internal static string TrainingIdentity(byte[] dataset, int eos,
        QwenLoraTrainingConfiguration config, bool? shuffleExamples = null)
    {
        // MaxSteps is a total-step ceiling that can grow on resume. Output path,
        // device placement and save frequency do not change the update contract.
        string contract = JsonSerializer.Serialize(new
        {
            Version = 1, DataSha256 = Convert.ToHexString(SHA256.HashData(dataset)),
            Objective = "response-only-shifted-ce-eos", Eos = eos, config.ContextLength,
            config.PromptPrefix, config.ResponsePrefix, config.Rank, config.Alpha,
            config.LearningRate, config.WeightDecay, config.GradientClip, config.Seed,
            Layers = config.Layers?.Order().ToArray(),
            Targets = config.Targets.Order(StringComparer.Ordinal).ToArray(), config.IncludeOutput
        }, LoraConfiguration.Json);
        // Preserve the identity of existing exact checkpoints. FP16 updates
        // use a different numerical trajectory and must not resume an exact run.
        if (config.Iq2ForwardPrecision != "exact")
            contract += "|iq2ForwardPrecision=" + config.Iq2ForwardPrecision;
        if (shuffleExamples ?? config.ShuffleExamples ?? true)
            contract += "|exampleOrder=epoch-shuffle-splitmix64-v1";
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(contract)));
    }

    internal static int[] ShuffledExampleOrder(int count, int seed, long epoch)
    {
        if (count <= 0 || epoch < 0) throw new ArgumentOutOfRangeException(nameof(count));
        int[] order = Enumerable.Range(0, count).ToArray();
        // SplitMix64 makes the order stable across .NET runtime versions. Each
        // epoch is derived independently, so a saved step fully defines resume.
        ulong state = unchecked((uint)seed + (ulong)epoch * 0x9E3779B97F4A7C15UL);
        for (int i = count - 1; i > 0; --i)
        {
            state = unchecked(state + 0x9E3779B97F4A7C15UL);
            ulong value = state;
            value = unchecked((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9UL);
            value = unchecked((value ^ (value >> 27)) * 0x94D049BB133111EBUL);
            value ^= value >> 31;
            int swap = (int)(value % (ulong)(i + 1));
            (order[i], order[swap]) = (order[swap], order[i]);
        }
        return order;
    }

    private static string? ReadCheckpointTrainingIdentity(string path)
    {
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Span<byte> prefix = stackalloc byte[12];
        stream.ReadExactly(prefix);
        if (!prefix[..8].SequenceEqual("NNQ35LR1"u8))
            throw new InvalidDataException("Not an NNtrain Qwen3.5 LoRA checkpoint.");
        int headerLength = BinaryPrimitives.ReadInt32LittleEndian(prefix[8..]);
        if (headerLength < 1 || headerLength > 1024 * 1024 || headerLength > stream.Length - 44)
            throw new InvalidDataException("Invalid LoRA header length.");
        byte[] header = new byte[headerLength];
        stream.ReadExactly(header);
        using JsonDocument json = JsonDocument.Parse(header);
        return json.RootElement.TryGetProperty("TrainingIdentity", out JsonElement identity)
            ? identity.GetString() : null;
    }
}
