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
            var seen = new HashSet<string>(StringComparer.Ordinal);
            for (int i = 1; i < args.Length; ++i)
            {
                string option = args[i];
                if (!seen.Add(option)) throw new ArgumentException($"Duplicate qwen-lora option: {option}");
                if (option == "--resume") { resume = true; continue; }
                if (option is not ("--model" or "--config"))
                    throw new ArgumentException($"Unknown qwen-lora option: {option}");
                if (++i == args.Length) throw new ArgumentException($"Missing value for {option}.");
                if (option == "--model") modelPath = Path.GetFullPath(args[i]);
                else configPath = Path.GetFullPath(args[i]);
            }
            if (modelPath is null || configPath is null)
                throw new ArgumentException("Usage: qwen-lora --model <qwen35.gguf> --config <qwen-lora.json> [--resume]");
            QwenLoraTrainingConfiguration config = QwenLoraTrainingConfiguration.Load(configPath);
            config.Validate();
            string directory = Path.GetDirectoryName(configPath)!;
            string dataPath = Path.GetFullPath(config.DataPath, directory);
            string adapterPath = Path.GetFullPath(config.AdapterPath, directory);
            ValidateDistinctPaths(modelPath, configPath, dataPath, adapterPath);
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
            string identity = TrainingIdentity(dataset, tokenizer.EosTokenId!.Value, config);

            using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(modelPath, config.Devices,
                output.WriteLine, options: new Qwen35ExecutionOptions { LoraTraining = true });
            long[] encodedBeforeTraining = model.ResidentWeightBytes.ToArray();
            if (resume) model.LoadLora(adapterPath, identity);
            else model.AttachLora(config.AdapterOptions());
            output.WriteLine($"Qwen3.5 LoRA: rank={config.Rank}, trainable={model.LoraParameterCount}, "
                + $"examples={examples.Length}, context={config.ContextLength}, step={model.LoraStep}/{config.MaxSteps}");
            output.WriteLine("Frozen quantized GPU base; response-only next-token loss including EOS; one example per step.");
            long savedStep = resume ? model.LoraStep : -1;
            while (model.LoraStep < config.MaxSteps)
            {
                int index = (int)(model.LoraStep % examples.Length);
                TrainingExample example = examples[index];
                var timer = Stopwatch.StartNew();
                Qwen35LoraStepResult result = model.TrainLora(example.Tokens, example.ResponseStartIndex);
                output.WriteLine(FormattableString.Invariant(
                    $"LoRA step={result.Step}, example={index + 1}/{examples.Length}, loss={result.Loss:F6}, gradient-norm={result.GradientNorm:G6}, supervised-tokens={result.SupervisedTokens}, elapsed={timer.Elapsed.TotalSeconds:F3} s"));
                output.Flush();
                if (model.LoraStep % config.SaveEverySteps == 0) Save();
            }
            if (!encodedBeforeTraining.SequenceEqual(model.ResidentWeightBytes))
                throw new InvalidOperationException("Encoded GGUF weight residency changed during LoRA training.");
            if (savedStep != model.LoraStep) Save();
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
            throw new ArgumentException("Model, configuration, dataset and adapter paths must all be distinct.");
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
        QwenLoraTrainingConfiguration config)
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
        return Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(contract)));
    }
}
