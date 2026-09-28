using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace NNtrain;

internal static class QwenGgufCommand
{
    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        try
        {
            string? modelPath = null, prompt = null, adapterPath = null;
            int device = 0, maxNewTokens = 16;
            bool stream = true;
            int[]? devices = null;
            bool explicitDevice = false;
            for (int i = 1; i < args.Length; ++i)
            {
                string option = args[i];
                if (option == "--no-stream") { stream = false; continue; }
                if (i + 1 >= args.Length)
                    throw new ArgumentException($"Missing value for {option}.");
                string value = args[++i];
                switch (option)
                {
                    case "--model": modelPath = Path.GetFullPath(value); break;
                    case "--adapter": adapterPath = Path.GetFullPath(value); break;
                    case "--prompt": prompt = value; break;
                    case "--device": device = ParseNonnegativeInteger(value, option); explicitDevice = true; break;
                    case "--devices": devices = value.Split(',').Select(index => ParseNonnegativeInteger(index, option)).ToArray(); break;
                    case "--max-new-tokens": maxNewTokens = ParseNonnegativeInteger(value, option); break;
                    default: throw new ArgumentException($"Unknown qwen-gguf option: {option}");
                }
            }
            if (modelPath is null || prompt is null)
                throw new ArgumentException(
                    "Usage: qwen-gguf --model <model.gguf> --prompt <text> " +
                    "[--device 0 | --devices 0,1] [--max-new-tokens 16] [--no-stream] [--adapter <qwen35.adapter.bin>]");
            if (!File.Exists(modelPath)) throw new FileNotFoundException("GGUF model not found.", modelPath);
            if (device < 0 || maxNewTokens < 0)
                throw new ArgumentOutOfRangeException("Device and max-new-tokens must be nonnegative.");
            if (explicitDevice && devices is not null)
                throw new ArgumentException("Use either --device or --devices.");

            long startupStarted = Stopwatch.GetTimestamp();
            using var gguf = new GgufReader(modelPath);
            string architecture = gguf.Metadata.TryGetValue("general.architecture", out object? architectureValue)
                ? architectureValue as string ?? "" : "";
            if (architecture == "qwen35")
            {
                if (adapterPath is not null)
                {
                    QwenLoraCommand.ValidateDistinctPaths(modelPath, adapterPath);
                    if (!File.Exists(adapterPath)) throw new FileNotFoundException("LoRA adapter not found.", adapterPath);
                }
                return RunQwen35(gguf, prompt, maxNewTokens,
                    devices ?? (explicitDevice ? [device] : null), stream, output, error, startupStarted, adapterPath);
            }
            if (adapterPath is not null)
                throw new NotSupportedException("--adapter currently supports qwen35 GGUF models only.");
            if (devices is not null)
            {
                if (devices.Length != 1 || devices[0] < 0)
                    throw new ArgumentException("Qwen2 currently accepts one Arc device; --devices 0,1 is for Qwen3.5.");
                device = devices[0];
            }

            var loadTimer = Stopwatch.StartNew();
            Qwen2GgufDescriptor descriptor = Qwen2Gguf.Inspect(modelPath);
            Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(gguf);
            output.WriteLine(
                $"Qwen GGUF: layers={descriptor.LayerCount}, width={descriptor.EmbeddingLength}, " +
                $"heads={descriptor.HeadCount}/{descriptor.KvHeadCount}, vocab={descriptor.VocabularySize}");
            output.WriteLine(
                $"token ids: {string.Join(',', tokenizer.Encode(prompt))}");

            using var arc = Tensor.BeginArcExecution(
                device, TensorPrecisionMode.Float32);
            using Qwen2QuantizedForCausalLM model =
                Qwen2Gguf.LoadQuantizedModel(modelPath);
            model.to(new TorchDevice(TensorDevice.Arc, device));
            loadTimer.Stop();
            GenerationTiming timing = WriteGeneration(prompt, tokenizer, stream,
                onToken => model.GenerateStreaming(
                    prompt, tokenizer, maxNewTokens, onToken,
                    temperature: 0f, topK: 1, random: new Random(1)), output);
            string first = timing.FirstTokenMilliseconds.HasValue
                ? timing.FirstTokenMilliseconds.Value.ToString("F1", CultureInfo.InvariantCulture) + " ms"
                : "n/a";
            string decode = timing.DecodeTokensPerSecond.HasValue
                ? timing.DecodeTokensPerSecond.Value.ToString("F2", CultureInfo.InvariantCulture) + " tok/s"
                : "n/a";
            error.WriteLine(FormattableString.Invariant(
                $"timing: load={loadTimer.Elapsed.TotalSeconds:F3} s, prefill/first-token={first}, decode={decode}, generated={timing.GeneratedTokens}, generation={timing.TotalMilliseconds / 1000:F3} s"));
            error.Flush();
            return 0;
        }
        catch (Exception exception) when (
            exception is not OutOfMemoryException and not StackOverflowException)
        {
            error.WriteLine($"Error: {exception.Message}");
            error.WriteLine(exception.StackTrace);
            return 2;
        }
    }

    internal static int ParseNonnegativeInteger(string value, string option)
    {
        string normalized = value.Normalize(NormalizationForm.FormKC);
        if (!int.TryParse(normalized, NumberStyles.Integer, CultureInfo.InvariantCulture, out int result)
            || result < 0)
            throw new ArgumentException($"{option} must be a nonnegative integer; got '{value}'.");
        return result;
    }

    // The delegate is the model's single generation call. Keeping display
    // separate lets tests verify output occurs while that call is still active.
    internal static GenerationTiming WriteGeneration(
        string prompt, Qwen2GgufTokenizer tokenizer, bool stream,
        Func<Action<int>, string> generate, TextWriter output)
    {
        Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();
        if (stream)
        {
            output.WriteLine("generated:");
            output.Write(tokenizer.Decode(tokenizer.Encode(prompt)));
            output.Flush();
        }
        int count = 0;
        double? first = null;
        double last = 0;
        long started = Stopwatch.GetTimestamp();
        string generated = generate(token =>
        {
            last = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            first ??= last;
            ++count;
            if (!stream) return;
            WriteChunk(decoder.Append(token));
        });
        double total = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        if (stream)
        {
            WriteChunk(decoder.Complete());
            output.WriteLine();
        }
        else
        {
            output.WriteLine("generated:");
            output.WriteLine(generated);
        }
        output.Flush();
        double? decodeRate = count > 1 && last > first.GetValueOrDefault()
            ? (count - 1) * 1000.0 / (last - first!.Value) : null;
        return new GenerationTiming(count, total, first, decodeRate);

        void WriteChunk(string text)
        {
            if (text.Length == 0) return;
            output.Write(text);
            output.Flush();
        }
    }

    internal sealed record GenerationTiming(
        int GeneratedTokens,
        double TotalMilliseconds,
        double? FirstTokenMilliseconds,
        double? DecodeTokensPerSecond);

    private static int RunQwen35(GgufReader gguf, string prompt, int maxNewTokens,
        int[]? devices, bool stream, TextWriter output, TextWriter error, long startupStarted, string? adapterPath = null)
    {
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(gguf);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(gguf);
        int[] promptIds = tokenizer.Encode(prompt);
        if (promptIds.Length == 0 || promptIds.Length > d.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.");
        output.WriteLine($"Qwen3.5 GGUF: layers={d.LayerCount}, width={d.EmbeddingLength}, " +
            $"heads={d.HeadCount}/{d.KvHeadCount}, head-width={d.HeadWidth}, vocab={d.VocabularySize}");
        output.WriteLine($"token ids: {string.Join(',', promptIds)}");
        if (maxNewTokens == 0 && adapterPath is null)
        {
            output.WriteLine("generated:");
            output.WriteLine(tokenizer.Decode(promptIds));
            return 0;
        }
        output.WriteLine("Text inference: quantized weights, attention, recurrent state and greedy sampling on Arc (Float32).");
        output.WriteLine("GPU KV cache and Gated DeltaNet state reuse: enabled.");
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(gguf, devices, output.WriteLine,
            new() { ComputeModelFingerprintOnLoad = adapterPath is not null });
        if (adapterPath is not null)
        {
            model.LoadLora(adapterPath);
            output.WriteLine($"LoRA adapter = {adapterPath}, step {model.LoraStep}, parameters={model.LoraParameterCount}");
        }
        double loadSeconds = Stopwatch.GetElapsedTime(startupStarted).TotalSeconds;
        int[] generated = [];
        GenerationTiming timing = WriteGeneration(prompt, tokenizer, stream, onToken =>
        {
            generated = model.GenerateTokenIds(promptIds, maxNewTokens, tokenizer.EosTokenId, onToken);
            return tokenizer.Decode(generated);
        }, output);
        if (!stream)
            output.WriteLine($"generated token ids: {string.Join(',', generated.Skip(promptIds.Length))}");
        string decode = timing.DecodeTokensPerSecond.HasValue
            ? FormattableString.Invariant($"{timing.DecodeTokensPerSecond.Value:F2} tok/s") : "n/a";
        string firstToken = timing.FirstTokenMilliseconds.HasValue
            ? FormattableString.Invariant($"{timing.FirstTokenMilliseconds.Value / 1000:F3} s") : "n/a";
        error.WriteLine(FormattableString.Invariant(
            $"timing: load={loadSeconds:F3} s, prefill/first-token={firstToken}, decode={decode}, generated={timing.GeneratedTokens}, total-generation={timing.TotalMilliseconds / 1000:F3} s"));
        error.Flush();
        return 0;
    }
}
