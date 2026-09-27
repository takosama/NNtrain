namespace NNtrain;

internal static class QwenGgufCommand
{
    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        try
        {
            string? modelPath = null, prompt = null;
            int device = 0, maxNewTokens = 16;
            int[]? devices = null;
            bool explicitDevice = false;
            for (int i = 1; i < args.Length; ++i)
            {
                string option = args[i];
                if (i + 1 >= args.Length)
                    throw new ArgumentException($"Missing value for {option}.");
                string value = args[++i];
                switch (option)
                {
                    case "--model": modelPath = Path.GetFullPath(value); break;
                    case "--prompt": prompt = value; break;
                    case "--device": device = int.Parse(value); explicitDevice = true; break;
                    case "--devices": devices = value.Split(',').Select(int.Parse).ToArray(); break;
                    case "--max-new-tokens": maxNewTokens = int.Parse(value); break;
                    default: throw new ArgumentException($"Unknown qwen-gguf option: {option}");
                }
            }
            if (modelPath is null || prompt is null)
                throw new ArgumentException(
                    "Usage: qwen-gguf --model <model.gguf> --prompt <text> " +
                    "[--device 0 | --devices 0,1] [--max-new-tokens 16]");
            if (!File.Exists(modelPath)) throw new FileNotFoundException("GGUF model not found.", modelPath);
            if (device < 0 || maxNewTokens < 0)
                throw new ArgumentOutOfRangeException("Device and max-new-tokens must be nonnegative.");
            if (explicitDevice && devices is not null)
                throw new ArgumentException("Use either --device or --devices.");

            string architecture;
            using (var gguf = new GgufReader(modelPath))
                architecture = gguf.Metadata.TryGetValue("general.architecture", out object? value)
                    ? value as string ?? "" : "";
            if (architecture == "qwen35")
                return RunQwen35(modelPath, prompt, maxNewTokens,
                    devices ?? (explicitDevice ? [device] : null), output);
            if (devices is not null)
            {
                if (devices.Length != 1 || devices[0] < 0)
                    throw new ArgumentException("Qwen2 currently accepts one Arc device; --devices 0,1 is for Qwen3.5.");
                device = devices[0];
            }

            Qwen2GgufDescriptor descriptor = Qwen2Gguf.Inspect(modelPath);
            Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(modelPath);
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
            string generated = model.Generate(
                prompt, tokenizer, maxNewTokens,
                temperature: 0f, topK: 1, random: new Random(1));
            output.WriteLine("generated:");
            output.WriteLine(generated);
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

    private static int RunQwen35(string path, string prompt, int maxNewTokens,
        int[]? devices, TextWriter output)
    {
        Qwen35GgufDescriptor d = Qwen35Gguf.Inspect(path);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(path);
        int[] promptIds = tokenizer.Encode(prompt);
        if (promptIds.Length == 0 || promptIds.Length > d.ContextLength)
            throw new ArgumentException("Prompt must contain 1..context-length tokens.");
        output.WriteLine($"Qwen3.5 GGUF: layers={d.LayerCount}, width={d.EmbeddingLength}, " +
            $"heads={d.HeadCount}/{d.KvHeadCount}, head-width={d.HeadWidth}, vocab={d.VocabularySize}");
        output.WriteLine($"token ids: {string.Join(',', promptIds)}");
        if (maxNewTokens == 0)
        {
            output.WriteLine("generated:");
            output.WriteLine(tokenizer.Decode(promptIds));
            return 0;
        }
        output.WriteLine("Text inference: quantized weights, attention, recurrent state and greedy sampling on Arc (Float32).");
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(path, devices, output.WriteLine);
        int[] generated = model.GenerateTokenIds(promptIds, maxNewTokens, tokenizer.EosTokenId);
        output.WriteLine($"generated token ids: {string.Join(',', generated.Skip(promptIds.Length))}");
        output.WriteLine("generated:");
        output.WriteLine(tokenizer.Decode(generated));
        return 0;
    }
}
