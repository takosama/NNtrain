namespace NNtrain;

internal static class QwenGgufCommand
{
    internal static int Run(string[] args, TextWriter output, TextWriter error)
    {
        try
        {
            string? modelPath = null, prompt = null;
            int device = 0, maxNewTokens = 16;
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
                    case "--device": device = int.Parse(value); break;
                    case "--max-new-tokens": maxNewTokens = int.Parse(value); break;
                    default: throw new ArgumentException($"Unknown qwen-gguf option: {option}");
                }
            }
            if (modelPath is null || prompt is null)
                throw new ArgumentException(
                    "Usage: qwen-gguf --model <model.gguf> --prompt <text> " +
                    "[--device 0] [--max-new-tokens 16]");
            if (!File.Exists(modelPath)) throw new FileNotFoundException("GGUF model not found.", modelPath);
            if (device < 0 || maxNewTokens < 0)
                throw new ArgumentOutOfRangeException("Device and max-new-tokens must be nonnegative.");

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
}
