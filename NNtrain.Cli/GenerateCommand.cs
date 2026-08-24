namespace NNtrain;

internal static class GenerateCommand
{
    internal static int Run(
        string generationConfigurationPath,
        TextWriter output,
        TextWriter error)
    {
        try
        {
            GenerationConfiguration generation =
                GenerationConfiguration.Load(generationConfigurationPath);
            WikiTrainingConfiguration modelConfiguration =
                WikiTrainingConfiguration.Load(generation.TrainingConfigPath);
            string tokenizerPath = generation.TokenizerPath
                ?? modelConfiguration.TokenizerPath;

            if (!File.Exists(tokenizerPath))
                throw new FileNotFoundException("BPE tokenizer file was not found.", tokenizerPath);
            if (!File.Exists(generation.SafeTensorsPath))
                throw new FileNotFoundException("SafeTensors file was not found.", generation.SafeTensorsPath);

            torch.manual_seed(generation.Seed ?? modelConfiguration.Seed);
            Tensor.SimdEnabled = modelConfiguration.UseSimd;
            Tensor.MaxDegreeOfParallelism = modelConfiguration.MaxDegreeOfParallelism;
            Tensor.ExecutionDevice = modelConfiguration.GetExecutionDevice();
            Tensor.CudaDeviceIndices = modelConfiguration.DeviceIndices
                ?? [modelConfiguration.DeviceIndex];

            BpeTokenizer tokenizer = tokenizers.load_bpe(tokenizerPath);
            LanguageModel model = WikiLanguageModelCommand.CreateModel(
                modelConfiguration,
                tokenizer.VocabularySize);
            model.load_state_dict(
                safetensors.torch.load_file(generation.SafeTensorsPath));
            if (Tensor.ExecutionDevice == TensorDevice.Cuda)
                model.to(TensorDevice.Cuda);

            int topK = generation.IsGreedy ? 1 : generation.TopK;
            float temperature = generation.IsGreedy
                ? 0f
                : generation.EffectiveTemperature;
            output.WriteLine($"safetensors = {generation.SafeTensorsPath}");
            output.WriteLine(
                $"sampling = {(generation.IsGreedy ? "greedy" : $"topK ({topK})")}, " +
                $"temperature = {temperature}");
            output.WriteLine("generated text:");
            output.Write(generation.Prompt);
            output.Flush();

            StreamGeneration(
                model,
                tokenizer,
                generation.Prompt,
                generation.MaxNewTokens,
                temperature,
                topK,
                new Random((generation.Seed ?? modelConfiguration.Seed) ^ 0x27D4EB2D),
                output);
            output.WriteLine();
            return 0;
        }
        catch (Exception exception) when (
            exception is not OutOfMemoryException
            and not StackOverflowException
            and not OperationCanceledException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }

    internal static void StreamGeneration(
        LanguageModel model,
        BpeTokenizer tokenizer,
        string prompt,
        int maxNewTokens,
        float temperature,
        int topK,
        Random random,
        TextWriter output)
    {
        var tokenIds = tokenizer.Encode(prompt, addBos: true).ToList();
        BpeTokenizer.IncrementalDecoder decoder =
            tokenizer.CreateIncrementalDecoder();

        void WriteToken(int token)
        {
            if (token == BpeTokenizer.EosTokenId)
                return;
            string text = decoder.Append(token);
            if (text.Length == 0)
                return;
            output.Write(text);
            output.Flush();
        }

        // ForgetMemory models expose their recurrent token loop through the
        // callback, preserving O(prompt + generated) inference while streaming.
        if (model is ForgetMemoryV2Gpt)
        {
            model.generate_token_ids(
                tokenIds,
                maxNewTokens,
                temperature,
                topK,
                BpeTokenizer.EosTokenId,
                random,
                WriteToken);
            output.Write(decoder.Flush());
            output.Flush();
            return;
        }

        for (int generated = 0; generated < maxNewTokens; generated++)
        {
            int[] result = model.generate_token_ids(
                tokenIds,
                1,
                temperature,
                topK,
                BpeTokenizer.EosTokenId,
                random);
            if (result.Length == tokenIds.Count)
                break;
            int token = result[^1];
            tokenIds.Add(token);
            if (token == BpeTokenizer.EosTokenId)
                break;
            WriteToken(token);
        }
        output.Write(decoder.Flush());
        output.Flush();
    }
}
