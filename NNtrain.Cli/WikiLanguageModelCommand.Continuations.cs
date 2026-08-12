namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static DatasetContinuation CreateDatasetContinuation(
        IWikiLanguageModel model,
        BpeTokenizer tokenizer,
        IReadOnlyList<string> documents,
        WikiTrainingConfiguration config,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);
        if (documents.Count == 0)
        {
            throw new ArgumentException(
                "At least one dataset sample document is required.",
                nameof(documents));
        }

        string document = documents[random.Next(documents.Count)];
        if (!TryGetDocumentSplit(document, out int split))
        {
            throw new ArgumentException(
                "Dataset sample document must have non-empty first and " +
                "second halves.",
                nameof(documents));
        }

        string prompt = document[..split];
        string expected = document[split..];
        int[] promptIds = tokenizer.Encode(prompt, addBos: true);
        int[] generatedIds = model.GenerateTokenIds(
            promptIds,
            config.MaxNewTokens,
            config.Temperature,
            config.TopK,
            BpeTokenizer.EosTokenId,
            random);
        string generated = tokenizer.Decode(
            generatedIds.Skip(promptIds.Length));

        const int displayCharacters = 400;
        return new DatasetContinuation(
            document.Length,
            split,
            TakeTail(prompt, displayCharacters),
            TakeHead(expected, displayCharacters),
            generated);
    }

    private static void WriteDatasetContinuation(
        long step,
        DatasetContinuation sample,
        TextWriter output)
    {
        output.WriteLine();
        output.WriteLine(
            $"dataset continuation sample at step {step:N0} " +
            $"(split {sample.SplitIndex:N0}/{sample.DocumentLength:N0} chars):");
        output.WriteLine("[prompt: first half, final context]");
        output.WriteLine(sample.PromptTail);
        output.WriteLine("[dataset continuation: second half excerpt]");
        output.WriteLine(sample.ExpectedContinuation);
        output.WriteLine("[model continuation]");
        output.WriteLine(sample.GeneratedContinuation);
        output.WriteLine();
    }

    private static void WriteFinalDatasetContinuation(
        IWikiLanguageModel model,
        BpeTokenizer tokenizer,
        IReadOnlyList<string> documents,
        WikiTrainingConfiguration config,
        Random random,
        long step,
        TextWriter output)
    {
        if (documents.Count == 0)
        {
            output.WriteLine(
                "dataset continuation unavailable: no splittable sample " +
                "document was retained");
            return;
        }

        DatasetContinuation sample = CreateDatasetContinuation(
            model,
            tokenizer,
            documents,
            config,
            random);
        WriteDatasetContinuation(step, sample, output);
    }

    private static bool TryGetDocumentSplit(
        string document,
        out int split)
    {
        split = document.Length / 2;
        if (split > 0
            && split < document.Length
            && char.IsLowSurrogate(document[split]))
        {
            split--;
        }
        return split > 0 && split < document.Length;
    }

    internal readonly record struct DatasetContinuation(
        int DocumentLength,
        int SplitIndex,
        string PromptTail,
        string ExpectedContinuation,
        string GeneratedContinuation);
}
