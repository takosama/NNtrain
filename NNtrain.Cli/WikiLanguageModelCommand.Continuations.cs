namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static bool ShouldGenerateDatasetContinuation(
        long committedGlobalStep,
        int everySteps,
        int retainedDocumentCount)
        => committedGlobalStep > 0
            && everySteps > 0
            && retainedDocumentCount > 0
            && committedGlobalStep % everySteps == 0;

    private static void RunDatasetContinuationAfterCommittedStep(
        long committedGlobalStep,
        LanguageModel model,
        BpeTokenizer tokenizer,
        IReadOnlyList<string> documents,
        WikiTrainingConfiguration config,
        Random random,
        TextWriter output,
        TextWriter warning)
    {
        RunDatasetContinuationAfterCommittedStep(
            committedGlobalStep,
            config.DatasetSampleEverySteps,
            documents.Count,
            () => StreamDatasetContinuation(
                committedGlobalStep,
                model,
                tokenizer,
                documents,
                config,
                random,
                output),
            warning);
    }

    internal static void RunDatasetContinuationAfterCommittedStep(
        long committedGlobalStep,
        int everySteps,
        int retainedDocumentCount,
        Action generate,
        TextWriter warning)
    {
        ArgumentNullException.ThrowIfNull(generate);
        ArgumentNullException.ThrowIfNull(warning);
        if (!ShouldGenerateDatasetContinuation(
            committedGlobalStep,
            everySteps,
            retainedDocumentCount))
        {
            return;
        }

        try
        {
            generate();
        }
        catch (Exception exception) when (
            !IsFatalDatasetContinuationFailure(exception))
        {
            warning.WriteLine(
                $"Warning: dataset continuation generation at step " +
                $"{committedGlobalStep:N0} failed: {exception.Message}");
        }
    }

    internal static bool IsFatalDatasetContinuationFailure(
        Exception exception)
    {
        ArgumentNullException.ThrowIfNull(exception);
        for (Exception? current = exception;
            current is not null;
            current = current.InnerException)
        {
            if (current is OutOfMemoryException
                or StackOverflowException
                or OperationCanceledException
                or AccessViolationException
                or System.Runtime.InteropServices.SEHException)
            {
                return true;
            }

            if (HasFatalCudaStatus(current.Message))
                return true;
        }
        return false;
    }

    private static bool HasFatalCudaStatus(string message)
    {
        ReadOnlySpan<int> fatalStatuses =
        [600, 700, 702, 710, 714, 715, 716, 717, 718, 719];
        foreach (int status in fatalStatuses)
        {
            if (message.Contains(
                $"CUDA error {status}",
                StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    internal static DatasetContinuation CreateDatasetContinuation(
        LanguageModel model,
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
        int[] generatedIds = model.generate_token_ids(
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

    /// <summary>
    /// Writes a dataset-continuation sample, streaming the model's own
    /// continuation to <paramref name="output"/> character by character as it
    /// is generated.
    /// </summary>
    /// <remarks>
    /// The unit is a character, not a token. The vocabulary is byte-level, so
    /// writing each token's bytes as they arrive would emit broken characters
    /// wherever a multi-byte character straddles a merge boundary.
    /// </remarks>
    internal static void StreamDatasetContinuation(
        long step,
        LanguageModel model,
        BpeTokenizer tokenizer,
        IReadOnlyList<string> documents,
        WikiTrainingConfiguration config,
        Random random,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(random);
        ArgumentNullException.ThrowIfNull(output);
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
                "Dataset sample document must have non-empty first and "
                + "second halves.",
                nameof(documents));
        }

        string prompt = document[..split];
        string expected = document[split..];
        const int displayCharacters = 400;

        output.WriteLine();
        output.WriteLine(
            $"dataset continuation sample at step {step:N0} "
            + $"(split {split:N0}/{document.Length:N0} chars):");
        output.WriteLine("[prompt: first half, final context]");
        output.WriteLine(TakeTail(prompt, displayCharacters));
        output.WriteLine("[dataset continuation: second half excerpt]");
        output.WriteLine(TakeHead(expected, displayCharacters));
        output.WriteLine("[model continuation]");
        output.Flush();

        BpeTokenizer.IncrementalDecoder decoder =
            tokenizer.CreateIncrementalDecoder();
        int[] promptIds = tokenizer.Encode(prompt, addBos: true);
        model.generate_token_ids(
            promptIds,
            config.MaxNewTokens,
            config.Temperature,
            config.TopK,
            BpeTokenizer.EosTokenId,
            random,
            token =>
            {
                string text = decoder.Append(token);
                foreach (char character in text)
                {
                    output.Write(character);
                    output.Flush();
                }
            });
        output.Write(decoder.Flush());
        output.WriteLine();
        output.WriteLine();
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
        LanguageModel model,
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

        StreamDatasetContinuation(
            step,
            model,
            tokenizer,
            documents,
            config,
            random,
            output);
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
