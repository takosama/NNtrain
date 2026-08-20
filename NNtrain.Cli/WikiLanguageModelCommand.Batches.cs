using System.Runtime.InteropServices;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static LanguageBatch CreateBatch(
        IReadOnlyList<int> tokens,
        IReadOnlyList<int> sequenceOrder,
        int orderStart,
        int count,
        int contextLength)
    {
        var input = new int[checked(count * contextLength)];
        var target = new int[input.Length];
        Array.Fill(target, Tensor.DefaultCrossEntropyIgnoreIndex);
        int validTargetCount = 0;
        if (tokens is int[] tokenArray)
        {
            void CopySequence(int item)
            {
                int sequence = sequenceOrder[orderStart + item];
                int tokenStart = checked(sequence * contextLength);
                int validLength = Math.Min(
                    contextLength,
                    tokenArray.Length - tokenStart - 1);
                if (validLength <= 0)
                    return;
                Array.Copy(
                    tokenArray,
                    tokenStart,
                    input,
                    item * contextLength,
                    validLength);
                Array.Copy(
                    tokenArray,
                    tokenStart + 1,
                    target,
                    item * contextLength,
                    validLength);
                Interlocked.Add(ref validTargetCount, validLength);
            }

            for (int item = 0; item < count; item++)
                CopySequence(item);
            return new LanguageBatch(input, target, validTargetCount);
        }

        for (int item = 0; item < count; item++)
        {
            int sequence = sequenceOrder[orderStart + item];
            int tokenStart = checked(sequence * contextLength);
            int validLength = Math.Min(
                contextLength,
                tokens.Count - tokenStart - 1);
            for (int position = 0; position < validLength; position++)
            {
                input[item * contextLength + position] =
                    tokens[tokenStart + position];
                target[item * contextLength + position] =
                    tokens[tokenStart + position + 1];
            }
            validTargetCount += validLength;
        }
        return new LanguageBatch(input, target, validTargetCount);
    }

    /// <summary>Keeps the shuffle stream independent of other seeded draws.</summary>
    internal const int ShuffleSeedSalt = 0x5A17;

    /// <summary>
    /// Randomizes the order of a document stream with a fixed-size shuffle
    /// buffer, so training does not consume the corpus in file order.
    /// </summary>
    /// <remarks>
    /// The corpus does not fit in memory, so this is local mixing rather than
    /// a true permutation: a document can move at most about
    /// <paramref name="bufferSize"/> positions. That is enough to break the
    /// dump's page-id ordering within a batch, which is what correlates
    /// consecutive training windows. The order is fully determined by
    /// <paramref name="random"/>, so a resumed run that replays the epoch and
    /// skips the documents it already processed sees the same sequence.
    /// </remarks>
    internal static IEnumerable<string> ShuffleDocuments(
        IEnumerable<string> documents,
        int bufferSize,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(documents);
        ArgumentNullException.ThrowIfNull(random);
        if (bufferSize < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(bufferSize),
                bufferSize,
                "The shuffle buffer size must be non-negative.");
        }

        if (bufferSize <= 1)
        {
            foreach (string document in documents)
                yield return document;
            yield break;
        }

        var buffer = new List<string>(bufferSize);
        foreach (string document in documents)
        {
            if (buffer.Count < bufferSize)
            {
                buffer.Add(document);
                continue;
            }

            int index = random.Next(buffer.Count);
            yield return buffer[index];
            buffer[index] = document;
        }

        // Drain what is left in a random order rather than in arrival order.
        for (int remaining = buffer.Count; remaining > 0; remaining--)
        {
            int index = random.Next(remaining);
            yield return buffer[index];
            buffer[index] = buffer[remaining - 1];
        }
    }

    /// <summary>
    /// Appends one document to the streaming token buffer as
    /// <c>&lt;bos&gt; tokens &lt;eos&gt;</c>.
    /// </summary>
    /// <param name="maxDocumentTokens">
    /// Zero keeps the whole document. A positive value truncates it, and a
    /// truncated document deliberately gets no <c>&lt;eos&gt;</c>: the article
    /// did not end there, so marking an end teaches the model to stop
    /// mid-sentence. The finite-corpus path already behaves this way when it
    /// runs out of its token budget.
    /// </param>
    /// <returns>Whether the document was written in full.</returns>
    internal static bool AppendDocument(
        List<int> buffer,
        BpeTokenizer tokenizer,
        string document,
        int maxDocumentTokens)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        ArgumentNullException.ThrowIfNull(tokenizer);
        ArgumentNullException.ThrowIfNull(document);
        if (maxDocumentTokens < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxDocumentTokens),
                maxDocumentTokens,
                "The document token limit must be non-negative.");
        }

        buffer.Add(BpeTokenizer.BosTokenId);
        int[] documentTokens = tokenizer.Encode(document);
        bool truncated = maxDocumentTokens > 0
            && documentTokens.Length > maxDocumentTokens;
        int tokenCount = truncated ? maxDocumentTokens : documentTokens.Length;
        for (int index = 0; index < tokenCount; index++)
            buffer.Add(documentTokens[index]);
        if (!truncated)
            buffer.Add(BpeTokenizer.EosTokenId);
        return !truncated;
    }

    internal static string FormatDocumentTokenLimit(int maxDocumentTokens)
        => maxDocumentTokens > 0
            ? $"up to {maxDocumentTokens:N0} tokens/document (longer " +
                "documents are truncated without <eos>)"
            : "whole documents, no truncation";

    internal static LanguageBatch CreateStreamingBatch(
        List<int> buffer,
        int batchSize,
        int sequenceLength)
    {
        int targetCount = checked(batchSize * sequenceLength);
        if (buffer.Count < 2)
        {
            throw new ArgumentException(
                "Streaming token buffer does not contain a token pair.",
                nameof(buffer));
        }

        var input = new int[targetCount];
        var target = new int[targetCount];
        Array.Fill(target, Tensor.DefaultCrossEntropyIgnoreIndex);
        int validTargetCount = Math.Min(targetCount, buffer.Count - 1);
        Span<int> values = CollectionsMarshal.AsSpan(buffer);
        values[..validTargetCount].CopyTo(input);
        values.Slice(1, validTargetCount).CopyTo(target);
        buffer.RemoveRange(0, validTargetCount);
        return new LanguageBatch(input, target, validTargetCount);
    }

    private static IEnumerable<string> ReadDocuments(
        string path,
        string textColumn,
        int? maxDocuments)
    {
        IAsyncEnumerator<string> enumerator = datasets
            .wikipedia(
                root: path,
                text_column: textColumn,
                max_documents: maxDocuments)
            .GetAsyncEnumerator();
        try
        {
            while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
                yield return enumerator.Current;
        }
        finally
        {
            enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    private static void AddReservoirSample(
        string document,
        List<string> samples,
        ref int eligibleDocuments,
        int capacity,
        Random random)
    {
        eligibleDocuments++;
        if (samples.Count < capacity)
        {
            samples.Add(document);
            return;
        }

        int replacement = random.Next(eligibleDocuments);
        if (replacement < samples.Count)
            samples[replacement] = document;
    }

    internal readonly record struct LanguageBatch(
        int[] Input,
        int[] Target,
        int ValidTargetCount);
}
