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
