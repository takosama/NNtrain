using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;

namespace NNtrain.Benchmarks;

/// <summary>A bounded, read-only corpus probe. Never writes training artifacts.</summary>
internal sealed record DrnRealDataProbe(CudaLanguageModelMicroBatch[] Training,
    CudaLanguageModelMicroBatch[] Evaluation, string TokenHash, double LoadSeconds)
{
    internal static DrnRealDataProbe Load(WikiTrainingConfiguration config, int updates)
    {
        var timer = Stopwatch.StartNew();
        BpeTokenizer tokenizer = BpeTokenizer.Load(config.TokenizerPath);
        if (tokenizer.VocabularySize != config.VocabularySize)
            throw new InvalidDataException("Probe will not retrain or overwrite a mismatched tokenizer.");
        int sequence = config.ContextLength;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var evaluationDocuments = new HashSet<string>(StringComparer.Ordinal);
        IAsyncEnumerator<string> source = WikiParquetCorpus.ReadTextsAsync(
            config.DataPath, config.TextColumn, shuffleSeed: config.Seed ^ 0x57A913)
            .GetAsyncEnumerator();
        int documents = 0;
        try
        {
            CudaLanguageModelMicroBatch[] Collect(int count, int batch, bool evaluation)
            {
                var buffer = new List<int>();
                var result = new List<CudaLanguageModelMicroBatch>(count);
                int length = checked(batch * sequence);
                while (result.Count < count)
                {
                    if (++documents > 100_000 || !source.MoveNextAsync().AsTask().GetAwaiter().GetResult())
                        throw new InvalidDataException("Not enough documents within the bounded probe.");
                    string text = source.Current;
                    string documentHash = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text)));
                    if (evaluation) evaluationDocuments.Add(documentHash);
                    else if (evaluationDocuments.Contains(documentHash)) continue;
                    WikiLanguageModelCommand.AppendDocument(buffer, tokenizer, text, config.MaxDocumentTokens);
                    while (result.Count < count && buffer.Count > length)
                    {
                        var values = WikiLanguageModelCommand.CreateStreamingBatch(buffer, batch, sequence);
                        result.Add(new(values.Input, values.Target, batch, sequence));
                        hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.Input.AsSpan()));
                        hash.AppendData(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.Target.AsSpan()));
                    }
                    if (documents % 500 == 0)
                        Console.WriteLine($"real corpus loading: {documents} documents, {result.Count}/{count} batches");
                }
                // Drop the remainder, so no document straddles probe train/eval.
                return result.ToArray();
            }
            var evaluation = Collect(8, 4, evaluation: true);
            var training = Collect(checked(updates * config.GradientAccumulationSteps), config.BatchSize, evaluation: false);
            return new(training, evaluation, Convert.ToHexString(hash.GetHashAndReset()), timer.Elapsed.TotalSeconds);
        }
        finally { source.DisposeAsync().AsTask().GetAwaiter().GetResult(); }
    }

    internal static float Evaluate(LanguageModel model, CudaLanguageModelMicroBatch[] batches)
    {
        bool wasTraining = model.IsTraining;
        model.eval();
        try
        {
            using IDisposable noGrad = AutogradContext.NoGrad();
            double sum = 0;
            foreach (var batch in batches)
            {
                using CudaInferenceScope scope = CudaInferenceScope.Begin(resetPool: true, clearPoolOnDispose: true);
                sum += model.ForwardLoss(batch.Input, batch.Target, batch.BatchSize, batch.SequenceLength).item();
            }
            return (float)(sum / batches.Length);
        }
        finally { if (wasTraining) model.train(); }
    }
}
