using System.Diagnostics;
using NNtrain;

namespace NNtrain.Benchmarks;

internal static class GenerationKvCacheProfiler
{
    internal static void Run(int warmup, int iterations, bool productionShape)
    {
        Tensor.ExecutionDevice = TensorDevice.Cuda;
        Tensor.CudaDeviceIndices = [0];
        int vocabulary = productionShape ? 11500 : 4096;
        int context = productionShape ? 512 : 128;
        int generatedTokens = productionShape ? 16 : 32;
        int width = productionShape ? 384 : 128;
        int heads = productionShape ? 12 : 4;
        int hidden = productionShape ? 1152 : 256;
        int layers = productionShape ? 16 : 2;
        int promptLength = productionShape ? 256 : 64;
        var model = new GptRinWikiJp(
            vocabulary,
            context,
            dModel: width,
            numHeads: heads,
            dHidden: hidden,
            numLayers: layers,
            rng: new Random(71),
            dropout: 0f,
            dtype: TensorDType.BFloat16,
            tieWordEmbeddings: true);
        model.SetPrecisionMode(TensorPrecisionMode.BFloat16);
        int[] prompt = Enumerable.Range(0, promptLength)
            .Select(index => 1 + index % (vocabulary - 1))
            .ToArray();

        (double Milliseconds, int[] Tokens) Measure(bool disableCache)
        {
            using IDisposable dispatch = CudaDispatchPolicy.Push(
                CudaDispatchPolicy.Defaults with
                {
                    DisableKvCache = disableCache,
                });
            int[] tokens = [];
            var samples = new double[iterations];
            for (int run = -warmup; run < iterations; ++run)
            {
                long start = Stopwatch.GetTimestamp();
                tokens = model.GenerateTokenIds(
                    prompt,
                    generatedTokens,
                    temperature: 0f,
                    topK: 1,
                    stopTokenId: null,
                    random: new Random(99));
                double elapsed = Stopwatch.GetElapsedTime(start)
                    .TotalMilliseconds;
                if (run >= 0)
                    samples[run] = elapsed;
            }
            return (samples.Average(), tokens);
        }

        var full = Measure(disableCache: true);
        var cached = Measure(disableCache: false);
        Console.WriteLine(
            $"generation BF16 CUDA [prompt={promptLength}," +
            $"new={generatedTokens},context={context},width={width}," +
            $"heads={heads},layers={layers}]: full-window " +
            $"{full.Milliseconds:F2} ms, K/V cache " +
            $"{cached.Milliseconds:F2} ms, speedup " +
            $"{full.Milliseconds / cached.Milliseconds:F2}x, " +
            $"greedy-match={full.Tokens.SequenceEqual(cached.Tokens)}");
    }
}
