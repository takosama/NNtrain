using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using System.Reflection;
using Xunit;

public sealed class ArcGenerationKvCacheTests
{
    // Relative RMS limits cover the existing Arc BF16/BFP8 publication and
    // differing QK reduction order: 0.01% Float32, 0.5% BF16, 2% BFP8.
    // Every logit is compared, not just the winner.
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false, 1e-4)]
    [InlineData(TensorPrecisionMode.Mix16_32, false, 5e-3)]
    [InlineData(TensorPrecisionMode.Mix8_32, false, 2e-2)]
    [InlineData(TensorPrecisionMode.Mix8_16, false, 2e-2)]
    [InlineData(TensorPrecisionMode.Float32, true, 1e-4)]
    [InlineData(TensorPrecisionMode.Mix16_32, true, 5e-3)]
    [InlineData(TensorPrecisionMode.Mix8_32, true, 2e-2)]
    [InlineData(TensorPrecisionMode.Mix8_16, true, 2e-2)]
    public void EveryCachedStepMatchesFullLastLogitsForDifferentTokens(
        TensorPrecisionMode precision, bool tensorParallel, double maximumRelativeRms)
    {
        RequireDevices(tensorParallel);
        float[][] reference = CollectLastLogits(precision, tensorParallel, cached: false);
        float[][] candidate = CollectLastLogits(precision, tensorParallel, cached: true);
        Assert.Equal(reference.Length, candidate.Length);
        for (int step = 0; step < reference.Length; ++step)
        {
            float[] full = reference[step], cached = candidate[step];
            Assert.Equal(full.Length, cached.Length);
            Assert.All(full, value => Assert.True(float.IsFinite(value)));
            Assert.All(cached, value => Assert.True(float.IsFinite(value)));
            Assert.True(full.Max() - full.Min() > 1e-4f,
                "Seeded untied model produced trivial logits.");
            double squaredError = 0, squaredReference = 0;
            for (int i = 0; i < full.Length; ++i)
            {
                double difference = (double)cached[i] - full[i];
                squaredError += difference * difference;
                squaredReference += (double)full[i] * full[i];
            }
            double relativeRms = Math.Sqrt(squaredError / squaredReference);
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"{precision}, TP={tensorParallel}, step={step}, relative RMS={relativeRms:G6}");
            Assert.True(relativeRms <= maximumRelativeRms,
                $"{precision}, TP={tensorParallel}, step={step}: logits relative RMS " +
                $"{relativeRms:G6} exceeds {maximumRelativeRms:G6}.");
        }
    }

    private static float[][] CollectLastLogits(
        TensorPrecisionMode precision, bool tensorParallel, bool cached)
    {
        using IDisposable execution = BeginExecution(precision, tensorParallel, cached);
        // Untied output weights avoid a trivial repeated embedding winner.
        var model = new GptRinWikiJp(64, 12, 64, 4, 128, 2,
            new Random(157), dropout: 0f, tieWordEmbeddings: false);
        model.to(precision, 32);
        model.eval();
        model.ArcTensorParallelEnabled = tensorParallel;
        var rows = new List<float[]>();
        int[] prompt = [1, 5, 9, 13], appended = [21, 7, 31, 2];
        var prefix = new List<int>(prompt);
        using var noGrad = AutogradContext.NoGrad();
        if (!cached)
        {
            for (int step = 0; step <= appended.Length; ++step)
            {
                using (Tensor.BeginArcInferenceFrame())
                    rows.Add(model.forward(prefix.ToArray(), 1, prefix.Count).Data
                        .TakeLast(model.VocabularySize).ToArray());
                if (step < appended.Length) prefix.Add(appended[step]);
            }
            return rows.ToArray();
        }

        Linear head = (Linear)typeof(GptRinWikiJp)
            .GetField("_languageModelHead", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(model)!;
        MethodInfo cachedForward = typeof(GptRinWikiJp).GetMethod(
            tensorParallel ? "ForwardHiddenArcTensorParallel" : "ForwardHiddenArcCached",
            BindingFlags.Instance | BindingFlags.NonPublic)!;
        var owned = new List<ArcAttentionKvCache>();
        ArcAttentionKvCache[] MakeCaches(int device)
        {
            using var scope = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Arc, device));
            var caches = new ArcAttentionKvCache[2];
            for (int i = 0; i < caches.Length; ++i)
            {
                caches[i] = new ArcAttentionKvCache(
                    tensorParallel ? 32 : 64, tensorParallel ? 2 : 4, capacity: 8);
                owned.Add(caches[i]);
            }
            return caches;
        }

        try
        {
            ArcAttentionKvCache[] first = MakeCaches(0);
            ArcAttentionKvCache[]? second = tensorParallel ? MakeCaches(1) : null;
            for (int step = 0; step <= appended.Length; ++step)
            {
                int position = step == 0 ? 0 : prefix.Count - 1;
                int[] input = step == 0 ? prompt : [appended[step - 1]];
                using (Tensor.BeginArcInferenceFrame())
                {
                    Tensor hidden = tensorParallel
                        ? (Tensor)cachedForward.Invoke(model,
                            [input, input.Length, first, second, position])!
                        : (Tensor)cachedForward.Invoke(model, [input, position, first])!;
                    rows.Add(head.ForwardBatch(hidden.SelectLastSequenceToken()).Data.ToArray());
                }
                if (step < appended.Length) prefix.Add(appended[step]);
            }
            return rows.ToArray();
        }
        finally
        {
            foreach (ArcAttentionKvCache cache in owned) cache.Dispose();
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_16, true)]
    public void CachedGreedyMatchesFullGenerationAndStreamsInOrder(
        TensorPrecisionMode precision, bool tensorParallel)
    {
        RequireDevices(tensorParallel);
        int[] prompt = [1, 5, 9, 13];
        int[] full = Generate(precision, tensorParallel, cached: false,
            contextLength: 12, prompt, newTokens: 4);
        int[] cached = Generate(precision, tensorParallel, cached: true,
            contextLength: 12, prompt, newTokens: 4);
        Assert.Equal(full, cached);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, true)]
    public void SeededSamplingMatchesFullGenerationAndCallbackOrder(
        TensorPrecisionMode precision, bool tensorParallel)
    {
        RequireDevices(tensorParallel);
        int[] prompt = [1, 5, 9, 13];
        int[] full = Generate(precision, tensorParallel, cached: false,
            contextLength: 12, prompt, newTokens: 4, temperature: .75f, topK: 4);
        int[] cached = Generate(precision, tensorParallel, cached: true,
            contextLength: 12, prompt, newTokens: 4, temperature: .75f, topK: 4);
        Assert.Equal(full, cached);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_16, true)]
    public void CachedGenerationMatchesFullAfterSlidingWindowStarts(
        TensorPrecisionMode precision, bool tensorParallel)
    {
        RequireDevices(tensorParallel);
        int[] prompt = [1, 5, 9, 13];
        int[] full = Generate(precision, tensorParallel, cached: false,
            contextLength: 6, prompt, newTokens: 4);
        int[] cached = Generate(precision, tensorParallel, cached: true,
            contextLength: 6, prompt, newTokens: 4);
        Assert.Equal(full, cached);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, true)]
    public void RepeatedCachedGenerationsReleaseAllCacheBuffers(
        TensorPrecisionMode precision, bool tensorParallel)
    {
        RequireDevices(tensorParallel);
        using IDisposable execution = BeginExecution(precision, tensorParallel, cached: true);
        GptRinWikiJp model = CreateModel(precision, tensorParallel, contextLength: 12);
        ArcExecutionLane[] lanes = Lanes(tensorParallel);
        int[] prompt = [1, 5, 9, 13];

        int[] first = RunGeneration(model, prompt, newTokens: 4);
        foreach (ArcExecutionLane lane in lanes) lane.Synchronize();
        long[] retained = lanes.Select(lane => lane.AllocatedBytes).ToArray();
        int[] second = RunGeneration(model, prompt, newTokens: 4);
        foreach (ArcExecutionLane lane in lanes) lane.Synchronize();

        Assert.Equal(first, second);
        Assert.Equal(retained, lanes.Select(lane => lane.AllocatedBytes).ToArray());
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, true)]
    public void StopZeroTokenAndCallbackFailureReleaseCaches(
        TensorPrecisionMode precision, bool tensorParallel)
    {
        RequireDevices(tensorParallel);
        using IDisposable execution = BeginExecution(precision, tensorParallel, cached: true);
        GptRinWikiJp model = CreateModel(precision, tensorParallel, contextLength: 12);
        ArcExecutionLane[] lanes = Lanes(tensorParallel);
        int[] prompt = [1, 5, 9, 13];
        int[] first = RunGeneration(model, prompt, newTokens: 1);
        foreach (ArcExecutionLane lane in lanes) lane.Synchronize();
        long[] retained = lanes.Select(lane => lane.AllocatedBytes).ToArray();

        var streamed = new List<int>();
        int[] unchanged = model.GenerateTokenIds(prompt, 0, 0f, 1,
            stopTokenId: null, random: new Random(73), onToken: streamed.Add);
        Assert.Equal(prompt, unchanged);
        Assert.Empty(streamed);

        int[] stopped = model.GenerateTokenIds(prompt, 4, 0f, 1,
            stopTokenId: first[^1], random: new Random(73), onToken: streamed.Add);
        Assert.Equal(first, stopped);
        Assert.Equal(new[] { first[^1] }, streamed.ToArray());
        foreach (ArcExecutionLane lane in lanes) lane.Synchronize();
        Assert.Equal(retained, lanes.Select(lane => lane.AllocatedBytes).ToArray());

        var failure = new InvalidOperationException("stream consumer failed");
        InvalidOperationException actual = Assert.Throws<InvalidOperationException>(() =>
            model.GenerateTokenIds(prompt, 4, 0f, 1, stopTokenId: null,
                random: new Random(73), onToken: _ => throw failure));
        Assert.Same(failure, actual);
        foreach (ArcExecutionLane lane in lanes) lane.Synchronize();
        Assert.Equal(retained, lanes.Select(lane => lane.AllocatedBytes).ToArray());
    }

    private static int[] Generate(TensorPrecisionMode precision, bool tensorParallel,
        bool cached, int contextLength, int[] prompt, int newTokens,
        float temperature = 0f, int topK = 1)
    {
        using IDisposable execution = BeginExecution(precision, tensorParallel, cached);
        GptRinWikiJp model = CreateModel(precision, tensorParallel, contextLength);
        ArcExecutionLane[] lanes = Lanes(tensorParallel);
        long[] launches = lanes.Select(lane => lane.KernelLaunchCount).ToArray();
        int[] generated = RunGeneration(model, prompt, newTokens, temperature, topK);
        Assert.Equal(prompt.Length + newTokens, generated.Length);
        for (int i = 0; i < lanes.Length; ++i)
            Assert.True(lanes[i].KernelLaunchCount > launches[i],
                $"Arc device {lanes[i].DeviceIndex} did not execute generation kernels.");
        return generated;
    }

    private static int[] RunGeneration(GptRinWikiJp model, int[] prompt, int newTokens,
        float temperature = 0f, int topK = 1)
    {
        var streamed = new List<int>();
        int[] generated = model.GenerateTokenIds(prompt, newTokens,
            temperature: temperature, topK: topK, stopTokenId: null,
            random: new Random(73), onToken: streamed.Add);
        Assert.Equal(generated.Skip(prompt.Length).ToArray(), streamed.ToArray());
        return generated;
    }

    private static GptRinWikiJp CreateModel(
        TensorPrecisionMode precision, bool tensorParallel, int contextLength)
    {
        var model = new GptRinWikiJp(64, contextLength, 64, 4, 128, 2,
            new Random(43), dropout: 0f, tieWordEmbeddings: true);
        model.to(precision, 32);
        model.eval();
        model.ArcTensorParallelEnabled = tensorParallel;
        return model;
    }

    private static IDisposable BeginExecution(
        TensorPrecisionMode precision, bool tensorParallel, bool cached)
    {
        // Keep the reference independent of every new inference optimization.
        // The candidate exercises their production combination, not KV alone.
        var options = new ArcExecutionOptions
        {
            InferenceKvCache = cached,
            InferenceGemv = cached,
            InferencePackedEmbedding = cached,
            InferenceSmallRowNorm = cached,
            InferenceFusedGemv = cached,
        };
        return tensorParallel
            ? Tensor.BeginArcInferenceExecution([0, 1], precision, options)
            : Tensor.BeginArcExecution(0, precision, options);
    }

    private static ArcExecutionLane[] Lanes(bool tensorParallel)
    {
        ExecutionSession session = ExecutionSession.Current!;
        return tensorParallel
            ? [
                (ArcExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Arc, 0),
                (ArcExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Arc, 1),
            ]
            : [Tensor.ArcLane];
    }

    private static void RequireDevices(bool tensorParallel)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        if (tensorParallel)
            Assert.SkipWhen(!Tensor.IsArcAvailable(1), "Two Intel Arc GPUs are required.");
    }
}
