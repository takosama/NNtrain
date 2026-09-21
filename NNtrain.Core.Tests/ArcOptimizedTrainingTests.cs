using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcOptimizedTrainingTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void OptimizedTrainingPreservesLossGradientsUpdatesAndResidency(
        TensorPrecisionMode precision, bool muon)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        AssertTrainingClose(RunTraining(precision, muon, false, false),
            RunTraining(precision, muon, true, false));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void BlockNormIntegratedTrainingPreservesExistingTransformerTolerance(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        // Check residual/dropout fusion separately from the other optimized
        // paths. The fused kernels retain the legacy serial reduction order;
        // this is the existing transformer comparison tolerance, not a new
        // relaxed low-precision tolerance.
        AssertTrainingClose(RunTraining(precision, false, false, false),
            RunTraining(precision, false, true, true));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void D32AttentionSpecialTailPreservesExactOutputAndAccumulatedGradient(
        TensorPrecisionMode precision, bool causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 3, sequence = 137, width = 128, heads = 4;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray();
        (float[][] Outputs, float[] Gradient) Run(bool optimized)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: Options(optimized, false));
            ArcExecutionLane lane = Tensor.ArcLane;
            var input = new Tensor((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(precision.ToStorageDType(),
                precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Mix8_32 : null);
            var outputs = new float[2][];
            long retained = -1;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Tensor output = input.FusedMultiHeadAttention(heads, causal);
                outputs[repeat] = output.Data.ToArray();
                float[] seed = Enumerable.Range(0, batch * sequence * width)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .01f).ToArray();
                long downloads = lane.D2HBytes;
                output.BackwardAndRelease(seed);
                lane.Synchronize();
                Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(0, lane.RetiredBytes);
                Assert.InRange(lane.CachedBytes, 0, lane.Options.BufferPoolBytes);
                if (repeat == 0) retained = lane.AllocatedBytes;
                else Assert.Equal(retained, lane.AllocatedBytes);
            }
            if (optimized)
            {
                Assert.Contains("attention_fp32_pv_d32_special", lane.KernelTimings.Keys);
                Assert.Contains("attention_fp32_dp_d32_special", lane.KernelTimings.Keys);
                Assert.Contains("attention_fp32_dq_d32_special", lane.KernelTimings.Keys);
                Assert.Contains("attention_dkv_fused_candidate", lane.KernelTimings.Keys);
            }
            return (outputs, input.Grad.ToArray());
        }
        var expected = Run(false);
        var actual = Run(true);
        for (int i = 0; i < expected.Outputs.Length; i++) Assert.Equal(expected.Outputs[i], actual.Outputs[i]);
        Assert.Equal(expected.Gradient, actual.Gradient);
    }

    private static StepSnapshot[] RunTraining(TensorPrecisionMode precision, bool muon, bool optimized, bool blockNorm)
    {
        const int batch = 4, sequence = 128, width = 128, heads = 4, vocabulary = 129, microbatches = 2;
        using var execution = Tensor.BeginArcExecution(precision: precision, options: Options(optimized, blockNorm));
        ArcExecutionLane lane = Tensor.ArcLane;
        var model = new GptRinWikiJp(vocabulary, sequence, width, heads, 384, 2, new Random(57),
            dropout: .1f, tieWordEmbeddings: true);
        model.to(precision, 32);
        Parameter[] parameters = model.parameters().ToArray();
        IOptimizer[] optimizers = muon
            ? [new NekoMuon(model.HiddenWeightParameters, new NekoMuonOptions {
                    LearningRate = .001f, WeightDecay = .01f, BetaFast = .95f, Nesterov = true,
                    NewtonSchulzInterval = 1, NewtonSchulzDepthMode = NekoMuonNewtonSchulzDepthMode.Fixed,
                    NewtonSchulzDepth = 5 }),
                new AdamW(model.AuxiliaryParameters, new AdamWOptions { LearningRate = .0003f, WeightDecay = .01f })]
            : [new AdamW(parameters, new AdamWOptions { LearningRate = .0003f, WeightDecay = .01f })];
        var snapshots = new StepSnapshot[2];
        long retained = -1;
        try
        {
            for (int step = 0; step < snapshots.Length; step++)
            {
                long upload = lane.H2DBytes, download = lane.D2HBytes;
                foreach (IOptimizer optimizer in optimizers) optimizer.zero_grad();
                var losses = new float[microbatches];
                for (int micro = 0; micro < microbatches; micro++)
                {
                    int[] tokens = Enumerable.Range(0, batch * sequence)
                        .Select(i => (i * 7 + step * 11 + micro * 3) % vocabulary).ToArray();
                    int[] labels = Enumerable.Range(0, tokens.Length)
                        .Select(i => i % 17 == 0 ? -1 : (tokens[i] + 1 + micro) % vocabulary).ToArray();
                    Tensor loss = model.forward_loss(tokens, labels, batch, sequence);
                    losses[micro] = loss.item();
                    Assert.True(float.IsFinite(losses[micro]));
                    loss.BackwardAndRelease([1f / microbatches]);
                }
                float norm = nn.utils.clip_grad_norm_(parameters, 1f);
                Assert.True(float.IsFinite(norm));
                foreach (IOptimizer optimizer in optimizers) optimizer.step();
                lane.Synchronize();
                Assert.Equal(0, lane.RetiredBytes);
                Assert.InRange(lane.CachedBytes, 0, lane.Options.BufferPoolBytes);
                if (step == 0) retained = lane.AllocatedBytes;
                else
                {
                    // Inspect weights and gradients only outside this measured
                    // region. A warm update needs token/target uploads and the
                    // established loss/clip/Muon scalar diagnostics, nothing else.
                    Assert.Equal(microbatches * 2L * batch * sequence * sizeof(int), lane.H2DBytes - upload);
                    long scalarBytes = microbatches * sizeof(float) + 12L
                        + (muon ? model.HiddenWeightParameters.Count * 20L : 0);
                    Assert.Equal(scalarBytes, lane.D2HBytes - download);
                    Assert.Equal(retained, lane.AllocatedBytes);
                }
                snapshots[step] = new(losses, norm,
                    parameters.Select(p => p.T.Grad.ToArray()).ToArray(),
                    parameters.Select(p => p.T.CaptureData(preferMaster: true)).ToArray());
            }
            Assert.Contains(Enumerable.Range(0, parameters.Length),
                p => !snapshots[0].Masters[p].SequenceEqual(snapshots[1].Masters[p]));
            Assert.All(parameters, p => Assert.Equal(precision.ToStorageDType(), p.T.DType));
            Assert.Equal(optimized && blockNorm, lane.KernelTimings.ContainsKey("norm_residual_serial"));
            Assert.Equal(optimized && !blockNorm, lane.KernelTimings.ContainsKey("norm_residual_back_accumulate"));
            if (optimized)
            {
                Assert.Contains("attention_fp32_pv_d32_aligned", lane.KernelTimings.Keys);
                Assert.Contains("attention_fp32_dp_d32_aligned", lane.KernelTimings.Keys);
                Assert.Contains("attention_fp32_dq_d32_aligned", lane.KernelTimings.Keys);
                Assert.Contains("attention_dkv_d32_k32_q32_causal", lane.KernelTimings.Keys);
            }
            return snapshots;
        }
        finally
        {
            foreach (IOptimizer optimizer in optimizers)
                if (optimizer is IDisposable disposable) disposable.Dispose();
        }
    }

    private static ArcExecutionOptions Options(bool optimized, bool blockNorm) => new()
    {
        // Freeze both sides explicitly: changing a production default must
        // never silently turn the legacy comparison into the candidate route.
        InlineMatrixGradient = optimized,
        DirectXmxMatrices = optimized,
        PackedMatrixStorage = optimized,
        FusedAttentionDkv = optimized,
        TunedFp32Attention = optimized,
        LruBufferPool = optimized,
        ReuseRetiredBuffers = optimized,
        BlockResidualNorm = optimized && blockNorm,
        FusedNormGradient = optimized,
        // DQ/PV fusion supersedes the tuned kernels rather than composing with
        // them. Exercise the selected tiled candidate, not those A/B alternatives.
        FusedAttentionDq = false,
        FusedAttentionPv = false,
        SubgroupAttentionReduction = false,
        QueuedKernelLimit = optimized ? 512 : 128,
        BufferPoolBytes = 64L * 1024 * 1024,
    };

    private static void AssertTrainingClose(StepSnapshot[] expected, StepSnapshot[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int step = 0; step < expected.Length; step++)
        {
            AssertClose(expected[step].Losses, actual[step].Losses);
            Assert.InRange(MathF.Abs(expected[step].Norm - actual[step].Norm), 0, 1e-4f);
            Assert.Equal(expected[step].Gradients.Length, actual[step].Gradients.Length);
            Assert.Equal(expected[step].Masters.Length, actual[step].Masters.Length);
            for (int p = 0; p < expected[step].Gradients.Length; p++)
            {
                AssertClose(expected[step].Gradients[p], actual[step].Gradients[p]);
                AssertClose(expected[step].Masters[p], actual[step].Masters[p]);
            }
        }
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0, 1e-4f);
    }

    private sealed record StepSnapshot(float[] Losses, float Norm, float[][] Gradients, float[][] Masters);
}
