using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcFlashAttentionTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 65, 32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, 17, 32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, 65, 128, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, 65, 32, false)]
    public void UnsupportedShapesAndDisabledXmxKeepTheExistingArcPath(TensorPrecisionMode precision, int sequence, int d, bool xmx)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        float[][] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                FlashAttention = enabled, FlashAttentionXmxProducts = true, XmxMatrices = xmx });
            var input = new Tensor(Enumerable.Range(0, sequence * d * 3)
                .Select(i => MathF.Sin(i * .013f) * .1f).ToArray(), [1, sequence, 3 * d]);
            input.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            var output = input.FusedMultiHeadAttention(1, true);
            float[] values = output.Data.ToArray();
            output.BackwardAndRelease(Enumerable.Repeat(.01f, sequence * d).ToArray());
            Assert.DoesNotContain(Tensor.ArcLane.KernelTimings.Keys, k => k.StartsWith("attention_flash_"));
            return [values,input.Grad.ToArray()];
        }
        var expected = Run(false);var actual = Run(true);
        Assert.Equal(expected[0],actual[0]);Assert.Equal(expected[1],actual[1]);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void PublicResultUsesTheRequestedStorageAndTheExistingQuantizer(TensorPrecisionMode precision)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
            FlashAttention = true, FlashAttentionXmxProducts = true, FlashAttentionAsyncCopy = true });
        const int batch = 2, sequence = 65, heads = 3, width = 96;
        var input = new Tensor(Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .0317f) * .7f).ToArray(), [batch, sequence, 3 * width]);
        var descriptor = precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null;
        input.ConvertStorageInPlace(precision.ToStorageDType(), descriptor);
        var raw = input.ArcFlashAttention(batch, sequence, width, heads, true, TensorDType.Float32);
        float[] rawValues = raw.Data.ToArray();
        raw.BackwardAndRelease(new float[batch * sequence * width]);
        var expected = new Tensor(rawValues, [batch, sequence, width]);
        expected.ConvertStorageInPlace(precision.ToStorageDType(), descriptor);
        var actual = input.FusedMultiHeadAttention(heads, true);
        Assert.Equal(precision.ToStorageDType(), actual.DType);
        Assert.Equal(expected.Data.ToArray(), actual.Data.ToArray());
        actual.BackwardAndRelease(new float[batch * sequence * width]);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [Trait("Category", "ArcAttentionExperimentalAcceptance")]
    public void ModelLossGradientUpdateAndCheckpointReplayStayWithinExistingTolerance(TensorPrecisionMode precision, bool fused)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        (float[] Losses, float[][] Gradients, float[][] Weights) Run(bool flash, bool checkpoint)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                FlashAttention = flash && fused, XmxAttentionProducts = flash && !fused,
                FlashAttentionXmxProducts = true, FlashAttentionAsyncCopy = true,
                TransformerCheckpointing = checkpoint });
            var model = new GptRinWikiJp(129, 65, 96, 3, 288, 2, new Random(57), dropout: .1f, tieWordEmbeddings: true);
            model.to(precision, 32);
            var parameters = model.parameters().ToArray();
            var optimizer = new AdamW(parameters, new AdamWOptions { LearningRate = .0003f, WeightDecay = .01f });
            var losses = new List<float>();
            long retained = -1;
            for (int step = 0; step < 3; step++)
            {
                optimizer.zero_grad();
                long upload = Tensor.ArcLane.H2DBytes, download = Tensor.ArcLane.D2HBytes;
                for (int micro = 0; micro < 2; micro++)
                {
                    int[] tokens = Enumerable.Range(0, 130).Select(i => (i * 7 + step * 11 + micro * 3) % 129).ToArray();
                    var loss = model.forward_loss(tokens, tokens.Select(i => (i + 1) % 129).ToArray(), 2, 65);
                    losses.Add(loss.item());
                    loss.BackwardAndRelease([.5f]);
                }
                optimizer.step();Tensor.ArcLane.Synchronize();
                if (step == 0) retained = Tensor.ArcLane.AllocatedBytes;
                else {
                    Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
                    Assert.Equal(2L * 2 * 130 * sizeof(int), Tensor.ArcLane.H2DBytes - upload);
                    Assert.Equal(2 * sizeof(float), Tensor.ArcLane.D2HBytes - download);
                }
            }
            return (losses.ToArray(), parameters.Select(p => p.T.Grad.ToArray()).ToArray(),
                parameters.Select(p => p.T.CaptureData(true)).ToArray());
        }
        void Close(float[] a, float[] b) {
            Assert.Equal(a.Length,b.Length);
            for (int i=0;i<a.Length;i++) Assert.InRange(MathF.Abs(a[i]-b[i]),0,1e-4f);
        }
        var expected = Run(false, false);
        foreach (bool checkpoint in new[] { false, true })
        {
            var actual = Run(true, checkpoint);
            Close(expected.Losses, actual.Losses);
            for (int p=0;p<expected.Gradients.Length;p++) {
                Close(expected.Gradients[p],actual.Gradients[p]);Close(expected.Weights[p],actual.Weights[p]);
            }
        }
    }

    public static IEnumerable<object[]> Cases()
    {
        foreach (var precision in new[] { TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
        foreach (bool causal in new[] { true, false })
        foreach (var (sequence, d) in new[] { (33, 7), (65, 17), (137, 32), (129, 35), (128, 64), (1024, 32) })
            yield return [precision, causal, sequence, d];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FlashStagesPreserveForwardAccumulatedGradientAndResidency(TensorPrecisionMode precision, bool causal, int sequence, int d)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
        const int batch = 2, heads = 3;
        int width = heads * d;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .0317f) * .2f).ToArray();
        float[][] Run(int stage)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                FlashAttention = stage is > 0 and < 4, FlashAttentionXmxProducts = stage > 1, FlashAttentionAsyncCopy = stage == 3,
                XmxAttentionProducts = stage >= 4, DirectXmxAttentionProducts = stage == 5 });
            var input = new Tensor((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            float[] outputValues = [];
            long allocated = -1;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                long downloads = Tensor.ArcLane.D2HBytes, uploads = Tensor.ArcLane.H2DBytes;
                // Compare FP32 before storage quantization: a tiny association
                // change at a BF16/BFP8 rounding boundary is one storage ULP,
                // not the underlying arithmetic error. Gradients remain FP32.
                var output = stage == 0 || stage >= 4
                    ? input.ArcBatchedAttention(batch, sequence, width, heads, causal, TensorDType.Float32)
                    : input.ArcFlashAttention(batch, sequence, width, heads, causal, TensorDType.Float32);
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
                if (repeat > 0) Assert.Equal(uploads, Tensor.ArcLane.H2DBytes);
                outputValues = output.Data.ToArray();
                downloads = Tensor.ArcLane.D2HBytes;
                output.BackwardAndRelease(Enumerable.Range(0, batch * sequence * width)
                    .Select(i => MathF.Cos(i * .0173f + repeat) * .1f).ToArray());
                Tensor.ArcLane.Synchronize();
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
                Assert.Equal(0, Tensor.ArcLane.RetiredBytes);
                if (repeat == 0) allocated = Tensor.ArcLane.AllocatedBytes;
                else Assert.Equal(allocated, Tensor.ArcLane.AllocatedBytes);
            }
            return [outputValues, input.Grad.ToArray()];
        }
        var expected = Run(0);
        for (int stage = 1; stage <= 5; stage++)
        {
            var actual = Run(stage);
            for (int field = 0; field < 2; field++)
                for (int i = 0; i < expected[field].Length; i++)
                    Assert.True(float.IsFinite(actual[field][i]) && MathF.Abs(expected[field][i] - actual[field][i]) <= 1e-4f,
                        $"stage={stage} field={field} i={i}: {expected[field][i]} vs {actual[field][i]}");
        }
    }
}
