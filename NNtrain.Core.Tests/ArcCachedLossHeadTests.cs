using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcCachedLossHeadTests
{
    [Fact]
    public void CacheAdmissionBoundsRetainedBytesAndLiveHeadroom()
    {
        const ulong global = 12UL * 1024 * 1024 * 1024;
        const ulong maximumAllocation = 4UL * 1024 * 1024 * 1024;
        Assert.True(Tensor.CanCacheArcLossHeadLogits(16384, 11500,
            5L * 1024 * 1024 * 1024, global, maximumAllocation));
        Assert.False(Tensor.CanCacheArcLossHeadLogits(25000, 11500,
            5L * 1024 * 1024 * 1024, global, maximumAllocation));
        Assert.False(Tensor.CanCacheArcLossHeadLogits(16384, 11500,
            9L * 1024 * 1024 * 1024, global, maximumAllocation));
        Assert.False(Tensor.CanCacheArcLossHeadLogits(16384, 11500,
            5L * 1024 * 1024 * 1024, global, 128UL * 1024 * 1024));
        Assert.False(Tensor.CanCacheArcLossHeadLogits(16384, 11500,
            5L * 1024 * 1024 * 1024, global, maximumAllocation,
            6L * 1024 * 1024 * 1024));
    }

    [Fact]
    public void LegacyPoolAdmissionCountsNonEvictableCachedAndRetiredBytes()
    {
        const ulong global = 12UL * 1024 * 1024 * 1024;
        const ulong maximumAllocation = 4UL * 1024 * 1024 * 1024;
        const long gib = 1024L * 1024 * 1024;
        long allocated = 8 * gib, cached = gib * 3 / 4, retired = gib / 4;

        Assert.True(Tensor.CanCacheArcLossHeadLogits(16384, 11500,
            Tensor.ArcLossHeadAdmissionLiveBytes(allocated, cached, retired, true),
            global, maximumAllocation));
        Assert.False(Tensor.CanCacheArcLossHeadLogits(16384, 11500,
            Tensor.ArcLossHeadAdmissionLiveBytes(allocated, cached, retired, false),
            global, maximumAllocation));
    }

    [Fact]
    public void CachedBf16LogitsPreserveLossAndAccumulatedGradients()
    {
        RequireXmx();
        const int rows = 640, width = 65, vocabulary = 133;
        Snapshot Run(bool cached)
        {
            using var execution = Tensor.BeginArcExecution(
                precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    DirectXmxMatrices = true,
                    PackedMatrixStorage = true,
                    InlineMatrixGradient = true,
                    MixedBackwardMatrixOperands = true,
                    LossChunkRows = 512,
                    Mix8_16CachedLossLogits = cached,
                    Mix8_16ReluDualGradientPack = false,
                    Mix8_16BiasOnlyGradientReduction = false,
                });
            Tensor input = Make([rows, width], 1);
            Tensor weight = Make([vocabulary, width], 2);
            Tensor bias = Make([vocabulary], 3);
            float[] losses = new float[2];
            long[] retained = new long[2];
            long[] forwardLive = new long[2];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                int[] labels = Enumerable.Range(0, rows)
                    .Select(i => i % 17 == 0 ? -1 : (i * 13 + repeat) % vocabulary).ToArray();
                Tensor loss = input.ArcLinearCrossEntropy(weight, bias, labels, -1);
                losses[repeat] = loss.item();
                forwardLive[repeat] = Tensor.ArcLane.AllocatedBytes;
                loss.BackwardAndRelease([repeat == 0 ? .375f : .625f]);
                Tensor.ArcLane.Synchronize();
                retained[repeat] = Tensor.ArcLane.AllocatedBytes;
            }
            Assert.Equal(retained[0], retained[1]);
            return new(losses,
                [input.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()],
                forwardLive,
                Tensor.ArcLane.KernelTimings.ContainsKey("loss_head_pack_bf16_at"),
                Tensor.ArcLane.KernelTimings.ContainsKey("loss_head_unpack_bf16_at"));
        }

        Snapshot baseline = Run(false), candidate = Run(true);
        Assert.False(baseline.Packed);
        Assert.False(baseline.Unpacked);
        Assert.True(candidate.Packed);
        Assert.True(candidate.Unpacked);
        Assert.Equal((long)rows * vocabulary * sizeof(ushort)
            + (long)rows * 2 * sizeof(float),
            candidate.ForwardLive[0] - baseline.ForwardLive[0]);
        Assert.Equal(baseline.Losses, candidate.Losses);
        for (int i = 0; i < baseline.Gradients.Length; i++)
            Assert.Equal(baseline.Gradients[i], candidate.Gradients[i]);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix8_16, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    public void CacheDoesNotRunWithoutGraphOrMatchingPrecision(
        TensorPrecisionMode precision, bool noGrad)
    {
        RequireXmx();
        const int rows = 256, width = 65, vocabulary = 133;
        using var execution = Tensor.BeginArcExecution(precision: precision,
            options: new ArcExecutionOptions
            {
                DirectXmxMatrices = true,
                PackedMatrixStorage = true,
                MixedBackwardMatrixOperands = true,
                LossChunkRows = 256,
                Mix8_16CachedLossLogits = true,
            });
        Tensor input = Make([rows, width], 1);
        Tensor weight = Make([vocabulary, width], 2);
        Tensor bias = Make([vocabulary], 3);
        int[] labels = Enumerable.Range(0, rows).Select(i => i % vocabulary).ToArray();
        if (noGrad)
        {
            using (AutogradContext.NoGrad())
                _ = input.ArcLinearCrossEntropy(weight, bias, labels, -1).item();
        }
        else
        {
            Tensor loss = input.ArcLinearCrossEntropy(weight, bias, labels, -1);
            loss.BackwardAndRelease();
        }
        Tensor.ArcLane.Synchronize();
        Assert.Contains("cross_entropy_rows", Tensor.ArcLane.KernelTimings.Keys);
        Assert.DoesNotContain("loss_head_pack_bf16_at", Tensor.ArcLane.KernelTimings.Keys);
        Assert.DoesNotContain("loss_head_unpack_bf16_at", Tensor.ArcLane.KernelTimings.Keys);
    }

    private static Tensor Make(int[] shape, int phase)
    {
        int count = shape.Aggregate(1, (a, b) => checked(a * b));
        var tensor = new Tensor(Enumerable.Range(0, count)
            .Select(i => MathF.Sin(i * .013f + phase) * .031f).ToArray(), shape);
        tensor.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
        return tensor;
    }

    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }

    private sealed record Snapshot(float[] Losses, float[][] Gradients, long[] ForwardLive,
        bool Packed, bool Unpacked);
}
