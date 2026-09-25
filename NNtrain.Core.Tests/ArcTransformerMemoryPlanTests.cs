using NNtrain;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class ArcTransformerMemoryPlanTests
{
    private const ulong GiB = 1024UL * 1024 * 1024;
    private const long Parameters = 90_507_500;
    private const long LargestParameter = 20_000_000;
    private const long LargestBackwardProducerLeaves = 21_000_000;
    private const long LargestMuonScratch = 30_000_000;

    [Fact]
    public void CurrentB16FitsWithoutRecomputation()
    {
        ArcTransformerMemoryPlan plan = Plan(16, TensorDType.Bfp8, 12 * GiB);
        Assert.Equal(0, plan.CheckpointPrefixLayers);
        Assert.False(plan.CheckpointFfn);
        Assert.Equal(plan.UncheckpointedActivationBytes, plan.EstimatedActivationBytes);
        Assert.True(plan.FitsEstimatedBudget);
    }

    [Fact]
    public void CurrentB64SelectsFfnAndMinimumFullPrefix()
    {
        ArcTransformerMemoryPlan plan = Plan(64, TensorDType.Bfp8, 12 * GiB);
        Assert.True(plan.CheckpointFfn);
        Assert.InRange(plan.CheckpointPrefixLayers, 9, 12);
        Assert.True(plan.FitsEstimatedBudget);
        Assert.InRange(plan.EstimatedActivationBytes, 6L * (long)GiB, 8L * (long)GiB);
        long fullBlockIncrement = 7L * 64 * 1024 * 512 * 36 / 32 + 64L * 1024 * (16 * 8 + 16);
        Assert.True(plan.EstimatedActivationBytes + fullBlockIncrement > plan.ActivationBudgetBytes);
    }

    [Theory]
    [InlineData(TensorDType.Float32)]
    [InlineData(TensorDType.BFloat16)]
    [InlineData(TensorDType.Bfp8)]
    public void LowerMemoryNeverReducesRecomputation(TensorDType dtype)
    {
        ArcTransformerMemoryPlan large = Plan(64, dtype, 24 * GiB);
        ArcTransformerMemoryPlan medium = Plan(64, dtype, 12 * GiB);
        ArcTransformerMemoryPlan small = Plan(64, dtype, 4 * GiB);
        Assert.True(large.CheckpointPrefixLayers <= medium.CheckpointPrefixLayers);
        Assert.True(medium.CheckpointPrefixLayers <= small.CheckpointPrefixLayers);
        Assert.True(large.EstimatedActivationBytes >= medium.EstimatedActivationBytes);
        Assert.True(medium.EstimatedActivationBytes >= small.EstimatedActivationBytes);
        Assert.Equal(32, small.CheckpointPrefixLayers);
        Assert.True(small.CheckpointFfn);
        Assert.False(small.FitsEstimatedBudget);
    }

    [Fact]
    public void WiderStorageAccountsForBothActivationsAndPersistentWeights()
    {
        ArcTransformerMemoryPlan mix8 = Plan(64, TensorDType.Bfp8, 12 * GiB);
        ArcTransformerMemoryPlan bf16 = Plan(64, TensorDType.BFloat16, 12 * GiB);
        ArcTransformerMemoryPlan fp32 = Plan(64, TensorDType.Float32, 12 * GiB);
        Assert.True(mix8.UncheckpointedActivationBytes < bf16.UncheckpointedActivationBytes);
        Assert.True(bf16.UncheckpointedActivationBytes < fp32.UncheckpointedActivationBytes);
        Assert.True(mix8.EstimatedPersistentBytes < bf16.EstimatedPersistentBytes);
        Assert.True(bf16.EstimatedPersistentBytes < fp32.EstimatedPersistentBytes);
        Assert.True(mix8.CheckpointPrefixLayers < bf16.CheckpointPrefixLayers);
        Assert.True(bf16.CheckpointPrefixLayers < fp32.CheckpointPrefixLayers);
    }

    [Fact]
    public void Mix8_16AccountsForPackedStateAndTemporaryFloatGradient()
    {
        ArcTransformerMemoryPlan mix8_32 = Plan(16, TensorDType.Bfp8, 12 * GiB, PrecisionMode.Mix8_32);
        ArcTransformerMemoryPlan mix8_16 = Plan(16, TensorDType.Bfp8, 12 * GiB, PrecisionMode.Mix8_16);

        Assert.Equal(mix8_32.UncheckpointedActivationBytes, mix8_16.UncheckpointedActivationBytes);
        Assert.Equal(Parameters * 8 - LargestBackwardProducerLeaves * 4,
            mix8_32.EstimatedPersistentBytes - mix8_16.EstimatedPersistentBytes);
        Assert.Equal(Parameters + (Parameters + 31) / 32 * 4 + Parameters * 8
            + LargestBackwardProducerLeaves * 4,
            mix8_16.EstimatedPersistentBytes);
        Assert.True(mix8_16.ActivationBudgetBytes >= mix8_32.ActivationBudgetBytes);
    }

    [Fact]
    public void Mix8_16ReservesTheLargerMuonScratchWhenItExceedsBackwardGradients()
    {
        long packed = Parameters + (Parameters + 31) / 32 * 4;
        ArcTransformerMemoryPlan plan = ArcTransformerMemoryPlan.Create(
            16, 1024, 512, 16, 1536, 32,
            TensorDType.Bfp8, PrecisionMode.Mix8_16, 32,
            Parameters, packed, LargestParameter, LargestBackwardProducerLeaves,
            100_000_000, 12 * GiB);
        Assert.Equal(packed + Parameters * 8 + 100_000_000,
            plan.EstimatedPersistentBytes);
    }

    [Fact]
    public void Mix8_16RowDeltaReservesSavedBFloat16OutputAcrossAllLayers()
    {
        long packedParameters = Parameters + (Parameters + 31) / 32 * 4;
        ArcTransformerMemoryPlan Create(bool retainOutput) => ArcTransformerMemoryPlan.Create(
            8, 2048, 512, 16, 1536, 32,
            TensorDType.Bfp8, PrecisionMode.Mix8_16, 32,
            Parameters, packedParameters, LargestParameter, LargestBackwardProducerLeaves,
            LargestMuonScratch, 12 * GiB, retainAttentionOutputBFloat16: retainOutput);
        var baseline = Create(false);
        var selected = Create(true);
        Assert.Equal(512L * 1024 * 1024,
            selected.UncheckpointedActivationBytes - baseline.UncheckpointedActivationBytes);
        Assert.Equal(baseline.EstimatedPersistentBytes, selected.EstimatedPersistentBytes);
        Assert.True(selected.FitsEstimatedBudget);
        Assert.False(selected.CheckpointFfn);
        Assert.Equal(0, selected.CheckpointPrefixLayers);
    }

    [Fact]
    public void Mix8_16BFloat16ActivationPlanKeepsFusedLinearOutputsPacked()
    {
        const int batch = 16, sequence = 1024, width = 512, hidden = 1536, layers = 32;
        long rows = (long)batch * sequence;
        long packedParameters = Parameters + (Parameters + 31) / 32 * 4;
        ArcTransformerMemoryPlan PlanBf16(int fusedWidth, bool fusedFfn) =>
            ArcTransformerMemoryPlan.Create(batch, sequence, width, 16, hidden, layers,
                TensorDType.Bfp8, PrecisionMode.Mix8_16, 32,
                Parameters, packedParameters, LargestParameter,
                LargestBackwardProducerLeaves, LargestMuonScratch, 12 * GiB,
                publishBFloat16Activations: true,
                fusedBfp8LinearOutputWidthPerLayer: fusedWidth,
                fusedBfp8FfnIntermediate: fusedFfn);

        ArcTransformerMemoryPlan unfused = PlanBf16(0, false);
        ArcTransformerMemoryPlan fused = PlanBf16(5 * width + hidden, true);
        long fusedElements = rows * (5 * width + hidden);
        long fusedPackedBytes = fusedElements + fusedElements / 32 * 4;
        long expectedPerLayer = rows * 3 * width * 2 + fusedPackedBytes
            + rows * 16 * 8 + rows * 16;
        long expectedOutside = rows * width * 3 * 2 + rows * 16;

        Assert.Equal(expectedPerLayer * layers + expectedOutside,
            fused.UncheckpointedActivationBytes);
        Assert.True(fused.UncheckpointedActivationBytes < unfused.UncheckpointedActivationBytes);
        Assert.Equal(0, fused.CheckpointPrefixLayers);
        Assert.False(fused.CheckpointFfn);
    }

    [Fact]
    public void PlanIsDeterministicAndIndependentOfAccumulationWarmupOrAllocations()
    {
        var first = Plan(64, TensorDType.Bfp8, 12 * GiB);
        for (int step = 0; step < 5; step++) Assert.Equal(first, Plan(64, TensorDType.Bfp8, 12 * GiB));
        var secondShape = Plan(16, TensorDType.Bfp8, 12 * GiB);
        Assert.Equal(first, Plan(64, TensorDType.Bfp8, 12 * GiB));
        Assert.False(secondShape.CheckpointFfn);
    }

    [Fact]
    public void NoMemoryReportsUnfittablePlanWithoutChangingTheRequestedShape()
    {
        ArcTransformerMemoryPlan plan = Plan(64, TensorDType.Bfp8, 0);
        Assert.Equal(32, plan.CheckpointPrefixLayers);
        Assert.Equal(0, plan.ActivationBudgetBytes);
        Assert.True(plan.EstimatedActivationBytes > 0);
        Assert.False(plan.FitsEstimatedBudget);
    }

    [Theory]
    [InlineData(0, 512, 16)]
    [InlineData(-1, 512, 16)]
    [InlineData(16, 513, 16)]
    public void InvalidShapesFailBeforeAnyDeviceAccess(int batch, int width, int heads)
    {
        Assert.ThrowsAny<ArgumentException>(() => ArcTransformerMemoryPlan.Create(
            batch, 1024, width, heads, 1536, 32, TensorDType.Bfp8,
            PrecisionMode.Mix8_32, 32, Parameters, Parameters * 9 / 8,
            LargestParameter, LargestBackwardProducerLeaves,
            LargestMuonScratch, 12 * GiB));
    }

    private static ArcTransformerMemoryPlan Plan(int batch, TensorDType dtype, ulong memory,
        PrecisionMode? precisionMode = null)
    {
        long packed = dtype switch
        {
            TensorDType.Float32 => Parameters * 4,
            TensorDType.BFloat16 => Parameters * 2,
            _ => Parameters + (Parameters + 31) / 32 * 4,
        };
        return ArcTransformerMemoryPlan.Create(batch, 1024, 512, 16, 1536, 32,
            dtype, precisionMode ?? (dtype switch
            {
                TensorDType.Float32 => PrecisionMode.Float32,
                TensorDType.BFloat16 => PrecisionMode.Mix16_32,
                _ => PrecisionMode.Mix8_32,
            }), 32, Parameters, packed, LargestParameter,
            LargestBackwardProducerLeaves, LargestMuonScratch, memory);
    }
}
