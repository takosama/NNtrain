using NNtrain;
using Xunit;

public sealed class ArcTransformerMemoryPlanTests
{
    private const ulong GiB = 1024UL * 1024 * 1024;
    private const long Parameters = 90_507_500;

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
            batch, 1024, width, heads, 1536, 32, TensorDType.Bfp8, 32, Parameters, Parameters * 9 / 8, 12 * GiB));
    }

    private static ArcTransformerMemoryPlan Plan(int batch, TensorDType dtype, ulong memory)
    {
        long packed = dtype switch
        {
            TensorDType.Float32 => Parameters * 4,
            TensorDType.BFloat16 => Parameters * 2,
            _ => Parameters + (Parameters + 31) / 32 * 4,
        };
        return ArcTransformerMemoryPlan.Create(batch, 1024, 512, 16, 1536, 32,
            dtype, 32, Parameters, packed, memory);
    }
}
