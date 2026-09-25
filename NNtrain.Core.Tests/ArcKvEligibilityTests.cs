using NNtrain;
using Xunit;

public sealed class ArcKvEligibilityTests
{
    [Theory]
    [InlineData(64, 128, 32, false, true)]
    [InlineData(64, 128, 32, true, true)]
    [InlineData(32, 64, 32, false, true)]
    [InlineData(32, 64, 32, true, false)]
    [InlineData(64, 96, 32, false, true)]
    [InlineData(64, 96, 32, true, false)]
    [InlineData(8, 32, 32, false, false)]
    [InlineData(64, 65, 32, false, false)]
    [InlineData(64, 128, 128, false, false)]
    public void QuantizedRowsMustContainWholeBlocksIncludingTensorParallelShards(
        int width, int hidden, int blockSize, bool tensorParallel, bool expected)
    {
        var model = new GptRinWikiJp(32, 8, width, 2, hidden, 1, new Random(3));
        model.to(TensorPrecisionMode.Mix8_32, blockSize);
        model.ArcTensorParallelEnabled = tensorParallel;
        Assert.Equal(expected, model.HasArcKvCompatibleQuantization());
    }

    [Fact]
    public void TensorWideQuantizationCannotPreserveCachedRows()
    {
        var model = new GptRinWikiJp(32, 8, 64, 2, 128, 1, new Random(3));
        model.to(TensorPrecisionMode.Bfp8);
        Assert.False(model.HasArcKvCompatibleQuantization());
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32)]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    public void NonBfp8FormatsDoNotNeedBlockAlignment(TensorPrecisionMode precision)
    {
        var model = new GptRinWikiJp(32, 8, 8, 2, 19, 1, new Random(3));
        model.to(precision);
        Assert.True(model.HasArcKvCompatibleQuantization());
    }

    [Fact]
    public void Mix8_16ConservativelyRetainsAlignmentRequirement()
    {
        var model = new GptRinWikiJp(32, 8, 32, 2, 64, 1, new Random(3));
        model.to(TensorPrecisionMode.Mix8_16, 32);
        Assert.True(model.HasArcKvCompatibleQuantization());
        model.ArcTensorParallelEnabled = true;
        Assert.False(model.HasArcKvCompatibleQuantization());
    }
}
