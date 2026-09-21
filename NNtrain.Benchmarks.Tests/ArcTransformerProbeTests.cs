using Xunit;

namespace NNtrain.Benchmarks;

public sealed class ArcTransformerProbeTests
{
    [Theory]
    [InlineData(64, 4, 1024, 32, true)]
    [InlineData(32, 8, 1024, 32, true)]
    [InlineData(16, 16, 1024, 32, true)]
    [InlineData(8, 32, 1024, 32, true)]
    [InlineData(16, 4, 512, 2, true)]
    [InlineData(16, 17, 1024, 32, false)]
    [InlineData(128, 2, 1024, 32, false)]
    [InlineData(16, 16, 2048, 32, false)]
    [InlineData(16, 16, 1024, 33, false)]
    [InlineData(16, 0, 1024, 32, false)]
    [InlineData(64, int.MaxValue, 1024, 32, false)]
    public void MicrobatchMayBeExchangedForAccumulationWithoutIncreasingWork(
        int batch, int accumulation, int sequence, int layers, bool expected)
        => Assert.Equal(expected, ArcTransformerProbe.IsShapeOverrideWithinBudget(
            (64, 4, 1024, 32), (batch, accumulation, sequence, layers)));
}
