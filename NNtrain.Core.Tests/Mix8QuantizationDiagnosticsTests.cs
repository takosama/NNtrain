using NNtrain;
using Xunit;

public sealed class Mix8QuantizationDiagnosticsTests
{
    [Fact]
    public void RatiosUseElementWeightedRootMeanSquare()
    {
        var diagnostics = new Mix8QuantizationDiagnostics(
            ChangedWeightCount: 25,
            ElementCount: 100,
            ResidualStepRatioSquaredSum: 9d,
            UpdateStepRatioSquaredSum: 16d);

        Assert.Equal(0.25d, diagnostics.QuantizedWeightChangeRate, 12);
        Assert.Equal(0.3d, diagnostics.ResidualRmsPerQuantStep, 12);
        Assert.Equal(0.4d, diagnostics.UpdateRmsPerQuantStep, 12);
    }

    [Fact]
    public void CombinePreservesRawSumsBeforeComputingRatios()
    {
        Mix8QuantizationDiagnostics combined =
            Mix8QuantizationDiagnostics.Combine(
            [
                new Mix8QuantizationDiagnostics(1, 4, 1d, 4d),
                new Mix8QuantizationDiagnostics(5, 6, 3d, 5d),
            ]);

        Assert.Equal((ulong)6, combined.ChangedWeightCount);
        Assert.Equal((ulong)10, combined.ElementCount);
        Assert.Equal(0.6d, combined.QuantizedWeightChangeRate, 12);
        Assert.Equal(Math.Sqrt(0.4d),
            combined.ResidualRmsPerQuantStep, 12);
        Assert.Equal(Math.Sqrt(0.9d),
            combined.UpdateRmsPerQuantStep, 12);
    }

    [Fact]
    public void EmptyDiagnosticsAreFiniteZeros()
    {
        Mix8QuantizationDiagnostics diagnostics = default;

        Assert.False(diagnostics.HasValues);
        Assert.Equal(0d, diagnostics.QuantizedWeightChangeRate);
        Assert.Equal(0d, diagnostics.ResidualRmsPerQuantStep);
        Assert.Equal(0d, diagnostics.UpdateRmsPerQuantStep);
    }
}
