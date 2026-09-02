using NNtrain;
using Xunit;

public sealed class WikiOptimizerDiagnosticsTests
{
    [Fact]
    public void Mix8DiagnosticsAggregateLeafSumsOnlyWhenFormatted()
    {
        var first = new DiagnosticOptimizer(
            new Mix8QuantizationDiagnostics(
                ChangedWeightCount: 1,
                ElementCount: 2,
                ResidualStepRatioSquaredSum: 8d,
                UpdateStepRatioSquaredSum: 18d));
        var second = new DiagnosticOptimizer(
            new Mix8QuantizationDiagnostics(
                ChangedWeightCount: 2,
                ElementCount: 6,
                ResidualStepRatioSquaredSum: 16d,
                UpdateStepRatioSquaredSum: 54d));
        IOptimizer optimizer = new CompositeOptimizer(first, second);
        var config = new WikiTrainingConfiguration
        {
            Optimizer = "adamw",
        };

        Assert.Equal(0, first.ReadCount);
        Assert.Equal(0, second.ReadCount);

        string formatted = WikiLanguageModelCommand
            .FormatOptimizerDiagnostics(optimizer, config);

        Assert.Equal(1, first.ReadCount);
        Assert.Equal(1, second.ReadCount);
        Assert.Equal(
            ", quantized_weight_change_rate = 0.375, " +
            "residual_rms / quant_step = 1.73205, " +
            "update_rms / quant_step = 3",
            formatted);
    }

    [Fact]
    public void Mix8DiagnosticsKeepOrdinaryMuonText()
    {
        IOptimizer optimizer = new DiagnosticOptimizer(
            new Mix8QuantizationDiagnostics(
                ChangedWeightCount: 1,
                ElementCount: 1,
                ResidualStepRatioSquaredSum: 0.25d,
                UpdateStepRatioSquaredSum: 4d));
        var config = new WikiTrainingConfiguration
        {
            Optimizer = "muon",
        };

        string formatted = WikiLanguageModelCommand
            .FormatOptimizerDiagnostics(optimizer, config);

        Assert.Equal(
            ", muon NS depth = 5, " +
            "quantized_weight_change_rate = 1, " +
            "residual_rms / quant_step = 0.5, " +
            "update_rms / quant_step = 2",
            formatted);
    }

    private sealed class DiagnosticOptimizer(
        Mix8QuantizationDiagnostics diagnostics)
        : IOptimizer, IMix8QuantizationDiagnosticsProvider
    {
        internal int ReadCount { get; private set; }

        public void zero_grad()
        {
        }

        public void step()
        {
        }

        public bool TryGetMix8QuantizationDiagnostics(
            out Mix8QuantizationDiagnostics value)
        {
            ReadCount++;
            value = diagnostics;
            return value.HasValues;
        }
    }
}
