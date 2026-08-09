using NNtrain;
using Xunit;

public sealed class LearningRateAdjustmentTests
{
    [Fact]
    public void AllBuiltInOptimizersSupportLearningRateAdjustment()
    {
        foreach (IOptimizer optimizer in CreateOptimizers())
        {
            var adjustable = Assert.IsAssignableFrom<ILearningRateAdjustable>(
                optimizer);

            adjustable.SetLearningRate(0.025f);

            Assert.Equal(0.025f, adjustable.LearningRate);
        }
    }

    [Fact]
    public void LearningRateAdjustmentIsIncludedInCapturedState()
    {
        var parameter = CreateParameter();
        var adamW = new AdamW([parameter]);
        adamW.SetLearningRate(0.004f);

        Assert.Equal(0.004f, adamW.CaptureState().Options.LearningRate);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-0.1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void AllBuiltInOptimizersRejectInvalidLearningRates(float value)
    {
        foreach (IOptimizer optimizer in CreateOptimizers())
        {
            var adjustable = (ILearningRateAdjustable)optimizer;
            Assert.Throws<ArgumentOutOfRangeException>(
                () => adjustable.SetLearningRate(value));
        }
    }

    private static IOptimizer[] CreateOptimizers()
        =>
        [
            new AdamW([CreateParameter()]),
            new GainShareAdamW([[CreateParameter()]]),
            new Lion([CreateParameter()]),
            new NekoMuon([CreateParameter()]),
        ];

    private static Parameter CreateParameter()
        => new(
            [1f],
            [1],
            "weight",
            WeightDecayPolicy.Apply);
}
