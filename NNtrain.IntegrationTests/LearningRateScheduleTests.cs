using NNtrain;
using Xunit;

public sealed class LearningRateScheduleTests
{
    [Fact]
    public void FiveEpochWarmupReachesTheConfiguredLearningRate()
    {
        Assert.Equal(
            0.2f,
            Program.CalculateLearningRateFactor(1, 200, 5, 0.01f),
            6);
        Assert.Equal(
            1f,
            Program.CalculateLearningRateFactor(5, 200, 5, 0.01f),
            6);
    }

    [Fact]
    public void CosineDecayReachesTheConfiguredFloorAtTheFinalEpoch()
    {
        float afterWarmup = Program.CalculateLearningRateFactor(
            6,
            200,
            5,
            0.01f);
        float final = Program.CalculateLearningRateFactor(
            200,
            200,
            5,
            0.01f);

        Assert.InRange(afterWarmup, 0.99f, 1f);
        Assert.Equal(0.01f, final, 6);
    }

    [Fact]
    public void CompositeGroupsKeepTheirConfiguredLearningRateRatio()
    {
        var firstParameter = new Parameter(
            [1f], [1], "first", WeightDecayPolicy.Apply);
        var secondParameter = new Parameter(
            [1f], [1], "second", WeightDecayPolicy.Apply);
        var primary = new NekoMuon([firstParameter]);
        var auxiliary = new AdamW([secondParameter]);
        var composite = new CompositeOptimizer(primary, auxiliary);
        var configuration = new TrainingConfiguration
        {
            Epochs = 10,
            WarmupEpochs = 2,
            LearningRate = 0.01f,
            AuxiliaryLearningRate = 0.001f,
        };

        Program.LearningRates rates = Program.SetScheduledLearningRates(
            composite,
            configuration,
            1);

        Assert.Equal(0.005f, rates.Primary, 7);
        Assert.Equal(0.0005f, rates.Auxiliary!.Value, 7);
        Assert.Equal(rates.Primary, primary.LearningRate);
        Assert.Equal(rates.Auxiliary.Value, auxiliary.LearningRate);
    }
}
