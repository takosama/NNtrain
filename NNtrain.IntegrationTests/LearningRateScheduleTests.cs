using NNtrain;
using Xunit;

public sealed class LearningRateScheduleTests
{
    [Fact]
    public void FiveEpochWarmupReachesTheConfiguredLearningRate()
    {
        Assert.Equal(
            0.2f,
            LinearWarmupCosineLRScheduler.CalculateFactor(
                1, 200, 5, 0.01f),
            6);
        Assert.Equal(
            1f,
            LinearWarmupCosineLRScheduler.CalculateFactor(
                5, 200, 5, 0.01f),
            6);
    }

    [Fact]
    public void CosineDecayReachesTheConfiguredFloorAtTheFinalEpoch()
    {
        float afterWarmup = LinearWarmupCosineLRScheduler.CalculateFactor(
            6,
            200,
            5,
            0.01f);
        float final = LinearWarmupCosineLRScheduler.CalculateFactor(
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
        var primary = new NekoMuon(
            [firstParameter],
            new NekoMuonOptions { LearningRate = 0.01f });
        var auxiliary = new AdamW(
            [secondParameter],
            new AdamWOptions { LearningRate = 0.001f });
        var composite = new CompositeOptimizer(primary, auxiliary);
        ILRScheduler scheduler =
            lr_scheduler.LinearWarmupCosineAnnealingLR(
            composite,
            total_epochs: 10,
            warmup_epochs: 2,
            min_lr_ratio: 0.01f);

        IReadOnlyList<float> rates = scheduler.step();

        Assert.Equal(0.005f, rates[0], 7);
        Assert.Equal(0.0005f, rates[1], 7);
        Assert.Equal(rates[0], primary.LearningRate);
        Assert.Equal(rates[1], auxiliary.LearningRate);
    }
}
