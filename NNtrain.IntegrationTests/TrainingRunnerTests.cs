using NNtrain;
using Xunit;

public sealed class TrainingRunnerTests
{
    [Fact]
    public void EpochsCarryResumeUnitOnlyIntoFirstEpoch()
    {
        TrainingEpoch[] epochs = TrainingRunner.Epochs(2, 4, 7).ToArray();

        Assert.Equal(
            [
                new TrainingEpoch(2, 7),
                new TrainingEpoch(3, 0),
                new TrainingEpoch(4, 0),
            ],
            epochs);
    }

    [Fact]
    public void ShuffleIsDeterministicForASeed()
    {
        int[] first = Enumerable.Range(0, 20).ToArray();
        int[] second = Enumerable.Range(0, 20).ToArray();

        TrainingRunner.Shuffle(first, new Random(17));
        TrainingRunner.Shuffle(second, new Random(17));

        Assert.Equal(first, second);
        Assert.Equal(Enumerable.Range(0, 20), first.Order());
    }

    [Theory]
    [InlineData(1, 20, false)]
    [InlineData(1, 10, true)]
    [InlineData(10, 10, true)]
    public void CheckpointBoundaryUsesTenths(
        int completed,
        int total,
        bool expected)
    {
        Assert.Equal(
            expected,
            TrainingRunner.ShouldSaveCheckpoint(completed, total));
    }
}
