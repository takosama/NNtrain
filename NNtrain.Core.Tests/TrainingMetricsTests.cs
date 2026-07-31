using NNtrain;
using Xunit;

public sealed class TrainingMetricsTests
{
    [Fact]
    public void EpochResultGroupsLossAccuracyAndElapsedInMetrics()
    {
        var metrics = new TrainingMetrics(
            Loss: 0.25f,
            Accuracy: 0.75f,
            Elapsed: TimeSpan.FromSeconds(2));
        var result = new TrainingEpochResult(
            Epoch: 3,
            TrainingSteps: 10,
            EvaluationSamples: 4,
            Training: metrics,
            Evaluation: metrics);

        Assert.Same(metrics, result.Training);
        Assert.Same(metrics, result.Evaluation);
        Assert.Equal(
            new TrainingMetrics(0.25f, 0.75f, TimeSpan.FromSeconds(2)),
            result.Training);
        Assert.Null(typeof(TrainingEpochResult).GetProperty("AverageLoss"));
        Assert.Null(typeof(TrainingEpochResult).GetProperty("Accuracy"));
        Assert.Null(typeof(TrainingEpochResult).GetProperty("Elapsed"));
    }
}
