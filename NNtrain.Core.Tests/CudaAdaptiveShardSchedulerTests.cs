using NNtrain;
using Xunit;

public sealed class CudaAdaptiveShardSchedulerTests
{
    [Fact]
    public void StartsEvenThenMovesOneBatchTowardFasterDevice()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions
            {
                EmaAlpha = 1d,
                MinimumRelativeShardSize = 0.5d,
                MaximumBatchAdjustmentPerStep = 1,
                MinimumObservationsBeforeAdjustment = 1,
                RequiredConsecutiveCandidateObservations = 1,
                MinimumStepsBetweenAdjustments = 0,
                MinimumPredictedStepTimeImprovement = 0d,
                OversizedGraphMinimumPredictedImprovement = 0d,
            });

        int[] first = scheduler.Allocate(72, [0, 1]);
        scheduler.Observe(first, [360d, 432d]);
        int[] second = scheduler.Allocate(72, [0, 1]);

        Assert.Equal([36, 36], first);
        Assert.Equal([37, 35], second);
    }

    [Fact]
    public void ConvergesWhileRespectingConfiguredMinimum()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions
            {
                EmaAlpha = 1d,
                MinimumRelativeShardSize = 0.5d,
                MaximumBatchAdjustmentPerStep = 32,
                MinimumObservationsBeforeAdjustment = 1,
                RequiredConsecutiveCandidateObservations = 1,
                MinimumStepsBetweenAdjustments = 0,
                MinimumPredictedStepTimeImprovement = 0d,
                OversizedGraphMinimumPredictedImprovement = 0d,
            });

        int[] initial = scheduler.Allocate(72, [0, 1]);
        scheduler.Observe(initial, [36d, 3600d]);
        int[] allocation = scheduler.Allocate(72, [0, 1]);

        Assert.Equal(72, allocation.Sum());
        Assert.All(allocation, value => Assert.InRange(value, 18, 54));
        Assert.Equal([54, 18], allocation);
    }

    [Fact]
    public void InvalidTimingFallsBackDeterministicallyToEvenShards()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions());

        int[] initial = scheduler.Allocate(73, [0, 1]);
        scheduler.Observe(initial, [double.NaN, 10d]);
        int[] fallback = scheduler.Allocate(73, [0, 1]);

        Assert.Equal([37, 36], initial);
        Assert.Equal([37, 36], fallback);
    }

    [Fact]
    public void DisabledModeAlwaysUsesEvenShards()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions { Enabled = false });

        int[] first = scheduler.Allocate(72, [0, 1]);
        scheduler.Observe(first, [3600d, 36d]);

        Assert.Equal([36, 36], scheduler.Allocate(72, [0, 1]));
    }

    [Fact]
    public void OversizedGraphRejectsOscillatingMarginalCandidates()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions
            {
                EmaAlpha = 1d,
                MinimumObservationsBeforeAdjustment = 1,
                RequiredConsecutiveCandidateObservations = 3,
                MinimumStepsBetweenAdjustments = 0,
                MinimumPredictedStepTimeImprovement = 0d,
                OversizedGraphMinimumPredictedImprovement = 0.15d,
            });

        int[] allocation = scheduler.Allocate(72, [0, 1]);
        scheduler.ObserveCompiledGraph(
            allocation,
            graphPinnedBytes: 6L * 1024 * 1024 * 1024,
            graphCacheBudgetBytes: 512L * 1024 * 1024);

        for (int step = 0; step < 20; step++)
        {
            // Alternating 8.3% preferences previously rebuilt [37,35] and
            // [36,36] graphs continuously.
            double[] elapsed = step % 2 == 0
                ? [360d, 432d]
                : [432d, 360d];
            scheduler.Observe(allocation, elapsed);
            allocation = scheduler.Allocate(72, [0, 1]);
            Assert.Equal([36, 36], allocation);
        }
    }

    [Fact]
    public void ConfirmedLargeImbalanceCanReplaceOversizedGraph()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions
            {
                EmaAlpha = 1d,
                MinimumRelativeShardSize = 0.5d,
                MaximumBatchAdjustmentPerStep = 32,
                MinimumObservationsBeforeAdjustment = 1,
                RequiredConsecutiveCandidateObservations = 3,
                MinimumStepsBetweenAdjustments = 0,
                MinimumPredictedStepTimeImprovement = 0d,
                OversizedGraphMinimumPredictedImprovement = 0.15d,
            });

        int[] allocation = scheduler.Allocate(72, [0, 1]);
        scheduler.ObserveCompiledGraph(
            allocation,
            graphPinnedBytes: 6L * 1024 * 1024 * 1024,
            graphCacheBudgetBytes: 512L * 1024 * 1024);
        for (int observation = 0; observation < 2; observation++)
        {
            scheduler.Observe(allocation, [36d, 3600d]);
            Assert.Equal([36, 36], scheduler.Allocate(72, [0, 1]));
        }

        scheduler.Observe(allocation, [36d, 3600d]);
        Assert.Equal([54, 18], scheduler.Allocate(72, [0, 1]));
    }

    [Fact]
    public void DuplicateAllocateDoesNotConfirmCandidateWithoutObservation()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions
            {
                EmaAlpha = 1d,
                MinimumObservationsBeforeAdjustment = 1,
                RequiredConsecutiveCandidateObservations = 2,
                MinimumStepsBetweenAdjustments = 0,
                MinimumPredictedStepTimeImprovement = 0d,
                OversizedGraphMinimumPredictedImprovement = 0d,
            });

        int[] allocation = scheduler.Allocate(72, [0, 1]);
        scheduler.Observe(allocation, [360d, 720d]);
        Assert.Equal([36, 36], scheduler.Allocate(72, [0, 1]));
        Assert.Equal([36, 36], scheduler.Allocate(72, [0, 1]));

        scheduler.Observe(allocation, [360d, 720d]);
        Assert.Equal([37, 35], scheduler.Allocate(72, [0, 1]));
    }
}
