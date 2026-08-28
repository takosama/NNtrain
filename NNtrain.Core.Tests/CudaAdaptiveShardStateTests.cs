using NNtrain;
using Xunit;

public sealed class CudaAdaptiveShardStateTests
{
    [Fact]
    public void RestoredSchedulerContinuesWithIdenticalAllocations()
    {
        var options = new CudaAdaptiveShardingOptions
        {
            Enabled = true,
            EmaAlpha = 0.25d,
            MinimumRelativeShardSize = 0.5d,
            MaximumBatchAdjustmentPerStep = 1,
            MinimumObservationsBeforeAdjustment = 1,
            RequiredConsecutiveCandidateObservations = 1,
            MinimumStepsBetweenAdjustments = 0,
            MinimumPredictedStepTimeImprovement = 0d,
            OversizedGraphMinimumPredictedImprovement = 0d,
        };
        int[] devices = [0, 1];
        var uninterrupted = new CudaAdaptiveShardScheduler(options);
        Assert.Equal([6, 6], uninterrupted.Allocate(12, devices));
        uninterrupted.Observe([6, 6], [6d, 12d]);
        Assert.Equal([7, 5], uninterrupted.Allocate(12, devices));

        CudaAdaptiveShardState checkpoint = uninterrupted.CaptureState();
        var resumed = new CudaAdaptiveShardScheduler(options);
        resumed.RestoreState(checkpoint, devices);

        for (int step = 0; step < 4; step++)
        {
            int[] expected = uninterrupted.Allocate(12, devices);
            int[] actual = resumed.Allocate(12, devices);
            Assert.Equal(expected, actual);
            double[] elapsed = step % 2 == 0
                ? [7d, 10d]
                : [9d, 8d];
            uninterrupted.Observe(expected, elapsed);
            resumed.Observe(actual, elapsed);
        }
        CudaAdaptiveShardState expectedState = uninterrupted.CaptureState();
        CudaAdaptiveShardState actualState = resumed.CaptureState();
        Assert.Equal(expectedState.FormatVersion, actualState.FormatVersion);
        Assert.Equal(expectedState.Devices, actualState.Devices);
        Assert.Equal(expectedState.LastAllocation, actualState.LastAllocation);
        Assert.Equal(expectedState.ThroughputEma, actualState.ThroughputEma);
        Assert.Equal(expectedState.HasObservation, actualState.HasObservation);
        Assert.Equal(
            expectedState.ObservationCount,
            actualState.ObservationCount);
        Assert.Equal(
            expectedState.LastAdjustmentObservation,
            actualState.LastAdjustmentObservation);
        Assert.Equal(
            expectedState.PendingAllocation,
            actualState.PendingAllocation);
        Assert.Equal(
            expectedState.PendingConfirmationCount,
            actualState.PendingConfirmationCount);
        Assert.Equal(
            expectedState.LastCandidateObservation,
            actualState.LastCandidateObservation);
        Assert.Equal(
            expectedState.OversizedGraphAllocation,
            actualState.OversizedGraphAllocation);
        Assert.Equal(
            expectedState.OversizedGraphPinnedBytes,
            actualState.OversizedGraphPinnedBytes);
    }

    [Fact]
    public void RestoreRejectsAStateForDifferentDevices()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions());
        var state = new CudaAdaptiveShardState(
            CudaAdaptiveShardState.CurrentFormatVersion,
            [0, 1],
            [4, 4],
            [1d, 1d],
            HasObservation: true);

        Assert.Throws<ArgumentException>(
            () => scheduler.RestoreState(state, [0, 2]));
    }

    [Theory]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    [InlineData(-1d)]
    public void RestoreRejectsInvalidThroughput(double invalid)
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions());
        var state = new CudaAdaptiveShardState(
            CudaAdaptiveShardState.CurrentFormatVersion,
            [0, 1],
            [4, 4],
            [1d, invalid],
            HasObservation: true);

        Assert.Throws<ArgumentException>(
            () => scheduler.RestoreState(state, [0, 1]));
    }

    [Fact]
    public void RestoredPendingCandidateCommitsOnSameObservation()
    {
        var options = new CudaAdaptiveShardingOptions
        {
            EmaAlpha = 1d,
            MinimumObservationsBeforeAdjustment = 1,
            RequiredConsecutiveCandidateObservations = 3,
            MinimumStepsBetweenAdjustments = 0,
            MinimumPredictedStepTimeImprovement = 0d,
            OversizedGraphMinimumPredictedImprovement = 0d,
        };
        int[] devices = [0, 1];
        var uninterrupted = new CudaAdaptiveShardScheduler(options);
        int[] allocation = uninterrupted.Allocate(72, devices);
        for (int index = 0; index < 2; index++)
        {
            uninterrupted.Observe(allocation, [360d, 720d]);
            Assert.Equal(
                [36, 36],
                uninterrupted.Allocate(72, devices));
        }

        CudaAdaptiveShardState checkpoint = uninterrupted.CaptureState();
        var resumed = new CudaAdaptiveShardScheduler(options);
        resumed.RestoreState(checkpoint, devices);

        uninterrupted.Observe(allocation, [360d, 720d]);
        resumed.Observe(allocation, [360d, 720d]);
        Assert.Equal(
            uninterrupted.Allocate(72, devices),
            resumed.Allocate(72, devices));
        Assert.Equal([37, 35], resumed.LastAllocation);
    }

    [Fact]
    public void VersionOneStateRestoresWithSafeStabilizationDefaults()
    {
        var scheduler = new CudaAdaptiveShardScheduler(
            new CudaAdaptiveShardingOptions());
        var versionOne = new CudaAdaptiveShardState(
            FormatVersion: 1,
            Devices: [0, 1],
            LastAllocation: [38, 34],
            ThroughputEma: [0.1d, 0.09d],
            HasObservation: true);

        scheduler.RestoreState(versionOne, [0, 1]);

        Assert.Equal([38, 34], scheduler.LastAllocation);
        Assert.Equal(
            CudaAdaptiveShardState.CurrentFormatVersion,
            scheduler.CaptureState().FormatVersion);
    }
}
