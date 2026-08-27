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
}
