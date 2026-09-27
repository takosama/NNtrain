using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferenceQueueTests
{
    private const int Width = 17;

    [Theory]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(33)]
    [InlineData(512)]
    public void SparseEventsPreserveMultipleWindowsPooledReuseCopiesAndReadReset(int queueLimit)
    {
        RequireArc();
        using var lane = new ArcExecutionLane(0, Options(queueLimit));
        using var ones = lane.Upload(Enumerable.Repeat(1f, Width).ToArray());
        using var total = lane.Allocate(Width);
        using var copy = lane.Allocate(Width);
        lane.Run("q35a_zero", Width, 0, total, Width);
        int completed = 0;

        // Only the two final reads synchronize fully. Every batch crosses
        // several half-window boundaries, including the odd-sized windows.
        foreach (int count in new[] { 2 * queueLimit + 5, queueLimit + 7 })
        {
            for (int i = 0; i < count; i++)
            {
                using var temporary = lane.Allocate(Width);
                lane.Run("q35a_zero", Width, 0, temporary, Width);
                lane.Run("q35a_add_in_place", Width, 0, temporary, ones, Width);
                lane.Run("q35a_add_in_place", Width, 0, total, temporary, Width);
                lane.CopyBytes(total, copy, 0, 0, Width * sizeof(float));
                // The next iteration reuses temporary while older commands
                // may still be queued; their ordered use must remain valid.
                completed++;
            }
            var actual = new float[Width];
            lane.Read(copy, actual);
            Assert.All(actual, value => Assert.Equal((float)completed, value));
            Assert.Equal(0L, lane.RetiredBytes);
            Assert.Empty(lane.KernelTimings);
            Assert.Equal(0d, lane.KernelMilliseconds);
        }

        Assert.True(lane.PoolHits > 0);
        Assert.Equal(1L + completed * 3L, lane.KernelLaunchCount);
        Assert.Equal(lane.NativeAllocatedBytes - lane.NativeReleasedBytes,
            lane.AllocatedBytes + lane.CachedBytes);
        lane.Dispose();
        Assert.Equal(lane.NativeAllocatedBytes, lane.NativeReleasedBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SparseEventsKeepRetiredCopySourcesAliveUntilTheFullFence(bool pipeline)
    {
        RequireArc();
        using var lane = new ArcExecutionLane(0, Options(17) with
        {
            PipelineEventCollection = pipeline,
            BufferPoolBytes = 0,
            ReuseRetiredBuffers = false,
            DeferredReleaseBytes = 4 * 1024 * 1024
        });
        using var ones = lane.Upload(Enumerable.Repeat(1f, Width).ToArray());
        using var total = lane.Allocate(Width);
        using var copy = lane.Allocate(Width);
        lane.Run("q35a_zero", Width, 0, total, Width);

        const int iterations = 71;
        for (int i = 0; i < iterations; i++)
        {
            using var source = lane.Allocate(Width);
            lane.Run("q35a_zero", Width, 0, source, Width);
            lane.Run("q35a_add_in_place", Width, 0, source, ones, Width);
            lane.CopyBytes(source, copy, 0, 0, Width * sizeof(float));
            lane.Run("q35a_add_in_place", Width, 0, total, copy, Width);
        }

        Assert.True(lane.RetiredBytes > 0);
        var actual = new float[Width];
        lane.Read(total, actual);
        Assert.All(actual, value => Assert.Equal((float)iterations, value));
        Assert.Equal(0L, lane.RetiredBytes);
        Assert.Equal(0L, lane.PoolHits);
        Assert.Equal(0L, lane.RetiredReuseCount);
        Assert.Empty(lane.KernelTimings);
        Assert.Equal(lane.NativeAllocatedBytes - lane.NativeReleasedBytes, lane.AllocatedBytes);
        lane.Dispose();
        Assert.Equal(lane.NativeAllocatedBytes, lane.NativeReleasedBytes);
    }

    [Fact]
    public void DetailedProfilingOverridesTimingSuppressionAndCollectsEveryKernel()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(0, Options(17) with { DetailedProfiling = true });
        using var ones = lane.Upload(Enumerable.Repeat(1f, Width).ToArray());
        using var total = lane.Allocate(Width);
        lane.Run("q35a_zero", Width, 0, total, Width);
        const int additions = 72;
        for (int i = 0; i < additions; i++)
            lane.Run("q35a_add_in_place", Width, 0, total, ones, Width);
        var actual = new float[Width];
        lane.Read(total, actual);

        Assert.All(actual, value => Assert.Equal((float)additions, value));
        Assert.True(lane.KernelMilliseconds > 0);
        Assert.Contains("q35a_zero", lane.KernelTimings.Keys);
        Assert.Contains("q35a_add_in_place", lane.KernelTimings.Keys);
        Assert.Equal(additions + 1L, lane.DetailedProfiler!.Snapshot()
            .Where(entry => entry.Kind == "gpu-kernel").Sum(entry => entry.Count));
    }

    [Fact]
    public void TimelineCanStartAfterSuppressedCommandsAndThenReturnToSuppression()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(0, Options(17));
        using var ones = lane.Upload(Enumerable.Repeat(1f, Width).ToArray());
        using var total = lane.Allocate(Width);
        lane.Run("q35a_zero", Width, 0, total, Width);
        for (int i = 0; i < 41; i++)
            lane.Run("q35a_add_in_place", Width, 0, total, ones, Width);

        // BeginTimeline must drain any sparse completion events before
        // switching to a complete stream of profiling events.
        var timeline = lane.BeginTimeline();
        Assert.Empty(lane.KernelTimings);
        timeline.MarkStart();
        lane.Run("q35a_zero", Width, 0, total, Width);
        const int tracedAdditions = 36;
        for (int i = 0; i < tracedAdditions; i++)
            lane.Run("q35a_add_in_place", Width, 0, total, ones, Width);
        lane.Synchronize();
        timeline.MarkEnd();
        Assert.Same(timeline, lane.EndTimeline());
        Assert.True(lane.KernelMilliseconds > 0);
        Assert.Equal(0, timeline.MissingEvents);

        string tracePath = Path.Combine(Path.GetTempPath(), "qwen35-sparse-events-" + Guid.NewGuid() + ".json.gz");
        try
        {
            var report = timeline.Export(tracePath);
            Assert.Equal(tracedAdditions + 1, report.DeviceEvents);
            Assert.Equal(0, report.MissingEvents);
        }
        finally { File.Delete(tracePath); }

        double tracedMilliseconds = lane.KernelMilliseconds;
        var tracedTimings = lane.KernelTimings.ToArray();
        const int suppressedAdditions = 39;
        for (int i = 0; i < suppressedAdditions; i++)
            lane.Run("q35a_add_in_place", Width, 0, total, ones, Width);
        var actual = new float[Width];
        lane.Read(total, actual);
        Assert.All(actual, value => Assert.Equal((float)(tracedAdditions + suppressedAdditions), value));
        Assert.Equal(tracedMilliseconds, lane.KernelMilliseconds);
        Assert.Equal(tracedTimings, lane.KernelTimings.ToArray());
    }

    private static ArcExecutionOptions Options(int queueLimit) => new()
    {
        Qwen35InferenceKernelsOnly = true,
        CollectKernelTimings = false,
        QueuedKernelLimit = queueLimit,
        PipelineEventCollection = true,
        BufferPoolBytes = 4096,
        BatchDispatch = true
    };

    private static void RequireArc()
        => Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
}
