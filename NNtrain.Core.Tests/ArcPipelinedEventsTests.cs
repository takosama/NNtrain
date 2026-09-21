using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcPipelinedEventsTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HalfWindowKeepsCopiesAndRetiredOwnersAliveUntilFullFence(bool pipeline)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() { PipelineEventCollection = pipeline,
            QueuedKernelLimit = 16, BufferPoolBytes = 0, DeferredReleaseBytes = 1048576, ReuseRetiredBuffers = false, DetailedProfiling = true });
        using var destination = lane.Allocate(4096);
        for (int step = 0; step < 100; step++)
        {
            using var source = lane.Upload(Enumerable.Repeat((float)step, 4096).ToArray());
            lane.CopyBytes(source, destination, 0, 0, 4096 * 4);
            // A later unrelated kernel cannot make the source disposable early.
            using var scratch = lane.Allocate(17);
            lane.Run("resident_zero", 17, 0, scratch, 17);
        }
        float[] actual = new float[4096]; lane.Read(destination, actual);
        Assert.All(actual, value => Assert.Equal(99, value));
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(lane.NativeAllocatedBytes - lane.NativeReleasedBytes, lane.AllocatedBytes + lane.CachedBytes);
        Assert.Equal(100, lane.KernelLaunchCount);
        lane.Dispose(); Assert.Equal(lane.NativeAllocatedBytes, lane.NativeReleasedBytes);
    }
    [Fact]
    public void ParallelSubmissionsAreBoundedAndEventsAreCollectedExactlyOnce()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() { PipelineEventCollection = true, QueuedKernelLimit = 16, DetailedProfiling = true });
        var t = lane.BeginTimeline(); t.MarkStart();
        using (t.Host("parallel-dispatch"))
        {
            Parallel.For(0, 96, i => { using var value = lane.Allocate(64); lane.Run("resident_zero", 64, 0, value, 64); });
            lane.Synchronize();
        }
        t.MarkEnd(); lane.EndTimeline();
        string path = Path.Combine(Path.GetTempPath(), "arc-events-" + Guid.NewGuid() + ".json.gz");
        try { var r = t.Export(path); Assert.Equal(96, r.DeviceEvents); Assert.Equal(0, r.MissingEvents); Assert.Equal(1, r.Partition.CoverageFraction, 10); }
        finally { File.Delete(path); }
        Assert.Equal(96, lane.DetailedProfiler!.Snapshot().Where(p => p.Kind == "gpu-kernel").Sum(p => p.Count));
    }
}
