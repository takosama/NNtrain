using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcTimelineTests
{
    [Fact]
    public void NestedWaitOverlappingGpuIsNeverCountedTwice()
    {
        ArcTimelineInterval[] host = [new(0, 100, "managed", "step", 1), new(10, 90, "wait", "finish", 2), new(30, 40, "allocate", "buffer", 3)];
        ArcTimelineInterval[] gpu = [new(20, 70, "GPU/kernel", "gemm", 0), new(75, 85, "GPU/transfer", "D2D", 0)];
        var r = ArcTimelinePartition.Build(100, host, gpu);
        Assert.Equal(1, r.CoverageFraction, 12);
        Assert.Equal(60 / 1e6, r.WallCategories.Where(s => s.Name.StartsWith("GPU/")).Sum(s => s.Milliseconds), 12);
        Assert.Equal(40 / 1e6, r.WallCategories.Where(s => s.Name.StartsWith("GPU-idle/")).Sum(s => s.Milliseconds), 12);
        Assert.Equal(70 / 1e6, r.HostExclusive.Single(s => s.Name == "wait").Milliseconds, 12);
        Assert.Equal(0, r.GpuOverlapMs); Assert.Equal(0, r.UnattributedHostMs);
    }
    [Fact]
    public void MissingHostAndOverlappingGpuAreReportedNotHidden()
    {
        var r = ArcTimelinePartition.Build(100, [], [new(-10, 40, "gpu", "a", 0), new(30, 110, "gpu", "b", 0)]);
        Assert.Equal(1, r.CoverageFraction, 12); Assert.Equal(10 / 1e6, r.GpuOverlapMs, 12); Assert.Equal(100 / 1e6, r.UnattributedHostMs, 12);
    }
    [Fact]
    public void ZeroLengthAndSimultaneousEdgesPreservePartition()
    {
        var r = ArcTimelinePartition.Build(100, [new(0, 100, "host", "step", 1)],
            [new(0, 50, "gpu", "a", 0), new(50, 100, "gpu", "b", 0), new(25, 25, "gpu", "empty", 0)]);
        Assert.Equal(1, r.CoverageFraction, 12); Assert.Equal(0, r.GpuOverlapMs); Assert.Equal(1, r.WallCategories.Single().Fraction);
    }
    [Fact]
    public void PoolMissUploadIsAnExplicitVisibleTransferWithExactContents()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() { BufferPoolBytes = 0 });
        float[] expected = Enumerable.Range(0, 257).Select(i => i * .03125f - 2).ToArray();
        var t = lane.BeginTimeline(); t.MarkStart();
        float[] actual = new float[expected.Length];
        using (t.Host("upload-pool-miss"))
        {
            using var value = lane.Upload(expected);
            lane.Read(value, actual);
            lane.Synchronize();
        }
        t.MarkEnd(); lane.EndTimeline();
        Assert.Equal(expected, actual);
        string path = Path.Combine(Path.GetTempPath(), "arc-upload-trace-" + Guid.NewGuid() + ".json.gz");
        try
        {
            var r = t.Export(path);
            Assert.Equal(2, r.DeviceEvents);
            Assert.Equal(0, r.OpaqueAllocationCopies);
            Assert.Equal(0, r.MissingEvents);
            Assert.Equal(0, r.Partition.UnattributedHostMs);
            Assert.Contains(r.Partition.WallCategories, c => c.Name == "GPU/transfer/H2D");
            Assert.Contains(r.Partition.WallCategories, c => c.Name == "GPU/transfer/D2H");
        }
        finally { File.Delete(path); }
    }
    [Fact]
    public void ArcKernelAndCopyEventsShareCalibratedWallClock()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() { DetailedProfiling = true });
        using var a = lane.Allocate(4096); using var b = lane.Allocate(4096);
        lane.Run("resident_zero", 4096, 0, a, 4096); lane.Synchronize();
        var t = lane.BeginTimeline(); t.MarkStart();
        using (t.Host("test-step"))
        {
            lane.Write(a, new float[4096]); lane.Run("resident_zero", 4096, 0, a, 4096);
            lane.CopyBytes(a, b, 0, 0, 4096 * 4); lane.Read(b, new float[4096]); lane.Synchronize();
        }
        t.MarkEnd(); lane.EndTimeline();
        string path = Path.Combine(Path.GetTempPath(), "arc-timeline-" + Guid.NewGuid() + ".json.gz");
        try
        {
            var r = t.Export(path);
            Assert.Equal(4, r.DeviceEvents); Assert.Equal(0, r.MissingEvents); Assert.Equal(0, r.OpaqueAllocationCopies);
            Assert.Equal(1, r.Partition.CoverageFraction, 10);
            Assert.InRange(r.Partition.GpuOverlapMs, 0, .001);
            Assert.Contains(r.Partition.WallCategories, c => c.Name == "GPU/transfer/H2D");
            Assert.Contains(r.Partition.WallCategories, c => c.Name == "GPU/transfer/D2H");
            Assert.Contains(r.Partition.WallCategories, c => c.Name == "GPU/transfer/D2D");
        }
        finally { File.Delete(path); }
    }
}
