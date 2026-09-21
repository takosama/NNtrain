using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcRetiredBufferReuseTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PendingKernelAndCopyFinishBeforeReusedBufferIsOverwritten(bool reuse)
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: Options(reuse));
        using var source = lane.Upload(Enumerable.Repeat(1f, 16).ToArray());
        using var result = lane.Allocate(16);
        using var old = lane.Allocate(16);
        using var borrowed = old.Borrow();
        lane.Run("copy_scale", 16, 0, source, old, 16, 2f, 0);
        lane.CopyBytes(old, result, 0, 0, 64);
        old.Dispose();
        Assert.False(old.IsAlive); Assert.False(borrowed.IsAlive);
        Assert.Equal(64, lane.RetiredBytes);
        long allocations = lane.AllocationCount;
        using var next = lane.Allocate(16);
        Assert.Equal(allocations + (reuse ? 0 : 1), lane.AllocationCount);
        Assert.Equal(reuse ? 1 : 0, lane.RetiredReuseCount);
        Assert.Equal(reuse ? 0 : 64, lane.RetiredBytes);
        Assert.Equal(0, lane.PoolHits);
        old.Dispose(); borrowed.Dispose();
        Assert.True(next.IsAlive);
        Assert.Throws<ArgumentException>(() => lane.CopyBytes(old, result, 0, 0, 64));
        lane.Run("copy_scale", 16, 0, source, next, 16, 9f, 0);
        lane.Run("add", 16, 0, result, next, result, 16);
        var actual = new float[16]; lane.Read(result, actual);
        Assert.All(actual, value => Assert.Equal(11f, value));
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(3 * 64, lane.AllocatedBytes);
        Assert.Equal(64, lane.H2DBytes);
        Assert.Equal(64, lane.D2HBytes);
        if (reuse) Assert.Contains(lane.DetailedProfiler!.Snapshot(), e => e.Kind == "retired-reuse");
    }

    [Fact]
    public void ReclaimedUploadUsesOrderedWriteAndPreservesEarlierCopy()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: Options(true));
        using var source = lane.Upload(Enumerable.Repeat(3f, 16).ToArray());
        using var result = lane.Allocate(16);
        using var old = lane.Allocate(16);
        lane.CopyBytes(source, old, 0, 0, 64);
        lane.CopyBytes(old, result, 0, 0, 64);
        old.Dispose();
        long allocations = lane.AllocationCount, upload = lane.H2DBytes;
        using var next = lane.Upload(Enumerable.Repeat(7f, 16).ToArray());
        Assert.Equal(allocations, lane.AllocationCount);
        Assert.Equal(1, lane.RetiredReuseCount);
        Assert.Equal(upload + 64, lane.H2DBytes);
        Assert.Equal(0, lane.RetiredBytes);
        var earlier = new float[16]; var later = new float[16];
        lane.Read(result, earlier); lane.Read(next, later);
        Assert.All(earlier, value => Assert.Equal(3f, value));
        Assert.All(later, value => Assert.Equal(7f, value));
        old.Dispose(); Assert.True(next.IsAlive);
    }

    [Fact]
    public void RetiredLookupRequiresExactSizeAndDoesNotConsumeUnrelatedEntries()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: Options(true));
        using var old = lane.Allocate(16);
        lane.Run("resident_zero", 16, 0, old, 16);
        old.Dispose();
        long allocations = lane.AllocationCount;
        using var small = lane.Allocate(8);
        Assert.Equal(allocations + 1, lane.AllocationCount);
        Assert.Equal(64, lane.RetiredBytes);
        Assert.Equal(0, lane.RetiredReuseCount);
        using var exact = lane.Allocate(16);
        Assert.Equal(allocations + 1, lane.AllocationCount);
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(1, lane.RetiredReuseCount);
        Assert.Equal(96, lane.AllocatedBytes);
        Assert.Equal(0, lane.CachedBytes);
        Assert.Equal(96, lane.PeakAllocatedBytes);
        Assert.Throws<ArgumentException>(() => lane.Run("resident_zero", 16, 0, exact, "invalid"));
        Assert.True(exact.IsAlive);
        Assert.Equal(0, lane.RetiredBytes);
    }

    [Fact]
    public void CachedHitTakesPriorityOverAnEvictedRetiredHandle()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: Options(true) with { BufferPoolBytes = 64 });
        using var old = lane.Allocate(16);
        using var recent = lane.Allocate(16);
        lane.Run("resident_zero", 16, 0, old, 16);
        old.Dispose(); recent.Dispose();
        Assert.Equal(64, lane.CachedBytes);
        Assert.Equal(64, lane.RetiredBytes);
        long allocations = lane.AllocationCount;
        using var cached = lane.Allocate(16);
        Assert.Equal(1, lane.PoolHits);
        Assert.Equal(0, lane.RetiredReuseCount);
        Assert.Equal(64, lane.RetiredBytes);
        using var retired = lane.Allocate(16);
        Assert.Equal(allocations, lane.AllocationCount);
        Assert.Equal(1, lane.PoolHits);
        Assert.Equal(1, lane.RetiredReuseCount);
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(0, lane.CachedBytes);
        Assert.Equal(128, lane.AllocatedBytes);
        Assert.Equal(128, lane.PeakAllocatedBytes);
    }

    [Fact]
    public void LaneDisposalDrainsReclaimedOwnerAndRepeatedDisposalsAreHarmless()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: Options(true));
        using var original = lane.Allocate(16);
        using var borrowed = original.Borrow();
        lane.Run("resident_zero", 16, 0, original, 16);
        original.Dispose();
        using var current = lane.Allocate(16);
        lane.Run("resident_zero", 16, 0, current, 16);
        Assert.Equal(1, lane.AllocationCount);
        Assert.Equal(1, lane.RetiredReuseCount);
        lane.Dispose(); lane.Dispose();
        Assert.False(original.IsAlive); Assert.False(borrowed.IsAlive); Assert.False(current.IsAlive);
        original.Dispose(); borrowed.Dispose(); current.Dispose();
        Assert.Equal(0, lane.AllocatedBytes);
        Assert.Equal(0, lane.CachedBytes);
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Throws<ObjectDisposedException>(() => lane.Allocate(16));
    }

    private static ArcExecutionOptions Options(bool reuse) => new() {
        ReuseRetiredBuffers = reuse, LruBufferPool = true, BufferPoolBytes = 0,
        DeferredReleaseBytes = 4096, QueuedKernelLimit = 4096, DetailedProfiling = true };

    private static void RequireArc() => Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
}
