using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcLruBufferPoolTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RecentlyReusedSizeSurvivesAndLegacyAdmissionRemainsUnchanged(bool lru)
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: new() {
            LruBufferPool = lru, BufferPoolBytes = 128, DetailedProfiling = true });
        using var first = lane.Allocate(16);
        using var cold = lane.Allocate(8);
        using var incoming = lane.Allocate(16);
        first.Dispose(); cold.Dispose();
        Assert.Equal(96, lane.CachedBytes);
        long allocations = lane.AllocationCount;
        using var recent = lane.Allocate(16);
        Assert.Equal(allocations, lane.AllocationCount);
        recent.Dispose(); // Refresh the 64-byte entry after the 32-byte entry.
        incoming.Dispose(); incoming.Dispose();
        Assert.Equal(lru ? 128 : 96, lane.CachedBytes);
        Assert.Equal(0, lane.AllocatedBytes);
        using var small = lane.Allocate(8);
        Assert.Equal(allocations + (lru ? 1 : 0), lane.AllocationCount);
        using var large = lane.Allocate(16);
        Assert.Equal(allocations + (lru ? 1 : 0), lane.AllocationCount);
        long evicted = lane.DetailedProfiler!.Snapshot().Where(e => e.Kind == "cache-eviction").Sum(e => e.Bytes);
        Assert.Equal(lru ? 32 : 0, evicted);
        Assert.InRange(lane.CachedBytes, 0, 128);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(2048)]
    public void PendingEvictionsPreserveCopiesAndBoundPhysicalBytes(long deferred)
    {
        RequireArc();
        const int n = 131;
        const long budget = 1536;
        using var lane = new ArcExecutionLane(options: new() {
            LruBufferPool = true, ReuseRetiredBuffers = false, BufferPoolBytes = budget, DeferredReleaseBytes = deferred,
            QueuedKernelLimit = 4096, DetailedProfiling = true });
        using var one = lane.Upload(Enumerable.Repeat(1f, n).ToArray());
        using var result = lane.Upload(new float[n]);
        long uploads = lane.H2DBytes;
        bool hadRetired = false;
        for (int i = 0; i < 64; i++)
        {
            int capacity = i % 2 == 0 ? n : 257;
            using var temporary = lane.Allocate(capacity);
            using var copy = lane.Allocate(capacity);
            lane.Run("copy_scale", n, 0, one, temporary, n, (float)i, 0);
            lane.CopyBytes(temporary, copy, 0, 0, n * 4);
            lane.Run("add", n, 0, copy, result, result, n);
            temporary.Dispose(); copy.Dispose();
            Assert.False(temporary.IsAlive); Assert.False(copy.IsAlive);
            Assert.InRange(lane.CachedBytes, 0, budget);
            Assert.InRange(lane.RetiredBytes, 0, deferred);
            Assert.Equal(n * 8, lane.AllocatedBytes);
            Assert.True(lane.AllocatedBytes + lane.CachedBytes + lane.RetiredBytes <= lane.PeakAllocatedBytes);
            hadRetired |= lane.RetiredBytes > 0;
        }
        Assert.Equal(deferred > 0, hadRetired);
        Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(0, lane.D2HBytes);
        // Invalid submission must drain earlier work, including evicted handles.
        Assert.Throws<ArgumentException>(() => lane.Run("copy_scale", n, 0, one, result, "invalid"));
        Assert.Equal(0, lane.RetiredBytes);
        var actual = new float[n]; lane.Read(result, actual);
        Assert.All(actual, value => Assert.Equal(2016f, value));
        Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(n * 4, lane.D2HBytes);
        Assert.InRange(lane.PeakAllocatedBytes, 0, n * 8 + 2 * 257 * 4 + budget + deferred);
        var profile = lane.DetailedProfiler!.Snapshot();
        Assert.Contains(profile, e => e.Kind == "cache-eviction");
        Assert.Contains(profile, e => e.Kind == "synchronize" && e.Detail.EndsWith("pool-lru-eviction"));
    }

    [Fact]
    public void DisposingLaneDrainsPendingCopyRetirementAndBorrowedViewsExactlyOnce()
    {
        RequireArc();
        var lane = new ArcExecutionLane(options: new() {
            LruBufferPool = true, BufferPoolBytes = 64, DeferredReleaseBytes = 64 });
        using var source = lane.Upload(Enumerable.Repeat(1f, 16).ToArray());
        using var old = lane.Allocate(16);
        using var recent = lane.Allocate(8);
        using var borrowed = source.Borrow();
        try
        {
            lane.CopyBytes(source, old, 0, 0, 64);
            old.Dispose();
            recent.Dispose(); // Old cache entry becomes event-fenced retirement.
            Assert.Equal(64, lane.AllocatedBytes);
            Assert.Equal(32, lane.CachedBytes);
            Assert.Equal(64, lane.RetiredBytes);
            Assert.True(source.IsAlive); Assert.True(borrowed.IsAlive);
            lane.Dispose(); lane.Dispose();
            Assert.Equal(0, lane.AllocatedBytes);
            Assert.Equal(0, lane.CachedBytes);
            Assert.Equal(0, lane.RetiredBytes);
            Assert.False(source.IsAlive); Assert.False(borrowed.IsAlive);
            old.Dispose(); recent.Dispose(); borrowed.Dispose(); source.Dispose();
            Assert.Throws<ObjectDisposedException>(() => lane.Allocate(1));
        }
        finally { lane.Dispose(); }
    }

    [Fact]
    public void HostTemporariesAndFailedDispatchReturnToBoundedPool()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: new() {
            LruBufferPool = true, BufferPoolBytes = 80, DeferredReleaseBytes = 80 });
        float[] left = Enumerable.Range(0, 17).Select(i => (float)i).ToArray();
        float[] right = Enumerable.Repeat(3f, 17).ToArray();
        var output = new float[17];
        lane.Run("add", 17, 0, ArcExecutionLane.In(left), ArcExecutionLane.In(right), ArcExecutionLane.Out(output), 17);
        Assert.Equal(left.Select(value => value + 3).ToArray(), output);
        Assert.Equal(0, lane.AllocatedBytes); Assert.Equal(68, lane.CachedBytes); Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(136, lane.H2DBytes); Assert.Equal(68, lane.D2HBytes);
        Assert.Throws<ArgumentException>(() => lane.Run("add", 17, 0,
            ArcExecutionLane.In(left), ArcExecutionLane.In(right), "invalid", 17));
        Assert.Equal(0, lane.AllocatedBytes); Assert.Equal(68, lane.CachedBytes); Assert.Equal(0, lane.RetiredBytes);
    }

    [Fact]
    public void OversizedReleaseDoesNotEvictReusableSmallBuffers()
    {
        RequireArc();
        using var lane = new ArcExecutionLane(options: new() {
            LruBufferPool = true, BufferPoolBytes = 64, DeferredReleaseBytes = 0 });
        using var small = lane.Allocate(16);
        using var large = lane.Allocate(65);
        small.Dispose();
        long allocations = lane.AllocationCount;
        large.Dispose();
        Assert.Equal(64, lane.CachedBytes); Assert.Equal(0, lane.AllocatedBytes);
        using var reuse = lane.Allocate(16);
        Assert.Equal(allocations, lane.AllocationCount);
        Assert.Equal(1, lane.PoolHits);
    }

    private static void RequireArc() => Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
}
