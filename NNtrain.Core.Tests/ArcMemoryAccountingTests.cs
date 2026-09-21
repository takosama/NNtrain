using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMemoryAccountingTests
{
    [Fact]
    public void PhysicalBudgetEvictsIdleOwnersWithoutInvalidatingLiveBuffers()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() {
            BufferPoolBytes = 4096, PhysicalBufferBudgetBytes = 192,
            DeferredReleaseBytes = 4096, LruBufferPool = true });
        using var live = lane.Upload(Enumerable.Repeat(7f, 16).ToArray());
        using var idle = lane.Allocate(16);
        lane.Run("resident_zero", 16, 0, idle, 16);
        idle.Dispose();
        using var next = lane.Allocate(32);
        Assert.True(live.IsAlive); Assert.True(next.IsAlive);
        Assert.Equal(0, lane.CachedBytes);
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(64, lane.NativeReleasedBytes);
        Assert.Equal(192, lane.PeakAllocatedBytes);
        var values = new float[16]; lane.Read(live, values);
        Assert.All(values, value => Assert.Equal(7f, value));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ChurnIsDistinctFromResidentBytesAndAllNativeOwnershipIsReleased(bool lru)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new() {
            BufferPoolBytes = 128, LruBufferPool = lru, DeferredReleaseBytes = 512,
            ReuseRetiredBuffers = true });
        for (int iteration = 0; iteration < 20; iteration++)
        {
            using var value = lane.Allocate(16);
            lane.Run("resident_zero", 16, 0, value, 16);
        }
        lane.Synchronize();
        Assert.Equal(20 * 64, lane.RequestedBytes);
        Assert.Equal(64, lane.NativeAllocatedBytes);
        Assert.Equal(64, lane.PeakAllocatedBytes);
        Assert.Equal(0, lane.NativeReleasedBytes);
        Assert.Equal(lane.NativeAllocatedBytes - lane.NativeReleasedBytes,
            lane.AllocatedBytes + lane.CachedBytes + lane.RetiredBytes);
        lane.Dispose();
        Assert.Equal(lane.NativeAllocatedBytes, lane.NativeReleasedBytes);
        Assert.Equal(lane.AllocationCount, lane.NativeReleaseCount);
        lane.Dispose();
        Assert.Equal(64, lane.NativeReleasedBytes);
    }
}
