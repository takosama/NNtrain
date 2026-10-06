using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMemoryAccountingTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ExternalReservationReclaimsIdleAndQueuedOwnersAndPreventsOvercommit(bool lru)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        using var lane = new ArcExecutionLane(options: new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35VisionKernelsOnly = true,
            CacheProgramBinary = true, BufferPoolBytes = 64, PhysicalBufferBudgetBytes = 256,
            DeferredReleaseBytes = 256, LruBufferPool = lru, ReuseRetiredBuffers = false
        });
        using var live = lane.Upload(Enumerable.Repeat(7f, 16).ToArray());
        using var result = lane.Allocate(16);
        using var idle = lane.Allocate(16);
        using var queued = lane.Allocate(16);
        lane.CopyBytes(live, queued, 0, 0, 64);
        lane.CopyBytes(queued, result, 0, 0, 64);
        idle.Dispose(); queued.Dispose();
        Assert.Equal(128, lane.AllocatedBytes);
        Assert.Equal(64, lane.CachedBytes);
        Assert.Equal(64, lane.RetiredBytes);

        lane.SetExternalMemoryReservation(96);
        Assert.Equal(96, lane.ExternalMemoryReservationBytes);
        Assert.Equal(160, lane.EffectivePhysicalBufferBudgetBytes);
        Assert.Equal(0, lane.CachedBytes);
        Assert.Equal(0, lane.RetiredBytes);
        Assert.Equal(128, lane.NativeReleasedBytes);
        Assert.True(live.IsAlive); Assert.True(result.IsAlive);
        var copied = new float[16]; lane.Read(result, copied);
        Assert.All(copied, value => Assert.Equal(7f, value));

        using var fitting = lane.Allocate(8);
        long nativeBytes = lane.NativeAllocatedBytes, allocations = lane.AllocationCount;
        Assert.Throws<InvalidOperationException>(() => lane.Allocate(1));
        Assert.Equal(nativeBytes, lane.NativeAllocatedBytes);
        Assert.Equal(allocations, lane.AllocationCount);
        Assert.Equal(160, lane.AllocatedBytes);
        Assert.Throws<InvalidOperationException>(() => lane.SetExternalMemoryReservation(128));
        Assert.Throws<ArgumentOutOfRangeException>(() => lane.SetExternalMemoryReservation(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => lane.SetExternalMemoryReservation(257));
        Assert.Equal(96, lane.ExternalMemoryReservationBytes);
        Assert.Equal(160, lane.EffectivePhysicalBufferBudgetBytes);
        Assert.True(live.IsAlive); Assert.True(result.IsAlive); Assert.True(fitting.IsAlive);

        lane.SetExternalMemoryReservation(0);
        Assert.Equal(256, lane.EffectivePhysicalBufferBudgetBytes);
        using var admitted = lane.Allocate(16);
        Assert.Equal(224, lane.AllocatedBytes);
        Assert.Equal(lane.NativeAllocatedBytes - lane.NativeReleasedBytes,
            lane.AllocatedBytes + lane.CachedBytes + lane.RetiredBytes);
    }

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
