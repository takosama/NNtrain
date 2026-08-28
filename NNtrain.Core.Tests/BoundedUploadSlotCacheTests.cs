using System.Collections.Concurrent;
using NNtrain;
using Xunit;

public sealed class BoundedUploadSlotCacheTests
{
    [Fact]
    public void TwoSlotsPerLengthKeepInputAndTargetIndependent()
    {
        int nextId = 0;
        var used = new List<int>();
        using var cache = new BoundedUploadSlotCache<FakeSlot>(
            length => new FakeSlot(length, ++nextId));

        cache.Use(128, slot => used.Add(slot.Id));
        cache.Use(128, slot => used.Add(slot.Id));
        cache.Use(128, slot => used.Add(slot.Id));

        Assert.Equal(3, used.Count);
        Assert.NotEqual(used[0], used[1]);
        Assert.Equal(used[0], used[2]);
        BoundedUploadSlotCacheTelemetry telemetry = cache.Telemetry;
        Assert.Equal(1, telemetry.ActiveLengthCount);
        Assert.Equal(2, telemetry.ActiveSlotCount);
        Assert.Equal(256, telemetry.ActiveElementCapacity);
        Assert.Equal(2, telemetry.CreatedSlotCount);
        Assert.Equal(0, telemetry.DisposedSlotCount);
        Assert.Equal(3, telemetry.UseCount);
        Assert.Equal([128], telemetry.ActiveLengths);
    }

    [Fact]
    public void HundredsOfAdaptiveUsesRetainOnlyThreeRecentLengths()
    {
        int nextId = 0;
        var slots = new ConcurrentBag<FakeSlot>();
        var cache = new BoundedUploadSlotCache<FakeSlot>(length =>
        {
            var slot = new FakeSlot(length, Interlocked.Increment(ref nextId));
            slots.Add(slot);
            return slot;
        });
        int[] lengths = [32, 64, 96, 32, 128];

        Parallel.For(0, 512, iteration =>
            cache.Use(lengths[iteration % lengths.Length], static _ => { }));

        BoundedUploadSlotCacheTelemetry active = cache.Telemetry;
        Assert.Equal(3, active.ActiveLengthCount);
        Assert.Equal(6, active.ActiveSlotCount);
        Assert.Equal(512, active.UseCount);
        Assert.True(active.CreatedSlotCount > active.ActiveSlotCount);
        Assert.Equal(
            active.CreatedSlotCount - active.ActiveSlotCount,
            active.DisposedSlotCount);

        cache.Dispose();
        cache.Dispose();

        BoundedUploadSlotCacheTelemetry disposed = cache.Telemetry;
        Assert.Equal(0, disposed.ActiveLengthCount);
        Assert.Equal(0, disposed.ActiveSlotCount);
        Assert.Equal(disposed.CreatedSlotCount, disposed.DisposedSlotCount);
        Assert.All(slots, slot => Assert.Equal(1, slot.DisposeCount));
    }

    [Fact]
    public void CleanupFailuresNeverPreventRemainingSlotsFromReleasing()
    {
        int nextId = 0;
        var slots = new List<FakeSlot>();
        var cache = new BoundedUploadSlotCache<FakeSlot>(
            length =>
            {
                var slot = new FakeSlot(
                    length,
                    ++nextId,
                    throwOnDispose: length == 16);
                slots.Add(slot);
                return slot;
            },
            maximumLengths: 1);
        bool newerBucketUsed = false;
        cache.Use(16, static _ => { });

        // Adding the second length evicts both failing slots from the first
        // bucket, records both failures, and still performs the requested use.
        cache.Use(32, _ => newerBucketUsed = true);
        Assert.True(newerBucketUsed);
        Assert.Equal(2, cache.ReleaseErrors.Count);

        AggregateException failure = Assert.Throws<AggregateException>(
            cache.Dispose);

        Assert.Equal(2, failure.Flatten().InnerExceptions.Count);
        Assert.All(slots, slot => Assert.Equal(1, slot.DisposeCount));
        Assert.Equal(
            cache.Telemetry.CreatedSlotCount,
            cache.Telemetry.DisposedSlotCount);
    }

    private sealed class FakeSlot(
        int length,
        int id,
        bool throwOnDispose = false) : IDisposable
    {
        private int _disposed;
        internal int Length { get; } = length;
        internal int Id { get; } = id;
        internal int DisposeCount => Volatile.Read(ref _disposed);

        public void Dispose()
        {
            Interlocked.Increment(ref _disposed);
            if (throwOnDispose)
                throw new InvalidOperationException($"slot {Id} failed");
        }
    }
}
