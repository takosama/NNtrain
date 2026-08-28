namespace NNtrain;

/// <summary>
/// Process-wide lifecycle counters for asynchronous pinned upload slots.
/// They deliberately exclude unrelated checkpoint staging and scalar
/// readbacks so a training session can prove that its batch-upload resources
/// are bounded and completely released.
/// </summary>
internal static class NativeCudaPinnedUploadTracker
{
    private static long _createdSlotCount;
    private static long _disposedSlotCount;
    private static long _activeSlotCount;
    private static long _hostAllocationCount;
    private static long _hostFreeCount;
    private static long _activePinnedBytes;
    private static long _eventCreateCount;
    private static long _eventDestroyCount;
    private static long _activeEventCount;
    private static long _reuseSynchronizationCount;

    internal static NativeCudaPinnedUploadTelemetry Telemetry => new(
        Interlocked.Read(ref _createdSlotCount),
        Interlocked.Read(ref _disposedSlotCount),
        Interlocked.Read(ref _activeSlotCount),
        Interlocked.Read(ref _hostAllocationCount),
        Interlocked.Read(ref _hostFreeCount),
        Interlocked.Read(ref _activePinnedBytes),
        Interlocked.Read(ref _eventCreateCount),
        Interlocked.Read(ref _eventDestroyCount),
        Interlocked.Read(ref _activeEventCount),
        Interlocked.Read(ref _reuseSynchronizationCount));

    internal static void RecordHostAllocation(nuint bytes)
    {
        Interlocked.Increment(ref _hostAllocationCount);
        Interlocked.Add(ref _activePinnedBytes, checked((long)bytes));
    }

    internal static void RecordHostFree(nuint bytes)
    {
        Interlocked.Increment(ref _hostFreeCount);
        Interlocked.Add(ref _activePinnedBytes, -checked((long)bytes));
    }

    internal static void RecordEventCreate()
    {
        Interlocked.Increment(ref _eventCreateCount);
        Interlocked.Increment(ref _activeEventCount);
    }

    internal static void RecordEventDestroy()
    {
        Interlocked.Increment(ref _eventDestroyCount);
        Interlocked.Decrement(ref _activeEventCount);
    }

    internal static void RecordSlotCreated()
    {
        Interlocked.Increment(ref _createdSlotCount);
        Interlocked.Increment(ref _activeSlotCount);
    }

    internal static void RecordSlotDisposed()
    {
        Interlocked.Increment(ref _disposedSlotCount);
        Interlocked.Decrement(ref _activeSlotCount);
    }

    internal static void RecordReuseSynchronization()
        => Interlocked.Increment(ref _reuseSynchronizationCount);
}

internal readonly record struct NativeCudaPinnedUploadTelemetry(
    long CreatedSlotCount,
    long DisposedSlotCount,
    long ActiveSlotCount,
    long HostAllocationCount,
    long HostFreeCount,
    long ActivePinnedBytes,
    long EventCreateCount,
    long EventDestroyCount,
    long ActiveEventCount,
    long ReuseSynchronizationCount)
{
    public static NativeCudaPinnedUploadTelemetry operator -(
        NativeCudaPinnedUploadTelemetry left,
        NativeCudaPinnedUploadTelemetry right)
        => new(
            left.CreatedSlotCount - right.CreatedSlotCount,
            left.DisposedSlotCount - right.DisposedSlotCount,
            left.ActiveSlotCount - right.ActiveSlotCount,
            left.HostAllocationCount - right.HostAllocationCount,
            left.HostFreeCount - right.HostFreeCount,
            left.ActivePinnedBytes - right.ActivePinnedBytes,
            left.EventCreateCount - right.EventCreateCount,
            left.EventDestroyCount - right.EventDestroyCount,
            left.ActiveEventCount - right.ActiveEventCount,
            left.ReuseSynchronizationCount - right.ReuseSynchronizationCount);
}
