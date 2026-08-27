using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Memory;

public enum CudaMemoryKind
{
    Persistent = 0,
    Transient = 1,
    Workspace = 2,
    PinnedStaging = 3,
}

/// <summary>Native allocation adapter used by the CUDA memory owner.</summary>
public interface ICudaMemoryAllocator
{
    nint Allocate(int deviceIndex, nuint byteLength, CudaMemoryKind kind);
    void Release(
        int deviceIndex,
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind);
}

/// <summary>
/// Completion primitive recorded after the final use of an allocation.
/// Implementations normally wrap a CUDA event, but the ownership contract is
/// deliberately CUDA-independent so it can be verified without a GPU.
/// </summary>
public interface ICudaCompletionFence : IDisposable
{
    bool IsCompleted { get; }

    void Wait();
}

/// <summary>Callback invoked between a failed allocation and its sole retry.</summary>
public delegate void CudaOutOfMemoryRecovery(
    int deviceIndex,
    nuint requestedByteLength,
    CudaMemoryKind kind);

/// <summary>
/// Per-lane allocation owner. A disposed lease with an unfinished completion
/// fence remains owned as a pending allocation until the fence completes.
/// Native and fence cleanup are non-throwing and continue after individual
/// failures.
/// </summary>
public sealed class CudaMemoryManager : IDeviceMemoryManager
{
    private readonly object _sync = new();
    private readonly object _allocationGate = new();
    private readonly ICudaMemoryAllocator _allocator;
    private readonly CudaOutOfMemoryRecovery? _oomRecovery;
    private readonly Dictionary<long, TrackedAllocation> _allocations = [];
    private readonly ConcurrentQueue<Exception> _releaseErrors = new();
    private readonly ConcurrentQueue<Exception> _recoveryErrors = new();
    private long _nextAllocationId;
    private long _activeCount;
    private long _pendingCount;
    private long _activeBytes;
    private long _pendingBytes;
    private int _disposed;

    public CudaMemoryManager(int deviceIndex, ICudaMemoryAllocator allocator)
        : this(deviceIndex, allocator, oomRecovery: null)
    {
    }

    public CudaMemoryManager(
        int deviceIndex,
        ICudaMemoryAllocator allocator,
        CudaOutOfMemoryRecovery? oomRecovery)
    {
        if (deviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        DeviceIndex = deviceIndex;
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        _oomRecovery = oomRecovery;
    }

    public int DeviceIndex { get; }

    /// <summary>Total number of native allocations still owned by this manager.</summary>
    public long AllocationCount
    {
        get
        {
            lock (_sync)
                return _allocations.Count;
        }
    }

    public long ActiveAllocationCount
    {
        get
        {
            lock (_sync)
                return _activeCount;
        }
    }

    public long PendingAllocationCount
    {
        get
        {
            lock (_sync)
                return _pendingCount;
        }
    }

    /// <summary>Total bytes still owned, including event-fenced pending bytes.</summary>
    public long AllocatedBytes
    {
        get
        {
            lock (_sync)
                return checked(_activeBytes + _pendingBytes);
        }
    }

    public long ActiveBytes
    {
        get
        {
            lock (_sync)
                return _activeBytes;
        }
    }

    public long PendingBytes
    {
        get
        {
            lock (_sync)
                return _pendingBytes;
        }
    }

    public IReadOnlyList<Exception> ReleaseErrors => _releaseErrors.ToArray();

    public IReadOnlyList<Exception> RecoveryErrors => _recoveryErrors.ToArray();

    public CudaMemoryLease Allocate(nuint byteLength, CudaMemoryKind kind)
    {
        ValidateAllocation(byteLength, kind);

        // Allocation is serialized with manager disposal. Native allocation is
        // intentionally outside _sync so recovery can collect completed leases.
        lock (_allocationGate)
        {
            ThrowIfDisposed();
            nint pointer = AllocateWithSingleRecovery(byteLength, kind);
            CudaMemoryLease? lease = null;
            try
            {
                lock (_sync)
                {
                    // Dispose may set the flag while it waits for
                    // _allocationGate. Do not publish after shutdown starts.
                    ThrowIfDisposed();

                    long id = checked(_nextAllocationId + 1);
                    long nextCount = checked(_activeCount + 1);
                    long nextBytes =
                        checked(_activeBytes + (long)byteLength);
                    lease = new CudaMemoryLease(
                        this,
                        id,
                        pointer,
                        byteLength,
                        kind);
                    var allocation = new TrackedAllocation(
                        id,
                        pointer,
                        byteLength,
                        kind,
                        lease);

                    // Every potentially throwing calculation is complete before
                    // publishing the entry. State assignments after Add cannot
                    // strand a closed SafeHandle without its native pointer.
                    _allocations.Add(id, allocation);
                    _nextAllocationId = id;
                    _activeCount = nextCount;
                    _activeBytes = nextBytes;
                    return lease;
                }
            }
            catch
            {
                lease?.SetHandleAsInvalid();
                ReleaseNative(pointer, byteLength, kind);
                throw;
            }
        }
    }

    /// <summary>
    /// Releases pending allocations whose fences are already complete.
    /// Fence queries, fence disposal, and native release failures are recorded
    /// and never stop other pending allocations from being processed.
    /// </summary>
    public int CollectCompleted()
    {
        TrackedAllocation[] candidates;
        lock (_sync)
        {
            candidates = _allocations.Values
                .Where(static allocation =>
                    allocation.State == AllocationState.Pending)
                .ToArray();
        }

        int released = 0;
        foreach (TrackedAllocation candidate in candidates)
        {
            if (!TryClaimCompleted(
                    candidate.AllocationId,
                    out TrackedAllocation owned))
                continue;

            ReleasePending(owned);
            released++;
        }

        return released;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        // Wait for any native allocation currently in progress before taking
        // the active snapshot. Subsequent Allocate calls observe _disposed.
        lock (_allocationGate)
        {
        }

        while (true)
        {
            CudaMemoryLease[] leases;
            lock (_sync)
            {
                leases = _allocations.Values
                    .Where(static allocation =>
                        allocation.State == AllocationState.Active)
                    .Select(static allocation => allocation.Lease)
                    .ToArray();
            }
            if (leases.Length == 0)
                break;

            foreach (CudaMemoryLease lease in leases)
            {
                try
                {
                    lease.Dispose();
                }
                catch (Exception exception)
                {
                    // SafeHandle.ReleaseHandle is non-throwing. Keep this guard
                    // so a future lease implementation cannot stop shutdown.
                    RecordReleaseError(exception);
                }
            }
        }

        while (true)
        {
            TrackedAllocation[] pending;
            lock (_sync)
            {
                pending = _allocations.Values
                    .Where(static allocation =>
                        allocation.State == AllocationState.Pending)
                    .ToArray();
            }
            if (pending.Length == 0)
                break;

            foreach (TrackedAllocation candidate in pending)
            {
                if (TryClaimPending(
                        candidate.AllocationId,
                        out TrackedAllocation owned))
                {
                    WaitFence(owned.Fence!);
                    ReleasePending(owned);
                }
            }
        }
    }

    internal void ReleaseLease(
        long allocationId,
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind,
        ICudaCompletionFence? fence)
    {
        try
        {
            bool defer = fence is not null && !IsFenceCompleted(fence);
            bool ownsAllocation;

            lock (_sync)
            {
                ownsAllocation = _allocations.TryGetValue(
                    allocationId,
                    out TrackedAllocation? allocation);
                if (ownsAllocation && allocation!.State == AllocationState.Active)
                {
                    long nextActiveCount = checked(_activeCount - 1);
                    long nextActiveBytes =
                        checked(_activeBytes - (long)byteLength);
                    if (defer)
                    {
                        long nextPendingCount = checked(_pendingCount + 1);
                        long nextPendingBytes =
                            checked(_pendingBytes + (long)byteLength);
                        allocation.Fence = fence;
                        allocation.State = AllocationState.Pending;
                        _pendingCount = nextPendingCount;
                        _pendingBytes = nextPendingBytes;
                    }
                    else
                    {
                        _allocations.Remove(allocationId);
                    }

                    _activeCount = nextActiveCount;
                    _activeBytes = nextActiveBytes;
                }
                else
                {
                    ownsAllocation = false;
                }
            }

            if (!ownsAllocation)
            {
                DisposeFence(fence);
                return;
            }

            if (!defer)
            {
                ReleaseNative(pointer, byteLength, kind);
                DisposeFence(fence);
            }
        }
        catch (Exception exception)
        {
            // This is called by SafeHandle.ReleaseHandle and must never throw.
            RecordReleaseError(exception);
        }
    }

    private nint AllocateWithSingleRecovery(
        nuint byteLength,
        CudaMemoryKind kind)
    {
        try
        {
            return AllocateNative(byteLength, kind);
        }
        catch (OutOfMemoryException)
        {
            CollectCompleted();
            if (_oomRecovery is not null)
            {
                try
                {
                    _oomRecovery(DeviceIndex, byteLength, kind);
                }
                catch (Exception exception)
                {
                    // Recovery is best-effort; its failure must not create more
                    // retries or prevent the one allowed native retry.
                    RecordRecoveryError(exception);
                }
            }

            return AllocateNative(byteLength, kind);
        }
    }

    private nint AllocateNative(nuint byteLength, CudaMemoryKind kind)
    {
        nint pointer = _allocator.Allocate(DeviceIndex, byteLength, kind);
        if (pointer == nint.Zero)
        {
            throw new OutOfMemoryException(
                $"CUDA allocation of {byteLength} bytes returned a null pointer.");
        }

        return pointer;
    }

    private bool TryClaimPending(
        long allocationId,
        out TrackedAllocation pending)
    {
        lock (_sync)
        {
            return TryClaimPendingLocked(allocationId, out pending);
        }
    }

    private bool TryClaimCompleted(
        long allocationId,
        out TrackedAllocation pending)
    {
        lock (_sync)
        {
            if (!_allocations.TryGetValue(
                    allocationId,
                    out TrackedAllocation? allocation) ||
                allocation.State != AllocationState.Pending ||
                !IsFenceCompleted(allocation.Fence!))
            {
                pending = null!;
                return false;
            }

            return TryClaimPendingLocked(allocationId, out pending);
        }
    }

    private bool TryClaimPendingLocked(
        long allocationId,
        out TrackedAllocation pending)
    {
        if (!_allocations.TryGetValue(
                allocationId,
                out TrackedAllocation? allocation) ||
            allocation.State != AllocationState.Pending)
        {
            pending = null!;
            return false;
        }

        long nextPendingCount = checked(_pendingCount - 1);
        long nextPendingBytes =
            checked(_pendingBytes - (long)allocation.ByteLength);
        _allocations.Remove(allocationId);
        _pendingCount = nextPendingCount;
        _pendingBytes = nextPendingBytes;
        pending = allocation;
        return true;
    }

    private void ReleasePending(TrackedAllocation pending)
    {
        ReleaseNative(pending.Pointer, pending.ByteLength, pending.Kind);
        DisposeFence(pending.Fence!);
    }

    private void ReleaseNative(
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind)
    {
        try
        {
            _allocator.Release(DeviceIndex, pointer, byteLength, kind);
        }
        catch (Exception exception)
        {
            RecordReleaseError(exception);
        }
    }

    private bool IsFenceCompleted(ICudaCompletionFence fence)
    {
        try
        {
            return fence.IsCompleted;
        }
        catch (Exception exception)
        {
            RecordReleaseError(exception);
            return false;
        }
    }

    private void WaitFence(ICudaCompletionFence fence)
    {
        try
        {
            fence.Wait();
        }
        catch (Exception exception)
        {
            // During manager shutdown no more work may use the allocation. Even
            // if event waiting fails, continue releasing every native buffer.
            RecordReleaseError(exception);
        }
    }

    private void DisposeFence(ICudaCompletionFence? fence)
    {
        if (fence is null)
            return;
        try
        {
            fence.Dispose();
        }
        catch (Exception exception)
        {
            RecordReleaseError(exception);
        }
    }

    private void RecordReleaseError(Exception exception)
    {
        try
        {
            _releaseErrors.Enqueue(exception);
        }
        catch
        {
            // Error reporting must never make SafeHandle cleanup throwing.
        }
    }

    private void RecordRecoveryError(Exception exception)
    {
        try
        {
            _recoveryErrors.Enqueue(exception);
        }
        catch
        {
            // OOM recovery remains best-effort even when diagnostics cannot be
            // retained because the managed process is also under pressure.
        }
    }

    private static void ValidateAllocation(
        nuint byteLength,
        CudaMemoryKind kind)
    {
        if (byteLength == 0)
            throw new ArgumentOutOfRangeException(nameof(byteLength));
        if (byteLength > long.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(byteLength));
        if (!Enum.IsDefined(kind))
            throw new ArgumentOutOfRangeException(nameof(kind));
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);

    private enum AllocationState
    {
        Active,
        Pending,
    }

    private sealed class TrackedAllocation(
        long allocationId,
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind,
        CudaMemoryLease lease)
    {
        public long AllocationId { get; } = allocationId;

        public nint Pointer { get; } = pointer;

        public nuint ByteLength { get; } = byteLength;

        public CudaMemoryKind Kind { get; } = kind;

        public CudaMemoryLease Lease { get; } = lease;

        public AllocationState State { get; set; }

        public ICudaCompletionFence? Fence { get; set; }
    }
}

/// <summary>A single allocation whose lifetime is owned by its manager.</summary>
public sealed class CudaMemoryLease : SafeHandle
{
    private readonly object _lifetimeSync = new();
    private readonly CudaMemoryManager _owner;
    private readonly long _allocationId;
    private ICudaCompletionFence? _releaseFence;
    private bool _releaseStarted;

    internal CudaMemoryLease(
        CudaMemoryManager owner,
        long allocationId,
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind)
        : base(nint.Zero, ownsHandle: true)
    {
        _owner = owner;
        _allocationId = allocationId;
        ByteLength = byteLength;
        Kind = kind;
        SetHandle(pointer);
    }

    public override bool IsInvalid => handle == nint.Zero;

    public nuint ByteLength { get; }

    public CudaMemoryKind Kind { get; }

    public nint Pointer
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsClosed, this);
            return DangerousGetHandle();
        }
    }

    /// <summary>
    /// Transfers ownership of a completion fence to this lease. The fence must
    /// be attached before disposal and may be attached only once.
    /// </summary>
    public void SetReleaseFence(ICudaCompletionFence fence)
    {
        ArgumentNullException.ThrowIfNull(fence);
        lock (_lifetimeSync)
        {
            if (_releaseStarted || IsClosed)
                throw new ObjectDisposedException(nameof(CudaMemoryLease));
            if (_releaseFence is not null)
            {
                throw new InvalidOperationException(
                    "A completion fence is already attached to this CUDA memory lease.");
            }

            _releaseFence = fence;
        }
    }

    /// <summary>Attaches a fence and disposes the lease in one operation.</summary>
    public void DisposeAfter(ICudaCompletionFence fence)
    {
        SetReleaseFence(fence);
        Dispose();
    }

    protected override bool ReleaseHandle()
    {
        try
        {
            ICudaCompletionFence? fence;
            lock (_lifetimeSync)
            {
                _releaseStarted = true;
                fence = _releaseFence;
                _releaseFence = null;
            }

            _owner.ReleaseLease(
                _allocationId,
                handle,
                ByteLength,
                Kind,
                fence);
        }
        catch
        {
            // SafeHandle cleanup must never allow an exception to escape a
            // finalizer or interrupt cleanup of sibling allocations.
        }

        return true;
    }
}
