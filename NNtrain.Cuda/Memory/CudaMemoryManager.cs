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
/// Process-wide count of real allocator calls made by CUDA lane memory
/// managers.  Lease construction/disposal and reusable-cache hits are
/// deliberately excluded: this reports cudaMalloc/cudaFree equivalents, not
/// managed wrapper churn.
/// </summary>
public readonly record struct CudaNativeMemoryTelemetry(
    long AllocationCount,
    long AllocationBytes,
    long ReleaseCount,
    long ReleaseBytes);

/// <summary>Atomic lifecycle snapshot for one lane memory manager.</summary>
public readonly record struct CudaMemoryManagerTelemetry(
    long AllocationCount,
    long AllocatedBytes,
    long ActiveAllocationCount,
    long ActiveBytes,
    long PendingAllocationCount,
    long PendingBytes,
    long CachedAllocationCount,
    long CachedBytes,
    long GraphPinnedAllocationCount,
    long GraphPinnedBytes);

/// <summary>
/// Per-lane allocation owner. A disposed lease with an unfinished completion
/// fence remains owned as a pending allocation until the fence completes.
/// Native and fence cleanup are non-throwing and continue after individual
/// failures.
/// </summary>
public sealed class CudaMemoryManager : IDeviceMemoryManager
{
    // The production language-model head owns one ~404 MiB BF16 transient on
    // each GPU. Retain one hot instance so every step does not fall back to a
    // synchronizing cudaMalloc/cudaFree pair; adaptive shape variants remain
    // bounded by both this byte budget and the entry limit below.
    public const long DefaultMaximumReusableCacheBytes =
        512L * 1024 * 1024;
    // A training graph commonly has several hundred simultaneously-live
    // typed activation slices, even when their aggregate size is small.  The
    // byte budget remains the authoritative VRAM bound; a 32-entry limit
    // forced otherwise reusable buffers through cudaFree/cudaMalloc every
    // step.  This count only prevents pathological zero/small-size metadata
    // growth.
    public const int DefaultMaximumReusableCacheEntries = 1024;

    private static long _nativeAllocationCount;
    private static long _nativeAllocationBytes;
    private static long _nativeReleaseCount;
    private static long _nativeReleaseBytes;

    private readonly object _sync = new();
    private readonly object _allocationGate = new();
    private readonly ICudaMemoryAllocator _allocator;
    private readonly CudaOutOfMemoryRecovery? _oomRecovery;
    private readonly Dictionary<long, TrackedAllocation> _allocations = [];
    private readonly Dictionary<long, GraphCaptureState> _graphCaptures = [];
    private readonly ConcurrentQueue<Exception> _releaseErrors = new();
    private readonly ConcurrentQueue<Exception> _recoveryErrors = new();
    private long _nextAllocationId;
    private long _cacheSequence;
    private long _nextGraphCaptureId;
    private long? _activeGraphCaptureId;
    private long _activeCount;
    private long _pendingCount;
    private long _cachedCount;
    private long _activeBytes;
    private long _pendingBytes;
    private long _cachedBytes;
    private long _graphPinnedCount;
    private long _graphPinnedBytes;
    private int _disposed;

    public CudaMemoryManager(int deviceIndex, ICudaMemoryAllocator allocator)
        : this(
            deviceIndex,
            allocator,
            oomRecovery: null,
            DefaultMaximumReusableCacheBytes,
            DefaultMaximumReusableCacheEntries)
    {
    }

    public CudaMemoryManager(
        int deviceIndex,
        ICudaMemoryAllocator allocator,
        CudaOutOfMemoryRecovery? oomRecovery,
        long maximumReusableCacheBytes = DefaultMaximumReusableCacheBytes,
        int maximumReusableCacheEntries =
            DefaultMaximumReusableCacheEntries)
    {
        if (deviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        if (maximumReusableCacheBytes < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maximumReusableCacheBytes));
        }
        if (maximumReusableCacheEntries < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maximumReusableCacheEntries));
        }
        DeviceIndex = deviceIndex;
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        _oomRecovery = oomRecovery;
        MaximumReusableCacheBytes = maximumReusableCacheBytes;
        MaximumReusableCacheEntries = maximumReusableCacheEntries;
    }

    public int DeviceIndex { get; }

    public static CudaNativeMemoryTelemetry NativeTelemetry => new(
        Interlocked.Read(ref _nativeAllocationCount),
        Interlocked.Read(ref _nativeAllocationBytes),
        Interlocked.Read(ref _nativeReleaseCount),
        Interlocked.Read(ref _nativeReleaseBytes));

    /// <summary>
    /// Hard per-lane bound for idle transient/workspace storage. A single
    /// allocation larger than this value is released instead of cached.
    /// </summary>
    public long MaximumReusableCacheBytes { get; }

    /// <summary>Hard per-lane bound for the number of idle exact-size leases.</summary>
    public int MaximumReusableCacheEntries { get; }

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

    /// <summary>
    /// Native allocations retained by this lane for an exact-size transient
    /// or workspace reuse. Cached allocations are never shared across lanes.
    /// </summary>
    public long CachedAllocationCount
    {
        get
        {
            lock (_sync)
                return _cachedCount;
        }
    }

    /// <summary>Total bytes still owned, including event-fenced pending bytes.</summary>
    public long AllocatedBytes
    {
        get
        {
            lock (_sync)
                return checked(
                    _activeBytes
                    + _pendingBytes
                    + _cachedBytes
                    + _graphPinnedBytes);
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

    public long CachedBytes
    {
        get
        {
            lock (_sync)
                return _cachedBytes;
        }
    }

    /// <summary>
    /// Capture-local idle buffers and committed graph-reservation buffers.
    /// Active leases remain represented by the active counters until they are
    /// returned to the capture-local pool.
    /// </summary>
    public long GraphPinnedAllocationCount
    {
        get
        {
            lock (_sync)
                return _graphPinnedCount;
        }
    }

    public long GraphPinnedBytes
    {
        get
        {
            lock (_sync)
                return _graphPinnedBytes;
        }
    }

    public CudaMemoryManagerTelemetry Telemetry
    {
        get
        {
            lock (_sync)
            {
                return new CudaMemoryManagerTelemetry(
                    _allocations.Count,
                    checked(
                        _activeBytes
                        + _pendingBytes
                        + _cachedBytes
                        + _graphPinnedBytes),
                    _activeCount,
                    _activeBytes,
                    _pendingCount,
                    _pendingBytes,
                    _cachedCount,
                    _cachedBytes,
                    _graphPinnedCount,
                    _graphPinnedBytes);
            }
        }
    }

    public IReadOnlyList<Exception> ReleaseErrors => _releaseErrors.ToArray();

    public IReadOnlyList<Exception> RecoveryErrors => _recoveryErrors.ToArray();

    public CudaMemoryLease Allocate(nuint byteLength, CudaMemoryKind kind)
        => Acquire(byteLength, kind, reusable: false);

    /// <summary>
    /// Rents exact-size lane-owned storage. Disposing the returned lease keeps
    /// the native allocation in this manager for a later exact-size rent. An
    /// OOM recovery or manager disposal trims the cache deterministically.
    /// </summary>
    public CudaMemoryLease Rent(nuint byteLength, CudaMemoryKind kind)
    {
        if (kind is not CudaMemoryKind.Transient
            and not CudaMemoryKind.Workspace)
        {
            throw new ArgumentException(
                "Only transient and workspace memory may be rented for reuse.",
                nameof(kind));
        }
        return Acquire(byteLength, kind, reusable: true);
    }

    /// <summary>
    /// Starts a thread-affine CUDA Graph capture allocation reservation. Only
    /// transient/workspace allocations made by the calling thread participate;
    /// other threads and other managers retain their ordinary lifecycle.
    /// Dispose without Commit rolls the capture back and reclaims every
    /// participating lease. Commit returns the graph-lifetime reservation.
    /// </summary>
    public CudaGraphCaptureScope BeginGraphCaptureReservation()
    {
        lock (_allocationGate)
        {
            ThrowIfDisposed();
            lock (_sync)
            {
                ThrowIfDisposed();
                if (_activeGraphCaptureId is not null)
                {
                    throw new InvalidOperationException(
                        "This CUDA memory manager already has an active graph capture reservation.");
                }

                long id = checked(_nextGraphCaptureId + 1);
                var capture = new GraphCaptureState(
                    id,
                    Environment.CurrentManagedThreadId);
                _graphCaptures.Add(id, capture);
                _activeGraphCaptureId = id;
                _nextGraphCaptureId = id;
                return new CudaGraphCaptureScope(this, id);
            }
        }
    }

    internal IDisposable CommitGraphCapture(long captureId)
    {
        lock (_allocationGate)
        {
            ThrowIfDisposed();
            lock (_sync)
            {
                ThrowIfDisposed();
                GraphCaptureState capture = RequireGraphCaptureLocked(
                    captureId,
                    GraphCaptureStatus.Capturing);
                if (capture.OwnerManagedThreadId
                    != Environment.CurrentManagedThreadId)
                {
                    throw new InvalidOperationException(
                        "A CUDA Graph capture must be committed on the thread that began it.");
                }

                int activeCaptureLeases = _allocations.Values.Count(
                    allocation => ReferenceEquals(
                            allocation.GraphCapture,
                            capture)
                        && allocation.State == AllocationState.Active);
                if (activeCaptureLeases != 0)
                {
                    throw new InvalidOperationException(
                        $"CUDA Graph capture {captureId} still owns " +
                        $"{activeCaptureLeases} active transient/workspace " +
                        "lease(s). Dispose them before Commit().");
                }

                foreach (TrackedAllocation allocation in _allocations.Values)
                {
                    if (ReferenceEquals(allocation.GraphCapture, capture)
                        && allocation.State
                            == AllocationState.CaptureAvailable)
                    {
                        allocation.State = AllocationState.GraphPinned;
                    }
                }
                capture.Status = GraphCaptureStatus.Committed;
                _activeGraphCaptureId = null;
                return new CudaGraphMemoryReservation(this, captureId);
            }
        }
    }

    internal void RollbackGraphCapture(long captureId)
    {
        GraphCaptureState? capture;
        CudaMemoryLease[] activeLeases;
        lock (_allocationGate)
        {
            lock (_sync)
            {
                if (!_graphCaptures.TryGetValue(
                        captureId,
                        out capture)
                    || capture.Status != GraphCaptureStatus.Capturing)
                {
                    return;
                }
                capture.Status = GraphCaptureStatus.RollingBack;
                if (_activeGraphCaptureId == captureId)
                    _activeGraphCaptureId = null;
                activeLeases = _allocations.Values
                    .Where(allocation => ReferenceEquals(
                            allocation.GraphCapture,
                            capture)
                        && allocation.State == AllocationState.Active)
                    .Select(static allocation => allocation.Lease)
                    .ToArray();
            }

            foreach (CudaMemoryLease lease in activeLeases)
            {
                try
                {
                    lease.Dispose();
                }
                catch (Exception exception)
                {
                    RecordReleaseError(exception);
                }
            }

            ReleaseGraphIdleAllocations(
                capture,
                AllocationState.CaptureAvailable);
            DrainGraphCaptureFences(capture);
            lock (_sync)
            {
                capture.Status = GraphCaptureStatus.Disposed;
                _graphCaptures.Remove(captureId);
            }
        }
    }

    internal void DisposeGraphReservation(long captureId)
    {
        GraphCaptureState? capture;
        lock (_allocationGate)
        {
            lock (_sync)
            {
                if (!_graphCaptures.TryGetValue(
                        captureId,
                        out capture)
                    || capture.Status != GraphCaptureStatus.Committed)
                {
                    return;
                }
                capture.Status = GraphCaptureStatus.Disposing;
            }

            ReleaseGraphIdleAllocations(
                capture,
                AllocationState.GraphPinned);
            DrainGraphCaptureFences(capture);
            lock (_sync)
            {
                capture.Status = GraphCaptureStatus.Disposed;
                _graphCaptures.Remove(captureId);
            }
        }
    }

    private GraphCaptureState RequireGraphCaptureLocked(
        long captureId,
        GraphCaptureStatus expectedStatus)
    {
        if (!_graphCaptures.TryGetValue(
                captureId,
                out GraphCaptureState? capture)
            || capture.Status != expectedStatus
            || _activeGraphCaptureId != captureId)
        {
            throw new InvalidOperationException(
                $"CUDA Graph capture reservation {captureId} is not active.");
        }
        return capture;
    }

    private void ReleaseGraphIdleAllocations(
        GraphCaptureState capture,
        AllocationState expectedState)
    {
        while (TryClaimGraphIdleAllocation(
            capture,
            expectedState,
            out TrackedAllocation? allocation))
        {
            if (allocation.Reusable && TryCacheClaimed(allocation))
                continue;
            ReleaseNative(
                allocation.Pointer,
                allocation.ByteLength,
                allocation.Kind);
        }
    }

    private bool TryClaimGraphIdleAllocation(
        GraphCaptureState capture,
        AllocationState expectedState,
        out TrackedAllocation allocation)
    {
        lock (_sync)
        {
            TrackedAllocation? candidate = _allocations.Values
                .FirstOrDefault(item => ReferenceEquals(
                        item.GraphCapture,
                        capture)
                    && item.State == expectedState);
            if (candidate is null)
            {
                allocation = null!;
                return false;
            }
            _allocations.Remove(candidate.AllocationId);
            _graphPinnedCount = checked(_graphPinnedCount - 1);
            _graphPinnedBytes = checked(
                _graphPinnedBytes - (long)candidate.ByteLength);
            allocation = candidate;
            return true;
        }
    }

    private void DrainGraphCaptureFences(GraphCaptureState capture)
    {
        ICudaCompletionFence[] fences;
        lock (_sync)
        {
            fences = capture.OwnedFences.ToArray();
            capture.OwnedFences.Clear();
        }
        foreach (ICudaCompletionFence fence in fences)
            DisposeFence(fence);
    }

    /// <summary>Releases every currently cached reusable allocation.</summary>
    public int TrimReusableCache()
    {
        TrackedAllocation[] cached;
        lock (_sync)
        {
            cached = _allocations.Values
                .Where(static allocation =>
                    allocation.State == AllocationState.Cached)
                .ToArray();
        }

        int released = 0;
        foreach (TrackedAllocation candidate in cached)
        {
            if (!TryClaimCached(
                    candidate.AllocationId,
                    out TrackedAllocation owned))
            {
                continue;
            }
            ReleaseNative(owned.Pointer, owned.ByteLength, owned.Kind);
            released++;
        }
        return released;
    }

    private CudaMemoryLease Acquire(
        nuint byteLength,
        CudaMemoryKind kind,
        bool reusable)
    {
        ValidateAllocation(byteLength, kind);

        // Allocation is serialized with manager disposal. Native allocation is
        // intentionally outside _sync so recovery can collect completed leases.
        lock (_allocationGate)
        {
            ThrowIfDisposed();
            GraphCaptureState? capture;
            lock (_sync)
                capture = GetActiveGraphCaptureForCurrentThreadLocked();
            if (capture is not null
                && kind is not CudaMemoryKind.Transient
                    and not CudaMemoryKind.Workspace)
            {
                throw new InvalidOperationException(
                    "A graph capture may allocate only transient or workspace memory. " +
                    "Create persistent and pinned-staging storage before capture begins.");
            }

            bool effectiveReusable = reusable;
            nint pointer;
            if (capture is not null
                && TryTakeGraphCaptureAvailable(
                    capture,
                    byteLength,
                    kind,
                    out nint graphPointer,
                    out bool graphReusable))
            {
                pointer = graphPointer;
                effectiveReusable |= graphReusable;
            }
            else if (reusable
                && TryTakeCached(byteLength, kind, out nint cachedPointer))
            {
                pointer = cachedPointer;
            }
            else
            {
                pointer = AllocateWithSingleRecovery(byteLength, kind);
            }
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
                        effectiveReusable,
                        lease,
                        capture);

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
            AbortGraphCapturesForManagerDispose();
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

        TrimReusableCache();
    }

    private void AbortGraphCapturesForManagerDispose()
    {
        GraphCaptureState[] captures;
        TrackedAllocation[] idle;
        lock (_sync)
        {
            captures = _graphCaptures.Values.ToArray();
            foreach (GraphCaptureState capture in captures)
                capture.Status = GraphCaptureStatus.Aborted;
            _activeGraphCaptureId = null;
            _graphCaptures.Clear();

            idle = _allocations.Values
                .Where(static allocation => allocation.State is
                    AllocationState.CaptureAvailable
                    or AllocationState.GraphPinned)
                .ToArray();
            foreach (TrackedAllocation allocation in idle)
            {
                _allocations.Remove(allocation.AllocationId);
                _graphPinnedCount = checked(_graphPinnedCount - 1);
                _graphPinnedBytes = checked(
                    _graphPinnedBytes - (long)allocation.ByteLength);
            }
        }

        foreach (TrackedAllocation allocation in idle)
        {
            ReleaseNative(
                allocation.Pointer,
                allocation.ByteLength,
                allocation.Kind);
        }
        foreach (GraphCaptureState capture in captures)
        {
            DrainGraphCaptureFences(capture);
            capture.Status = GraphCaptureStatus.Disposed;
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
            if (TryReturnLeaseToGraphCapture(
                allocationId,
                byteLength,
                fence))
            {
                return;
            }

            if (IsAbortedGraphCaptureLease(allocationId))
            {
                DisposeFence(fence);
                fence = null;
            }

            bool defer = fence is not null && !IsFenceCompleted(fence);
            bool cache = false;
            bool ownsAllocation;
            List<TrackedAllocation>? evicted = null;

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
                    else if (allocation.Reusable
                        && Volatile.Read(ref _disposed) == 0
                        && CanCache(byteLength)
                        && TryPrepareCacheAdmissionLocked(
                            byteLength,
                            1,
                            out evicted))
                    {
                        long nextCachedCount = checked(_cachedCount + 1);
                        long nextCachedBytes = checked(
                            _cachedBytes + (long)byteLength);
                        allocation.Fence = null;
                        allocation.State = AllocationState.Cached;
                        allocation.CacheSequence = checked(
                            ++_cacheSequence);
                        _cachedCount = nextCachedCount;
                        _cachedBytes = nextCachedBytes;
                        cache = true;
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

            if (!defer && !cache)
            {
                ReleaseNative(pointer, byteLength, kind);
            }
            if (evicted is not null)
            {
                foreach (TrackedAllocation candidate in evicted)
                {
                    ReleaseNative(
                        candidate.Pointer,
                        candidate.ByteLength,
                        candidate.Kind);
                }
            }
            if (!defer)
                DisposeFence(fence);
        }
        catch (Exception exception)
        {
            // This is called by SafeHandle.ReleaseHandle and must never throw.
            RecordReleaseError(exception);
        }
    }

    private bool IsAbortedGraphCaptureLease(long allocationId)
    {
        lock (_sync)
        {
            return _allocations.TryGetValue(
                    allocationId,
                    out TrackedAllocation? allocation)
                && allocation.State == AllocationState.Active
                && allocation.GraphCapture?.Status is
                    GraphCaptureStatus.RollingBack
                    or GraphCaptureStatus.Aborted;
        }
    }

    private bool TryReturnLeaseToGraphCapture(
        long allocationId,
        nuint byteLength,
        ICudaCompletionFence? fence)
    {
        lock (_sync)
        {
            if (Volatile.Read(ref _disposed) != 0
                || !_allocations.TryGetValue(
                    allocationId,
                    out TrackedAllocation? allocation)
                || allocation.State != AllocationState.Active
                || allocation.GraphCapture is not { } capture
                || capture.Status != GraphCaptureStatus.Capturing)
            {
                return false;
            }

            if (fence is not null)
                capture.OwnedFences.Add(fence);
            allocation.State = AllocationState.CaptureAvailable;
            _activeCount = checked(_activeCount - 1);
            _activeBytes = checked(_activeBytes - (long)byteLength);
            _graphPinnedCount = checked(_graphPinnedCount + 1);
            _graphPinnedBytes = checked(
                _graphPinnedBytes + (long)byteLength);
            return true;
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
            TrimReusableCache();
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

        Interlocked.Increment(ref _nativeAllocationCount);
        Interlocked.Add(
            ref _nativeAllocationBytes,
            checked((long)byteLength));

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

    private bool TryTakeCached(
        nuint byteLength,
        CudaMemoryKind kind,
        out nint pointer)
    {
        lock (_sync)
        {
            TrackedAllocation? allocation = _allocations.Values.FirstOrDefault(
                candidate => candidate.State == AllocationState.Cached
                    && candidate.ByteLength == byteLength
                    && candidate.Kind == kind);
            if (allocation is null)
            {
                pointer = nint.Zero;
                return false;
            }

            _allocations.Remove(allocation.AllocationId);
            _cachedCount = checked(_cachedCount - 1);
            _cachedBytes = checked(
                _cachedBytes - (long)allocation.ByteLength);
            pointer = allocation.Pointer;
            return true;
        }
    }

    private bool TryTakeGraphCaptureAvailable(
        GraphCaptureState capture,
        nuint byteLength,
        CudaMemoryKind kind,
        out nint pointer,
        out bool reusable)
    {
        lock (_sync)
        {
            if (capture.Status != GraphCaptureStatus.Capturing
                || _activeGraphCaptureId != capture.CaptureId)
            {
                pointer = nint.Zero;
                reusable = false;
                return false;
            }

            TrackedAllocation? allocation = _allocations.Values
                .FirstOrDefault(candidate =>
                    ReferenceEquals(candidate.GraphCapture, capture)
                    && candidate.State == AllocationState.CaptureAvailable
                    && candidate.ByteLength == byteLength
                    && candidate.Kind == kind);
            if (allocation is null)
            {
                pointer = nint.Zero;
                reusable = false;
                return false;
            }

            _allocations.Remove(allocation.AllocationId);
            _graphPinnedCount = checked(_graphPinnedCount - 1);
            _graphPinnedBytes = checked(
                _graphPinnedBytes - (long)allocation.ByteLength);
            pointer = allocation.Pointer;
            reusable = allocation.Reusable;
            return true;
        }
    }

    private GraphCaptureState? GetActiveGraphCaptureForCurrentThreadLocked()
    {
        if (_activeGraphCaptureId is not long captureId
            || !_graphCaptures.TryGetValue(
                captureId,
                out GraphCaptureState? capture)
            || capture.Status != GraphCaptureStatus.Capturing
            || capture.OwnerManagedThreadId
                != Environment.CurrentManagedThreadId)
        {
            return null;
        }
        return capture;
    }

    private bool TryClaimCached(
        long allocationId,
        out TrackedAllocation cached)
    {
        lock (_sync)
        {
            if (!_allocations.TryGetValue(
                    allocationId,
                    out TrackedAllocation? allocation)
                || allocation.State != AllocationState.Cached)
            {
                cached = null!;
                return false;
            }

            _allocations.Remove(allocationId);
            _cachedCount = checked(_cachedCount - 1);
            _cachedBytes = checked(
                _cachedBytes - (long)allocation.ByteLength);
            cached = allocation;
            return true;
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
        DisposeFence(pending.Fence!);
        pending.Fence = null;
        if (pending.Reusable && TryCacheClaimed(pending))
            return;
        ReleaseNative(pending.Pointer, pending.ByteLength, pending.Kind);
    }

    private bool TryCacheClaimed(TrackedAllocation allocation)
    {
        List<TrackedAllocation>? evicted;
        lock (_sync)
        {
            if (Volatile.Read(ref _disposed) != 0
                || !CanCache(allocation.ByteLength))
            {
                return false;
            }
            if (!TryPrepareCacheAdmissionLocked(
                    allocation.ByteLength,
                    1,
                    out evicted))
            {
                return false;
            }
            allocation.State = AllocationState.Cached;
            allocation.CacheSequence = checked(++_cacheSequence);
            _allocations.Add(allocation.AllocationId, allocation);
            _cachedCount = checked(_cachedCount + 1);
            _cachedBytes = checked(
                _cachedBytes + (long)allocation.ByteLength);
        }
        foreach (TrackedAllocation candidate in evicted)
        {
            ReleaseNative(
                candidate.Pointer,
                candidate.ByteLength,
                candidate.Kind);
        }
        return true;
    }

    private bool CanCache(nuint byteLength)
        => MaximumReusableCacheEntries > 0
            && MaximumReusableCacheBytes > 0
            && byteLength <= (nuint)MaximumReusableCacheBytes;

    /// <summary>
    /// Selects and removes idle leases needed to admit a new cache entry.
    /// An allocation larger than half the byte budget is the lane's protected
    /// large slot. Smaller arrivals may evict other small entries, but are
    /// rejected instead of evicting that slot. A new protected-large arrival
    /// replaces the prior large shape and then evicts the oldest small entries
    /// needed to satisfy the exact byte/count bounds. Must be called while
    /// holding <see cref="_sync"/>; native frees are deliberately performed by
    /// the caller after releasing the lock.
    /// </summary>
    private bool TryPrepareCacheAdmissionLocked(
        nuint incomingBytes,
        int incomingEntries,
        out List<TrackedAllocation> evicted)
    {
        evicted = [];
        long requestedBytes = checked((long)incomingBytes);
        bool incomingIsProtectedLarge = IsProtectedLarge(incomingBytes);

        List<TrackedAllocation> candidates = _allocations.Values
            .Where(static candidate =>
                candidate.State == AllocationState.Cached)
            .Where(candidate => incomingIsProtectedLarge
                || !IsProtectedLarge(candidate.ByteLength))
            .OrderBy(candidate => incomingIsProtectedLarge
                    && IsProtectedLarge(candidate.ByteLength)
                ? 0
                : 1)
            .ThenBy(static candidate => candidate.CacheSequence)
            .ToList();

        long retainedCount = _cachedCount;
        long retainedBytes = _cachedBytes;
        int candidateIndex = 0;
        while (retainedCount + incomingEntries
                    > MaximumReusableCacheEntries
            || retainedBytes + requestedBytes
                    > MaximumReusableCacheBytes)
        {
            if (candidateIndex >= candidates.Count)
            {
                // Admission would require evicting the protected-large slot.
                // Keep the existing hot allocation and release the incoming
                // small allocation instead.
                evicted.Clear();
                return false;
            }

            TrackedAllocation candidate = candidates[candidateIndex++];
            retainedCount = checked(retainedCount - 1);
            retainedBytes = checked(
                retainedBytes - (long)candidate.ByteLength);
            evicted.Add(candidate);
        }

        foreach (TrackedAllocation candidate in evicted)
        {
            _allocations.Remove(candidate.AllocationId);
            _cachedCount = checked(_cachedCount - 1);
            _cachedBytes = checked(
                _cachedBytes - (long)candidate.ByteLength);
        }
        return true;
    }

    private bool IsProtectedLarge(nuint byteLength)
        => (long)byteLength > MaximumReusableCacheBytes / 2;

    private void ReleaseNative(
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind)
    {
        try
        {
            _allocator.Release(DeviceIndex, pointer, byteLength, kind);
            Interlocked.Increment(ref _nativeReleaseCount);
            Interlocked.Add(
                ref _nativeReleaseBytes,
                checked((long)byteLength));
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
        Cached,
        CaptureAvailable,
        GraphPinned,
    }

    private enum GraphCaptureStatus
    {
        Capturing,
        RollingBack,
        Committed,
        Disposing,
        Aborted,
        Disposed,
    }

    private sealed class TrackedAllocation(
        long allocationId,
        nint pointer,
        nuint byteLength,
        CudaMemoryKind kind,
        bool reusable,
        CudaMemoryLease lease,
        GraphCaptureState? graphCapture)
    {
        public long AllocationId { get; } = allocationId;

        public nint Pointer { get; } = pointer;

        public nuint ByteLength { get; } = byteLength;

        public CudaMemoryKind Kind { get; } = kind;

        public bool Reusable { get; } = reusable;

        public CudaMemoryLease Lease { get; } = lease;

        public GraphCaptureState? GraphCapture { get; } = graphCapture;

        public AllocationState State { get; set; }

        public ICudaCompletionFence? Fence { get; set; }

        public long CacheSequence { get; set; }
    }

    private sealed class GraphCaptureState(
        long captureId,
        int ownerManagedThreadId)
    {
        public long CaptureId { get; } = captureId;

        public int OwnerManagedThreadId { get; } = ownerManagedThreadId;

        public GraphCaptureStatus Status { get; set; } =
            GraphCaptureStatus.Capturing;

        public List<ICudaCompletionFence> OwnedFences { get; } = [];
    }
}

/// <summary>
/// Uncommitted CUDA Graph capture memory scope. Dispose rolls back all
/// participating allocations. Commit may be called once and returns the
/// independent graph-lifetime reservation.
/// </summary>
public sealed class CudaGraphCaptureScope : IDisposable
{
    private readonly object _sync = new();
    private readonly CudaMemoryManager _owner;
    private readonly long _captureId;
    private ScopeState _state;

    internal CudaGraphCaptureScope(
        CudaMemoryManager owner,
        long captureId)
    {
        _owner = owner;
        _captureId = captureId;
    }

    public IDisposable Commit()
    {
        lock (_sync)
        {
            if (_state == ScopeState.Committed)
            {
                throw new InvalidOperationException(
                    "This CUDA Graph capture reservation was already committed.");
            }
            ObjectDisposedException.ThrowIf(
                _state == ScopeState.Disposed,
                this);
            IDisposable reservation = _owner.CommitGraphCapture(_captureId);
            _state = ScopeState.Committed;
            return reservation;
        }
    }

    public void Dispose()
    {
        bool rollback;
        lock (_sync)
        {
            if (_state == ScopeState.Disposed)
                return;
            rollback = _state == ScopeState.Active;
            _state = ScopeState.Disposed;
        }
        if (rollback)
            _owner.RollbackGraphCapture(_captureId);
    }

    private enum ScopeState
    {
        Active,
        Committed,
        Disposed,
    }
}

internal sealed class CudaGraphMemoryReservation : IDisposable
{
    private CudaMemoryManager? _owner;
    private readonly long _captureId;

    internal CudaGraphMemoryReservation(
        CudaMemoryManager owner,
        long captureId)
    {
        _owner = owner;
        _captureId = captureId;
    }

    public void Dispose()
    {
        CudaMemoryManager? owner = Interlocked.Exchange(ref _owner, null);
        owner?.DisposeGraphReservation(_captureId);
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
