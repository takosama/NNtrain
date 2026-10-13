using System.Runtime.InteropServices;
using System.Diagnostics;

namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    private sealed class CachedBuffer
    {
        internal readonly nint Handle;
        internal readonly long Bytes;
        internal readonly LinkedListNode<CachedBuffer> SizeNode, RecencyNode;
        internal CachedBuffer(nint handle, long bytes)
        {
            Handle = handle; Bytes = bytes;
            SizeNode = new(this); RecencyNode = new(this);
        }
    }

    public ArcBuffer Allocate(int elements)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elements);
        lock (_sync) return AllocateCore(checked((long)Math.Max(1, elements) * 4), null);
    }

    public ArcBuffer Upload(float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        lock (_sync) return AllocateCore(Math.Max(4L, values.LongLength * 4), values);
    }

    public ArcBuffer AllocateBytes(int bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        lock (_sync) return AllocateCore(Math.Max(4, bytes), null);
    }

    public ArcBuffer UploadRaw(Array values)
    {
        if (values is not (float[] or int[] or ushort[] or short[] or byte[] or sbyte[]))
            throw new ArgumentException("Arc raw upload requires a primitive numeric array.");
        lock (_sync) return AllocateCore(Math.Max(4, Buffer.ByteLength(values)), values);
    }

    private ArcBuffer AllocateCore(long bytes, Array? data)
    {
        using var allocationScope = Timeline?.Host("allocation-pool", DetailedProfiler?.Phase ?? "");
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((ulong)bytes > Device.MaximumAllocationBytes)
            throw new InvalidOperationException($"Arc allocation {bytes:N0} exceeds device maximum {Device.MaximumAllocationBytes:N0} bytes.");
        RequestedBytes += bytes;
        GCHandle pin = default;
        try
        {
            long start = Stopwatch.GetTimestamp();
            bool poolHit = TryTakeCached(bytes, out nint pointer);
            bool retiredHit = false;
            if (poolHit)
            {
                CachedBytes -= bytes;
                PoolHits++;
            }
            else if (Options.ReuseRetiredBuffers && TryTakeRetired(bytes, out pointer))
            {
                retiredHit = true;
                RetiredReuseCount++;
            }
            else
            {
                TrimCacheForAllocation(bytes);
                if (_externalMemoryReservationBytes > 0
                    && AllocatedBytes + CachedBytes + RetiredBytes + bytes > EffectivePhysicalBufferBudgetBytes)
                    throw new InvalidOperationException("Arc allocation exceeds the device budget after external memory reservation.");
                // COPY_HOST_PTR copies the full allocation, not the array length.
                // Tiny packed payloads must not read padding beyond their managed array.
                if (!Options.ExplicitHostUploads && data is { Length: > 0 } && Buffer.ByteLength(data) == bytes)
                    pin = GCHandle.Alloc(data, GCHandleType.Pinned);
                int error;
                using (Timeline?.Host(pin.IsAllocated ? "native-allocation-with-upload" : "native-allocation"))
                    pointer = OpenClNative.clCreateBuffer(_context, pin.IsAllocated ? 33UL : 1UL,
                        (nuint)bytes, pin.IsAllocated ? pin.AddrOfPinnedObject() : 0, out error);
                if (pin.IsAllocated && Timeline is { } timeline) timeline.OpaqueAllocationCopies++;
                OpenClNative.Check(error, $"allocate {bytes:N0} bytes");
                AllocationCount++;
                NativeAllocatedBytes += bytes;
            }
            double elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            AllocationMilliseconds += elapsed;
            DetailedProfiler?.Add(poolHit ? "pool-hit" : retiredHit ? "retired-reuse" : "native-allocation",
                $"{DetailedProfiler.Phase}/{bytes}B", elapsed, bytes);
            if (pin.IsAllocated)
            {
                H2DBytes += bytes;
                DetailedProfiler?.Add("H2D", $"{DetailedProfiler.Phase}/allocation-copy/{bytes}B", elapsed, bytes);
            }
            var buffer = new ArcBuffer(this, pointer, bytes);
            _buffers.Add(buffer);
            AllocatedBytes += bytes;
            PeakAllocatedBytes = Math.Max(PeakAllocatedBytes, AllocatedBytes + CachedBytes + RetiredBytes);
            if ((poolHit || retiredHit || !pin.IsAllocated) && data is { Length: > 0 })
            {
                try { Transfer(buffer, data, read: false); }
                catch { buffer.Dispose(); throw; }
            }
            return buffer;
        }
        finally { if (pin.IsAllocated) pin.Free(); }
    }

    private void Release(ArcBuffer buffer)
    {
        lock (_sync)
        {
            using var releaseScope = Timeline?.Host("release-pool");
            nint handle = buffer.Handle;
            if (handle == 0) return;
            buffer.Handle = 0;
            _buffers.Remove(buffer);
            AllocatedBytes -= buffer.Bytes;
            if (!_disposed && buffer.Bytes <= Options.BufferPoolBytes
                && (Options.LruBufferPool || buffer.Bytes <= Options.BufferPoolBytes - CachedBytes))
            {
                if (Options.LruBufferPool)
                {
                    // Prepare managed metadata before changing native ownership.
                    // A cache miss may evict only free entries, never live buffers.
                    CachedBuffer incoming;
                    try
                    {
                        incoming = new(handle, buffer.Bytes);
                        while (buffer.Bytes > Options.BufferPoolBytes - CachedBytes)
                        {
                            CachedBuffer victim = _freeLru.First?.Value
                                ?? throw new InvalidOperationException("Arc LRU cache accounting is inconsistent.");
                            RemoveLruEntry(victim);
                            CachedBytes -= victim.Bytes;
                            RetireOrRelease(victim.Handle, victim.Bytes, "pool-lru-eviction");
                            DetailedProfiler?.Add("cache-eviction", $"{DetailedProfiler.Phase}/{victim.Bytes}B", bytes: victim.Bytes);
                        }
                        if (!_lruPool.TryGetValue(buffer.Bytes, out LinkedList<CachedBuffer>? free))
                            _lruPool.Add(buffer.Bytes, free = new());
                        free.AddLast(incoming.SizeNode);
                        _freeLru.AddLast(incoming.RecencyNode);
                        CachedBytes += buffer.Bytes;
                    }
                    catch (Exception cacheFailure)
                    {
                        // Eviction can encounter a queue error. The incoming
                        // native handle must still be retired/freed exactly once.
                        try { RetireOrRelease(handle, buffer.Bytes, "pool-lru-failure"); }
                        catch (Exception releaseFailure) { throw new AggregateException(cacheFailure, releaseFailure); }
                        throw;
                    }
                }
                else
                {
                    if (!_pool.TryGetValue(buffer.Bytes, out Stack<nint>? free)) _pool.Add(buffer.Bytes, free = new());
                    free.Push(handle);
                    CachedBytes += buffer.Bytes;
                }
            }
            else RetireOrRelease(handle, buffer.Bytes, "pool-budget-release");
        }
    }

    // Called under _sync. Taking a free buffer keeps existing in-order queue
    // reuse semantics; no fence or host transfer is added on a pool hit.
    private bool TryTakeCached(long bytes, out nint handle)
    {
        if (Options.LruBufferPool)
        {
            if (_lruPool.TryGetValue(bytes, out var lruFree) && lruFree.Last is { } node)
            {
                CachedBuffer entry = node.Value;
                RemoveLruEntry(entry);
                handle = entry.Handle;
                return true;
            }
        }
        else if (_pool.TryGetValue(bytes, out Stack<nint>? legacyFree) && legacyFree.Count > 0)
        {
            handle = legacyFree.Pop();
            return true;
        }
        handle = 0;
        return false;
    }

    private void RemoveLruEntry(CachedBuffer entry)
    {
        LinkedList<CachedBuffer> free = _lruPool[entry.Bytes];
        free.Remove(entry.SizeNode);
        _freeLru.Remove(entry.RecencyNode);
        if (free.Count == 0) _lruPool.Remove(entry.Bytes);
    }

    private void TrimCacheForAllocation(long bytes)
    {
        if (!Options.LruBufferPool && _externalMemoryReservationBytes == 0) return;
        // Retired bytes can be reclaimed at the single fence below. Do not
        // evict additional useful cache entries merely because that fence has
        // not yet completed.
        TrimCacheToPhysicalBudget(bytes, EffectivePhysicalBufferBudgetBytes, "physical-budget");
    }

    // Called only under _sync after an exact-size pool miss. Retired entries
    // still own their native reference; their original ArcBuffer and all its
    // borrowed views are already invalid. Earlier commands and every new use
    // share this lane's in-order queue, so reuse needs no completion fence.
    // Move, rather than duplicate, the handle and its physical-byte accounting.
    private bool TryTakeRetired(long bytes, out nint handle)
    {
        for (int i = _retired.Count - 1; i >= 0; i--)
        {
            var entry = _retired[i];
            if (entry.Bytes != bytes) continue;
            _retired.RemoveAt(i);
            RetiredBytes -= bytes;
            handle = entry.Handle;
            return true;
        }
        handle = 0;
        return false;
    }

    private void RetireOrRelease(nint handle, long bytes, string fenceReason)
    {
        // Retiring an evicted cache entry preserves its physical-byte accounting
        // until the same queue-complete boundary used for ordinary releases.
        DetailedProfiler?.Add("uncached-release", $"{DetailedProfiler.Phase}/{bytes}B", bytes: bytes);
        if (!_disposed && (_pendingEvents.Count != 0 || _queuedCopies)
            && bytes <= Options.DeferredReleaseBytes - RetiredBytes)
        {
            _retired.Add((handle, bytes)); RetiredBytes += bytes;
            DetailedProfiler?.Add("deferred-release", DetailedProfiler.Phase, bytes: bytes);
            return;
        }
        try { if (!_disposed && (_pendingEvents.Count != 0 || _queuedCopies)) SynchronizeCore(fenceReason); }
        finally { ReleaseNative(handle, bytes); }
    }

    private void ReleaseNative(nint handle, long bytes)
    {
        InvalidateKernelArgumentCaches();
        using var releaseScope = Timeline?.Host("native-free");
        int status = OpenClNative.clReleaseMemObject(handle);
        if (status == 0)
        {
            NativeReleasedBytes += bytes;
            NativeReleaseCount++;
            DetailedProfiler?.Add("native-release", $"{DetailedProfiler.Phase}/{bytes}B", bytes: bytes);
        }
    }

    public sealed class ArcBuffer : IDisposable
    {
        private readonly ArcBuffer? _borrowedFrom;
        private bool _borrowDisposed;
        private nint _handle;
        internal ArcExecutionLane Owner { get; }
        internal nint Handle { get => _borrowedFrom is null ? _handle : _borrowDisposed ? 0 : _borrowedFrom.Handle; set => _handle = value; }
        internal long Bytes { get; }
        public long ByteLength => Bytes;
        public bool IsAlive => Handle != 0;
        internal ArcBuffer(ArcExecutionLane owner, nint handle, long bytes) => (Owner, Handle, Bytes) = (owner, handle, bytes);
        private ArcBuffer(ArcBuffer source) { _borrowedFrom = source; Owner = source.Owner; Bytes = source.Bytes; }
        public ArcBuffer Borrow() => IsAlive ? new(this) : throw new ObjectDisposedException(nameof(ArcBuffer));
        public void Dispose() { if (_borrowedFrom is null) Owner.Release(this); else _borrowDisposed = true; }
    }
}
