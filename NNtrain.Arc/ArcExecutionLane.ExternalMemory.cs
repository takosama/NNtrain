namespace NNtrain.Arc;

public sealed partial class ArcExecutionLane
{
    private long _externalMemoryReservationBytes;

    private long NormalPhysicalBufferBudgetBytes => Options.PhysicalBufferBudgetBytes > 0
        ? Options.PhysicalBufferBudgetBytes : checked((long)(Device.GlobalMemoryBytes / 10 * 9));

    /// <summary>Memory assigned to another owner on this device, such as a vision encoder.</summary>
    public long ExternalMemoryReservationBytes
    {
        get { lock (_sync) return _externalMemoryReservationBytes; }
    }

    /// <summary>The lane's physical allocation budget after other owners' reservations.</summary>
    public long EffectivePhysicalBufferBudgetBytes
    {
        get { lock (_sync) return NormalPhysicalBufferBudgetBytes - _externalMemoryReservationBytes; }
    }

    /// <summary>
    /// Reserve device memory for a separate allocator. Live buffers are never
    /// released; idle buffers are trimmed and retired handles are fenced first.
    /// An invalid reservation leaves the previous reservation unchanged.
    /// </summary>
    public void SetExternalMemoryReservation(long bytes)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentOutOfRangeException.ThrowIfNegative(bytes);
            long normalBudget = NormalPhysicalBufferBudgetBytes;
            if (bytes > normalBudget)
                throw new ArgumentOutOfRangeException(nameof(bytes), "External memory exceeds the lane's device budget.");
            long effectiveBudget = normalBudget - bytes;
            if (AllocatedBytes > effectiveBudget)
                throw new InvalidOperationException("External memory reservation would exceed the device budget with live buffers.");
            TrimCacheToPhysicalBudget(0, effectiveBudget, "external-memory-reservation");
            _externalMemoryReservationBytes = bytes;
        }
    }

    // Called under _sync. Keep live allocations intact, and reclaim all idle
    // cache policies before fencing retired ownership at most once.
    private void TrimCacheToPhysicalBudget(long bytes, long budget, string reason)
    {
        while (AllocatedBytes + CachedBytes + bytes > budget && CachedBytes > 0)
        {
            nint handle;
            long victimBytes;
            if (Options.LruBufferPool)
            {
                CachedBuffer victim = _freeLru.First?.Value
                    ?? throw new InvalidOperationException("Arc LRU cache accounting is inconsistent.");
                RemoveLruEntry(victim);
                handle = victim.Handle; victimBytes = victim.Bytes;
            }
            else
            {
                var victim = _pool.FirstOrDefault(pair => pair.Value.Count > 0);
                if (victim.Value is null)
                    throw new InvalidOperationException("Arc buffer cache accounting is inconsistent.");
                handle = victim.Value.Pop(); victimBytes = victim.Key;
            }
            CachedBytes -= victimBytes;
            RetireOrRelease(handle, victimBytes, reason);
            DetailedProfiler?.Add("cache-budget-trim", $"{DetailedProfiler.Phase}/{victimBytes}B", bytes: victimBytes);
        }
        if (RetiredBytes > 0 && AllocatedBytes + CachedBytes + RetiredBytes + bytes > budget)
            SynchronizeCore(reason + "-retired");
    }
}
