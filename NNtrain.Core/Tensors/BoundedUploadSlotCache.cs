namespace NNtrain;

/// <summary>
/// Bounded, least-recently-used upload-slot owner.  Slots are keyed by the
/// flattened batch length rather than by a transient device-buffer wrapper.
/// Keeping two slots per length lets input and target copies remain pending at
/// the same time without reusing pinned host memory.
/// </summary>
internal sealed class BoundedUploadSlotCache<TSlot> : IDisposable
    where TSlot : class, IDisposable
{
    private readonly object _sync = new();
    private readonly Func<int, TSlot> _factory;
    private readonly int _slotsPerLength;
    private readonly int _maximumLengths;
    private readonly Dictionary<int, SlotBucket> _buckets = [];
    private readonly List<Exception> _releaseErrors = [];
    private long _sequence;
    private long _createdSlotCount;
    private long _disposedSlotCount;
    private long _useCount;
    private int _disposed;

    internal BoundedUploadSlotCache(
        Func<int, TSlot> factory,
        int slotsPerLength = 2,
        int maximumLengths = 3)
    {
        _factory = factory ?? throw new ArgumentNullException(nameof(factory));
        ArgumentOutOfRangeException.ThrowIfLessThan(slotsPerLength, 2);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumLengths);
        _slotsPerLength = slotsPerLength;
        _maximumLengths = maximumLengths;
    }

    internal IReadOnlyList<Exception> ReleaseErrors
    {
        get
        {
            lock (_sync)
                return _releaseErrors.ToArray();
        }
    }

    internal BoundedUploadSlotCacheTelemetry Telemetry
    {
        get
        {
            lock (_sync)
            {
                int[] activeLengths = _buckets.Values
                    .OrderBy(static bucket => bucket.LastUsedSequence)
                    .Select(static bucket => bucket.Length)
                    .ToArray();
                long elementCapacity = _buckets.Values.Sum(
                    bucket => checked(
                        (long)bucket.Length * bucket.Slots.Length));
                return new BoundedUploadSlotCacheTelemetry(
                    _buckets.Count,
                    _buckets.Values.Sum(static bucket => bucket.Slots.Length),
                    elementCapacity,
                    _createdSlotCount,
                    _disposedSlotCount,
                    _useCount,
                    activeLengths);
            }
        }
    }

    /// <summary>
    /// Selects a slot and keeps it exclusively owned until <paramref name="use"/>
    /// returns.  CUDA event synchronization performed by a reused slot therefore
    /// cannot race another managed caller on the same lane.
    /// </summary>
    internal void Use(int length, Action<TSlot> use)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        ArgumentNullException.ThrowIfNull(use);
        lock (_sync)
        {
            ThrowIfDisposed();
            SlotBucket bucket = GetOrCreateBucket(length);
            bucket.LastUsedSequence = NextSequence();
            TSlot slot = bucket.Slots[bucket.NextSlot];
            bucket.NextSlot = (bucket.NextSlot + 1) % bucket.Slots.Length;
            _useCount++;
            use(slot);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        TSlot[] slots;
        lock (_sync)
        {
            slots = _buckets.Values
                .SelectMany(static bucket => bucket.Slots)
                .ToArray();
            _buckets.Clear();
            DisposeAll(slots);
            if (_releaseErrors.Count != 0)
            {
                throw new AggregateException(
                    "One or more bounded upload slots failed to dispose.",
                    _releaseErrors.ToArray());
            }
        }
    }

    private SlotBucket GetOrCreateBucket(int length)
    {
        if (_buckets.TryGetValue(length, out SlotBucket? existing))
            return existing;

        var created = new List<TSlot>(_slotsPerLength);
        try
        {
            for (int index = 0; index < _slotsPerLength; index++)
            {
                created.Add(_factory(length));
                _createdSlotCount++;
            }
        }
        catch (Exception creationFailure)
        {
            int priorErrorCount = _releaseErrors.Count;
            DisposeAll(created);
            if (_releaseErrors.Count == priorErrorCount)
                throw;
            throw new AggregateException(
                "An upload-slot bucket could not be constructed or rolled back.",
                [creationFailure, .. _releaseErrors.Skip(priorErrorCount)]);
        }

        var bucket = new SlotBucket(
            length,
            created.ToArray(),
            NextSequence());
        _buckets.Add(length, bucket);
        EvictLeastRecentlyUsedIfNeeded(length);
        return bucket;
    }

    private void EvictLeastRecentlyUsedIfNeeded(int retainedLength)
    {
        while (_buckets.Count > _maximumLengths)
        {
            SlotBucket retired = _buckets.Values
                .Where(bucket => bucket.Length != retainedLength)
                .MinBy(static bucket => bucket.LastUsedSequence)
                ?? throw new InvalidOperationException(
                    "No upload-slot bucket was available for eviction.");
            _buckets.Remove(retired.Length);
            DisposeAll(retired.Slots);
        }
    }

    private void DisposeAll(IEnumerable<TSlot> slots)
    {
        foreach (TSlot slot in slots)
        {
            try
            {
                slot.Dispose();
            }
            catch (Exception exception)
            {
                _releaseErrors.Add(exception);
            }
            finally
            {
                _disposedSlotCount++;
            }
        }
    }

    private long NextSequence()
    {
        if (_sequence == long.MaxValue)
        {
            long next = 0;
            foreach (SlotBucket bucket in _buckets.Values
                         .OrderBy(static bucket => bucket.LastUsedSequence))
            {
                bucket.LastUsedSequence = ++next;
            }
            _sequence = next;
        }
        return ++_sequence;
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);

    private sealed class SlotBucket(
        int length,
        TSlot[] slots,
        long lastUsedSequence)
    {
        internal int Length { get; } = length;
        internal TSlot[] Slots { get; } = slots;
        internal int NextSlot { get; set; }
        internal long LastUsedSequence { get; set; } = lastUsedSequence;
    }
}

internal readonly record struct BoundedUploadSlotCacheTelemetry(
    int ActiveLengthCount,
    int ActiveSlotCount,
    long ActiveElementCapacity,
    long CreatedSlotCount,
    long DisposedSlotCount,
    long UseCount,
    IReadOnlyList<int> ActiveLengths);
