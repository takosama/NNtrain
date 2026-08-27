namespace NNtrain;

/// <summary>
/// A small LRU cache whose values can be leased while native work is being
/// enqueued. Eviction prevents new leases immediately, but disposal is
/// deferred until the last outstanding lease is returned.
/// </summary>
internal sealed class BoundedDisposableLeaseCache<TKey, TValue> : IDisposable
    where TKey : notnull
    where TValue : class, IDisposable
{
    private readonly int _capacity;
    private readonly object _sync = new();
    private readonly Dictionary<TKey, Entry> _entries = new();
    private readonly LinkedList<TKey> _lru = new();
    private bool _disposed;

    internal BoundedDisposableLeaseCache(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        _capacity = capacity;
    }

    internal int Capacity => _capacity;

    internal int Count
    {
        get
        {
            lock (_sync)
                return _entries.Count;
        }
    }

    internal Lease? Acquire(TKey key, Func<TKey, TValue?> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);

        TValue? dispose = null;
        Lease? lease;
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_entries.TryGetValue(key, out Entry? cached))
            {
                Touch(cached);
                cached.ActiveLeases++;
                return new Lease(this, cached);
            }

            // Creation is intentionally serialized with cache lookup. Plan
            // construction is a cold-path operation and duplicate native
            // descriptors are more expensive than briefly holding this lock.
            TValue? value = factory(key);
            if (value is null)
                return null;

            // Allocate the node/entry/lease before mutating either collection.
            // If Dictionary.Add (including a custom comparer) fails, roll the
            // native value back immediately instead of orphaning it.
            LinkedListNode<TKey> node = new(key);
            var added = new Entry(value, node) { ActiveLeases = 1 };
            lease = new Lease(this, added);
            _lru.AddFirst(node);
            try
            {
                _entries.Add(key, added);
            }
            catch (Exception insertionException)
            {
                _lru.Remove(node);
                try
                {
                    value.Dispose();
                }
                catch (Exception rollbackException)
                {
                    throw new AggregateException(
                        insertionException, rollbackException);
                }
                throw;
            }

            while (_entries.Count > _capacity)
            {
                LinkedListNode<TKey> victimNode = _lru.Last!;
                Entry victim = _entries[victimNode.Value];
                _lru.Remove(victimNode);
                _entries.Remove(victimNode.Value);
                victim.Node = null;
                victim.Retired = true;
                if (victim.ActiveLeases == 0)
                    dispose = victim.Value;
            }
        }

        try
        {
            dispose?.Dispose();
        }
        catch
        {
            // The new entry remains a valid cached plan, but the caller never
            // received its lease, so return that lease before propagating the
            // failed cleanup.
            lease.Dispose();
            throw;
        }
        return lease;
    }

    public void Dispose()
    {
        List<TValue>? dispose = null;
        lock (_sync)
        {
            if (_disposed)
                return;
            _disposed = true;

            foreach (Entry entry in _entries.Values)
            {
                entry.Node = null;
                entry.Retired = true;
                if (entry.ActiveLeases == 0)
                    (dispose ??= []).Add(entry.Value);
            }
            _entries.Clear();
            _lru.Clear();
        }

        DisposeValues(dispose);
    }

    private void Touch(Entry entry)
    {
        LinkedListNode<TKey> node = entry.Node!;
        if (node != _lru.First)
        {
            _lru.Remove(node);
            _lru.AddFirst(node);
        }
    }

    private void Release(Entry entry)
    {
        TValue? dispose = null;
        lock (_sync)
        {
            if (entry.ActiveLeases <= 0)
                throw new InvalidOperationException("Cache lease underflow.");
            entry.ActiveLeases--;
            if (entry.Retired && entry.ActiveLeases == 0)
                dispose = entry.Value;
        }

        dispose?.Dispose();
    }

    private static void DisposeValues(List<TValue>? values)
    {
        if (values is null)
            return;
        List<Exception>? exceptions = null;
        foreach (TValue value in values)
        {
            try
            {
                value.Dispose();
            }
            catch (Exception exception)
            {
                (exceptions ??= []).Add(exception);
            }
        }
        if (exceptions is null)
            return;
        if (exceptions.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(exceptions[0])
                .Throw();
        }
        throw new AggregateException(exceptions);
    }

    internal sealed class Entry(
        TValue value,
        LinkedListNode<TKey> node)
    {
        internal TValue Value { get; } = value;
        internal LinkedListNode<TKey>? Node = node;
        internal int ActiveLeases;
        internal bool Retired;
    }

    internal sealed class Lease : IDisposable
    {
        private BoundedDisposableLeaseCache<TKey, TValue>? _owner;
        private Entry? _entry;

        internal Lease(
            BoundedDisposableLeaseCache<TKey, TValue> owner,
            Entry entry)
        {
            _owner = owner;
            _entry = entry;
            Value = entry.Value;
        }

        internal TValue Value { get; }

        public void Dispose()
        {
            BoundedDisposableLeaseCache<TKey, TValue>? owner =
                Interlocked.Exchange(ref _owner, null);
            Entry? entry = Interlocked.Exchange(ref _entry, null);
            if (owner is not null && entry is not null)
                owner.Release(entry);
        }
    }
}
