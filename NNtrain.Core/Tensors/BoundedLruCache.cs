namespace NNtrain;

/// <summary>
/// Small thread-safe cache for immutable managed values. Evicted values remain
/// valid for callers that already obtained them, while the cache itself keeps
/// at most <see cref="Capacity"/> strong references.
/// </summary>
internal sealed class BoundedLruCache<TKey, TValue>
    where TKey : notnull
    where TValue : class
{
    private readonly object _sync = new();
    private readonly Dictionary<TKey, Entry> _entries;
    private readonly LinkedList<TKey> _lru = [];

    internal BoundedLruCache(
        int capacity,
        IEqualityComparer<TKey>? comparer = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        Capacity = capacity;
        _entries = new Dictionary<TKey, Entry>(comparer);
    }

    internal int Capacity { get; }

    internal int Count
    {
        get
        {
            lock (_sync)
                return _entries.Count;
        }
    }

    internal TValue GetOrAdd(TKey key, Func<TKey, TValue> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        lock (_sync)
        {
            if (_entries.TryGetValue(key, out Entry? existing))
            {
                Touch(existing);
                return existing.Value;
            }

            TValue value = factory(key)
                ?? throw new InvalidOperationException(
                    "A bounded cache factory returned null.");
            LinkedListNode<TKey> node = _lru.AddFirst(key);
            try
            {
                _entries.Add(key, new Entry(value, node));
            }
            catch
            {
                _lru.Remove(node);
                throw;
            }

            if (_entries.Count > Capacity)
            {
                LinkedListNode<TKey> victim = _lru.Last!;
                _lru.RemoveLast();
                _entries.Remove(victim.Value);
            }
            return value;
        }
    }

    internal void Clear()
    {
        lock (_sync)
        {
            _entries.Clear();
            _lru.Clear();
        }
    }

    private void Touch(Entry entry)
    {
        if (entry.Node == _lru.First)
            return;
        _lru.Remove(entry.Node);
        _lru.AddFirst(entry.Node);
    }

    private sealed record Entry(
        TValue Value,
        LinkedListNode<TKey> Node);
}
