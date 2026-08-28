namespace NNtrain;

/// <summary>
/// A bounded native-resource fallback whose current generation can be
/// explicitly retired. Outstanding leases keep retired values alive until
/// their caller has finished enqueueing work; later callers transparently use
/// a fresh generation.
/// </summary>
internal sealed class ResettableBoundedDisposableLeaseCache<TKey, TValue>
    : IDisposable
    where TKey : notnull
    where TValue : class, IDisposable
{
    private readonly int _capacity;
    private BoundedDisposableLeaseCache<TKey, TValue> _current;

    internal ResettableBoundedDisposableLeaseCache(int capacity)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(capacity, 1);
        _capacity = capacity;
        _current = new BoundedDisposableLeaseCache<TKey, TValue>(capacity);
    }

    internal int Capacity => _capacity;

    internal int Count => Volatile.Read(ref _current).Count;

    internal BoundedDisposableLeaseCache<TKey, TValue>.Lease? Acquire(
        TKey key,
        Func<TKey, TValue?> factory)
    {
        ArgumentNullException.ThrowIfNull(factory);
        while (true)
        {
            BoundedDisposableLeaseCache<TKey, TValue> generation =
                Volatile.Read(ref _current);
            try
            {
                return generation.Acquire(key, factory);
            }
            catch (ObjectDisposedException)
                when (!ReferenceEquals(
                    generation,
                    Volatile.Read(ref _current)))
            {
                // Reset won the race before Acquire entered the generation.
                // Retry against the replacement rather than exposing an
                // incidental cache-lifecycle failure to a legacy caller.
            }
        }
    }

    /// <summary>
    /// Retires every cached value. Active leases defer their value's disposal;
    /// subsequent acquisitions use a new empty bounded generation.
    /// </summary>
    public void Dispose()
    {
        var replacement =
            new BoundedDisposableLeaseCache<TKey, TValue>(_capacity);
        BoundedDisposableLeaseCache<TKey, TValue> retired =
            Interlocked.Exchange(ref _current, replacement);
        retired.Dispose();
    }
}
