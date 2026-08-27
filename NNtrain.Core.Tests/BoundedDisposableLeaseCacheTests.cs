using NNtrain;
using Xunit;

public sealed class BoundedDisposableLeaseCacheTests
{
    [Fact]
    public void EvictsLeastRecentlyUsedIdleValue()
    {
        using var cache =
            new BoundedDisposableLeaseCache<int, TrackedDisposable>(2);
        var first = new TrackedDisposable();
        var second = new TrackedDisposable();
        var third = new TrackedDisposable();

        cache.Acquire(1, _ => first)!.Dispose();
        cache.Acquire(2, _ => second)!.Dispose();
        cache.Acquire(1, _ => throw new InvalidOperationException())!.Dispose();
        cache.Acquire(3, _ => third)!.Dispose();

        Assert.Equal(2, cache.Count);
        Assert.Equal(0, first.DisposeCount);
        Assert.Equal(1, second.DisposeCount);
        Assert.Equal(0, third.DisposeCount);
    }

    [Fact]
    public void DefersEvictedValueDisposalUntilLastLeaseReturns()
    {
        using var cache =
            new BoundedDisposableLeaseCache<int, TrackedDisposable>(1);
        var first = new TrackedDisposable();
        var second = new TrackedDisposable();
        BoundedDisposableLeaseCache<int, TrackedDisposable>.Lease firstLease =
            cache.Acquire(1, _ => first)!;

        cache.Acquire(2, _ => second)!.Dispose();

        Assert.Equal(1, cache.Count);
        Assert.Equal(0, first.DisposeCount);

        firstLease.Dispose();
        firstLease.Dispose();

        Assert.Equal(1, first.DisposeCount);
    }

    [Fact]
    public void CacheDisposeAlsoWaitsForOutstandingLease()
    {
        var cache =
            new BoundedDisposableLeaseCache<int, TrackedDisposable>(1);
        var value = new TrackedDisposable();
        BoundedDisposableLeaseCache<int, TrackedDisposable>.Lease lease =
            cache.Acquire(1, _ => value)!;

        cache.Dispose();

        Assert.Equal(0, cache.Count);
        Assert.Equal(0, value.DisposeCount);

        lease.Dispose();

        Assert.Equal(1, value.DisposeCount);
    }

    [Fact]
    public void FailedCreationIsNotCached()
    {
        using var cache =
            new BoundedDisposableLeaseCache<int, TrackedDisposable>(1);
        int attempts = 0;

        Assert.Null(cache.Acquire(1, _ =>
        {
            attempts++;
            return null;
        }));
        Assert.Null(cache.Acquire(1, _ =>
        {
            attempts++;
            return null;
        }));

        Assert.Equal(2, attempts);
        Assert.Equal(0, cache.Count);
    }

    [Fact]
    public void InsertionFailureDisposesNewValue()
    {
        using var cache = new BoundedDisposableLeaseCache<
            ThrowOnFirstHashKey,
            TrackedDisposable>(1);
        var key = new ThrowOnFirstHashKey();
        var value = new TrackedDisposable();

        Assert.Throws<InvalidOperationException>(
            () => cache.Acquire(key, _ => value));

        Assert.Equal(0, cache.Count);
        Assert.Equal(1, value.DisposeCount);
    }

    [Fact]
    public void DisposeContinuesAfterOneValueThrows()
    {
        var cache =
            new BoundedDisposableLeaseCache<int, TrackedDisposable>(2);
        var throwing = new TrackedDisposable(throwOnDispose: true);
        var normal = new TrackedDisposable();
        cache.Acquire(1, _ => throwing)!.Dispose();
        cache.Acquire(2, _ => normal)!.Dispose();

        Assert.Throws<InvalidOperationException>(() => cache.Dispose());

        Assert.Equal(1, throwing.DisposeCount);
        Assert.Equal(1, normal.DisposeCount);
        Assert.Equal(0, cache.Count);
    }

    private sealed class TrackedDisposable(bool throwOnDispose = false)
        : IDisposable
    {
        internal int DisposeCount;

        public void Dispose()
        {
            Interlocked.Increment(ref DisposeCount);
            if (throwOnDispose)
                throw new InvalidOperationException("Expected disposal failure.");
        }
    }

    private sealed class ThrowOnFirstHashKey
    {
        private int _hashCount;

        public override int GetHashCode()
        {
            if (Interlocked.Increment(ref _hashCount) == 1)
                throw new InvalidOperationException("Expected insertion failure.");
            return 1;
        }
    }
}
