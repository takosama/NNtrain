using NNtrain;
using Xunit;

public sealed class BoundedLruCacheTests
{
    [Fact]
    public void RepeatedShapeChangesNeverExceedCapacity()
    {
        var cache = new BoundedLruCache<int, CacheValue>(capacity: 3);

        for (int key = 0; key < 1_000; key++)
            _ = cache.GetOrAdd(key, static value => new CacheValue(value));

        Assert.Equal(3, cache.Count);
    }

    [Fact]
    public void ParallelLookupsCreateOneValuePerRetainedKey()
    {
        int creations = 0;
        var cache = new BoundedLruCache<int, CacheValue>(capacity: 8);

        Parallel.For(0, 1_024, _ =>
            cache.GetOrAdd(
                7,
                value =>
                {
                    Interlocked.Increment(ref creations);
                    return new CacheValue(value);
                }));

        Assert.Equal(1, creations);
        Assert.Equal(1, cache.Count);
    }

    private sealed record CacheValue(int Key);
}
