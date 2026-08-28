using NNtrain;
using Xunit;

public sealed class CudaResourceCleanupTests
{
    [Fact]
    public void RunAllAttemptsEveryReleaseAndFlattensFailures()
    {
        int attempts = 0;

        AggregateException failure = Assert.Throws<AggregateException>(() =>
            CudaResourceCleanup.RunAll(
                "cleanup",
                () =>
                {
                    attempts++;
                    throw new InvalidOperationException("first");
                },
                () => attempts++,
                () =>
                {
                    attempts++;
                    throw new AggregateException(
                        new InvalidOperationException("third-a"),
                        new InvalidOperationException("third-b"));
                }));

        Assert.Equal(3, attempts);
        Assert.Equal(3, failure.InnerExceptions.Count);
    }
}
