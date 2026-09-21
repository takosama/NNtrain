using NNtrain;
using Xunit;

public sealed class ParallelGenerationQueueTests
{
    private sealed record Request(int Id);
    private sealed class Worker(Func<Request, CancellationToken, int> generate) : ParallelGenerationQueue<Request, int>.IWorker
    {
        public int Generate(Request r, CancellationToken token) => generate(r, token);
        public void Dispose() { }
    }
    [Fact]
    public void WorkersOverlapAndShortCompletionIsConsumedBeforeLongTask()
    {
        using var both = new CountdownEvent(2);
        using var releaseLong = new ManualResetEventSlim();
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        timeout.CancelAfter(TimeSpan.FromSeconds(20));
        int issued = 0;
        using var queue = new ParallelGenerationQueue<Request, int>(2, 2, _ => new Worker((r, token) =>
        {
            both.Signal(); both.Wait(token);
            if (r.Id == 0) releaseLong.Wait(token);
            return r.Id;
        }), () => issued < 2 ? new Request(issued++) : null, [], timeout.Token);
        queue.Resume();
        Assert.Equal(new[] { 1 }, queue.Take(1));
        Assert.Equal(2, queue.PeakActive);
        releaseLong.Set();
        Assert.Equal(new[] { 0 }, queue.Take(1));
        Assert.Empty(queue.Take(1));
    }
    [Fact]
    public void PausedSnapshotIncludesAllIssuedWorkAndRestoresWithoutDuplicates()
    {
        int issued = 0;
        using var queue = new ParallelGenerationQueue<Request, int>(2, 4, _ => new Worker((r, _) => r.Id),
            () => new Request(issued++), [], TestContext.Current.CancellationToken);
        queue.Resume();
        int[] consumed = queue.Take(2);
        int[] saved = queue.PauseAndSnapshot();
        Assert.Equal(issued, consumed.Length + saved.Length);
        Assert.InRange(saved.Length, 0, 4);
        Assert.Equal(issued, consumed.Concat(saved).Distinct().Count());
        using var restored = new ParallelGenerationQueue<Request, int>(2, 4, _ => new Worker((r, _) => r.Id),
            () => null, saved, TestContext.Current.CancellationToken);
        restored.Resume();
        Assert.Equal(saved, restored.Take(4));
    }
    [Fact]
    public void WorkerFailureUnblocksConsumer()
    {
        int issued = 0;
        var queue = new ParallelGenerationQueue<Request, int>(1, 1, _ => new Worker((_, _) => throw new InvalidOperationException("boom")),
            () => issued++ == 0 ? new Request(0) : null, [], TestContext.Current.CancellationToken);
        queue.Resume();
        Assert.Throws<AggregateException>(() => queue.Take(1));
        Assert.Throws<AggregateException>(() => queue.Dispose());
    }

    [Fact]
    public void SmallerResumedCapacityPreservesBacklog()
    {
        int issued = 5;
        using var queue = new ParallelGenerationQueue<Request, int>(1, 2, _ => new Worker((r, _) => r.Id),
            () => issued < 7 ? new Request(issued++) : null, [0, 1, 2, 3, 4], TestContext.Current.CancellationToken);
        queue.Resume();
        var results = new List<int>();
        while (results.Count < 7) results.AddRange(queue.Take(2));
        Assert.Equal(Enumerable.Range(0, 7), results);
        Assert.Empty(queue.Take(2));
    }
}
