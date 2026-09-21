using NNtrain;
using Xunit;

public sealed class CompletedGenerationQueueTests
{
    private sealed class Job(int id, int remaining) : CompletedGenerationQueue<int>.IJob
    {
        internal bool Disposed;
        public bool Advance(out int result) { result = id; return --remaining == 0; }
        public void Dispose() => Disposed = true;
    }
    [Fact]
    public void CompletionOrderBoundsAndTailAreIndependentOfAdmissionOrder()
    {
        using var q = new CompletedGenerationQueue<int>(3, 3);
        Job[] jobs = [new(0, 5), new(1, 1), new(2, 2), new(3, 1)];
        int created = 0;
        q.Fill(() => created < jobs.Length ? jobs[created++] : null);
        while (q.Count < 2) q.Tick(TestContext.Current.CancellationToken);
        Assert.Equal(new[] { 1, 2 }, q.Take(2));
        Assert.False(jobs[0].Disposed);
        q.Fill(() => created < jobs.Length ? jobs[created++] : null);
        q.Drain(TestContext.Current.CancellationToken);
        Assert.Equal(new[] { 3, 0 }, q.Take(2));
        Assert.All(jobs, j => Assert.True(j.Disposed));
    }
    [Fact]
    public void CancelAndDisposeReleaseActiveJobs()
    {
        Job job = new(0, 10);
        var q = new CompletedGenerationQueue<int>(1, 1);
        q.Fill(() => job);
        Assert.Throws<OperationCanceledException>(() => q.Tick(new CancellationToken(true)));
        q.Dispose(); Assert.True(job.Disposed);
    }
    [Fact]
    public void PairPackingMasksPromptAndPadding()
    {
        var p = DpoCommand.Pack([new(0, [1, 7], [8, 9], [4])]);
        Assert.Equal(new[] { -1, 8, 9 }, p.Labels[0]);
        Assert.Equal(new[] { -1, 4, -1 }, p.Labels[1]);
        Assert.Equal(new[] { 2, 1 }, p.Counts);
    }
    private sealed class FailedJob : CompletedGenerationQueue<int>.IJob
    {
        internal bool Disposed;
        public bool Advance(out int result) => throw new InvalidOperationException("generation failed");
        public void Dispose() => Disposed = true;
    }
    [Fact]
    public void GenerationFailurePropagatesAndAllJobsAreReleased()
    {
        var failed = new FailedJob(); Job other = new(1, 5);
        var q = new CompletedGenerationQueue<int>(2, 2);
        int calls = 0;
        q.Fill(() => calls++ == 0 ? failed : other);
        Assert.Throws<InvalidOperationException>(() => q.Tick(TestContext.Current.CancellationToken));
        q.Dispose();
        Assert.True(failed.Disposed); Assert.True(other.Disposed);
    }
}
