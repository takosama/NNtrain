namespace NNtrain;

// Dedicated threads keep CUDA native TLS and execution streams bound for their lifetime.
// Admission + completed results share one capacity bound; training never holds this lock.
internal sealed class ParallelGenerationQueue<TRequest, TResult> : IDisposable where TRequest : class
{
    internal interface IWorker : IDisposable { TResult Generate(TRequest request, CancellationToken token); }
    private readonly object sync = new();
    private readonly Queue<TResult> ready = new();
    private readonly List<Exception> failures = [];
    private readonly Thread[] threads;
    private readonly CancellationTokenSource stop;
    private readonly int capacity;
    private bool paused = true, ended, disposed;
    private int active, peakActive, initialized;
    internal ParallelGenerationQueue(int workers, int capacity, Func<int, IWorker> initialize,
        Func<TRequest?> next, IEnumerable<TResult> restored, CancellationToken cancellation)
    {
        if (workers < 1 || capacity < workers) throw new ArgumentException("Invalid parallel generation capacity.");
        this.capacity = capacity;
        foreach (TResult item in restored) ready.Enqueue(item);
        // A smaller resumed capacity must not discard already completed CPU token pairs.
        // Admission stays blocked until the restored backlog falls below the new bound.
        stop = CancellationTokenSource.CreateLinkedTokenSource(cancellation);
        threads = Enumerable.Range(0, workers).Select(index => new Thread(() => Run(index))
            { IsBackground = true, Name = $"DPO generation {index}" }).ToArray();
        int started = 0;
        try
        {
            foreach (Thread thread in threads)
            {
                if (ExecutionContext.IsFlowSuppressed()) thread.Start();
                else { using var suppressed = ExecutionContext.SuppressFlow(); thread.Start(); }
                started++;
            }
            lock (sync)
                while (initialized < workers) Wait();
        }
        catch
        {
            stop.Cancel();
            lock (sync) Monitor.PulseAll(sync);
            for (int i = 0; i < started; i++) threads[i].Join();
            stop.Dispose();
            throw;
        }
        void Run(int index)
        {
            try
            {
                using var worker = initialize(index);
                lock (sync) { initialized++; Monitor.PulseAll(sync); }
                while (true)
                {
                    TRequest? request;
                    lock (sync)
                    {
                        while (paused || ready.Count + active >= capacity) Wait();
                        stop.Token.ThrowIfCancellationRequested();
                        if (ended) return;
                        request = next();
                        if (request is null) { ended = true; Monitor.PulseAll(sync); return; }
                        active++;
                        peakActive = Math.Max(peakActive, active);
                    }
                    try
                    {
                        TResult result = worker.Generate(request, stop.Token);
                        lock (sync) ready.Enqueue(result);
                    }
                    finally { lock (sync) { active--; Monitor.PulseAll(sync); } }
                }
            }
            catch (OperationCanceledException) when (stop.IsCancellationRequested) { }
            catch (Exception error) { lock (sync) { failures.Add(error); Monitor.PulseAll(sync); } }
        }
    }
    private void Wait()
    {
        ThrowFailure();
        Monitor.Wait(sync, 50);
        ThrowFailure();
    }
    private void ThrowFailure()
    {
        if (failures.Count != 0) throw new AggregateException("DPO generation worker failed.", failures);
        stop.Token.ThrowIfCancellationRequested();
    }
    internal int Count { get { lock (sync) return ready.Count; } }
    internal int ActiveCount { get { lock (sync) return active; } }
    internal int PeakActive { get { lock (sync) return peakActive; } }
    internal TResult[] Take(int batchSize)
    {
        if (batchSize < 1 || batchSize > capacity) throw new ArgumentOutOfRangeException(nameof(batchSize));
        lock (sync)
        {
            ThrowFailure();
            while (ready.Count < batchSize && !(ended && active == 0)) Wait();
            var batch = Enumerable.Range(0, Math.Min(batchSize, ready.Count)).Select(_ => ready.Dequeue()).ToArray();
            Monitor.PulseAll(sync);
            return batch;
        }
    }
    internal TResult[] PauseAndSnapshot()
    {
        lock (sync)
        {
            paused = true;
            ThrowFailure();
            while (active != 0) Wait();
            return ready.ToArray();
        }
    }
    internal void Resume() { lock (sync) { ThrowFailure(); paused = false; Monitor.PulseAll(sync); } }
    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        stop.Cancel();
        lock (sync) Monitor.PulseAll(sync);
        foreach (Thread thread in threads) thread.Join();
        stop.Dispose();
        lock (sync) if (failures.Count > 0) throw new AggregateException("DPO generation cleanup failed.", failures);
    }
}
