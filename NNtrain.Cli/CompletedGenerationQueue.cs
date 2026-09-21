namespace NNtrain;

// Cooperative tasks on a single execution lane: no native TLS or model-mode races.
// A turn advances one task; completed tasks leave immediately, independent of other lengths.
internal sealed class CompletedGenerationQueue<T> : IDisposable
{
    internal interface IJob : IDisposable { bool Advance(out T result); }
    private readonly List<IJob> active = [];
    private readonly Queue<T> ready = new();
    private readonly int slots, capacity;
    private int cursor;
    internal CompletedGenerationQueue(int slots, int capacity, IEnumerable<T>? restored = null)
    {
        if (slots <= 0 || capacity < slots) throw new ArgumentException("Invalid generation queue capacity.");
        this.slots = slots; this.capacity = capacity;
        foreach (T item in restored ?? []) ready.Enqueue(item);
        if (ready.Count > capacity) throw new InvalidDataException("Saved queue exceeds capacity.");
    }
    internal int ActiveCount => active.Count;
    internal int Count => ready.Count;
    internal T[] Snapshot() => ready.ToArray();
    internal T[] Take(int count)
    {
        if (count <= 0 || count > ready.Count) throw new ArgumentOutOfRangeException(nameof(count));
        return Enumerable.Range(0, count).Select(_ => ready.Dequeue()).ToArray();
    }
    internal void Fill(Func<IJob?> create)
    {
        while (active.Count < slots && active.Count + ready.Count < capacity)
        {
            var job = create();
            if (job is null) break;
            active.Add(job);
        }
    }
    internal void Tick(CancellationToken cancellation)
    {
        cancellation.ThrowIfCancellationRequested();
        if (active.Count == 0) return;
        cursor %= active.Count;
        IJob job = active[cursor];
        if (job.Advance(out T item))
        {
            active.RemoveAt(cursor);
            if (active.Count == 0) cursor = 0;
            job.Dispose();
            ready.Enqueue(item);
        }
        else cursor++;
    }
    internal void Drain(CancellationToken cancellation)
    {
        while (active.Count != 0) Tick(cancellation);
    }
    public void Dispose()
    {
        List<Exception> errors = [];
        foreach (var job in active)
            try { job.Dispose(); } catch (Exception e) { errors.Add(e); }
        active.Clear();
        if (errors.Count > 0) throw new AggregateException(errors);
    }
}
