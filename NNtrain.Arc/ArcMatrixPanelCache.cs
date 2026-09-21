namespace NNtrain.Arc;

// Tensor-owned, lane-budgeted immutable weight layouts. Clear BEFORE changing
// the source value. Already-enqueued consumers finish through the normal
// in-order buffer retirement rules; managed borrows expire on invalidation.
public sealed class ArcMatrixPanelCache(ArcExecutionLane lane) : IDisposable
{
    private readonly Dictionary<(int Outer, int Depth, bool Transpose), ArcExecutionLane.ArcBuffer> _panels = [];
    private bool _disposed;

    public ArcExecutionLane.ArcBuffer GetOrCreate(int outer, int depth, bool transpose, long bytes,
        Func<ArcExecutionLane.ArcBuffer> create)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var key = (outer, depth, transpose);
        if (_panels.TryGetValue(key, out var existing)) return existing.Borrow();
        if (_panels.Count >= 2 || !lane.TryReserveMatrixPanel(bytes)) return create();
        ArcExecutionLane.ArcBuffer? value = null;
        try
        {
            value = create();
            if (value.Bytes != bytes) throw new InvalidOperationException("Matrix panel reservation does not match allocation.");
            var borrowed = value.Borrow();
            try { _panels.Add(key, value); }
            catch { borrowed.Dispose(); throw; }
            return borrowed;
        }
        catch
        {
            try { value?.Dispose(); }
            finally { lane.ReleaseMatrixPanel(bytes); }
            throw;
        }
    }

    public void Clear()
    {
        var panels = _panels.Values.ToArray();
        _panels.Clear();
        List<Exception>? failures = null;
        foreach (var panel in panels)
        {
            try { panel.Dispose(); }
            catch (Exception failure) { (failures ??= []).Add(failure); }
            finally { lane.ReleaseMatrixPanel(panel.Bytes); }
        }
        if (failures is not null) throw new AggregateException("Matrix panel cleanup failed.", failures);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Clear();
    }
}

public sealed partial class ArcExecutionLane
{
    public long RetainedMatrixPanelBytes { get; private set; }
    internal bool TryReserveMatrixPanel(long bytes)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long budget = Options.MatrixPanelCacheMiB * 1048576L;
            if (bytes <= 0 || bytes > budget - RetainedMatrixPanelBytes) return false;
            RetainedMatrixPanelBytes += bytes;
            return true;
        }
    }
    internal void ReleaseMatrixPanel(long bytes)
    {
        lock (_sync) RetainedMatrixPanelBytes -= bytes;
    }
}
