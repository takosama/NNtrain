using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Session-bounded optimizer state; host arrays are refreshed only at explicit handoffs.</summary>
internal sealed class ArcOptimizerStateCache : IDisposable
{
    private readonly Dictionary<float[], ArcBuffer> _values = new(ReferenceEqualityComparer.Instance);
    private ArcExecutionLane? _lane;
    private IDisposable? _registration;

    internal ArcBuffer Get(float[] values)
    {
        var lane = Tensor.ArcLane;
        if (_lane is not null && !ReferenceEquals(_lane, lane)) Dispose();
        if (_lane is null)
        {
            _lane = lane;
            _registration = ExecutionSession.Current!.RegisterBeforeDispose(this, static owner => ((ArcOptimizerStateCache)owner).Dispose());
        }
        if (!_values.TryGetValue(values, out ArcBuffer? buffer)) _values.Add(values, buffer = lane.Upload(values));
        return buffer;
    }

    internal void Synchronize()
    {
        if (_lane is null) return;
        foreach (var (host, device) in _values) _lane.Read(device, host);
    }

    public void Dispose()
    {
        try { Synchronize(); }
        finally
        {
            foreach (ArcBuffer buffer in _values.Values) buffer.Dispose();
            _values.Clear(); _lane = null;
            _registration?.Dispose(); _registration = null;
        }
    }
}
