using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>
/// Persistent, head-major Arc K/V values for one causal attention layer.
/// The owner must dispose this cache before its inference session ends.
/// </summary>
internal sealed class ArcAttentionKvCache : IDisposable
{
    // The incremental kernel stages one scaled score per cached token in SLM.
    // At 8192 tokens this consumes 32 KiB plus its 64-lane reduction buffer.
    internal const int MaximumCapacity = 8192;

    private bool _disposed;

    internal ArcAttentionKvCache(int width, int heads, int capacity)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (heads <= 0 || width % heads != 0)
            throw new ArgumentException("The model width must be divisible by the head count.", nameof(heads));
        if (capacity is < 1 or > MaximumCapacity)
            throw new ArgumentOutOfRangeException(nameof(capacity),
                $"Arc incremental attention supports a capacity of 1..{MaximumCapacity} tokens.");

        Lane = Tensor.ArcLane;
        Width = width;
        Heads = heads;
        Capacity = capacity;
        int elements = checked(width * capacity);
        Key = Lane.Allocate(elements);
        try { Value = Lane.Allocate(elements); }
        catch { Key.Dispose(); throw; }
    }

    internal ArcExecutionLane Lane { get; }
    internal ArcBuffer Key { get; }
    internal ArcBuffer Value { get; }
    internal int Width { get; }
    internal int Heads { get; }
    internal int Capacity { get; }
    internal int Length { get; private set; }

    internal void Validate(ArcExecutionLane lane)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!ReferenceEquals(Lane, lane))
            throw new InvalidOperationException("Arc attention cache belongs to a different device lane.");
    }

    internal void MarkPrefilled(int sequence)
    {
        if (Length != 0 || sequence is < 1 || sequence > Capacity)
            throw new InvalidOperationException("Arc attention cache prefill must run once within capacity.");
        Length = sequence;
    }

    internal void MarkAppended(int position)
    {
        if (position != Length || position >= Capacity)
            throw new InvalidOperationException("Arc attention cache tokens must append in position order.");
        Length = position + 1;
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Value.Dispose();
        Key.Dispose();
    }
}
