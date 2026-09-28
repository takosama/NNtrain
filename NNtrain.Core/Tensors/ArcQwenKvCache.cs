using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

/// <summary>Per-generation, token-major rotary K/V storage for one Qwen layer.</summary>
internal sealed class ArcQwenKvCache : IDisposable
{
    private bool _disposed;

    internal ArcQwenKvCache(int queryHeads, int kvHeads, int headWidth, int capacity)
    {
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
            throw new ArgumentException("Query heads must be a positive multiple of KV heads.");
        if (headWidth <= 0 || (headWidth & 1) != 0)
            throw new ArgumentOutOfRangeException(nameof(headWidth));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        Lane = Tensor.ArcLane;
        QueryHeads = queryHeads;
        KvHeads = kvHeads;
        HeadWidth = headWidth;
        Capacity = capacity;
        int elements = checked(capacity * kvHeads * headWidth);
        Key = Lane.Allocate(elements);
        try { Value = Lane.Allocate(elements); }
        catch { Key.Dispose(); throw; }
    }

    internal ArcExecutionLane Lane { get; }
    internal ArcBuffer Key { get; }
    internal ArcBuffer Value { get; }
    internal int QueryHeads { get; }
    internal int KvHeads { get; }
    internal int HeadWidth { get; }
    internal int Capacity { get; }
    internal int Length { get; private set; }

    internal void Validate(ArcExecutionLane lane, int position, int tokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!ReferenceEquals(Lane, lane))
            throw new InvalidOperationException("Qwen K/V cache belongs to a different Arc lane.");
        if (position != Length || tokens < 1 || position > Capacity - tokens)
            throw new ArgumentOutOfRangeException(nameof(position),
                "Qwen K/V tokens must append in position order within capacity.");
    }

    internal void MarkAppended(int tokens) => Length += tokens;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        Value.Dispose();
        Key.Dispose();
    }
}
