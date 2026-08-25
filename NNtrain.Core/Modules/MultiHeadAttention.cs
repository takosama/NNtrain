namespace NNtrain;

class MultiHeadAttention : Module
{
    private readonly bool _causal;

    public Linear Qkv { get; }
    public Linear Wo { get; }

    public int DModel { get; }
    public int NumHeads { get; }
    public int DHead { get; }

    public MultiHeadAttention(
        int dModel,
        int numHeads,
        bool causal = false,
        Random? rng = null,
        float initScale = 0.02f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        if (numHeads <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(numHeads),
                numHeads,
                "Head count must be positive.");
        if (dModel % numHeads != 0)
            throw new ArgumentException("dModel must be divisible by numHeads");

        DModel = dModel;
        NumHeads = numHeads;
        DHead = dModel / numHeads;
        _causal = causal;

        rng ??= new Random(1);

        Qkv = RegisterModule(
            new Linear(dModel, 3 * dModel, rng, initScale, dtype));
        Wo = RegisterModule(new Linear(dModel, dModel, rng, initScale, dtype));
    }

    public Tensor Forward(Tensor x) // (T, D) or (B, T, D)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank is not 2 and not 3)
        {
            throw new InvalidOperationException(
                "Multi-head attention input must have rank 2 or rank 3.");
        }

        if (x.Shape[^1] != DModel)
        {
            throw new ArgumentException(
                $"Attention input width '{x.Shape[^1]}' " +
                $"does not match dModel '{DModel}'.",
                nameof(x));
        }

        Tensor projected = Qkv.ForwardBatch(x);
        Tensor attended = projected.FusedMultiHeadAttention(
            NumHeads,
            _causal);
        return Wo.ForwardBatch(attended);
    }

    internal CudaAttentionKvCache CreateIncrementalCache(int capacity)
        => new(capacity, DModel);

    internal Tensor ForwardIncremental(
        Tensor x,
        CudaAttentionKvCache cache,
        int position)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(cache);
        Tensor projected = Qkv.ForwardBatch(x);
        Tensor attended = projected.FusedMultiHeadAttentionIncremental(
            cache.Key,
            cache.Value,
            position,
            cache.Capacity,
            DModel,
            NumHeads);
        return Wo.ForwardBatch(attended);
    }

    internal Tensor ForwardPrefill(
        Tensor x,
        CudaAttentionKvCache cache,
        int sequence)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(cache);
        Tensor projected = Qkv.ForwardBatch(x);
        projected.PrefillMultiHeadAttentionCache(
            cache.Key,
            cache.Value,
            sequence,
            cache.Capacity,
            DModel);
        return Wo.ForwardBatch(
            projected.FusedMultiHeadAttention(NumHeads, _causal));
    }

}

internal sealed class CudaAttentionKvCache : IDisposable
{
    internal CudaAttentionKvCache(int capacity, int modelWidth)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(capacity);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(modelWidth);
        Capacity = capacity;
        NativeCudaDevice accelerator = ForgetMemoryV2Cuda.GetAccelerator();
        Key = accelerator.Allocate1D<ushort>(checked(capacity * modelWidth));
        Value = accelerator.Allocate1D<ushort>(checked(capacity * modelWidth));
    }

    internal int Capacity { get; }
    internal NativeCudaBuffer<ushort> Key { get; }
    internal NativeCudaBuffer<ushort> Value { get; }

    public void Dispose()
    {
        Key.Dispose();
        Value.Dispose();
    }
}
