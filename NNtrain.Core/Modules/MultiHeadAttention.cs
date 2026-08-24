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

}
