namespace NNtrain;

class TransformerBlock : Module
{
    public MultiHeadAttention Attn { get; }
    public Dropout AttnDropout { get; }
    public LayerNorm Ln1 { get; }
    public FeedForward Ffn { get; }
    public Dropout FfnDropout { get; }
    public LayerNorm Ln2 { get; }

    public TransformerBlock(
        int dModel,
        int numHeads,
        int dHidden,
        bool causal = false,
        Random? rng = null,
        float initScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        rng ??= new Random(1);

        Attn = RegisterModule(
            new MultiHeadAttention(
                dModel, numHeads, causal, rng, initScale, dtype));
        AttnDropout = RegisterModule(new Dropout(dropout, rng, dtype));
        Ln1 = RegisterModule(new LayerNorm(dModel, dtype: dtype));
        Ffn = RegisterModule(
            new FeedForward(dModel, dHidden, rng, initScale, dtype));
        FfnDropout = RegisterModule(new Dropout(dropout, rng, dtype));
        Ln2 = RegisterModule(new LayerNorm(dModel, dtype: dtype));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        // Keep CUDA-resident gradients in the regular add/layer-norm kernels.
        // The fused residual-layer-norm CPU backward assumes host gradient
        // buffers and is therefore not valid for a CUDA-resident branch.
        var h1 = Ln1.Forward(
            AttnDropout.AddResidual(x, Attn.Forward(x)));
        var h2 = Ln2.Forward(
            FfnDropout.AddResidual(h1, Ffn.Forward(h1)));
        return h2;
    }

}
