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
        var h1 = Ln1.ForwardResidualDropout(
            x,
            Attn.Forward(x),
            AttnDropout);
        var h2 = Ln2.ForwardResidualDropout(
            h1,
            Ffn.Forward(h1),
            FfnDropout);
        return h2;
    }

    internal Tensor ForwardIncremental(
        Tensor x,
        CudaAttentionKvCache cache,
        int position)
    {
        Tensor h1 = Ln1.ForwardResidualDropout(
            x,
            Attn.ForwardIncremental(x, cache, position),
            AttnDropout);
        return Ln2.ForwardResidualDropout(
            h1,
            Ffn.Forward(h1),
            FfnDropout);
    }

    internal Tensor ForwardPrefill(
        Tensor x,
        CudaAttentionKvCache cache,
        int sequence)
    {
        Tensor h1 = Ln1.ForwardResidualDropout(
            x,
            Attn.ForwardPrefill(x, cache, sequence),
            AttnDropout);
        return Ln2.ForwardResidualDropout(
            h1,
            Ffn.Forward(h1),
            FfnDropout);
    }

}
