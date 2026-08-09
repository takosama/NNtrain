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
        float dropout = 0f)
    {
        rng ??= new Random(1);

        Attn = RegisterModule(
            new MultiHeadAttention(dModel, numHeads, causal, rng, initScale));
        AttnDropout = RegisterModule(new Dropout(dropout, rng));
        Ln1 = RegisterModule(new LayerNorm(dModel));
        Ffn = RegisterModule(new FeedForward(dModel, dHidden, rng, initScale));
        FfnDropout = RegisterModule(new Dropout(dropout, rng));
        Ln2 = RegisterModule(new LayerNorm(dModel));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        var h1 = Ln1.ForwardResidual(
            x,
            AttnDropout.Forward(Attn.Forward(x)));
        var h2 = Ln2.ForwardResidual(
            h1,
            FfnDropout.Forward(Ffn.Forward(h1)));
        return h2;
    }

}
