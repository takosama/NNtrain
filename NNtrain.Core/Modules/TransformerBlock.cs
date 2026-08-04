namespace NNtrain;

class TransformerBlock : Module
{
    public MultiHeadAttention Attn { get; }
    public LayerNorm Ln1 { get; }
    public FeedForward Ffn { get; }
    public LayerNorm Ln2 { get; }

    public TransformerBlock(
        int dModel,
        int numHeads,
        int dHidden,
        bool causal = false,
        Random? rng = null,
        float initScale = 0.02f)
    {
        rng ??= new Random(1);

        Attn = RegisterModule(
            new MultiHeadAttention(dModel, numHeads, causal, rng, initScale));
        Ln1 = RegisterModule(new LayerNorm(dModel));
        Ffn = RegisterModule(new FeedForward(dModel, dHidden, rng, initScale));
        Ln2 = RegisterModule(new LayerNorm(dModel));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        var h1 = Ln1.ForwardResidual(x, Attn.Forward(x));
        var h2 = Ln2.ForwardResidual(h1, Ffn.Forward(h1));
        return h2;
    }

}
