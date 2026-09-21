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
        => ForwardArcPlanned(x, IsTraining && AutogradContext.IsRecordingEnabled
            && Tensor.ArcResident && Tensor.ArcLane.Options.TransformerFfnCheckpointing);

    // A shape-specific automatic plan is passed by value; it never mutates
    // shared lane options or leaks its policy into another model/forward.
    internal Tensor ForwardArcPlanned(Tensor x, bool checkpointFfn)
        => ForwardArcPlanned(x, checkpointFfn, AttnDropout, FfnDropout);

    internal Func<Tensor, Tensor> CaptureArcCheckpointForward(bool checkpointFfn)
    {
        // Backward remains valid after model.eval(), just as for a saved eager
        // graph. These private snapshots freeze effective dropout modes, not
        // the mutable model's Train/Eval state. Their Random is never consumed:
        // checkpoint replay supplies the recorded device-mask seeds instead.
        var attentionDropout = new Dropout(AttnDropout.IsTraining ? AttnDropout.Probability : 0f,
            dtype: AttnDropout.DType);
        var feedForwardDropout = new Dropout(FfnDropout.IsTraining ? FfnDropout.Probability : 0f,
            dtype: FfnDropout.DType);
        return input => Tensor.IsArcCheckpointReplay
            ? ForwardArcPlanned(input, checkpointFfn, attentionDropout, feedForwardDropout)
            : ForwardArcPlanned(input, checkpointFfn);
    }

    private Tensor ForwardArcPlanned(Tensor x, bool checkpointFfn,
        Dropout attentionDropout, Dropout feedForwardDropout)
    {
        var h1 = Ln1.ForwardResidualDropout(
            x,
            Attn.Forward(x),
            attentionDropout);
        bool applyCheckpoint = checkpointFfn && IsTraining && AutogradContext.IsRecordingEnabled
            && Tensor.ArcResident && !Tensor.IsArcCheckpointActive;
        Tensor feedForward = applyCheckpoint
            ? h1.ArcCheckpoint(Ffn.Forward, Ffn.Parameters().Select(parameter => parameter.T).ToArray())
            : Ffn.Forward(h1);
        var h2 = Ln2.ForwardResidualDropout(
            h1,
            feedForward,
            feedForwardDropout);
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
