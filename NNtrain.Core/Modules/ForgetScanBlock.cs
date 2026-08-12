namespace NNtrain;

/// <summary>
/// Pre-normalized residual ForgetScan mixer followed by a feed-forward block.
/// </summary>
class ForgetScanBlock : Module
{
    public LayerNorm Ln1 { get; }
    public ForgetScanMixer Mixer { get; }
    public Dropout MixerDropout { get; }
    public LayerNorm Ln2 { get; }
    public FeedForward Ffn { get; }
    public Dropout FfnDropout { get; }

    public ForgetScanBlock(
        int modelWidth,
        int hiddenWidth,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f)
    {
        random ??= new Random(1);
        Ln1 = RegisterModule(new LayerNorm(modelWidth));
        Mixer = RegisterModule(
            new ForgetScanMixer(
                modelWidth,
                random,
                initializationScale));
        MixerDropout = RegisterModule(new Dropout(dropout, random));
        Ln2 = RegisterModule(new LayerNorm(modelWidth));
        Ffn = RegisterModule(
            new FeedForward(
                modelWidth,
                hiddenWidth,
                random,
                initializationScale));
        FfnDropout = RegisterModule(new Dropout(dropout, random));
    }

    public Tensor Forward(Tensor input)
    {
        Tensor mixed = MixerDropout.AddResidual(
            input,
            Mixer.Forward(Ln1.Forward(input)));
        return FfnDropout.AddResidual(
            mixed,
            Ffn.Forward(Ln2.Forward(mixed)));
    }
}
