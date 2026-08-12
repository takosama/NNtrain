namespace NNtrain;

/// <summary>
/// A residual language-model block using Hyena instead of self-attention.
/// </summary>
class HyenaBlock : Module
{
    public HyenaOperator Mixer { get; }
    public Dropout MixerDropout { get; }
    public LayerNorm Ln1 { get; }
    public FeedForward Ffn { get; }
    public Dropout FfnDropout { get; }
    public LayerNorm Ln2 { get; }

    public HyenaBlock(
        int modelWidth,
        int contextLength,
        int hiddenWidth,
        int filterWidth,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f,
        HyenaConvolutionAlgorithm convolutionAlgorithm =
            HyenaConvolutionAlgorithm.Auto)
    {
        random ??= new Random(1);
        Mixer = RegisterModule(
            new HyenaOperator(
                modelWidth,
                contextLength,
                filterWidth,
                random,
                initializationScale,
                convolutionAlgorithm));
        MixerDropout = RegisterModule(new Dropout(dropout, random));
        Ln1 = RegisterModule(new LayerNorm(modelWidth));
        Ffn = RegisterModule(
            new FeedForward(
                modelWidth,
                hiddenWidth,
                random,
                initializationScale));
        FfnDropout = RegisterModule(new Dropout(dropout, random));
        Ln2 = RegisterModule(new LayerNorm(modelWidth));
    }

    public Tensor Forward(Tensor input)
    {
        Tensor mixed = Ln1.ForwardResidual(
            input,
            MixerDropout.Forward(Mixer.Forward(input)));
        return Ln2.ForwardResidual(
            mixed,
            FfnDropout.Forward(Ffn.Forward(mixed)));
    }
}
