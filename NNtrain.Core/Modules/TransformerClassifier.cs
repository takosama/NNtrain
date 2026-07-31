namespace NNtrain;

public class TransformerClassifier : Module, IClassificationModel
{
    private readonly TransformerBlock[] _blocks;

    internal IReadOnlyList<TransformerBlock> Blocks { get; }
    internal Linear Head { get; }
    internal Parameter Pos { get; } // (seqLen, dModel)

    public int DModel { get; }
    public int SeqLen { get; }
    public int NumClasses { get; }

    int IClassificationModel.InputRows => SeqLen;
    int IClassificationModel.InputColumns => DModel;
    int IClassificationModel.ClassCount => NumClasses;

    public TransformerClassifier(
        int seqLen,
        int dModel,
        int numHeads,
        int dHidden,
        int numLayers,
        int numClasses,
        Random? rng = null,
        float initScale = 0.02f)
    {
        rng ??= new Random(1);

        SeqLen = seqLen;
        DModel = dModel;
        NumClasses = numClasses;

        _blocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
            _blocks[i] = new TransformerBlock(dModel, numHeads, dHidden, false, rng, initScale);

        float[] pos = new float[seqLen * dModel];
        for (int i = 0; i < pos.Length; i++)
            pos[i] = ((float)rng.NextDouble() * 2f - 1f) * initScale;

        Pos = RegisterParameter(
            new Parameter(
                pos,
                new[] { seqLen, dModel },
                "Pos",
                WeightDecayPolicy.Apply));

        for (int i = 0; i < _blocks.Length; i++)
            RegisterModule(_blocks[i]);

        Head = RegisterModule(
            new Linear(seqLen * dModel, numClasses, rng, initScale));
        Blocks = Array.AsReadOnly(_blocks);
    }

    public Tensor Forward(Tensor x)
    {
        var h = x + Pos.T;

        for (int i = 0; i < _blocks.Length; i++)
            h = _blocks[i].Forward(h);

        var flat = h.Reshape(SeqLen * DModel);
        return Head.Forward(flat);
    }

}
