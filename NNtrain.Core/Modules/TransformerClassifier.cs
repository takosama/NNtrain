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
        return ForwardFromEmbedding(Embed(x));
    }

    public Tensor ForwardBatch(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank != 3)
        {
            throw new InvalidOperationException(
                "ForwardBatch requires input shaped " +
                "[batch, sequence, features].");
        }

        return Forward(x);
    }

    public Tensor Embed(Tensor x)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.Rank == 2)
            return x + Pos.T;
        if (x.Rank == 3)
            return x.AddBatchWise(Pos.T);

        throw new InvalidOperationException(
            "Transformer input must have rank 2 or rank 3.");
    }

    public Tensor ForwardFromEmbedding(Tensor embedding)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        Tensor h = embedding;

        for (int i = 0; i < _blocks.Length; i++)
            h = _blocks[i].Forward(h);

        if (h.Rank == 2)
        {
            var flat = h.Reshape(SeqLen * DModel);
            return Head.Forward(flat);
        }

        if (h.Rank == 3)
        {
            int batch = h.Shape[0];
            var flat = h.Reshape(batch, SeqLen * DModel);
            return Head.ForwardBatch(flat);
        }

        throw new InvalidOperationException(
            "Transformer embedding must have rank 2 or rank 3.");
    }

}
