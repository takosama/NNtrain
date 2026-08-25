namespace NNtrain;

public class TransformerClassifier : Module
{
    private readonly TransformerBlock[] _blocks;
    private readonly Parameter[] _hiddenWeightParameters;
    private readonly Parameter[] _auxiliaryParameters;

    internal IReadOnlyList<TransformerBlock> Blocks { get; }
    internal Linear Head { get; }
    internal Parameter Pos { get; } // (seqLen, dModel)

    public int DModel { get; }
    public int SeqLen { get; }
    public int NumClasses { get; }

    public TransformerClassifier(
        int seqLen,
        int dModel,
        int numHeads,
        int dHidden,
        int numLayers,
        int numClasses,
        Random? rng = null,
        float initScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        rng ??= new Random(1);

        if (!float.IsFinite(dropout) || dropout < 0f || dropout >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(dropout),
                dropout,
                "Dropout probability must be finite and in [0, 1).");
        }

        SeqLen = seqLen;
        DModel = dModel;
        NumClasses = numClasses;

        _blocks = new TransformerBlock[numLayers];
        for (int i = 0; i < numLayers; i++)
        {
            _blocks[i] = new TransformerBlock(
                dModel,
                numHeads,
                dHidden,
                false,
                rng,
                initScale,
                dropout,
                dtype);
        }

        float[] pos = new float[seqLen * dModel];
        for (int i = 0; i < pos.Length; i++)
            pos[i] = ((float)rng.NextDouble() * 2f - 1f) * initScale;

        Pos = RegisterParameter(
            new Parameter(
                pos,
                new[] { seqLen, dModel },
                "Pos",
                WeightDecayPolicy.Apply,
                dtype));

        for (int i = 0; i < _blocks.Length; i++)
            RegisterModule(_blocks[i]);

        Head = RegisterModule(
            new Linear(
                seqLen * dModel,
                numClasses,
                rng,
                initScale,
                dtype));
        Blocks = Array.AsReadOnly(_blocks);

        _hiddenWeightParameters = _blocks
            .SelectMany(block => block.Parameters())
            .Where(parameter => parameter.T.Rank >= 2)
            .ToArray();
        var hiddenWeightSet = new HashSet<Parameter>(
            _hiddenWeightParameters,
            ReferenceEqualityComparer.Instance);
        _auxiliaryParameters = Parameters()
            .Where(parameter => !hiddenWeightSet.Contains(parameter))
            .ToArray();
    }

    public IReadOnlyList<Parameter> HiddenWeightParameters
        => Array.AsReadOnly(_hiddenWeightParameters);

    public IReadOnlyList<Parameter> AuxiliaryParameters
        => Array.AsReadOnly(_auxiliaryParameters);

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

    public Tensor forward(Tensor input)
        => input.Rank == 3 ? ForwardBatch(input) : Forward(input);

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
