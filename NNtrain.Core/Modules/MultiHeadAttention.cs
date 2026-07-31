namespace NNtrain;

class MultiHeadAttention : Module
{
    private readonly AttentionHead[] _heads;

    public IReadOnlyList<AttentionHead> Heads { get; }
    public Linear Wo { get; }

    public int DModel { get; }
    public int NumHeads { get; }
    public int DHead { get; }

    public MultiHeadAttention(int dModel, int numHeads, bool causal = false, Random? rng = null, float initScale = 0.02f)
    {
        if (dModel % numHeads != 0)
            throw new ArgumentException("dModel must be divisible by numHeads");

        DModel = dModel;
        NumHeads = numHeads;
        DHead = dModel / numHeads;

        rng ??= new Random(1);

        _heads = new AttentionHead[numHeads];
        for (int h = 0; h < numHeads; h++)
        {
            _heads[h] = RegisterModule(
                new AttentionHead(dModel, DHead, causal, rng, initScale));
        }

        Heads = Array.AsReadOnly(_heads);
        Wo = RegisterModule(new Linear(dModel, dModel, rng, initScale));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        Tensor[] parts = new Tensor[NumHeads];
        for (int h = 0; h < NumHeads; h++)
            parts[h] = _heads[h].Forward(x);  // (T, DHead)

        Tensor cat = Tensor.Concat(1, parts); // (T, DModel)
        return Wo.ForwardBatch(cat);          // (T, DModel)
    }

}
