namespace NNtrain;

class AttentionHead : Module
{
    public Linear Wq { get; }
    public Linear Wk { get; }
    public Linear Wv { get; }

    private readonly float _scale;
    private readonly bool _causal;

    public AttentionHead(int dModel, int dHead, bool causal = false, Random? rng = null, float initScale = 0.02f)
    {
        rng ??= new Random(1);

        Wq = RegisterModule(new Linear(dModel, dHead, rng, initScale));
        Wk = RegisterModule(new Linear(dModel, dHead, rng, initScale));
        Wv = RegisterModule(new Linear(dModel, dHead, rng, initScale));

        _scale = 1f / MathF.Sqrt(dHead);
        _causal = causal;
    }

    public Tensor Forward(Tensor x) // x: (T, D)
    {
        Tensor q = Wq.ForwardBatch(x); // (T, Dh)
        Tensor k = Wk.ForwardBatch(x); // (T, Dh)
        Tensor v = Wv.ForwardBatch(x); // (T, Dh)

        Tensor scores = q.MatMulTransposedRight(k) * Tensor.Scalar(_scale); // (T, T)
        if (_causal)
            scores = scores.CausalMask();

        Tensor attention = scores.SoftmaxLastDim(); // (T, T)
        return attention.MatMul(v);                 // (T, Dh)
    }

}
