namespace NNtrain;

class AttentionHead : Module
{
    public Linear Wq { get; }
    public Linear Wk { get; }
    public Linear Wv { get; }

    private readonly float _scale;
    private readonly bool _causal;
    private readonly System.Collections.Concurrent.ConcurrentDictionary<
        TensorDType,
        Lazy<Tensor>> _scaleTensors = new();

    public AttentionHead(
        int dModel,
        int dHead,
        bool causal = false,
        Random? rng = null,
        float initScale = 0.02f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        rng ??= new Random(1);

        Wq = RegisterModule(new Linear(dModel, dHead, rng, initScale, dtype));
        Wk = RegisterModule(new Linear(dModel, dHead, rng, initScale, dtype));
        Wv = RegisterModule(new Linear(dModel, dHead, rng, initScale, dtype));

        _scale = 1f / MathF.Sqrt(dHead);
        _causal = causal;
    }

    public Tensor Forward(Tensor x) // x: (T, D)
    {
        Tensor q = Wq.ForwardBatch(x); // (T, Dh)
        Tensor k = Wk.ForwardBatch(x); // (T, Dh)
        Tensor v = Wv.ForwardBatch(x); // (T, Dh)

        // This scalar used to be recreated and uploaded on every forward.
        // Retain one logical tensor per storage dtype; its CUDA replica set is
        // then reused by every execution lane/device while model.to(...)
        // remains free to change the module's numeric contract.
        Tensor scale = _scaleTensors.GetOrAdd(
            DType,
            dtype => new Lazy<Tensor>(
                () => Tensor.Scalar(_scale, dtype: dtype),
                LazyThreadSafetyMode.ExecutionAndPublication)).Value;
        Tensor scores = q.MatMulTransposedRight(k) * scale; // (T, T)
        if (_causal)
            scores = scores.CausalMask();

        Tensor attention = scores.SoftmaxLastDim(); // (T, T)
        return attention.MatMul(v);                 // (T, Dh)
    }

}
