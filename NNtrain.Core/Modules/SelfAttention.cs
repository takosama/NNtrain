namespace NNtrain;

class SelfAttention : Module
{
    public AttentionHead Head { get; }
    public Linear Wo { get; }

    public SelfAttention(
        int dModel,
        int dHead = -1,
        bool causal = false,
        Random? rng = null,
        float initScale = 0.02f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        if (dHead <= 0) dHead = dModel;
        rng ??= new Random(1);

        Head = RegisterModule(
            new AttentionHead(dModel, dHead, causal, rng, initScale, dtype));
        Wo = RegisterModule(new Linear(dHead, dModel, rng, initScale, dtype));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        return Wo.ForwardBatch(Head.Forward(x)); // (T, D)
    }

}
