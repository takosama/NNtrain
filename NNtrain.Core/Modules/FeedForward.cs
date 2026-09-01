namespace NNtrain;

class FeedForward : Module
{
    public Linear Fc1 { get; }
    public Linear Fc2 { get; }

    public FeedForward(
        int dModel,
        int dHidden,
        Random? rng = null,
        float initScale = 0.02f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        rng ??= new Random(1);
        Fc1 = RegisterModule(
            new Linear(dModel, dHidden, rng, initScale, dtype));
        Fc2 = RegisterModule(
            new Linear(dHidden, dModel, rng, initScale, dtype));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        Tensor expanded = Fc1.ForwardBatchReluExclusiveOutputGradient(x);
        return Fc2.ForwardBatchExclusiveInputGradient(expanded);
    }

}
