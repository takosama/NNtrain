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
        TensorDType dtype = TensorDType.Float32,
        float? outputInitScale = null)
        : base(dtype)
    {
        rng ??= new Random(1);
        float fc2Scale = outputInitScale ?? initScale;
        if (!float.IsFinite(fc2Scale) || fc2Scale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(outputInitScale));
        Fc1 = RegisterModule(
            new Linear(dModel, dHidden, rng, initScale, dtype));
        Fc2 = RegisterModule(
            new Linear(dHidden, dModel, rng, fc2Scale, dtype));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        Tensor expanded = Fc1.ForwardBatchReluExclusiveOutputGradient(x);
        return Fc2.ForwardBatchExclusiveInputGradient(expanded);
    }

}
