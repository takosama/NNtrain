namespace NNtrain;

class FeedForward : Module
{
    public Linear Fc1 { get; }
    public Linear Fc2 { get; }

    public FeedForward(int dModel, int dHidden, Random? rng = null, float initScale = 0.02f)
    {
        rng ??= new Random(1);
        Fc1 = RegisterModule(new Linear(dModel, dHidden, rng, initScale));
        Fc2 = RegisterModule(new Linear(dHidden, dModel, rng, initScale));
    }

    public Tensor Forward(Tensor x) // (T, D)
    {
        return Fc2.ForwardBatch(Fc1.ForwardBatchRelu(x));
    }

}
