namespace NNtrain;

public interface IOptimizer
{
    void ZeroGrad();

    void Step();
}

public static class OptimizerTorchExtensions
{
    public static void zero_grad(this IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        optimizer.ZeroGrad();
    }

    public static void step(this IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        optimizer.Step();
    }
}

public interface ILearningRateAdjustable
{
    float LearningRate { get; }

    void SetLearningRate(float learningRate);
}
