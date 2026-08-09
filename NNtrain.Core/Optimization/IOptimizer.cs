namespace NNtrain;

public interface IOptimizer
{
    void ZeroGrad();

    void Step();
}

public interface ILearningRateAdjustable
{
    float LearningRate { get; }

    void SetLearningRate(float learningRate);
}
