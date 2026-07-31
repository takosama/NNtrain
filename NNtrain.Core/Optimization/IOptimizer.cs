namespace NNtrain;

public interface IOptimizer
{
    void ZeroGrad();

    void Step();
}
