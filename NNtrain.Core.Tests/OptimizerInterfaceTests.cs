using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class OptimizerInterfaceTests
{
    [Fact]
    public void AdamWImplementsTheOptimizerContract()
    {
        var parameter = CreateParameter("weight");

        IOptimizer optimizer = new AdamW([parameter]);

        Assert.IsType<AdamW>(optimizer);
    }

    [Fact]
    public void StepCanBeInvokedThroughTheOptimizerContract()
    {
        var parameter = CreateParameter("weight");
        parameter.T.MutableGrad[0] = 1f;
        IOptimizer optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0f,
            });

        optimizer.Step();

        AssertClose([0.9f], parameter.T.Data, 2e-5f);
    }

    [Fact]
    public void ZeroGradCanBeInvokedThroughTheOptimizerContract()
    {
        var first = CreateParameter("first");
        var second = CreateParameter("second");
        first.T.MutableGrad[0] = 3f;
        second.T.MutableGrad[0] = 4f;
        IOptimizer optimizer = new AdamW([first, second]);

        optimizer.ZeroGrad();

        AssertClose([0f], first.T.Grad);
        AssertClose([0f], second.T.Grad);
    }

    [Fact]
    public void OptimizerContractContainsOnlyLifecycleOperations()
    {
        string[] methods = typeof(IOptimizer)
            .GetMethods()
            .Select(method => method.Name)
            .Order()
            .ToArray();

        Assert.Equal(["Step", "ZeroGrad"], methods);
    }

    private static Parameter CreateParameter(string name)
    {
        return new Parameter(
            [1f],
            [1],
            name,
            WeightDecayPolicy.Exclude);
    }
}
