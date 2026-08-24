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
    public void LionImplementsTheOptimizerContract()
    {
        var parameter = CreateParameter("weight");

        IOptimizer optimizer = new Lion([parameter]);

        Assert.IsType<Lion>(optimizer);
    }

    [Fact]
    public void GainShareAdamWImplementsTheOptimizerContract()
    {
        var parameter = CreateParameter("weight");

        IOptimizer optimizer = new GainShareAdamW([[parameter]]);

        Assert.IsType<GainShareAdamW>(optimizer);
    }

    [Fact]
    public void NekoMuonImplementsTheOptimizerContract()
    {
        var parameter = CreateParameter("weight");

        IOptimizer optimizer = new NekoMuon([parameter]);

        Assert.IsType<NekoMuon>(optimizer);
    }

    [Fact]
    public void CompositeOptimizerForwardsEachLifecycleOperationOnce()
    {
        var first = new CountingOptimizer();
        var second = new CountingOptimizer();
        IOptimizer optimizer = new CompositeOptimizer(first, second);

        optimizer.zero_grad();
        optimizer.step();

        Assert.Equal(1, first.ZeroGradCalls);
        Assert.Equal(1, first.StepCalls);
        Assert.Equal(1, second.ZeroGradCalls);
        Assert.Equal(1, second.StepCalls);
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

        optimizer.step();

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

        optimizer.zero_grad();

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

        Assert.Equal(
            ["load_state_dict", "state_dict", "step", "zero_grad"],
            methods);
    }

    [Theory]
    [InlineData(typeof(AdamW))]
    [InlineData(typeof(Lion))]
    [InlineData(typeof(NekoMuon))]
    [InlineData(typeof(GainShareAdamW))]
    [InlineData(typeof(CompositeOptimizer))]
    public void ConcreteOptimizersDoNotRepublishPascalLifecycle(Type type)
    {
        string[] publicMethods = type
            .GetMethods()
            .Select(method => method.Name)
            .ToArray();

        Assert.DoesNotContain("Step", publicMethods);
        Assert.DoesNotContain("ZeroGrad", publicMethods);
        Assert.Contains("step", publicMethods);
        Assert.Contains("zero_grad", publicMethods);
    }

    private static Parameter CreateParameter(string name)
    {
        return new Parameter(
            [1f],
            [1],
            name,
            WeightDecayPolicy.Exclude);
    }

    private sealed class CountingOptimizer : IOptimizer
    {
        internal int ZeroGradCalls { get; private set; }

        internal int StepCalls { get; private set; }

        public void zero_grad() => ZeroGradCalls++;

        public void step() => StepCalls++;
    }
}
