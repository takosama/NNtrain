using NNtrain;
using Xunit;

public sealed class OptimizerStateCodecRegistryTests
{
    [Fact]
    public void RegistryRejectsDuplicateOptimizerType()
    {
        var registry = new OptimizerStateCodecRegistry();
        registry.Register(CreateCodec<FirstOptimizer>("First"));

        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(
            () => registry.Register(
                CreateCodec<FirstOptimizer>("FirstReplacement")));

        Assert.Contains(nameof(FirstOptimizer), exception.Message);
    }

    [Fact]
    public void RegistryRejectsDuplicateSerializedTypeName()
    {
        var registry = new OptimizerStateCodecRegistry();
        registry.Register(CreateCodec<FirstOptimizer>("Shared"));

        InvalidOperationException exception = Assert.Throws<InvalidOperationException>(
            () => registry.Register(CreateCodec<SecondOptimizer>("Shared")));

        Assert.Contains("Shared", exception.Message);
    }

    [Fact]
    public void RegistryResolvesOnlyTheExactRegisteredOptimizerType()
    {
        var registry = new OptimizerStateCodecRegistry();
        registry.Register(CreateCodec<FirstOptimizer>("First"));

        Assert.True(registry.TryResolve(
            new FirstOptimizer(),
            out IOptimizerStateCodec? codec));
        Assert.Equal("First", codec!.StateType);
        Assert.False(registry.TryResolve(
            new SecondOptimizer(),
            out _));
    }

    [Fact]
    public void UnknownOptimizerKeepsTheExistingExceptionContract()
    {
        var optimizer = new FirstOptimizer();

        NotSupportedException typeException = Assert.Throws<NotSupportedException>(
            () => OptimizerStateStream.GetStateType(optimizer));
        NotSupportedException saveException = Assert.Throws<NotSupportedException>(
            () => OptimizerStateStream.SaveStateJson(
                optimizer,
                new MemoryStream()));

        const string Expected =
            "Optimizer 'FirstOptimizer' does not support streaming " +
            "checkpoint state.";
        Assert.Equal(Expected, typeException.Message);
        Assert.Equal(Expected, saveException.Message);
    }

    [Fact]
    public void CompositeKeepsItsTypeAndLeafOnlyStreamingContract()
    {
        var first = new AdamW([CreateParameter("first")]);
        var second = new Lion([CreateParameter("second")]);
        var composite = new CompositeOptimizer(first, second);

        Assert.Equal(
            "CompositeOptimizer",
            OptimizerStateStream.GetStateType(composite));
        Assert.Equal(
            new IOptimizer[] { first, second },
            OptimizerStateStream.GetLeafOptimizers(composite));
        ArgumentException saveException = Assert.Throws<ArgumentException>(
            () => OptimizerStateStream.SaveStateJson(
                composite,
                new MemoryStream()));
        ArgumentException loadException = Assert.Throws<ArgumentException>(
            () => OptimizerStateStream.LoadStateJson(
                composite,
                new MemoryStream()));
        Assert.Equal("optimizer", saveException.ParamName);
        Assert.Equal("optimizer", loadException.ParamName);
    }

    private static OptimizerStateCodec<TOptimizer> CreateCodec<TOptimizer>(
        string stateType)
        where TOptimizer : class, IOptimizer
        => new(
            stateType,
            static (_, _) => { },
            static (_, _, _) => { },
            static (_, _) => { },
            static (_, _, _) => { });

    private static Parameter CreateParameter(string name)
        => new(
            [1f],
            [1],
            name,
            WeightDecayPolicy.Exclude);

    private sealed class FirstOptimizer : IOptimizer
    {
        public void zero_grad() { }

        public void step() { }
    }

    private sealed class SecondOptimizer : IOptimizer
    {
        public void zero_grad() { }

        public void step() { }
    }
}
