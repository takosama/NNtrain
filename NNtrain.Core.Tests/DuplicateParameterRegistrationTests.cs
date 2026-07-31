using NNtrain;
using Xunit;

public sealed class DuplicateParameterRegistrationTests
{
    [Fact]
    public void ModuleRejectsTheSameDirectParameterTwice()
    {
        var parameter = CreateParameter("shared");

        var exception = Assert.Throws<InvalidOperationException>(
            () => new DuplicateParameterModule(parameter));

        Assert.Contains("'shared'", exception.Message);
        Assert.Contains("already registered", exception.Message);
    }

    [Fact]
    public void ModuleRejectsTheSameDirectChildTwice()
    {
        var child = new ParameterModule(CreateParameter("child"));

        var exception = Assert.Throws<InvalidOperationException>(
            () => new DuplicateChildModule(child));

        Assert.Contains("already registered", exception.Message);
    }

    [Fact]
    public void ParameterEnumerationRejectsASharedChildThroughMultiplePaths()
    {
        var shared = new ParameterModule(CreateParameter("shared"));
        var root = new TwoBranchModule(
            new BranchModule(shared),
            new BranchModule(shared));

        var exception = Assert.Throws<InvalidOperationException>(
            () => root.Parameters().ToArray());

        Assert.Contains("multiple paths", exception.Message);
    }

    [Fact]
    public void ZeroGradDoesNotPartiallyClearAnInvalidModuleGraph()
    {
        var parameter = CreateParameter("shared");
        parameter.T.MutableGrad[0] = 3f;
        var shared = new ParameterModule(parameter);
        var root = new TwoBranchModule(
            new BranchModule(shared),
            new BranchModule(shared));

        Assert.Throws<InvalidOperationException>(() => root.ZeroGrad());

        Assert.Equal(3f, parameter.T.Grad[0]);
    }

    [Fact]
    public void AdamWRejectsTheSameParameterTwice()
    {
        var parameter = CreateParameter("weight");

        var exception = Assert.Throws<ArgumentException>(
            () => new AdamW([parameter, parameter]));

        Assert.Equal("parameters", exception.ParamName);
        Assert.Contains("'weight'", exception.Message);
        Assert.Contains("more than once", exception.Message);
    }

    [Fact]
    public void EqualNamesDoNotCountAsDuplicateParameters()
    {
        var first = CreateParameter("weight");
        var second = CreateParameter("weight");

        var optimizer = new AdamW([first, second]);

        Assert.NotNull(optimizer);
    }

    private static Parameter CreateParameter(string name)
    {
        return new Parameter(
            [1f],
            [1],
            name,
            WeightDecayPolicy.Exclude);
    }

    private sealed class DuplicateParameterModule : Module
    {
        internal DuplicateParameterModule(Parameter parameter)
        {
            RegisterParameter(parameter);
            RegisterParameter(parameter);
        }
    }

    private sealed class DuplicateChildModule : Module
    {
        internal DuplicateChildModule(Module child)
        {
            RegisterModule(child);
            RegisterModule(child);
        }
    }

    private sealed class ParameterModule : Module
    {
        internal ParameterModule(Parameter parameter)
        {
            RegisterParameter(parameter);
        }
    }

    private sealed class BranchModule : Module
    {
        internal BranchModule(Module child)
        {
            RegisterModule(child);
        }
    }

    private sealed class TwoBranchModule : Module
    {
        internal TwoBranchModule(Module first, Module second)
        {
            RegisterModule(first);
            RegisterModule(second);
        }
    }
}
