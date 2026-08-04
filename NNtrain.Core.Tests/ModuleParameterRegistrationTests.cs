using NNtrain;
using Xunit;

public sealed class ModuleParameterRegistrationTests
{
    [Fact]
    public void ParametersExpandMembersDepthFirstInRegistrationOrder()
    {
        var module = new InterleavedModule();

        Parameter[] parameters = module.Parameters().ToArray();

        Assert.Collection(
            parameters,
            parameter => Assert.Same(module.First, parameter),
            parameter => Assert.Same(module.Child.W, parameter),
            parameter => Assert.Same(module.Child.B, parameter),
            parameter => Assert.Same(module.Last, parameter));
    }

    [Fact]
    public void RepeatedEnumerationIsStable()
    {
        var module = new InterleavedModule();

        Parameter[] first = module.Parameters().ToArray();
        Parameter[] second = module.Parameters().ToArray();

        Assert.Equal(first.Length, second.Length);
        for (int index = 0; index < first.Length; index++)
            Assert.Same(first[index], second[index]);
    }

    [Fact]
    public void TransformerClassifierRetainsExpectedParameterCountAndBoundaries()
    {
        var model = new TransformerClassifier(
            seqLen: 2,
            dModel: 4,
            numHeads: 2,
            dHidden: 8,
            numLayers: 2,
            numClasses: 3,
            rng: new Random(31));

        Parameter[] parameters = model.Parameters().ToArray();

        Assert.Equal(27, parameters.Length);
        Assert.Same(model.Pos, parameters[0]);
        Assert.Same(model.Blocks[0].Attn.Qkv.W, parameters[1]);
        Assert.Same(model.Head.W, parameters[^2]);
        Assert.Same(model.Head.B, parameters[^1]);
        Assert.Equal(
            parameters.Length,
            parameters.Distinct(ReferenceEqualityComparer.Instance).Count());
    }

    [Fact]
    public void ZeroGradUsesTheCommonRecursiveEnumeration()
    {
        var module = new InterleavedModule();
        Parameter[] parameters = module.Parameters().ToArray();

        for (int index = 0; index < parameters.Length; index++)
            parameters[index].T.MutableGrad[0] = index + 1;

        module.ZeroGrad();

        Assert.All(
            parameters.SelectMany(parameter => parameter.T.Grad),
            gradient => Assert.Equal(0f, gradient));
    }

    private sealed class InterleavedModule : Module
    {
        internal InterleavedModule()
        {
            First = RegisterParameter(
                new Parameter(
                    [1f],
                    [1],
                    "First",
                    WeightDecayPolicy.Exclude));
            Child = RegisterModule(new Linear(1, 1, new Random(37)));
            Last = RegisterParameter(
                new Parameter(
                    [2f],
                    [1],
                    "Last",
                    WeightDecayPolicy.Exclude));
        }

        internal Parameter First { get; }
        internal Linear Child { get; }
        internal Parameter Last { get; }
    }
}
