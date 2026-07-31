using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class ParameterMetadataTests
{
    [Fact]
    public void LinearParametersExposeNameOwnerAndDecayPolicy()
    {
        var linear = new Linear(2, 3, new Random(41));

        Assert.Equal("W", linear.W.Name);
        Assert.Equal("W", linear.W.T.Name);
        Assert.Same(linear, linear.W.Owner);
        Assert.Equal(WeightDecayPolicy.Apply, linear.W.WeightDecay);

        Assert.Equal("B", linear.B.Name);
        Assert.Equal("B", linear.B.T.Name);
        Assert.Same(linear, linear.B.Owner);
        Assert.Equal(WeightDecayPolicy.Exclude, linear.B.WeightDecay);
    }

    [Fact]
    public void CompositeParametersAreOwnedByTheirDirectModule()
    {
        var model = new TransformerClassifier(
            seqLen: 2,
            dModel: 4,
            numHeads: 2,
            dHidden: 8,
            numLayers: 1,
            numClasses: 3,
            rng: new Random(43));

        Linear queryProjection = model.Blocks[0].Attn.Heads[0].Wq;

        Assert.Same(model, model.Pos.Owner);
        Assert.Equal(WeightDecayPolicy.Apply, model.Pos.WeightDecay);
        Assert.Same(queryProjection, queryProjection.W.Owner);
        Assert.NotSame(model, queryProjection.W.Owner);
        Assert.All(model.Parameters(), parameter => Assert.NotNull(parameter.Owner));
        Assert.All(
            model.Parameters(),
            parameter => Assert.Equal(parameter.Name, parameter.T.Name));
    }

    [Fact]
    public void LayerNormExplicitlyExcludesBothParametersFromDecay()
    {
        var layerNorm = new LayerNorm(3);

        Assert.Same(layerNorm, layerNorm.Gamma.Owner);
        Assert.Same(layerNorm, layerNorm.Beta.Owner);
        Assert.Equal(WeightDecayPolicy.Exclude, layerNorm.Gamma.WeightDecay);
        Assert.Equal(WeightDecayPolicy.Exclude, layerNorm.Beta.WeightDecay);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void ParameterRejectsMissingName(string? name)
    {
        Assert.Throws<ArgumentException>(
            () => new Parameter(
                [1f],
                [1],
                name!,
                WeightDecayPolicy.Exclude));
    }

    [Fact]
    public void StandaloneParameterHasExplicitMetadataWithoutOwner()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "standalone",
            WeightDecayPolicy.Exclude);

        Assert.Equal("standalone", parameter.Name);
        Assert.Null(parameter.Owner);
        Assert.Equal(WeightDecayPolicy.Exclude, parameter.WeightDecay);
    }

    [Fact]
    public void ParameterCannotBeOwnedByTwoDifferentModules()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "shared",
            WeightDecayPolicy.Exclude);
        var firstOwner = new ParameterOwnerModule(parameter);

        var exception = Assert.Throws<InvalidOperationException>(
            () => new ParameterOwnerModule(parameter));

        Assert.Same(firstOwner, parameter.Owner);
        Assert.Contains("already owned", exception.Message);
    }

    [Fact]
    public void AdamWUsesExplicitPolicyInsteadOfTensorRank()
    {
        var decayedVector = new Parameter(
            [1f],
            [1],
            "decayedVector",
            WeightDecayPolicy.Apply);
        var excludedMatrix = new Parameter(
            [1f],
            [1, 1],
            "excludedMatrix",
            WeightDecayPolicy.Exclude);
        var optimizer = new AdamW(
            [decayedVector, excludedMatrix],
            new AdamWOptions
            {
                LearningRate = 0.01f,
                WeightDecay = 0.1f,
                Decay1D = false,
            });

        optimizer.Step();

        AssertClose([0.999f], decayedVector.T.Data, 2e-6f);
        AssertClose([1f], excludedMatrix.T.Data);
    }

    private sealed class ParameterOwnerModule : Module
    {
        internal ParameterOwnerModule(Parameter parameter)
        {
            Registered = RegisterParameter(parameter);
        }

        internal Parameter Registered { get; }
    }
}
