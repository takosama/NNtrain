using Xunit;
using NNtrain;
using static TensorCharacterizationTests;

public sealed class ModuleCharacterizationTests
{
    [Fact]
    public void LinearForwardAndBackwardAreDeterministic()
    {
        var linear = new Linear(2, 2, new Random(123), initScale: 0.1f);
        var input = Tensor.From1D([0.5f, -1f]);

        var output = linear.Forward(input);
        AssertClose([-0.03310738f, -0.03797377f], output.Data, 1e-5f);

        output.Sum().Backward();
        AssertClose([0.5f, -1f, 0.5f, -1f], linear.W.T.Grad);
        AssertClose([1f, 1f], linear.B.T.Grad);
    }

    [Fact]
    public void AttentionAndTransformerPreserveDocumentedShapes()
    {
        var input = new Tensor(
            [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f],
            [2, 4]);

        var attention = new MultiHeadAttention(4, 2, causal: true, new Random(7));
        var attentionOutput = attention.Forward(input);
        Assert.Equal([2, 4], attentionOutput.Shape);
        Assert.All(attentionOutput.Data, value => Assert.True(float.IsFinite(value)));

        var block = new TransformerBlock(4, 2, 8, rng: new Random(7));
        var blockOutput = block.Forward(input);
        Assert.Equal([2, 4], blockOutput.Shape);

        var classifier = new TransformerClassifier(2, 4, 2, 8, 2, 3, new Random(7));
        var logits = classifier.Forward(input);
        Assert.Equal([3], logits.Shape);
        Assert.All(logits.Data, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void TransformerBackwardProducesFiniteGradientsForEveryParameter()
    {
        var model = new TransformerClassifier(2, 4, 2, 8, 1, 3, new Random(11));
        var input = new Tensor(
            [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f],
            [2, 4]);

        model.Forward(input).Sum().Backward();

        var parameters = model.Parameters().ToArray();
        Assert.NotEmpty(parameters);
        Assert.All(parameters, parameter =>
            Assert.All(parameter.T.Grad, value => Assert.True(float.IsFinite(value))));
        Assert.Contains(parameters.SelectMany(parameter => parameter.T.Grad), value => value != 0f);
    }

    [Fact]
    public void AdamWOneStepMatchesCurrentUpdateRule()
    {
        var parameter = new Parameter(
            [1f, -2f],
            [2],
            "p",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 0.5f;
        parameter.T.MutableGrad[1] = -0.25f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.01f,
                Beta1 = 0.9f,
                Beta2 = 0.999f,
                Epsilon = 1e-8f,
                WeightDecay = 0.1f,
                Decay1D = true,
            });

        optimizer.Step();

        AssertClose([0.989f, -1.988f], parameter.T.Data, 2e-6f);
    }

    [Fact]
    public void AdamWDoesNotDecayOneDimensionalParametersByDefault()
    {
        var parameter = new Parameter(
            [1f],
            [1],
            "p",
            WeightDecayPolicy.Exclude);
        parameter.T.MutableGrad[0] = 0f;
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.01f,
                WeightDecay = 0.1f,
            });

        optimizer.Step();

        AssertClose([1f], parameter.T.Data);
    }
}
