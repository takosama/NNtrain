using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class RepeatedBackwardTests
{
    [Fact]
    public void SameGraphCanRunBackwardRepeatedly()
    {
        var input = Tensor.Scalar(3f);
        var intermediate = input * input;
        var loss = intermediate.Sum();

        loss.Backward();
        AssertClose([6f], input.Grad);
        AssertClose([1f], intermediate.Grad);

        loss.Backward();

        AssertClose([12f], input.Grad);
        AssertClose([1f], intermediate.Grad);
        AssertClose([1f], loss.Grad);
    }

    [Fact]
    public void RepeatedBackwardCanUseDifferentSeeds()
    {
        var input = Tensor.From1D([1f, 2f]);
        var output = input * input;

        output.Backward([1f, 2f]);
        AssertClose([2f, 8f], input.Grad);

        output.Backward([3f, -1f]);

        AssertClose([8f, 4f], input.Grad);
        AssertClose([3f, -1f], output.Grad);
    }

    [Fact]
    public void ZeroGradStartsANewAccumulationWindowWithoutInvalidatingGraph()
    {
        var input = Tensor.Scalar(4f);
        var loss = input.Pow(2f).Sum();

        loss.Backward();
        AssertClose([8f], input.Grad);

        input.ZeroGrad();
        loss.Backward();

        AssertClose([8f], input.Grad);
    }

    [Fact]
    public void OptimizerStepInvalidatesPreviouslyBuiltGraph()
    {
        var parameter = new Parameter(
            [2f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        var loss = parameter.T.Pow(2f).Sum();
        loss.Backward();
        AssertClose([4f], parameter.T.Grad);
        AssertClose([1f], loss.Grad);

        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0f,
            });
        optimizer.Step();
        var exception = Assert.Throws<InvalidOperationException>(
            () => loss.Backward());

        Assert.Contains("'weight'", exception.Message);
        Assert.Contains("new forward graph", exception.Message);
        AssertClose([4f], parameter.T.Grad);
        AssertClose([1f], loss.Grad);
    }

    [Fact]
    public void FreshForwardGraphWorksAfterParameterUpdate()
    {
        var parameter = new Parameter(
            [2f],
            [1],
            "weight",
            WeightDecayPolicy.Exclude);
        var staleLoss = parameter.T.Pow(2f).Sum();
        staleLoss.Backward();
        var optimizer = new AdamW(
            [parameter],
            new AdamWOptions
            {
                LearningRate = 0.1f,
                WeightDecay = 0f,
            });
        optimizer.Step();
        parameter.ZeroGrad();

        var freshLoss = parameter.T.Pow(2f).Sum();
        freshLoss.Backward();

        AssertClose([1.9f], parameter.T.Data, 2e-5f);
        AssertClose([3.8f], parameter.T.Grad, 2e-5f);
        Assert.Throws<InvalidOperationException>(() => staleLoss.Backward());
    }
}
