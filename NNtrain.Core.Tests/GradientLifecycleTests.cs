using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class GradientLifecycleTests
{
    [Fact]
    public void LeafGradientsAccumulateAcrossIndependentGraphs()
    {
        var input = Tensor.From1D([1f, 2f]);

        (input * Tensor.Scalar(2f)).Sum().Backward();
        AssertClose([2f, 2f], input.Grad);

        (input * Tensor.Scalar(3f)).Sum().Backward();
        AssertClose([5f, 5f], input.Grad);
    }

    [Fact]
    public void TensorZeroGradClearsOnlyThatTensor()
    {
        var left = Tensor.Scalar(2f);
        var right = Tensor.Scalar(3f);
        (left * right).Backward();

        AssertClose([3f], left.Grad);
        AssertClose([2f], right.Grad);

        left.ZeroGrad();

        AssertClose([0f], left.Grad);
        AssertClose([2f], right.Grad);
    }

    [Fact]
    public void IntermediateGradientsAreClearedForEachBackwardTraversal()
    {
        var input = Tensor.Scalar(2f);
        var intermediate = input * Tensor.Scalar(2f);

        (intermediate * Tensor.Scalar(3f)).Backward();
        AssertClose([3f], intermediate.Grad);
        AssertClose([6f], input.Grad);

        input.ZeroGrad();
        (intermediate * Tensor.Scalar(4f)).Backward();

        AssertClose([4f], intermediate.Grad);
        AssertClose([8f], input.Grad);
    }

    [Fact]
    public void BackwardOnLeafAccumulatesExplicitSeeds()
    {
        var leaf = Tensor.From1D([1f, 2f]);

        leaf.Backward([1f, 2f]);
        leaf.Backward([3f, 4f]);

        AssertClose([4f, 6f], leaf.Grad);
    }

    [Fact]
    public void InvalidSeedDoesNotMutateExistingGradients()
    {
        var input = Tensor.From1D([1f, 2f]);
        var output = input * Tensor.Scalar(2f);
        input.MutableGrad[0] = 5f;
        input.MutableGrad[1] = 6f;
        output.MutableGrad[0] = 7f;
        output.MutableGrad[1] = 8f;

        Assert.Throws<ArgumentException>(() => output.Backward([1f]));

        AssertClose([5f, 6f], input.Grad);
        AssertClose([7f, 8f], output.Grad);
    }

    [Fact]
    public void OptimizerZeroGradClearsParameterLeafGradients()
    {
        var first = new Parameter(
            [1f],
            [1],
            "first",
            WeightDecayPolicy.Exclude);
        var second = new Parameter(
            [2f],
            [1],
            "second",
            WeightDecayPolicy.Exclude);
        first.T.MutableGrad[0] = 3f;
        second.T.MutableGrad[0] = 4f;
        var optimizer = new AdamW([first, second]);

        optimizer.ZeroGrad();

        AssertClose([0f], first.T.Grad);
        AssertClose([0f], second.T.Grad);
    }

    [Fact]
    public void ModuleZeroGradClearsEveryParameterLeaf()
    {
        var module = new Linear(2, 2, new Random(17));
        module.Forward(Tensor.From1D([1f, -1f])).Sum().Backward();
        Assert.Contains(
            module.Parameters().SelectMany(parameter => parameter.T.Grad),
            gradient => gradient != 0f);

        module.ZeroGrad();

        Assert.All(
            module.Parameters().SelectMany(parameter => parameter.T.Grad),
            gradient => Assert.Equal(0f, gradient));
    }
}
