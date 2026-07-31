using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class SharedTensorGradientTests
{
    [Fact]
    public void SameLeafUsedAsBothOperandsReceivesBothEdgeContributions()
    {
        var input = Tensor.Scalar(3f);
        var output = input * input + input;

        output.Backward();

        AssertClose([7f], input.Grad);
    }

    [Fact]
    public void DuplicateParentEdgesArePreservedInTheGraph()
    {
        var input = Tensor.Scalar(2f);
        var output = input + input;

        Assert.Collection(
            output.Node.Parents,
            parent => Assert.Same(input, parent),
            parent => Assert.Same(input, parent));
    }

    [Fact]
    public void DiamondGraphSumsBranchesBeforeRunningSharedIntermediate()
    {
        var input = Tensor.Scalar(2f);
        var shared = input * Tensor.Scalar(2f);
        var leftBranch = shared * Tensor.Scalar(3f);
        var rightBranch = shared.Pow(2f);
        var output = leftBranch + rightBranch;

        output.Backward();

        AssertClose([11f], shared.Grad);
        AssertClose([22f], input.Grad);
    }

    [Fact]
    public void SharedVectorIntermediateAccumulatesSeededBranchContributions()
    {
        var input = Tensor.From1D([1f, 2f]);
        var shared = input * input;
        var output = shared + shared;

        output.Backward([2f, -1f]);

        AssertClose([4f, -2f], shared.Grad);
        AssertClose([8f, -8f], input.Grad);
    }

    [Fact]
    public void SharedModuleParametersReceiveContributionsFromEveryCall()
    {
        var linear = new Linear(1, 1, new Random(23));
        var firstOutput = linear.Forward(Tensor.From1D([2f]));
        var secondOutput = linear.Forward(Tensor.From1D([3f]));
        var loss = firstOutput.Sum() + secondOutput.Sum();

        loss.Backward();

        AssertClose([5f], linear.W.T.Grad);
        AssertClose([2f], linear.B.T.Grad);
    }

    [Fact]
    public void EqualValuedDistinctLeavesRemainDistinctGraphNodes()
    {
        var left = Tensor.Scalar(2f);
        var right = Tensor.Scalar(2f);

        (left * right).Backward();

        Assert.NotSame(left, right);
        AssertClose([2f], left.Grad);
        AssertClose([2f], right.Grad);
    }
}
