using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class BroadcastGradientTests
{
    [Fact]
    public void SameShapeGradientsMatchEveryBinaryOperation()
    {
        AssertBinaryGradients(
            static (left, right) => left + right,
            leftData: [2f, 4f],
            rightData: [1f, 2f],
            seed: [3f, -1f],
            expectedLeft: [3f, -1f],
            expectedRight: [3f, -1f]);

        AssertBinaryGradients(
            static (left, right) => left - right,
            leftData: [2f, 4f],
            rightData: [1f, 2f],
            seed: [3f, -1f],
            expectedLeft: [3f, -1f],
            expectedRight: [-3f, 1f]);

        AssertBinaryGradients(
            static (left, right) => left * right,
            leftData: [2f, 4f],
            rightData: [1f, 2f],
            seed: [3f, -1f],
            expectedLeft: [3f, -2f],
            expectedRight: [6f, -4f]);

        AssertBinaryGradients(
            static (left, right) => left / right,
            leftData: [2f, 4f],
            rightData: [1f, 2f],
            seed: [3f, -1f],
            expectedLeft: [3f, -0.5f],
            expectedRight: [-6f, 1f]);
    }

    [Fact]
    public void LeftScalarGradientReducesEveryOutputContribution()
    {
        AssertBinaryGradients(
            static (left, right) => left + right,
            leftData: [2f],
            rightData: [1f, 2f, 4f],
            seed: [1f, 1f, 1f],
            expectedLeft: [3f],
            expectedRight: [1f, 1f, 1f]);

        AssertBinaryGradients(
            static (left, right) => left - right,
            leftData: [2f],
            rightData: [1f, 2f, 4f],
            seed: [1f, 1f, 1f],
            expectedLeft: [3f],
            expectedRight: [-1f, -1f, -1f]);

        AssertBinaryGradients(
            static (left, right) => left * right,
            leftData: [2f],
            rightData: [1f, 2f, 4f],
            seed: [1f, 1f, 1f],
            expectedLeft: [7f],
            expectedRight: [2f, 2f, 2f]);

        AssertBinaryGradients(
            static (left, right) => left / right,
            leftData: [2f],
            rightData: [1f, 2f, 4f],
            seed: [1f, 1f, 1f],
            expectedLeft: [1.75f],
            expectedRight: [-2f, -0.5f, -0.125f]);
    }

    [Fact]
    public void RightScalarGradientReducesEveryOutputContribution()
    {
        AssertBinaryGradients(
            static (left, right) => left + right,
            leftData: [1f, 2f, 4f],
            rightData: [2f],
            seed: [1f, 1f, 1f],
            expectedLeft: [1f, 1f, 1f],
            expectedRight: [3f]);

        AssertBinaryGradients(
            static (left, right) => left - right,
            leftData: [1f, 2f, 4f],
            rightData: [2f],
            seed: [1f, 1f, 1f],
            expectedLeft: [1f, 1f, 1f],
            expectedRight: [-3f]);

        AssertBinaryGradients(
            static (left, right) => left * right,
            leftData: [1f, 2f, 4f],
            rightData: [2f],
            seed: [1f, 1f, 1f],
            expectedLeft: [2f, 2f, 2f],
            expectedRight: [7f]);

        AssertBinaryGradients(
            static (left, right) => left / right,
            leftData: [1f, 2f, 4f],
            rightData: [2f],
            seed: [1f, 1f, 1f],
            expectedLeft: [0.5f, 0.5f, 0.5f],
            expectedRight: [-1.75f]);
    }

    [Fact]
    public void SameScalarOperandReceivesBothLocalDerivativeContributions()
    {
        var value = Tensor.Scalar(2f);
        var output = value / value;

        output.Backward();

        AssertClose([0f], value.Grad);
    }

    [Fact]
    public void RankTwoSingleElementTensorReducesLikeScalar()
    {
        var scalarLike = new Tensor([2f], [1, 1]);
        var vector = Tensor.From1D([1f, 2f, 3f]);
        var output = scalarLike * vector;

        output.Backward([1f, 2f, 3f]);

        Assert.Equal<int>([3], output.Shape);
        AssertClose([14f], scalarLike.Grad);
        AssertClose([2f, 4f, 6f], vector.Grad);
    }

    private static void AssertBinaryGradients(
        Func<Tensor, Tensor, Tensor> operation,
        float[] leftData,
        float[] rightData,
        float[] seed,
        float[] expectedLeft,
        float[] expectedRight)
    {
        var left = Tensor.From1D(leftData);
        var right = Tensor.From1D(rightData);

        operation(left, right).Backward(seed);

        AssertClose(expectedLeft, left.Grad);
        AssertClose(expectedRight, right.Grad);
    }
}
