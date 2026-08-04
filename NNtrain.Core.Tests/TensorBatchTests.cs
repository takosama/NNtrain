using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class TensorBatchTests
{
    [Fact]
    public void RankThreeSliceAndConcatPreserveValuesAndGradients()
    {
        var input = new Tensor(
            Enumerable.Range(1, 16).Select(value => (float)value).ToArray(),
            [2, 2, 4]);

        Tensor left = input.Slice(2, 0, 2);
        Tensor right = input.Slice(2, 2, 2);
        Tensor combined = Tensor.Concat(2, left, right);

        Assert.Equal<int>([2, 2, 4], combined.Shape);
        AssertClose(input.Data, combined.Data);

        combined.Sum().Backward();
        AssertClose(Enumerable.Repeat(1f, 16), input.Grad);
    }

    [Fact]
    public void BatchWiseAdditionReducesMatrixGradientAcrossTheBatch()
    {
        var input = new Tensor(new float[24], [3, 2, 4]);
        var matrix = new Tensor(
            Enumerable.Range(1, 8).Select(value => (float)value).ToArray(),
            [2, 4]);

        Tensor result = input.AddBatchWise(matrix);
        result.Sum().Backward();

        AssertClose(
            matrix.Data.Concat(matrix.Data).Concat(matrix.Data),
            result.Data);
        AssertClose(Enumerable.Repeat(1f, 24), input.Grad);
        AssertClose(Enumerable.Repeat(3f, 8), matrix.Grad);
    }

    [Fact]
    public void BatchedMatMulMatchesIndependentMatMulAndGradients()
    {
        float[] leftValues =
        [
            1f, 2f, 3f,
            4f, 5f, 6f,
            -1f, 0.5f, 2f,
            3f, -2f, 1f,
        ];
        float[] rightValues =
        [
            1f, 2f,
            3f, 4f,
            5f, 6f,
            -1f, 2f,
            0.5f, 3f,
            4f, -2f,
        ];
        var batchedLeft = new Tensor(leftValues, [2, 2, 3]);
        var batchedRight = new Tensor(rightValues, [2, 3, 2]);

        Tensor batched = batchedLeft.BatchedMatMul(batchedRight);
        batched.Sum().Backward();

        var expectedData = new List<float>();
        var expectedLeftGradient = new List<float>();
        var expectedRightGradient = new List<float>();
        for (int batch = 0; batch < 2; batch++)
        {
            var left = new Tensor(
                leftValues.Skip(batch * 6).Take(6).ToArray(),
                [2, 3]);
            var right = new Tensor(
                rightValues.Skip(batch * 6).Take(6).ToArray(),
                [3, 2]);
            Tensor result = left.MatMul(right);
            result.Sum().Backward();
            expectedData.AddRange(result.Data);
            expectedLeftGradient.AddRange(left.Grad);
            expectedRightGradient.AddRange(right.Grad);
        }

        Assert.Equal<int>([2, 2, 2], batched.Shape);
        AssertClose(expectedData, batched.Data);
        AssertClose(expectedLeftGradient, batchedLeft.Grad);
        AssertClose(expectedRightGradient, batchedRight.Grad);
    }

    [Fact]
    public void BatchedTransposedMatMulMatchesIndependentOperations()
    {
        float[] leftValues = Enumerable.Range(1, 12)
            .Select(value => value * 0.1f)
            .ToArray();
        float[] rightValues = Enumerable.Range(1, 18)
            .Select(value => (value - 9) * 0.05f)
            .ToArray();
        var batchedLeft = new Tensor(leftValues, [2, 2, 3]);
        var batchedRight = new Tensor(rightValues, [2, 3, 3]);

        Tensor batched =
            batchedLeft.BatchedMatMulTransposedRight(batchedRight);
        batched.Sum().Backward();

        var expectedData = new List<float>();
        var expectedLeftGradient = new List<float>();
        var expectedRightGradient = new List<float>();
        for (int batch = 0; batch < 2; batch++)
        {
            var left = new Tensor(
                leftValues.Skip(batch * 6).Take(6).ToArray(),
                [2, 3]);
            var right = new Tensor(
                rightValues.Skip(batch * 9).Take(9).ToArray(),
                [3, 3]);
            Tensor result = left.MatMulTransposedRight(right);
            result.Sum().Backward();
            expectedData.AddRange(result.Data);
            expectedLeftGradient.AddRange(left.Grad);
            expectedRightGradient.AddRange(right.Grad);
        }

        Assert.Equal<int>([2, 2, 3], batched.Shape);
        AssertClose(expectedData, batched.Data);
        AssertClose(expectedLeftGradient, batchedLeft.Grad);
        AssertClose(expectedRightGradient, batchedRight.Grad);
    }

    [Fact]
    public void RankThreeSoftmaxAndLayerNormOperateOnTheLastDimension()
    {
        var input = new Tensor(
            Enumerable.Range(0, 24)
                .Select(index => (index % 4 - 2) * 0.25f)
                .ToArray(),
            [2, 3, 4]);
        Tensor gamma = Tensor.From1D([1f, 1f, 1f, 1f]);
        Tensor beta = Tensor.From1D([0f, 0f, 0f, 0f]);

        Tensor normalized = input.LayerNormLastDim(gamma, beta);
        Tensor probabilities = normalized.SoftmaxLastDim();

        Assert.Equal<int>([2, 3, 4], probabilities.Shape);
        foreach (float[] row in probabilities.Data.Chunk(4))
            AssertClose(1f, row.Sum(), 2e-5f);

        probabilities.Sum().Backward();
        Assert.All(input.Grad, value => Assert.True(float.IsFinite(value)));
    }
}
