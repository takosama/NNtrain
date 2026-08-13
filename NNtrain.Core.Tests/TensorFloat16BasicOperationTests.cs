using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class TensorFloat16BasicOperationTests
{
    private const float HalfTolerance = 2e-3f;

    [Fact]
    public void EveryBinaryOperationUsesFloat16StorageWithSimdSizedInputs()
    {
        float[] leftValues = Enumerable.Range(1, 16)
            .Select(index => index * 0.25f)
            .ToArray();
        float[] rightValues = Enumerable.Range(1, 16)
            .Select(index => 1f + index * 0.125f)
            .ToArray();

        AssertBinary(
            static (left, right) => left + right,
            leftValues,
            rightValues,
            leftValues.Zip(rightValues, static (left, right) => left + right),
            Enumerable.Repeat(1f, leftValues.Length),
            Enumerable.Repeat(1f, leftValues.Length));
        AssertBinary(
            static (left, right) => left - right,
            leftValues,
            rightValues,
            leftValues.Zip(rightValues, static (left, right) => left - right),
            Enumerable.Repeat(1f, leftValues.Length),
            Enumerable.Repeat(-1f, leftValues.Length));
        AssertBinary(
            static (left, right) => left * right,
            leftValues,
            rightValues,
            leftValues.Zip(rightValues, static (left, right) => left * right),
            rightValues,
            leftValues);
        AssertBinary(
            static (left, right) => left / right,
            leftValues,
            rightValues,
            leftValues.Zip(rightValues, static (left, right) => left / right),
            rightValues.Select(static right => 1f / right),
            leftValues.Zip(
                rightValues,
                static (left, right) => -left / (right * right)));
    }

    [Fact]
    public void BothScalarBroadcastDirectionsReduceFloat16Gradients()
    {
        float[] values = Enumerable.Range(1, 16)
            .Select(index => index * 0.25f)
            .ToArray();
        var vectorForRightScalar = new Tensor(
            values,
            [values.Length],
            dtype: TensorDType.Float16);
        Tensor rightScalar = Tensor.Scalar(
            2f,
            dtype: TensorDType.Float16);

        Tensor rightScalarResult = vectorForRightScalar * rightScalar;
        rightScalarResult.Sum().Backward();

        var vectorForLeftScalar = new Tensor(
            values,
            [values.Length],
            dtype: TensorDType.Float16);
        Tensor leftScalar = Tensor.Scalar(
            2f,
            dtype: TensorDType.Float16);
        Tensor leftScalarResult = leftScalar / vectorForLeftScalar;
        leftScalarResult.Sum().Backward();

        Assert.Equal(TensorDType.Float16, rightScalarResult.DType);
        AssertClose(Enumerable.Repeat(2f, values.Length), vectorForRightScalar.Grad);
        AssertClose([values.Sum()], rightScalar.Grad, HalfTolerance);
        Assert.Equal(TensorDType.Float16, leftScalarResult.DType);
        AssertClose(
            [values.Sum(static value => 1f / value)],
            leftScalar.Grad,
            HalfTolerance);
        AssertClose(
            values.Select(static value => -2f / (value * value)),
            vectorForLeftScalar.Grad,
            HalfTolerance);
    }

    [Fact]
    public void MixedFloat16AndFloat32ArithmeticPromotesStorageToFloat32()
    {
        var half = new Tensor(
            [1f, 2f, 3f, 4f],
            [4],
            dtype: TensorDType.Float16);
        var single = new Tensor(
            [0.5f, 1f, 1.5f, 2f],
            [4],
            dtype: TensorDType.Float32);

        Tensor result = half + single;
        result.Sum().Backward();

        Assert.Equal(TensorDType.Float32, result.DType);
        AssertClose([1.5f, 3f, 4.5f, 6f], result.Data);
        AssertClose(Enumerable.Repeat(1f, 4), half.Grad);
        AssertClose(Enumerable.Repeat(1f, 4), single.Grad);
    }

    [Fact]
    public void ArithmeticAndBroadcastingPreserveFloat16StorageAndGradients()
    {
        float[] values = Enumerable.Range(1, 16)
            .Select(index => index * 0.25f)
            .ToArray();
        var left = new Tensor(
            values,
            [values.Length],
            dtype: TensorDType.Float16);
        var right = new Tensor(
            Enumerable.Repeat(2f, values.Length).ToArray(),
            [values.Length],
            dtype: TensorDType.Float16);
        Tensor scale = Tensor.Scalar(0.5f, dtype: TensorDType.Float16);

        Tensor result = ((left * right) + scale).Pow(2f);
        Tensor negated = -left;
        result.Sum().Backward();

        float[] expected = values
            .Select(value => MathF.Pow(value * 2f + 0.5f, 2f))
            .ToArray();
        float[] expectedLeftGradient = values
            .Select(value => 4f * (value * 2f + 0.5f))
            .ToArray();
        float expectedScaleGradient = values
            .Sum(value => 2f * (value * 2f + 0.5f));

        Assert.Equal(TensorDType.Float16, result.DType);
        Assert.Equal(TensorDType.Float16, negated.DType);
        AssertClose(expected, result.Data, HalfTolerance);
        AssertClose(values.Select(static value => -value), negated.Data);
        AssertClose(expectedLeftGradient, left.Grad, HalfTolerance);
        AssertClose([expectedScaleGradient], scale.Grad, HalfTolerance);
    }

    [Fact]
    public void ReductionsReturnFloat32AndAccumulateInFloat32()
    {
        var input = new Tensor(
            [1f, 2f, 3f, 4f],
            [4],
            dtype: TensorDType.Float16);

        Tensor sum = input.Sum();
        Tensor mean = input.Mean();
        mean.Backward();

        Assert.Equal(TensorDType.Float32, sum.DType);
        Assert.Equal(TensorDType.Float32, mean.DType);
        Assert.Equal(TensorDType.Float32, mean.AccumulationDType);
        AssertClose([10f], sum.Data);
        AssertClose([2.5f], mean.Data, HalfTolerance);
        AssertClose([0.25f, 0.25f, 0.25f, 0.25f], input.Grad);
    }

    [Fact]
    public void ShapeOperationsPreserveFloat16ValuesAndGradientRouting()
    {
        float[] values = Enumerable.Range(1, 16)
            .Select(value => (float)value)
            .ToArray();
        var input = new Tensor(
            values,
            [2, 2, 4],
            dtype: TensorDType.Float16);

        Tensor first = input.Slice(2, 0, 2);
        Tensor second = input.Slice(2, 2, 2);
        Tensor joined = Tensor.Concat(2, first, second);
        Tensor transposed = joined.Reshape(4, 4).Transpose();
        transposed.Sum().Backward();

        Assert.Equal(TensorDType.Float16, first.DType);
        Assert.Equal(TensorDType.Float16, joined.DType);
        Assert.Equal(TensorDType.Float16, transposed.DType);
        AssertClose(values, joined.Data, HalfTolerance);
        AssertClose(Enumerable.Repeat(1f, values.Length), input.Grad);
    }

    [Fact]
    public void SliceAndConcatCoverEveryRankThreeAxisForFloat16()
    {
        float[] values = Enumerable.Range(1, 24)
            .Select(value => value * 0.25f)
            .ToArray();

        for (int dimension = 0; dimension < 3; dimension++)
        {
            var input = new Tensor(
                values,
                [2, 3, 4],
                dtype: TensorDType.Float16);
            int size = input.Shape[dimension];
            Tensor joined = Tensor.Concat(
                dimension,
                input.Slice(dimension, 0, 1),
                input.Slice(dimension, 1, size - 1));

            joined.Sum().Backward();

            Assert.Equal(TensorDType.Float16, joined.DType);
            AssertClose(values, joined.Data, HalfTolerance);
            AssertClose(Enumerable.Repeat(1f, values.Length), input.Grad);
        }
    }

    [Fact]
    public void SliceAndConcatCoverRankOneAndRankTwoFloat16Copies()
    {
        float[] rankOneValues = Enumerable.Range(1, 8)
            .Select(value => value * 0.5f)
            .ToArray();
        var rankOne = new Tensor(
            rankOneValues,
            [8],
            dtype: TensorDType.Float16);
        Tensor rankOneJoined = Tensor.Concat(
            0,
            rankOne.Slice(0, 0, 3),
            rankOne.Slice(0, 3, 5));
        rankOneJoined.Sum().Backward();

        AssertClose(rankOneValues, rankOneJoined.Data, HalfTolerance);
        AssertClose(Enumerable.Repeat(1f, 8), rankOne.Grad);

        float[] rankTwoValues = Enumerable.Range(1, 12)
            .Select(value => value * 0.25f)
            .ToArray();
        for (int dimension = 0; dimension < 2; dimension++)
        {
            var rankTwo = new Tensor(
                rankTwoValues,
                [3, 4],
                dtype: TensorDType.Float16);
            int size = rankTwo.Shape[dimension];
            Tensor joined = Tensor.Concat(
                dimension,
                rankTwo.Slice(dimension, 0, 1),
                rankTwo.Slice(dimension, 1, size - 1));
            joined.Sum().Backward();

            AssertClose(rankTwoValues, joined.Data, HalfTolerance);
            AssertClose(Enumerable.Repeat(1f, 12), rankTwo.Grad);
        }
    }

    [Fact]
    public void BatchWiseAdditionReadsFloat16StorageAndReducesGradient()
    {
        var input = new Tensor(
            new float[24],
            [3, 2, 4],
            dtype: TensorDType.Float16);
        float[] matrixValues = Enumerable.Range(1, 8)
            .Select(value => value * 0.5f)
            .ToArray();
        var matrix = new Tensor(
            matrixValues,
            [2, 4],
            dtype: TensorDType.Float16);

        Tensor result = input.AddBatchWise(matrix);
        result.Sum().Backward();

        Assert.Equal(TensorDType.Float16, result.DType);
        AssertClose(
            matrixValues.Concat(matrixValues).Concat(matrixValues),
            result.Data,
            HalfTolerance);
        AssertClose(Enumerable.Repeat(1f, input.Numel), input.Grad);
        AssertClose(Enumerable.Repeat(3f, matrix.Numel), matrix.Grad);
    }

    private static void AssertBinary(
        Func<Tensor, Tensor, Tensor> operation,
        float[] leftValues,
        float[] rightValues,
        IEnumerable<float> expectedData,
        IEnumerable<float> expectedLeftGradient,
        IEnumerable<float> expectedRightGradient)
    {
        var left = new Tensor(
            leftValues,
            [leftValues.Length],
            dtype: TensorDType.Float16);
        var right = new Tensor(
            rightValues,
            [rightValues.Length],
            dtype: TensorDType.Float16);

        Tensor result = operation(left, right);
        result.Sum().Backward();

        Assert.Equal(TensorDType.Float16, result.DType);
        AssertClose(expectedData, result.Data, HalfTolerance);
        AssertClose(expectedLeftGradient, left.Grad, HalfTolerance);
        AssertClose(expectedRightGradient, right.Grad, HalfTolerance);
    }
}
