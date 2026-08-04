using Xunit;
using NNtrain;

public sealed class TensorCharacterizationTests
{
    private const float Tolerance = 1e-5f;

    [Fact]
    public void ArithmeticSupportsSameShapeAndScalarBroadcasting()
    {
        var a = Tensor.From1D([1f, 2f, 4f]);
        var b = Tensor.From1D([2f, 4f, 8f]);
        var scalar = Tensor.Scalar(2f);

        AssertClose([3f, 6f, 12f], (a + b).Data);
        AssertClose([-1f, -2f, -4f], (a - b).Data);
        AssertClose([2f, 8f, 32f], (a * b).Data);
        AssertClose([0.5f, 0.5f, 0.5f], (a / b).Data);
        AssertClose([3f, 4f, 6f], (a + scalar).Data);
        AssertClose([2f, 4f, 8f], (scalar * a).Data);
        AssertClose([-1f, -2f, -4f], (-a).Data);
        AssertClose([1f, 4f, 16f], a.Pow(2f).Data);
    }

    [Fact]
    public void ScalarBroadcastBackwardReducesGradient()
    {
        var x = Tensor.From1D([1f, 2f, 3f]);
        var scale = Tensor.Scalar(2f);

        var loss = (x * scale + scale).Sum();
        loss.Backward();

        AssertClose([2f, 2f, 2f], x.Grad);
        AssertClose([9f], scale.Grad);
    }

    [Fact]
    public void SumMeanAndReshapePreserveExpectedValuesAndGradients()
    {
        var x = Tensor.From1D([1f, 2f, 3f, 4f]);
        var reshaped = x.Reshape(2, 2);

        Assert.Equal([2, 2], reshaped.Shape);
        AssertClose([10f], reshaped.Sum().Data);
        AssertClose([2.5f], reshaped.Mean().Data);

        reshaped.Mean().Backward();
        AssertClose([0.25f, 0.25f, 0.25f, 0.25f], x.Grad);
    }

    [Fact]
    public void SliceAndConcatRoundTripRowsAndColumns()
    {
        var x = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);

        var firstColumn = x.Slice(1, 0, 1);
        var remainingColumns = x.Slice(1, 1, 2);
        var columns = Tensor.Concat(1, firstColumn, remainingColumns);
        var rows = Tensor.Concat(0, x.Slice(0, 0, 1), x.Slice(0, 1, 1));

        Assert.Equal([2, 3], columns.Shape);
        AssertClose(x.Data, columns.Data);
        AssertClose(x.Data, rows.Data);

        columns.Sum().Backward();
        AssertClose([1f, 1f, 1f, 1f, 1f, 1f], x.Grad);
    }

    [Fact]
    public void TransposeMapsValuesAndGradients()
    {
        var x = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);
        var y = x.Transpose();

        Assert.Equal([3, 2], y.Shape);
        AssertClose([1f, 4f, 2f, 5f, 3f, 6f], y.Data);

        y.Backward([1f, 2f, 3f, 4f, 5f, 6f]);
        AssertClose([1f, 3f, 5f, 2f, 4f, 6f], x.Grad);
    }

    [Fact]
    public void MatMulSupportsAllDocumentedRankCombinations()
    {
        var v1 = Tensor.From1D([1f, 2f, 3f]);
        var v2 = Tensor.From1D([4f, 5f, 6f]);
        AssertClose([32f], v1.MatMul(v2).Data);

        var matrix = new Tensor([1f, 2f, 3f, 4f, 5f, 6f], [2, 3]);
        AssertClose([14f, 32f], matrix.MatMul(v1).Data);

        var right = new Tensor([7f, 8f, 9f, 10f, 11f, 12f], [3, 2]);
        var product = matrix.MatMul(right);
        Assert.Equal([2, 2], product.Shape);
        AssertClose([58f, 64f, 139f, 154f], product.Data);
    }

    [Fact]
    public void MatMulBackwardMatchesKnownGradient()
    {
        var left = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var right = new Tensor([5f, 6f, 7f, 8f], [2, 2]);

        left.MatMul(right).Sum().Backward();

        AssertClose([11f, 15f, 11f, 15f], left.Grad);
        AssertClose([4f, 4f, 6f, 6f], right.Grad);
    }

    [Fact]
    public void MatMulTransposedRightMatchesExplicitTransposeAndGradients()
    {
        var optimizedLeft = Tensor.From2D(new float[,]
        {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f },
        });
        var optimizedRight = Tensor.From2D(new float[,]
        {
            { 7f, 8f, 9f },
            { 10f, 11f, 12f },
        });
        var referenceLeft = Tensor.From2D(new float[,]
        {
            { 1f, 2f, 3f },
            { 4f, 5f, 6f },
        });
        var referenceRight = Tensor.From2D(new float[,]
        {
            { 7f, 8f, 9f },
            { 10f, 11f, 12f },
        });

        Tensor optimized =
            optimizedLeft.MatMulTransposedRight(optimizedRight);
        Tensor reference =
            referenceLeft.MatMul(referenceRight.Transpose());
        optimized.Sum().Backward();
        reference.Sum().Backward();

        AssertClose(reference.Data, optimized.Data);
        AssertClose(referenceLeft.Grad, optimizedLeft.Grad);
        AssertClose(referenceRight.Grad, optimizedRight.Grad);
    }

    [Fact]
    public void FusedTransposedMatMulAndBiasMatchesSeparateOperations()
    {
        var optimizedInput = Tensor.From2D(new float[,]
        {
            { 1f, 2f },
            { 3f, 4f },
        });
        var optimizedWeight = Tensor.From2D(new float[,]
        {
            { 5f, 6f },
            { 7f, 8f },
            { 9f, 10f },
        });
        Tensor optimizedBias = Tensor.From1D([0.5f, 1f, 1.5f]);
        var referenceInput = Tensor.From2D(new float[,]
        {
            { 1f, 2f },
            { 3f, 4f },
        });
        var referenceWeight = Tensor.From2D(new float[,]
        {
            { 5f, 6f },
            { 7f, 8f },
            { 9f, 10f },
        });
        Tensor referenceBias = Tensor.From1D([0.5f, 1f, 1.5f]);

        Tensor optimized = optimizedInput.MatMulTransposedRightAddRow(
            optimizedWeight,
            optimizedBias);
        Tensor reference = referenceInput
            .MatMul(referenceWeight.Transpose())
            .AddRowWise(referenceBias);
        optimized.Sum().Backward();
        reference.Sum().Backward();

        AssertClose(reference.Data, optimized.Data);
        AssertClose(referenceInput.Grad, optimizedInput.Grad);
        AssertClose(referenceWeight.Grad, optimizedWeight.Grad);
        AssertClose(referenceBias.Grad, optimizedBias.Grad);
    }

    [Fact]
    public void ReshapeSharesStorageWithoutChangingGradientSemantics()
    {
        var parameter = new Parameter(
            [1f, 2f, 3f, 4f],
            [4],
            "value",
            WeightDecayPolicy.Exclude);
        Tensor reshaped = parameter.T.Reshape(2, 2);

        using (Tensor.DataMutation mutation = parameter.BeginUpdate())
            mutation.Values[2] = 9f;

        Assert.Equal(9f, reshaped.Data[2]);
        Assert.Throws<InvalidOperationException>(
            () => reshaped.Sum().Backward());
    }

    [Fact]
    public void ActivationsAndRowWiseAdditionHaveExpectedBehavior()
    {
        var x = new Tensor([-2f, 0f, 3f, -1f], [2, 2]);
        AssertClose([0f, 0f, 3f, 0f], x.Relu().Data);

        var matrix = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var bias = Tensor.From1D([10f, 20f]);
        var result = matrix.AddRowWise(bias);
        AssertClose([11f, 22f, 13f, 24f], result.Data);

        result.Sum().Backward();
        AssertClose([1f, 1f, 1f, 1f], matrix.Grad);
        AssertClose([2f, 2f], bias.Grad);
    }

    [Fact]
    public void SoftmaxAndLogSoftmaxAreStableAndNormalized()
    {
        var x = new Tensor([1000f, 1001f, 1002f, -1000f, -1000f, -1000f], [2, 3]);
        var softmax = x.SoftmaxLastDim();
        var logSoftmax = x.LogSoftmaxLastDim();

        AssertClose(1f, softmax.Data.Take(3).Sum());
        AssertClose(1f, softmax.Data.Skip(3).Sum());
        for (var i = 0; i < softmax.Numel; i++)
            AssertClose(softmax.Data[i], MathF.Exp(logSoftmax.Data[i]));
        Assert.All(softmax.Data, value => Assert.True(float.IsFinite(value)));
        Assert.All(logSoftmax.Data, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void LogSoftmaxPreservesProbabilitiesWhenLargeLogitsAreEqual()
    {
        Tensor result = Tensor.From1D([1e20f, 1e20f])
            .LogSoftmaxLastDim();

        float expected = -MathF.Log(2f);
        AssertClose([expected, expected], result.Data, 1e-6f);
    }

    [Fact]
    public void SoftmaxBackwardMatchesFiniteDifferences()
    {
        float[] values = [0.2f, -0.4f, 1.1f];
        float[] weights = [0.3f, -0.7f, 1.2f];

        var x = Tensor.From1D(values);
        var loss = (x.SoftmaxLastDim() * Tensor.From1D(weights)).Sum();
        loss.Backward();

        AssertGradient(
            values,
            x.Grad,
            data => Tensor.From1D(data).SoftmaxLastDim().Data
                .Zip(weights, (value, weight) => value * weight).Sum());
    }

    [Fact]
    public void LogSoftmaxBackwardMatchesFiniteDifferences()
    {
        float[] values = [0.2f, -0.4f, 1.1f];
        float[] weights = [0.3f, -0.7f, 1.2f];

        var x = Tensor.From1D(values);
        var loss = (x.LogSoftmaxLastDim() * Tensor.From1D(weights)).Sum();
        loss.Backward();

        AssertGradient(
            values,
            x.Grad,
            data => Tensor.From1D(data).LogSoftmaxLastDim().Data
                .Zip(weights, (value, weight) => value * weight).Sum());
    }

    [Fact]
    public void LayerNormNormalizesRowsAndPropagatesParameterGradients()
    {
        var x = new Tensor([1f, 2f, 3f, 4f, 6f, 8f], [2, 3]);
        var gamma = Tensor.From1D([1f, 1f, 1f]);
        var beta = Tensor.From1D([0f, 0f, 0f]);
        var y = x.LayerNormLastDim(gamma, beta);

        foreach (var row in y.Data.Chunk(3))
        {
            AssertClose(0f, row.Average(), 2e-5f);
            AssertClose(1f, row.Select(value => value * value).Average(), 2e-5f);
        }

        y.Backward([1f, 2f, 3f, -1f, 0.5f, 2f]);
        AssertClose([0f, 2.5f, 5f], beta.Grad);
        Assert.All(gamma.Grad, value => Assert.True(float.IsFinite(value)));
        Assert.All(x.Grad, value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void LayerNormInputGradientMatchesFiniteDifferences()
    {
        float[] values = [0.5f, -1f, 2f];
        float[] weights = [0.2f, -0.3f, 0.7f];
        var x = Tensor.From1D(values);
        var gamma = Tensor.From1D([1.2f, 0.8f, -0.5f]);
        var beta = Tensor.From1D([0.1f, -0.2f, 0.3f]);

        (x.LayerNormLastDim(gamma, beta) * Tensor.From1D(weights)).Sum().Backward();

        AssertGradient(
            values,
            x.Grad,
            data =>
            {
                var output = Tensor.From1D(data).LayerNormLastDim(
                    Tensor.From1D([1.2f, 0.8f, -0.5f]),
                    Tensor.From1D([0.1f, -0.2f, 0.3f]));
                return output.Data.Zip(weights, (value, weight) => value * weight).Sum();
            },
            tolerance: 2e-3f);
    }

    [Fact]
    public void CausalMaskBlocksFutureValuesAndGradients()
    {
        var x = new Tensor([1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f], [3, 3]);
        var masked = x.CausalMask(-99f);

        AssertClose([1f, -99f, -99f, 4f, 5f, -99f, 7f, 8f, 9f], masked.Data);
        masked.Sum().Backward();
        AssertClose([1f, 0f, 0f, 1f, 1f, 0f, 1f, 1f, 1f], x.Grad);
    }

    [Fact]
    public void BackwardRequiresSeedForNonScalarOutput()
    {
        var output = Tensor.From1D([1f, 2f]);
        Assert.Throws<InvalidOperationException>(() => output.Backward());
    }

    private static void AssertGradient(
        float[] values,
        IEnumerable<float> analytical,
        Func<float[], float> function,
        float epsilon = 1e-3f,
        float tolerance = 5e-4f)
    {
        var numerical = new float[values.Length];
        for (var i = 0; i < values.Length; i++)
        {
            var plus = (float[])values.Clone();
            var minus = (float[])values.Clone();
            plus[i] += epsilon;
            minus[i] -= epsilon;
            numerical[i] = (function(plus) - function(minus)) / (2f * epsilon);
        }

        AssertClose(numerical, analytical, tolerance);
    }

    internal static void AssertClose(
        IEnumerable<float> expected,
        IEnumerable<float> actual,
        float tolerance = Tolerance)
    {
        var expectedArray = expected.ToArray();
        var actualArray = actual.ToArray();
        Assert.Equal(expectedArray.Length, actualArray.Length);

        for (var i = 0; i < expectedArray.Length; i++)
            AssertClose(expectedArray[i], actualArray[i], tolerance);
    }

    internal static void AssertClose(float expected, float actual, float tolerance = Tolerance)
    {
        var difference = MathF.Abs(expected - actual);
        Assert.True(
            difference <= tolerance,
            $"Expected {expected}, actual {actual}, difference {difference}, tolerance {tolerance}.");
    }
}
