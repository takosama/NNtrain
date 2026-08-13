using NNtrain;
using Xunit;

public sealed class LinearLastDimTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RankThreeForwardBackwardMatchesFormerReshapeGraph(bool relu)
    {
        float[] values = Pattern(2 * 3 * 5, 0.037f);
        float[] seed = Pattern(2 * 3 * 20, 0.021f);
        var optimizedLinear = new Linear(5, 20, new Random(101), 0.1f);
        var referenceLinear = new Linear(5, 20, new Random(101), 0.1f);
        var optimizedInput = new Tensor(values, [2, 3, 5]);
        var referenceInput = new Tensor(values, [2, 3, 5]);

        Tensor optimized = relu
            ? optimizedLinear.ForwardBatchRelu(optimizedInput)
            : optimizedLinear.ForwardBatch(optimizedInput);
        Tensor reference = FormerReshapeForward(
            referenceLinear,
            referenceInput,
            relu);
        optimized.Backward(seed);
        reference.Backward(seed);

        Assert.Equal<int>([2, 3, 20], optimized.Shape);
        AssertClose(reference.Data, optimized.Data, 1e-6f);
        AssertClose(referenceInput.Grad, optimizedInput.Grad, 1e-6f);
        AssertClose(
            referenceLinear.W.T.Grad,
            optimizedLinear.W.T.Grad,
            1e-6f);
        AssertClose(
            referenceLinear.B.T.Grad,
            optimizedLinear.B.T.Grad,
            1e-6f);
        Assert.Collection(
            optimized.Node.Parents,
            parent => Assert.Same(optimizedInput, parent),
            parent => Assert.Same(optimizedLinear.W.T, parent),
            parent => Assert.Same(optimizedLinear.B.T, parent));
    }

    [Fact]
    public void RankFourProjectionPreservesEveryLeadingDimension()
    {
        float[] values = Pattern(2 * 2 * 3 * 5, 0.019f);
        float[] seed = Pattern(2 * 2 * 3 * 6, 0.013f);
        var optimizedLinear = new Linear(5, 6, new Random(103), 0.08f);
        var referenceLinear = new Linear(5, 6, new Random(103), 0.08f);
        var optimizedInput = new Tensor(values, [2, 2, 3, 5]);
        var referenceInput = new Tensor(values, [2, 2, 3, 5]);

        Tensor optimized = optimizedLinear.ForwardBatch(optimizedInput);
        Tensor reference = FormerReshapeForward(
            referenceLinear,
            referenceInput,
            relu: false);
        optimized.Backward(seed);
        reference.Backward(seed);

        Assert.Equal<int>([2, 2, 3, 6], optimized.Shape);
        AssertClose(reference.Data, optimized.Data, 1e-6f);
        AssertClose(referenceInput.Grad, optimizedInput.Grad, 1e-6f);
        AssertClose(
            referenceLinear.W.T.Grad,
            optimizedLinear.W.T.Grad,
            1e-6f);
        AssertClose(
            referenceLinear.B.T.Grad,
            optimizedLinear.B.T.Grad,
            1e-6f);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Float16ProjectionMatchesFormerFloat16Graph(bool relu)
    {
        float[] values = Pattern(2 * 4 * 9, 0.023f);
        float[] seed = Pattern(2 * 4 * 11, 0.017f);
        var optimizedLinear = new Linear(
            9,
            11,
            new Random(107),
            0.07f,
            TensorDType.Float16);
        var referenceLinear = new Linear(
            9,
            11,
            new Random(107),
            0.07f,
            TensorDType.Float16);
        var optimizedInput = new Tensor(
            values,
            [2, 4, 9],
            dtype: TensorDType.Float16);
        var referenceInput = new Tensor(
            values,
            [2, 4, 9],
            dtype: TensorDType.Float16);

        Tensor optimized = relu
            ? optimizedLinear.ForwardBatchRelu(optimizedInput)
            : optimizedLinear.ForwardBatch(optimizedInput);
        Tensor reference = FormerReshapeForward(
            referenceLinear,
            referenceInput,
            relu);
        optimized.Backward(seed);
        reference.Backward(seed);

        Assert.Equal(TensorDType.Float16, optimized.DType);
        AssertClose(reference.Data, optimized.Data, 0f);
        AssertClose(referenceInput.Grad, optimizedInput.Grad, 1e-6f);
        AssertClose(
            referenceLinear.W.T.Grad,
            optimizedLinear.W.T.Grad,
            1e-6f);
        AssertClose(
            referenceLinear.B.T.Grad,
            optimizedLinear.B.T.Grad,
            1e-6f);
    }

    [Fact]
    public void SharedInputAndRepeatedBackwardAccumulateLikeFormerGraph()
    {
        float[] values = Pattern(2 * 3 * 4, 0.031f);
        float[] firstSeed = Pattern(2 * 3 * 5, 0.029f);
        float[] secondSeed = Pattern(2 * 3 * 5, -0.017f);
        var optimizedLinear = new Linear(4, 5, new Random(109), 0.09f);
        var referenceLinear = new Linear(4, 5, new Random(109), 0.09f);
        var optimizedInput = new Tensor(values, [2, 3, 4]);
        var referenceInput = new Tensor(values, [2, 3, 4]);

        Tensor optimized = optimizedLinear.ForwardBatch(optimizedInput)
            + optimizedLinear.ForwardBatchRelu(optimizedInput);
        Tensor reference = FormerReshapeForward(
                referenceLinear,
                referenceInput,
                relu: false)
            + FormerReshapeForward(
                referenceLinear,
                referenceInput,
                relu: true);

        optimized.Backward(firstSeed);
        reference.Backward(firstSeed);
        AssertGradientsMatch(
            referenceLinear,
            optimizedLinear,
            referenceInput,
            optimizedInput);

        optimized.Backward(secondSeed);
        reference.Backward(secondSeed);
        AssertGradientsMatch(
            referenceLinear,
            optimizedLinear,
            referenceInput,
            optimizedInput);
    }

    [Fact]
    public void DirectNodeAllocatesFarLessThanFormerReshapeGraph()
    {
        const int batch = 2;
        const int sequence = 8;
        const int inputWidth = 16;
        const int outputWidth = 32;
        const int iterations = 64;
        var linear = new Linear(
            inputWidth,
            outputWidth,
            new Random(113),
            0.05f);
        var input = new Tensor(
            Pattern(batch * sequence * inputWidth, 0.011f),
            [batch, sequence, inputWidth]);

        _ = linear.ForwardBatch(input);
        _ = FormerReshapeForward(linear, input, relu: false);

        long optimizedBytes = MeasureAllocatedBytes(
            () => linear.ForwardBatch(input),
            iterations);
        long formerBytes = MeasureAllocatedBytes(
            () => FormerReshapeForward(linear, input, relu: false),
            iterations);
        long eliminatedGradientBytesPerIteration =
            ((long)input.Numel + batch * sequence * outputWidth)
            * sizeof(float);
        long minimumExpectedReduction =
            eliminatedGradientBytesPerIteration * iterations * 9 / 10;

        Assert.True(
            formerBytes - optimizedBytes >= minimumExpectedReduction,
            $"Expected to eliminate at least {minimumExpectedReduction:N0} " +
            $"bytes, but optimized={optimizedBytes:N0}, " +
            $"former={formerBytes:N0}.");
    }

    private static Tensor FormerReshapeForward(
        Linear linear,
        Tensor input,
        bool relu)
    {
        int inputWidth = input.Shape[^1];
        int rows = input.Numel / inputWidth;
        Tensor flattened = input.Reshape(rows, inputWidth);
        Tensor projected = relu
            ? flattened.MatMulTransposedRightAddRowRelu(
                linear.W.T,
                linear.B.T)
            : flattened.MatMulTransposedRightAddRow(
                linear.W.T,
                linear.B.T);
        int[] outputShape = input.Shape.ToArray();
        outputShape[^1] = linear.W.T.Shape[0];
        return projected.Reshape(outputShape);
    }

    private static long MeasureAllocatedBytes(
        Func<Tensor> operation,
        int iterations)
    {
        long before = GC.GetAllocatedBytesForCurrentThread();
        Tensor? last = null;
        for (int iteration = 0; iteration < iterations; iteration++)
            last = operation();
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        GC.KeepAlive(last);
        return allocated;
    }

    private static void AssertGradientsMatch(
        Linear expectedLinear,
        Linear actualLinear,
        Tensor expectedInput,
        Tensor actualInput)
    {
        AssertClose(expectedInput.Grad, actualInput.Grad, 2e-6f);
        AssertClose(expectedLinear.W.T.Grad, actualLinear.W.T.Grad, 2e-6f);
        AssertClose(expectedLinear.B.T.Grad, actualLinear.B.T.Grad, 2e-6f);
    }

    private static float[] Pattern(int length, float scale)
        => Enumerable.Range(0, length)
            .Select(index => ((index * 17) % 29 - 14) * scale)
            .ToArray();

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.True(
                MathF.Abs(expected[index] - actual[index]) <= tolerance,
                $"Mismatch at {index}: expected {expected[index]}, " +
                $"actual {actual[index]}, tolerance {tolerance}.");
        }
    }
}
