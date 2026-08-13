using NNtrain;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class TensorFloat16NativeF16cTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeF16cForwardMatchesManagedFallbackForRankThreeSimdTails(
        bool relu)
    {
        // The payload is deliberately optional: portable and non-Windows test
        // runs exercise the managed path elsewhere. When it is present, this
        // test fixes the native F16C contract against that fallback.
        if (!Tensor.IsFloat16NativeAccelerated)
            return;

        bool previousNativeEnabled = Tensor.Float16NativeEnabled;
        bool previousSimdEnabled = Tensor.SimdEnabled;
        int previousParallelism = Tensor.MaxDegreeOfParallelism;
        try
        {
            // 13 has an AVX/F16C eight-wide vector plus a scalar tail; 19
            // requires the native four-column block plus a column tail.
            // Six rank-3 rows also exercise the row-parallel native dispatch.
            Tensor.SimdEnabled = true;
            Tensor.MaxDegreeOfParallelism = 2;

            Tensor.Float16NativeEnabled = false;
            ProjectionResult managed = RunProjection(relu);

            Tensor.Float16NativeEnabled = true;
            Assert.True(Tensor.IsFloat16NativeAccelerated);
            ProjectionResult native = RunProjection(relu);

            Assert.Equal<int>([2, 3, 19], native.Shape);
            Assert.Equal(TensorDType.Float16, native.DType);
            AssertClose(managed.Output, native.Output, 2e-3f);
            AssertClose(managed.InputGradient, native.InputGradient, 1e-6f);
            AssertClose(managed.WeightGradient, native.WeightGradient, 1e-6f);
            AssertClose(managed.BiasGradient, native.BiasGradient, 1e-6f);

            if (relu)
            {
                Assert.Contains(native.Output, static value => value == 0f);
                Assert.Contains(native.Output, static value => value > 0f);
            }
        }
        finally
        {
            Tensor.MaxDegreeOfParallelism = previousParallelism;
            Tensor.SimdEnabled = previousSimdEnabled;
            Tensor.Float16NativeEnabled = previousNativeEnabled;
        }
    }

    [Fact]
    public void NativeF16cPreservesNonFiniteBiasAndScalarTailValues()
    {
        if (!Tensor.IsFloat16NativeAccelerated)
            return;

        bool previousNativeEnabled = Tensor.Float16NativeEnabled;
        try
        {
            Tensor.Float16NativeEnabled = true;
            var linear = new Linear(
                9,
                5,
                new Random(15019),
                initScale: 0f,
                dtype: TensorDType.Float16);
            using (Tensor.DataMutation weights = linear.W.T.BeginDataMutation())
            {
                weights.Values.Clear();
                // Column four is processed by the scalar output tail, and
                // element eight is the scalar K tail after F16C vectors.
                weights.Values[4 * 9 + 8] = float.PositiveInfinity;
            }
            using (Tensor.DataMutation biases = linear.B.T.BeginDataMutation())
            {
                biases.Values.Clear();
                // Columns zero and one are in the native 4-column kernel.
                biases.Values[0] = float.PositiveInfinity;
                biases.Values[1] = float.NaN;
            }

            float[] values = new float[9];
            values[8] = 1f;
            var input = new Tensor(values, [1, 1, 9], dtype: TensorDType.Float16);
            Tensor output = linear.ForwardBatch(input);

            Assert.True(float.IsPositiveInfinity(output.Data[0]));
            Assert.True(float.IsNaN(output.Data[1]));
            Assert.True(float.IsPositiveInfinity(output.Data[4]));
        }
        finally
        {
            Tensor.Float16NativeEnabled = previousNativeEnabled;
        }
    }

    [Fact]
    public void Float32InputAndFloat16ParametersUseTheManagedFallback()
    {
        bool previousNativeEnabled = Tensor.Float16NativeEnabled;
        try
        {
            Tensor.Float16NativeEnabled = true;
            var linear = new Linear(
                13,
                19,
                new Random(15023),
                initScale: 0.19f,
                dtype: TensorDType.Float16);
            var input = new Tensor(Pattern(2 * 3 * 13, 0.041f), [2, 3, 13]);
            Tensor output = linear.ForwardBatch(input);

            Assert.Equal(TensorDType.Float32, output.DType);
            Assert.All(output.Data, value => Assert.True(float.IsFinite(value)));
            output.Backward(Pattern(output.Numel, 0.023f));
            Assert.All(input.Grad, value => Assert.True(float.IsFinite(value)));
            Assert.All(linear.W.T.Grad, value => Assert.True(float.IsFinite(value)));
            Assert.All(linear.B.T.Grad, value => Assert.True(float.IsFinite(value)));
        }
        finally
        {
            Tensor.Float16NativeEnabled = previousNativeEnabled;
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeF16cAccumulatesOntoExistingGradients(bool relu)
    {
        if (!Tensor.IsFloat16NativeAccelerated)
            return;

        bool previousNativeEnabled = Tensor.Float16NativeEnabled;
        try
        {
            Tensor.Float16NativeEnabled = false;
            ProjectionResult managed = RunRepeatedBackward(relu);

            Tensor.Float16NativeEnabled = true;
            ProjectionResult native = RunRepeatedBackward(relu);

            AssertClose(managed.Output, native.Output, 2e-3f);
            AssertClose(managed.InputGradient, native.InputGradient, 1e-6f);
            AssertClose(managed.WeightGradient, native.WeightGradient, 1e-6f);
            AssertClose(managed.BiasGradient, native.BiasGradient, 1e-6f);
        }
        finally
        {
            Tensor.Float16NativeEnabled = previousNativeEnabled;
        }
    }

    private static ProjectionResult RunProjection(bool relu)
    {
        const int batch = 2;
        const int sequence = 3;
        const int inputWidth = 13;
        const int outputWidth = 19;
        float[] inputValues = Pattern(batch * sequence * inputWidth, 0.071f);
        float[] backwardSeed = Pattern(batch * sequence * outputWidth, 0.037f);
        var linear = new Linear(
            inputWidth,
            outputWidth,
            new Random(15017),
            initScale: 0.31f,
            dtype: TensorDType.Float16);
        var input = new Tensor(
            inputValues,
            [batch, sequence, inputWidth],
            dtype: TensorDType.Float16);

        Tensor output = relu
            ? linear.ForwardBatchRelu(input)
            : linear.ForwardBatch(input);
        output.Backward(backwardSeed);

        return new ProjectionResult(
            output.Shape.ToArray(),
            output.DType,
            output.Data.ToArray(),
            input.Grad.ToArray(),
            linear.W.T.Grad.ToArray(),
            linear.B.T.Grad.ToArray());
    }

    private static ProjectionResult RunRepeatedBackward(bool relu)
    {
        const int inputWidth = 13;
        const int outputWidth = 19;
        var linear = new Linear(
            inputWidth,
            outputWidth,
            new Random(15029),
            initScale: 0.21f,
            dtype: TensorDType.Float16);
        var input = new Tensor(
            Pattern(2 * 3 * inputWidth, 0.051f),
            [2, 3, inputWidth],
            dtype: TensorDType.Float16);
        Tensor output = relu
            ? linear.ForwardBatchRelu(input)
            : linear.ForwardBatch(input);
        output.Backward(Pattern(output.Numel, 0.017f));
        output.Backward(Pattern(output.Numel, -0.011f));

        return new ProjectionResult(
            output.Shape.ToArray(),
            output.DType,
            output.Data.ToArray(),
            input.Grad.ToArray(),
            linear.W.T.Grad.ToArray(),
            linear.B.T.Grad.ToArray());
    }

    private static float[] Pattern(int length, float scale)
        => Enumerable.Range(0, length)
            .Select(index => ((index * 17) % 31 - 15) * scale)
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

    private sealed record ProjectionResult(
        int[] Shape,
        TensorDType DType,
        float[] Output,
        float[] InputGradient,
        float[] WeightGradient,
        float[] BiasGradient);
}
