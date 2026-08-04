using System.Runtime.Intrinsics;
using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TensorSimdCollection
{
    public const string Name = "Tensor SIMD configuration";
}

[Collection(TensorSimdCollection.Name)]
public sealed class TensorSimdTests
{
    [Fact]
    public void SimdIsEnabledByDefaultAndCanBeDisabled()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            Assert.True(Tensor.SimdEnabled);

            Tensor.SimdEnabled = false;

            Assert.False(Tensor.SimdEnabled);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndSimdElementwisePathsProduceEquivalentGradients()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            ElementwiseResult scalar = RunElementwise(useSimd: false);
            ElementwiseResult simd = RunElementwise(useSimd: true);

            AssertClose(scalar.Output, simd.Output, 2e-5f);
            AssertClose(scalar.LeftGradient, simd.LeftGradient, 2e-5f);
            AssertClose(scalar.RightGradient, simd.RightGradient, 2e-5f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndSimdTransformerKernelsProduceEquivalentResults()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            KernelResult scalar = RunTransformerKernels(useSimd: false);
            KernelResult simd = RunTransformerKernels(useSimd: true);

            AssertClose(scalar.Output, simd.Output, 2e-4f);
            AssertClose(scalar.InputGradient, simd.InputGradient, 4e-4f);
            AssertClose(scalar.WeightGradient, simd.WeightGradient, 4e-4f);
            AssertClose(scalar.BiasGradient, simd.BiasGradient, 4e-4f);
            AssertClose(scalar.GammaGradient, simd.GammaGradient, 4e-4f);
            AssertClose(scalar.BetaGradient, simd.BetaGradient, 4e-4f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256BatchedBackwardProduceEquivalentResults()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            BatchedKernelResult scalar = RunBatchedKernels(useSimd: false);
            BatchedKernelResult vector256 = RunBatchedKernels(useSimd: true);

            AssertClose(scalar.Output, vector256.Output, 2e-4f);
            AssertClose(
                scalar.QueryGradient,
                vector256.QueryGradient,
                4e-4f);
            AssertClose(
                scalar.KeyGradient,
                vector256.KeyGradient,
                4e-4f);
            AssertClose(
                scalar.ValueGradient,
                vector256.ValueGradient,
                4e-4f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    [Fact]
    public void ScalarAndVector256ParallelAdamWProduceEquivalentStates()
    {
        bool previous = Tensor.SimdEnabled;
        try
        {
            AdamResult scalar = RunAdam(useSimd: false);
            AdamResult vector256 = RunAdam(useSimd: true);

            AssertClose(scalar.Parameters, vector256.Parameters, 3e-5f);
            AssertClose(scalar.FirstMoments, vector256.FirstMoments, 2e-6f);
            AssertClose(scalar.SecondMoments, vector256.SecondMoments, 2e-6f);
        }
        finally
        {
            Tensor.SimdEnabled = previous;
        }
    }

    private static ElementwiseResult RunElementwise(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        int length = Vector256<float>.Count * 2 + 3;
        float[] leftValues = Enumerable.Range(0, length)
            .Select(index => 0.25f + index * 0.01f)
            .ToArray();
        float[] rightValues = Enumerable.Range(0, length)
            .Select(index => 1.5f + index * 0.02f)
            .ToArray();
        var left = Tensor.From1D(leftValues);
        var right = Tensor.From1D(rightValues);

        Tensor output =
            ((left + right) * (left - right))
            / (right + Tensor.Scalar(2f));
        output.Sum().Backward();

        return new ElementwiseResult(
            output.Data.ToArray(),
            left.Grad.ToArray(),
            right.Grad.ToArray());
    }

    private static KernelResult RunTransformerKernels(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int rows = 3;
        const int inputWidth = 32;
        const int outputWidth = 24;

        var input = new Tensor(
            Enumerable.Range(0, rows * inputWidth)
                .Select(index => (index % 13 - 6) * 0.03f)
                .ToArray(),
            [rows, inputWidth]);
        var weight = new Tensor(
            Enumerable.Range(0, inputWidth * outputWidth)
                .Select(index => (index % 17 - 8) * 0.02f)
                .ToArray(),
            [inputWidth, outputWidth]);
        var bias = Tensor.From1D(
            Enumerable.Range(0, outputWidth)
                .Select(index => (index - 12) * 0.01f)
                .ToArray());
        var gamma = Tensor.From1D(
            Enumerable.Range(0, outputWidth)
                .Select(index => 0.8f + index * 0.01f)
                .ToArray());
        var beta = Tensor.From1D(new float[outputWidth]);
        var lossWeight = new Tensor(
            Enumerable.Range(0, rows * outputWidth)
                .Select(index => (index % 7 - 3) * 0.1f)
                .ToArray(),
            [rows, outputWidth]);

        Tensor output = input
            .MatMul(weight)
            .AddRowWise(bias)
            .Relu()
            .LayerNormLastDim(gamma, beta)
            .LogSoftmaxLastDim();
        (output * lossWeight).Sum().Backward();

        return new KernelResult(
            output.Data.ToArray(),
            input.Grad.ToArray(),
            weight.Grad.ToArray(),
            bias.Grad.ToArray(),
            gamma.Grad.ToArray(),
            beta.Grad.ToArray());
    }

    private static BatchedKernelResult RunBatchedKernels(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int batch = 3;
        const int sequence = 4;
        const int width = 16;
        int length = batch * sequence * width;
        var query = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 19 - 9) * 0.015f)
                .ToArray(),
            [batch, sequence, width]);
        var key = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 23 - 11) * 0.012f)
                .ToArray(),
            [batch, sequence, width]);
        var value = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 17 - 8) * 0.02f)
                .ToArray(),
            [batch, sequence, width]);
        var lossWeight = new Tensor(
            Enumerable.Range(0, length)
                .Select(index => (index % 13 - 6) * 0.03f)
                .ToArray(),
            [batch, sequence, width]);

        Tensor attention =
            (query.BatchedMatMulTransposedRight(key)
                * Tensor.Scalar(0.25f))
            .SoftmaxLastDim();
        Tensor output = attention.BatchedMatMul(value);
        (output * lossWeight).Sum().Backward();

        return new BatchedKernelResult(
            output.Data.ToArray(),
            query.Grad.ToArray(),
            key.Grad.ToArray(),
            value.Grad.ToArray());
    }

    private static AdamResult RunAdam(bool useSimd)
    {
        Tensor.SimdEnabled = useSimd;
        const int parameterLength = 20_000;
        var parameters = new[]
        {
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 31 - 15) * 0.01f)
                    .ToArray(),
                [parameterLength],
                "first",
                WeightDecayPolicy.Apply),
            new Parameter(
                Enumerable.Range(0, parameterLength)
                    .Select(index => (index % 23 - 11) * 0.013f)
                    .ToArray(),
                [parameterLength],
                "second",
                WeightDecayPolicy.Exclude),
        };
        for (int parameter = 0; parameter < parameters.Length; parameter++)
        {
            Span<float> gradient = parameters[parameter].T.MutableGrad;
            for (int index = 0; index < gradient.Length; index++)
            {
                gradient[index] =
                    (index % 17 - 8) * (0.002f + parameter * 0.001f);
            }
        }

        var optimizer = new AdamW(
            parameters,
            new AdamWOptions
            {
                LearningRate = 0.003f,
                WeightDecay = 0.01f,
                Decay1D = true,
            });
        optimizer.Step();
        optimizer.Step();
        AdamWState state = optimizer.CaptureState();

        return new AdamResult(
            parameters.SelectMany(parameter => parameter.T.Data).ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.FirstMoment)
                .ToArray(),
            state.ParameterStates
                .SelectMany(parameter => parameter.SecondMoment)
                .ToArray());
    }

    private sealed record ElementwiseResult(
        float[] Output,
        float[] LeftGradient,
        float[] RightGradient);

    private sealed record KernelResult(
        float[] Output,
        float[] InputGradient,
        float[] WeightGradient,
        float[] BiasGradient,
        float[] GammaGradient,
        float[] BetaGradient);

    private sealed record BatchedKernelResult(
        float[] Output,
        float[] QueryGradient,
        float[] KeyGradient,
        float[] ValueGradient);

    private sealed record AdamResult(
        float[] Parameters,
        float[] FirstMoments,
        float[] SecondMoments);
}
