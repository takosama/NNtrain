using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenRmsNormFastTests
{
    [Theory]
    [InlineData(1, 1, TensorDType.Float32)]
    [InlineData(3, 17, TensorDType.Float32)]
    [InlineData(2, 127, TensorDType.Float32)]
    [InlineData(1, 2048, TensorDType.Float32)]
    [InlineData(3, 2051, TensorDType.Float32)]
    [InlineData(1, 2048, TensorDType.BFloat16)]
    [InlineData(3, 97, TensorDType.BFloat16)]
    public void NoGradParallelNormMatchesScalarGpuAndCpuReference(
        int rows, int width, TensorDType dtype)
    {
        RequireArc();
        const float epsilon = 1e-6f;
        float[] inputs = Enumerable.Range(0, rows * width)
            .Select(index => MathF.Sin(index * .037f + .13f) * 1.7f).ToArray();
        float[] weights = Enumerable.Range(0, width)
            .Select(index => .7f + MathF.Cos(index * .019f) * .4f).ToArray();

        (float[] Output, float[] Reference) Run(bool fast)
        {
            using var execution = Tensor.BeginArcExecution(options:
                new ArcExecutionOptions { QwenRmsNormFast = fast });
            Tensor input = Make(inputs, [rows, width], dtype);
            Tensor weight = Make(weights, [width], dtype);
            // Reference consumes values rounded to the actual input storage.
            float[] x = input.Data.ToArray(), w = weight.Data.ToArray();
            var expected = new float[x.Length];
            for (int row = 0; row < rows; ++row)
            {
                double square = 0;
                for (int column = 0; column < width; ++column)
                    square += (double)x[row * width + column] * x[row * width + column];
                double inv = 1 / Math.Sqrt(square / width + epsilon);
                for (int column = 0; column < width; ++column)
                {
                    float value = (float)(x[row * width + column] * inv * w[column]);
                    expected[row * width + column] = dtype == TensorDType.BFloat16
                        ? TensorStorageCodec.RoundToBFloat16(value) : value;
                }
            }
            using var noGrad = AutogradContext.NoGrad();
            using var frame = Tensor.BeginArcInferenceFrame();
            Tensor normalized = input.RmsNormLastDim(weight, epsilon);
            Assert.Equal(dtype, normalized.DType);
            float[] actual = normalized.Data.ToArray();
            Assert.Equal(fast, Tensor.ArcLane.KernelTimings.ContainsKey("qwen_rmsnorm_fast"));
            Assert.Equal(!fast, Tensor.ArcLane.KernelTimings.ContainsKey("qwen_rmsnorm"));
            return (actual, expected);
        }

        var scalar = Run(false);
        var parallel = Run(true);
        AssertClose(scalar.Output, parallel.Output, dtype);
        AssertClose(parallel.Reference, parallel.Output, dtype);
    }

    [Fact]
    public void ParallelNormHandlesAllZeroInputWithoutNonfiniteValues()
    {
        RequireArc();
        using var execution = Tensor.BeginArcExecution(options:
            new ArcExecutionOptions { QwenRmsNormFast = true });
        using var noGrad = AutogradContext.NoGrad();
        using var frame = Tensor.BeginArcInferenceFrame();
        var input = new Tensor(new float[3 * 65], [3, 65]);
        var weight = new Tensor(Enumerable.Repeat(1f, 65).ToArray(), [65]);

        Tensor result = input.RmsNormLastDim(weight);

        Assert.All(result.Data, value => Assert.Equal(0f, value));
        Assert.Contains("qwen_rmsnorm_fast", Tensor.ArcLane.KernelTimings.Keys);
    }

    [Fact]
    public void RecordingGradientsKeepsScalarNormAndBackward()
    {
        RequireArc();
        using var execution = Tensor.BeginArcExecution(options:
            new ArcExecutionOptions { QwenRmsNormFast = true });
        var input = new Tensor(Enumerable.Range(0, 130).Select(index => .01f * index).ToArray(), [2, 65]);
        var weight = new Tensor(Enumerable.Repeat(1f, 65).ToArray(), [65]);

        Tensor result = input.RmsNormLastDim(weight);
        result.BackwardAndRelease(Enumerable.Repeat(1f, 130).ToArray());
        Tensor.ArcLane.Synchronize();

        Assert.DoesNotContain("qwen_rmsnorm_fast", Tensor.ArcLane.KernelTimings.Keys);
        Assert.Contains("qwen_rmsnorm", Tensor.ArcLane.KernelTimings.Keys);
        Assert.Contains("qwen_rmsnorm_dx", Tensor.ArcLane.KernelTimings.Keys);
        Assert.Contains("qwen_rmsnorm_dw", Tensor.ArcLane.KernelTimings.Keys);
        Assert.All(input.Grad, value => Assert.True(float.IsFinite(value)));
        Assert.All(weight.Grad, value => Assert.True(float.IsFinite(value)));
    }

    private static Tensor Make(float[] values, int[] shape, TensorDType dtype)
    {
        var tensor = new Tensor((float[])values.Clone(), shape);
        if (dtype != TensorDType.Float32) tensor.ConvertStorageInPlace(dtype);
        return tensor;
    }

    private static void AssertClose(float[] expected, float[] actual, TensorDType dtype)
    {
        Assert.Equal(expected.Length, actual.Length);
        double errorSquared = 0, expectedSquared = 0;
        for (int i = 0; i < actual.Length; ++i)
        {
            Assert.True(float.IsFinite(actual[i]));
            double error = Math.Abs(actual[i] - expected[i]);
            double allowance = dtype == TensorDType.BFloat16
                ? 1e-5 + Math.Abs(expected[i]) * .008
                : 1e-5 + Math.Abs(expected[i]) * 2e-5;
            Assert.True(error <= allowance,
                $"RMSNorm[{i}] expected={expected[i]:G9}, actual={actual[i]:G9}, error={error:G9}, limit={allowance:G9}");
            errorSquared += error * error;
            expectedSquared += (double)expected[i] * expected[i];
        }
        double relativeRms = Math.Sqrt(errorSquared / Math.Max(expectedSquared, 1e-30));
        Assert.True(relativeRms <= (dtype == TensorDType.BFloat16 ? .003 : 2e-5),
            $"RMSNorm relative RMS error {relativeRms:G9}.");
    }

    private static void RequireArc()
        => Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
}
