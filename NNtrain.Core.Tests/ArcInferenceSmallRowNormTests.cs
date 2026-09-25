using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcInferenceSmallRowNormTests
{
    private const int Width = 512;

    [Theory]
    [InlineData(1, TensorPrecisionMode.Float32)]
    [InlineData(8, TensorPrecisionMode.Float32)]
    [InlineData(1, TensorPrecisionMode.Mix16_32)]
    [InlineData(8, TensorPrecisionMode.Mix16_32)]
    [InlineData(1, TensorPrecisionMode.Mix8_32)]
    [InlineData(8, TensorPrecisionMode.Mix8_32)]
    [InlineData(1, TensorPrecisionMode.Mix8_16)]
    [InlineData(8, TensorPrecisionMode.Mix8_16)]
    public void NoGradSmallRowsUseOrderedKernelWithinStoragePrecision(
        int rows, TensorPrecisionMode precision)
    {
        RequireSg16Arc();
        float[] input = Values(rows * Width, 7, .27f, .03f);
        float[] branch = Values(rows * Width, 19, .14f, -.02f);
        float[] gamma = Values(Width, 31, .13f, 1f);
        float[] beta = Values(Width, 43, .025f, .01f);

        Snapshot Run(bool smallRows)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: Options(smallRows));
            TensorDType activationType = precision switch
            {
                TensorPrecisionMode.Float32 => TensorDType.Float32,
                TensorPrecisionMode.Mix16_32 => TensorDType.BFloat16,
                TensorPrecisionMode.Mix8_32 => TensorDType.Bfp8,
                _ => TensorDType.BFloat16,
            };
            TensorDType parameterType = precision.ToStorageDType();
            Tensor x = Make(input, [rows, Width], activationType);
            Tensor residual = Make(branch, [rows, Width], parameterType);
            Tensor gain = Make(gamma, [Width], parameterType);
            Tensor bias = Make(beta, [Width], parameterType);
            float[] plain, withResidual;
            TensorDType outputType;
            using (AutogradContext.NoGrad())
            {
                Tensor normalized = x.LayerNormLastDim(gain, bias);
                plain = normalized.Data.ToArray();
                outputType = normalized.DType;
                Tensor fused = x.AddDropoutLayerNormLastDim(residual, gain, bias,
                    probability: 0f);
                withResidual = fused.Data.ToArray();
                Assert.Equal(outputType, fused.DType);
            }
            Tensor.ArcLane.CheckNumericStatus();
            return new(plain, withResidual, outputType,
                Tensor.ArcLane.KernelTimings.Keys.ToArray());
        }

        Snapshot reference = Run(false);
        Snapshot candidate = Run(true);
        string selected = precision == TensorPrecisionMode.Mix8_16
            ? "norm_row_sg16_w64_parallel_bf16"
            : "norm_row_sg16_w64_candidate";
        Assert.Contains("norm", reference.Kernels);
        Assert.DoesNotContain(selected, reference.Kernels);
        Assert.Contains(selected, candidate.Kernels);
        Assert.DoesNotContain("norm", candidate.Kernels);
        Assert.Equal(reference.OutputType, candidate.OutputType);
        Assert.Equal(precision switch
        {
            TensorPrecisionMode.Float32 => TensorDType.Float32,
            TensorPrecisionMode.Mix16_32 => TensorDType.BFloat16,
            TensorPrecisionMode.Mix8_32 => TensorDType.Bfp8,
            _ => TensorDType.BFloat16,
        }, candidate.OutputType);
        CheckPrecision("plain", reference.Plain, candidate.Plain, candidate.OutputType);
        CheckPrecision("residual", reference.Residual, candidate.Residual, candidate.OutputType);
    }

    [Theory]
    [InlineData(1, TensorPrecisionMode.Float32)]
    [InlineData(8, TensorPrecisionMode.Mix8_16)]
    public void RecordingGradientsKeepsSmallRowsOnExistingPath(
        int rows, TensorPrecisionMode precision)
    {
        RequireSg16Arc();
        using var execution = Tensor.BeginArcExecution(precision: precision,
            options: Options(smallRows: true));
        TensorDType activationType = precision == TensorPrecisionMode.Float32
            ? TensorDType.Float32 : TensorDType.BFloat16;
        TensorDType parameterType = precision.ToStorageDType();
        Tensor x = Make(Values(rows * Width, 3, .2f, 0f),
            [rows, Width], activationType);
        Tensor branch = Make(Values(rows * Width, 11, .1f, .02f),
            [rows, Width], parameterType);
        Tensor gamma = Make(Values(Width, 23, .1f, 1f),
            [Width], parameterType);
        Tensor beta = Make(Values(Width, 37, .01f, 0f),
            [Width], parameterType);

        Tensor output = x.AddDropoutLayerNormLastDim(branch, gamma, beta,
            probability: 0f);
        Assert.All(output.Data.ToArray(), value => Assert.True(float.IsFinite(value)));
        Assert.Contains("norm", Tensor.ArcLane.KernelTimings.Keys);
        Assert.DoesNotContain("norm_row_sg16_w64_candidate", Tensor.ArcLane.KernelTimings.Keys);
        Assert.DoesNotContain("norm_row_sg16_w64_parallel_bf16", Tensor.ArcLane.KernelTimings.Keys);
        output.BackwardAndRelease(Values(rows * Width, 47, .012f, 0f));
        Tensor.ArcLane.CheckNumericStatus();
        Assert.All(x.Grad, value => Assert.True(float.IsFinite(value)));
        Assert.All(branch.Grad, value => Assert.True(float.IsFinite(value)));
        Assert.All(gamma.Grad, value => Assert.True(float.IsFinite(value)));
        Assert.All(beta.Grad, value => Assert.True(float.IsFinite(value)));
    }

    private static ArcExecutionOptions Options(bool smallRows) => new()
    {
        InferenceSmallRowNorm = smallRows,
        OrderedTiledNorm = true,
        BlockResidualNorm = false,
        FusedPackedResidualNorm = false,
        PackedNormInput = true,
        ParallelReductions = true,
        Mix8_16Bf16Activations = true,
        Mix8_16DirectBf16NormOutput = true,
        Mix8_16ParallelNormReduction = true,
    };

    private static void RequireSg16Arc()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel Arc is required.");
    }

    private static Tensor Make(float[] values, int[] shape, TensorDType storage)
    {
        var tensor = new Tensor((float[])values.Clone(), shape);
        if (storage == TensorDType.Bfp8)
            tensor.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
        else if (storage == TensorDType.BFloat16)
            tensor.ConvertStorageInPlace(storage);
        return tensor;
    }

    private static float[] Values(int count, int seed, float amplitude, float offset)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = MathF.Sin(i * .019f + seed * .17f) * amplitude + offset;
        return values;
    }

    private static void CheckPrecision(
        string label, float[] reference, float[] candidate, TensorDType storage)
    {
        Assert.Equal(reference.Length, candidate.Length);
        double errorSquared = 0, referenceSquared = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(float.IsFinite(reference[i]) && float.IsFinite(candidate[i]));
            float error = MathF.Abs(reference[i] - candidate[i]);
            float allowance = storage switch
            {
                TensorDType.Float32 => 1e-5f + 2e-5f * MathF.Abs(reference[i]),
                TensorDType.BFloat16 => MathF.Max(1e-4f, 2f * BFloat16Ulp(
                    MathF.Max(MathF.Abs(reference[i]), MathF.Abs(candidate[i])))),
                _ => 2f * Bfp8OutputStep(reference, i) + 1e-5f,
            };
            Assert.True(error <= allowance,
                $"{label}[{i}] differs by {error:G9}, allowance {allowance:G9}; "
                + $"expected {reference[i]:G9}, actual {candidate[i]:G9}.");
            errorSquared += (double)error * error;
            referenceSquared += (double)reference[i] * reference[i];
        }
        double relativeRms = Math.Sqrt(errorSquared / Math.Max(referenceSquared, 1e-30));
        double limit = storage switch
        {
            TensorDType.Float32 => 2e-5,
            TensorDType.BFloat16 => 2e-3,
            _ => 5e-3,
        };
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{label} {storage}: relative RMS={relativeRms:G6}");
        Assert.True(relativeRms <= limit,
            $"{label} relative RMS {relativeRms:G6} exceeds {limit:G6}.");
    }

    private static float BFloat16Ulp(float magnitude)
    {
        int exponent = (BitConverter.SingleToInt32Bits(magnitude) >> 23) & 255;
        return exponent == 0 ? 0f : MathF.ScaleB(1f, exponent - 127 - 7);
    }

    private static float Bfp8OutputStep(float[] values, int index)
    {
        int start = (index / 32) * 32;
        float maximum = 0f;
        for (int i = start; i < Math.Min(start + 32, values.Length); i++)
            maximum = MathF.Max(maximum, MathF.Abs(values[i]));
        return maximum / 127f;
    }

    private sealed record Snapshot(
        float[] Plain, float[] Residual, TensorDType OutputType, string[] Kernels);
}
