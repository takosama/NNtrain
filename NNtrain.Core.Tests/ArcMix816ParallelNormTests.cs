using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816ParallelNormTests
{
    [Theory]
    [InlineData(129, 65, false, .125f)]
    [InlineData(512, 512, false, 0f)]
    [InlineData(513, 512, true, .75f)]
    public void ParallelSubgroupSumsStayWithinBf16RoundingOfOrderedNorm(
        int rows, int width, bool aliasBranch, float dropout)
    {
        RequireXmx();
        Snapshot ordered = Run(TensorPrecisionMode.Mix8_16, rows, width,
            aliasBranch, dropout, parallel: false);
        Snapshot parallel = Run(TensorPrecisionMode.Mix8_16, rows, width,
            aliasBranch, dropout, parallel: true);

        Assert.DoesNotContain("norm_row_sg16_w64_parallel_bf16", ordered.Kernels);
        Assert.DoesNotContain("norm_dx_row_sg16_w64_parallel_residual_bf16", ordered.Kernels);
        Assert.Contains("norm_row_sg16_w64_parallel_bf16", parallel.Kernels);
        Assert.Contains("norm_dx_row_sg16_w64_parallel_residual_bf16", parallel.Kernels);
        Assert.Equal(ordered.RetainedBytes, parallel.RetainedBytes);
        for (int repeat = 0; repeat < ordered.Outputs.Length; repeat++)
            AssertBf16Close($"output {repeat}", ordered.Outputs[repeat], parallel.Outputs[repeat],
                ulpAllowance: 2);
        for (int index = 0; index < ordered.Gradients.Length; index++)
            AssertAccumulatedBf16Close($"gradient {index}",
                ordered.FirstGradients[index], parallel.FirstGradients[index],
                ordered.Gradients[index], parallel.Gradients[index], width);
    }

    [Fact]
    public void ParallelNormOptionKeepsMix8_32OnTheOrderedPath()
    {
        RequireXmx();
        Snapshot ordered = Run(TensorPrecisionMode.Mix8_32, 129, 65,
            aliasBranch: false, dropout: .125f, parallel: false);
        Snapshot optionEnabled = Run(TensorPrecisionMode.Mix8_32, 129, 65,
            aliasBranch: false, dropout: .125f, parallel: true);

        Assert.DoesNotContain("norm_row_sg16_w64_parallel_bf16", optionEnabled.Kernels);
        Assert.DoesNotContain("norm_dx_row_sg16_w64_parallel_residual_bf16", optionEnabled.Kernels);
        for (int repeat = 0; repeat < ordered.Outputs.Length; repeat++)
            Assert.Equal(ordered.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                optionEnabled.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
        for (int index = 0; index < ordered.Gradients.Length; index++)
            Assert.Equal(ordered.Gradients[index].Select(BitConverter.SingleToInt32Bits),
                optionEnabled.Gradients[index].Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AnchoredMeanMatchesDoubleReferenceAtLargeCommonOffset(bool constantRow)
    {
        RequireXmx();
        const int rows = 129, width = 65;
        const float common = 9984f, epsilon = 1e-5f;
        using var scope = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
            options: new ArcExecutionOptions
            {
                Mix8_16Bf16Activations = true,
                Mix8_16DirectBf16NormOutput = true,
                Mix8_16FusedNormResidualBackward = true,
                Mix8_16ParallelNormReduction = true,
                Mix8_16FusedNormParameterGradients = false,
                FusedPackedResidualNorm = false,
                BlockResidualNorm = false,
                OrderedTiledNorm = true,
                PackedNormInput = true,
            });
        Tensor Make(float[] source, int[] shape, TensorDType dtype)
        {
            var tensor = new Tensor(source, shape);
            tensor.ConvertStorageInPlace(dtype, dtype == TensorDType.Bfp8
                ? Bfp8QuantizationDescriptor.Block(32) : null);
            return tensor;
        }
        int length = rows * width;
        Tensor residual = Make(Enumerable.Repeat(common, length).ToArray(),
            [rows, width], TensorDType.BFloat16);
        Tensor branch = Make(Enumerable.Range(0, length).Select(i => constantRow
                ? 0f : (((i / width + (i % width) * 37) % 23) - 11) * .0115f).ToArray(),
            [rows, width], TensorDType.Bfp8);
        Tensor gamma = Make(Enumerable.Repeat(1f, width).ToArray(),
            [width], TensorDType.Bfp8);
        Tensor beta = Make(new float[width], [width], TensorDType.Bfp8);
        float[] residualValues = residual.Data.ToArray();
        float[] branchValues = branch.Data.ToArray();
        float[] gammaValues = gamma.Data.ToArray();
        float[] betaValues = beta.Data.ToArray();
        Tensor output = residual.AddDropoutLayerNormLastDim(branch, gamma, beta,
            probability: 0f, eps: epsilon);
        float[] actual = output.Data.ToArray();
        Assert.Contains("norm_row_sg16_w64_parallel_bf16", Tensor.ArcLane.KernelTimings.Keys);
        double worstError = 0, worstAllowanceRatio = 0;
        int worstIndex = -1;
        for (int row = 0; row < rows; row++)
        {
            float[] input = new float[width];
            for (int col = 0; col < width; col++)
            {
                int index = row * width + col;
                // Match the packed input's one FP32 FMA/store after both
                // operands have been decoded from their physical storage.
                input[col] = MathF.FusedMultiplyAdd(branchValues[index], 1f,
                    residualValues[index]);
            }
            double mean = input.Average(value => (double)value);
            double variance = input.Average(value =>
                ((double)value - mean) * ((double)value - mean));
            double inv = 1d / Math.Sqrt(variance + epsilon);
            float meanUlp = MathF.BitIncrement((float)mean) - (float)mean;
            for (int col = 0; col < width; col++)
            {
                int index = row * width + col;
                float expected = TensorStorageCodec.RoundToBFloat16((float)(
                    (input[col] - mean) * inv * gammaValues[col] + betaValues[col]));
                Assert.True(float.IsFinite(actual[index]));
                if (constantRow)
                {
                    Assert.Equal(BitConverter.SingleToInt32Bits(expected),
                        BitConverter.SingleToInt32Bits(actual[index]));
                    continue;
                }
                float magnitude = MathF.Max(MathF.Abs(expected), MathF.Abs(actual[index]));
                int exponent = (BitConverter.SingleToInt32Bits(magnitude) >> 23) & 255;
                float bf16Ulp = exponent == 0 ? 0f : MathF.ScaleB(1f, exponent - 127 - 7);
                // The final FP32 mean can differ from the double mean by one
                // FP32 ULP at this offset; propagate that through normalization.
                double allowance = 2d * bf16Ulp
                    + meanUlp * inv * Math.Abs(gammaValues[col]) + 1e-5d;
                double error = Math.Abs(expected - actual[index]);
                if (error / allowance > worstAllowanceRatio)
                {
                    worstAllowanceRatio = error / allowance;
                    worstError = error;
                    worstIndex = index;
                }
            }
        }
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"large-offset norm: constant={constantRow}, max normalized error={worstAllowanceRatio:G6}, "
            + $"error={worstError:G9}, index={worstIndex}");
        Assert.True(worstAllowanceRatio <= 1d,
            $"Large-offset LayerNorm differs from rounded-operand double reference by "
            + $"{worstError:G9} at {worstIndex}; allowance ratio {worstAllowanceRatio:G6}.");
        float[] seed = Enumerable.Range(0, length)
            .Select(i => MathF.Cos(i * .0167f) * .002f).ToArray();
        output.BackwardAndRelease(seed);
        Tensor.ArcLane.Synchronize();
        foreach (Tensor tensor in new[] { residual, branch, gamma, beta })
            Assert.All(tensor.Grad, value => Assert.True(float.IsFinite(value)));
    }

    private static Snapshot Run(TensorPrecisionMode precision, int rows, int width,
        bool aliasBranch, float dropout, bool parallel)
    {
        using var scope = Tensor.BeginArcExecution(precision: precision,
            options: new ArcExecutionOptions
            {
                Mix8_16Bf16Activations = true,
                Mix8_16DirectBf16NormOutput = true,
                Mix8_16FusedNormResidualBackward = true,
                Mix8_16ParallelNormReduction = parallel,
                // Isolate row reduction from the separately tested parameter fusion.
                Mix8_16FusedNormParameterGradients = false,
                Mix8_16BiasOnlyGradientReduction = false,
                FusedPackedResidualNorm = false,
                BlockResidualNorm = false,
                OrderedTiledNorm = true,
                PackedNormInput = true,
                ParallelReductions = true,
            });
        Tensor Make(int count, int[] shape, float amplitude, float offset,
            TensorDType dtype)
        {
            var tensor = new Tensor(Enumerable.Range(0, count)
                .Select(i => MathF.Sin(i * .0137f) * amplitude + offset).ToArray(), shape);
            tensor.ConvertStorageInPlace(dtype, dtype == TensorDType.Bfp8
                ? Bfp8QuantizationDescriptor.Block(32) : null);
            return tensor;
        }

        TensorDType residualType = precision == TensorPrecisionMode.Mix8_16
            ? TensorDType.BFloat16 : TensorDType.Bfp8;
        Tensor residual = Make(rows * width, [rows, width], .37f, .02f, residualType);
        Tensor branch = aliasBranch ? residual : Make(rows * width, [rows, width],
            .23f, -.04f, TensorDType.Bfp8);
        Tensor gamma = Make(width, [width], .16f, 1f, TensorDType.Bfp8);
        Tensor beta = Make(width, [width], .02f, 0f, TensorDType.Bfp8);
        float[] seed = Enumerable.Range(0, rows * width)
            .Select(i => MathF.Cos(i * .0167f) * .023f).ToArray();
        var outputs = new float[2][];
        float[][]? firstGradients = null;
        float[][] CaptureGradients() =>
            [residual.Grad.ToArray(), branch.Grad.ToArray(), gamma.Grad.ToArray(), beta.Grad.ToArray()];
        long retained = -1;
        for (int repeat = 0; repeat < 2; repeat++)
        {
            Tensor value = residual.AddDropoutLayerNormLastDim(branch, gamma, beta,
                dropout, new Random(123 + repeat));
            if (precision == TensorPrecisionMode.Mix8_16)
            {
                Assert.Equal(TensorDType.BFloat16, value.DType);
                Assert.Equal(rows * width * sizeof(ushort), value.StorageByteLength);
            }
            outputs[repeat] = value.Data.ToArray();
            value.BackwardAndRelease(seed);
            Tensor.ArcLane.Synchronize();
            if (repeat == 0) firstGradients = CaptureGradients();
            if (retained >= 0) Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
            retained = Tensor.ArcLane.AllocatedBytes;
        }
        Tensor.ArcLane.CheckNumericStatus();
        return new(outputs, firstGradients!, CaptureGradients(),
            Tensor.ArcLane.KernelTimings.Keys.ToArray(), retained);
    }

    // Forward publishes once, so two BF16 ULPs bound a midpoint crossing.
    private static void AssertBf16Close(string label, float[] expected, float[] actual,
        int ulpAllowance)
    {
        Assert.Equal(expected.Length, actual.Length);
        float maximumError = 0, maximumRatio = 0;
        float worstExpected = 0, worstActual = 0;
        double errorSquared = 0, referenceSquared = 0;
        int worstIndex = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            float error = MathF.Abs(expected[i] - actual[i]);
            maximumError = MathF.Max(maximumError, error);
            errorSquared += (double)error * error;
            referenceSquared += (double)expected[i] * expected[i];
            float magnitude = MathF.Max(MathF.Abs(expected[i]), MathF.Abs(actual[i]));
            int exponent = (BitConverter.SingleToInt32Bits(magnitude) >> 23) & 255;
            float ulp = exponent == 0 ? 0f : MathF.ScaleB(1f, exponent - 127 - 7);
            float allowance = MathF.Max(1e-4f, ulpAllowance * ulp);
            float ratio = error / allowance;
            if (ratio > maximumRatio)
            {
                maximumRatio = ratio;
                worstIndex = i;
                worstExpected = expected[i];
                worstActual = actual[i];
            }
        }
        double relativeRms = Math.Sqrt(errorSquared / Math.Max(referenceSquared, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{label}: max absolute error={maximumError:G9}, relative RMS={relativeRms:G6}, "
            + $"BF16 allowance ratio={maximumRatio:G6}, index={worstIndex}, "
            + $"expected={worstExpected:G9}, actual={worstActual:G9}");
        Assert.InRange(relativeRms, 0d, 5e-4d);
        Assert.True(maximumRatio <= 1f,
            $"{label} max error {maximumError:G9}, relative RMS {relativeRms:G6}; "
            + $"BF16 allowance ratio {maximumRatio:G6} at {worstIndex}, "
            + $"expected {worstExpected:G9}, actual {worstActual:G9}.");
    }

    private static void AssertAccumulatedBf16Close(string label,
        float[] expectedFirst, float[] actualFirst, float[] expectedFinal, float[] actualFinal,
        int reductionWidth)
    {
        Assert.Equal(expectedFirst.Length, actualFirst.Length);
        Assert.Equal(expectedFirst.Length, expectedFinal.Length);
        Assert.Equal(expectedFirst.Length, actualFinal.Length);
        float gradientScale = 0;
        for (int i = 0; i < expectedFinal.Length; i++)
        {
            float firstReference = expectedFirst[i], firstActual = actualFirst[i];
            float finalReference = expectedFinal[i], finalActual = actualFinal[i];
            Assert.True(float.IsFinite(firstReference) && float.IsFinite(firstActual)
                && float.IsFinite(finalReference) && float.IsFinite(finalActual));
            float secondReference = finalReference - firstReference;
            float secondActual = finalActual - firstActual;
            gradientScale = MathF.Max(gradientScale, MathF.Max(
                MathF.Max(MathF.Abs(firstReference), MathF.Abs(firstActual)),
                MathF.Max(MathF.Abs(secondReference), MathF.Abs(secondActual))));
        }
        // This compares two FP32 reduction orders. The BF16 publication
        // bound alone misses cancellation *inside* the row derivative: tiny
        // outputs can result from width terms at the gradient's typical scale.
        // One FP32 ULP per term gives a width-scaled comparison budget. This
        // is a test tolerance for changed summation order, not an error proof.
        double fp32SummationBudget = (double)reductionWidth
            * MathF.ScaleB(1f, -23) * gradientScale;
        float maximumError = 0;
        double maximumRatio = 0;
        float worstExpectedFirst = 0, worstActualFirst = 0;
        float worstExpectedSecond = 0, worstActualSecond = 0;
        float worstExpectedFinal = 0, worstActualFinal = 0;
        double worstAllowance = 0;
        double errorSquared = 0, referenceSquared = 0;
        int worstIndex = -1;
        for (int i = 0; i < expectedFinal.Length; i++)
        {
            float firstReference = expectedFirst[i], firstActual = actualFirst[i];
            float finalReference = expectedFinal[i], finalActual = actualFinal[i];
            float secondReference = finalReference - firstReference;
            float secondActual = finalActual - firstActual;
            float error = MathF.Abs(finalReference - finalActual);
            maximumError = MathF.Max(maximumError, error);
            errorSquared += (double)error * error;
            referenceSquared += (double)finalReference * finalReference;

            // The retained first-pass gradient and the inferred second-pass
            // contribution each undergo up to two BF16 additions when branch
            // aliases residual. Their magnitudes set the rounding scale;
            // the final value can be tiny after cancellation. One more ULP
            // covers the final BF16 publication. Add the width-scaled FP32
            // summation budget above for cancellation within each derivative;
            // RMS still limits aggregate drift to 0.05%.
            double allowance = 2d * Bf16Ulp(firstReference, firstActual)
                + 2f * Bf16Ulp(secondReference, secondActual)
                + Bf16Ulp(finalReference, finalActual) + fp32SummationBudget;
            double ratio = error / allowance;
            if (ratio > maximumRatio)
            {
                maximumRatio = ratio;
                worstIndex = i;
                worstExpectedFirst = firstReference;
                worstActualFirst = firstActual;
                worstExpectedSecond = secondReference;
                worstActualSecond = secondActual;
                worstExpectedFinal = finalReference;
                worstActualFinal = finalActual;
                worstAllowance = allowance;
            }
        }
        double relativeRms = Math.Sqrt(errorSquared / Math.Max(referenceSquared, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{label}: max absolute error={maximumError:G9}, relative RMS={relativeRms:G6}, "
            + $"accumulation allowance ratio={maximumRatio:G6}, index={worstIndex}, "
            + $"first={worstExpectedFirst:G9}/{worstActualFirst:G9}, "
            + $"second={worstExpectedSecond:G9}/{worstActualSecond:G9}, "
            + $"final={worstExpectedFinal:G9}/{worstActualFinal:G9}, "
            + $"allowance={worstAllowance:G9}, FP32 budget={fp32SummationBudget:G9}, "
            + $"gradient scale={gradientScale:G9}");
        Assert.InRange(relativeRms, 0d, 5e-4d);
        Assert.True(maximumRatio <= 1d,
            $"{label} max error {maximumError:G9}, relative RMS {relativeRms:G6}; "
            + $"accumulation allowance ratio {maximumRatio:G6} at {worstIndex}, "
            + $"first {worstExpectedFirst:G9}/{worstActualFirst:G9}, "
            + $"second {worstExpectedSecond:G9}/{worstActualSecond:G9}, "
            + $"final {worstExpectedFinal:G9}/{worstActualFinal:G9}, "
            + $"allowance {worstAllowance:G9}, FP32 budget {fp32SummationBudget:G9}, "
            + $"gradient scale {gradientScale:G9}.");
    }

    private static float Bf16Ulp(float left, float right)
    {
        float magnitude = MathF.Max(MathF.Abs(left), MathF.Abs(right));
        int exponent = (BitConverter.SingleToInt32Bits(magnitude) >> 23) & 255;
        return exponent == 0 ? MathF.ScaleB(1f, -133) : MathF.ScaleB(1f, exponent - 127 - 7);
    }

    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }

    private sealed record Snapshot(float[][] Outputs, float[][] FirstGradients, float[][] Gradients,
        string[] Kernels, long RetainedBytes);
}
