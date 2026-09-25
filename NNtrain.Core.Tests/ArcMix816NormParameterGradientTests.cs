using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816NormParameterGradientTests
{
    [Theory]
    [InlineData(512, false, 0f)]
    [InlineData(513, true, .25f)]
    [InlineData(1025, false, .125f)]
    public void FusedParameterPartialsMatchSeparateGradientScan(
        int rows, bool aliasBranch, float dropout)
    {
        RequireXmx();
        Snapshot separate = Run(rows, 512, aliasBranch, dropout, fused: false);
        Snapshot fused = Run(rows, 512, aliasBranch, dropout, fused: true);

        Assert.Contains("norm_dx_row_sg16_w64_parallel_residual_bf16", separate.Kernels);
        Assert.Contains("gradient_rows", separate.Kernels);
        Assert.DoesNotContain("norm_dx_row_sg16_w64_parallel_residual_parameter_bf16",
            separate.Kernels);
        Assert.Contains("norm_dx_row_sg16_w64_parallel_residual_parameter_bf16",
            fused.Kernels);
        Assert.Contains("norm_parameter_4row_partials_256row", fused.Kernels);
        Assert.Contains("gradient_rows_finish", fused.Kernels);
        Assert.DoesNotContain("gradient_rows", fused.Kernels);
        Assert.Equal(separate.RetainedBytes, fused.RetainedBytes);

        for (int repeat = 0; repeat < separate.Outputs.Length; repeat++)
        {
            Assert.Equal(separate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                fused.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            for (int tensor = 0; tensor < separate.Gradients[repeat].Length; tensor++)
            {
                if (tensor < 2)
                    Assert.Equal(separate.Gradients[repeat][tensor]
                            .Select(BitConverter.SingleToInt32Bits),
                        fused.Gradients[repeat][tensor]
                            .Select(BitConverter.SingleToInt32Bits));
                else
                    AssertParameterGradientClose($"repeat {repeat} tensor {tensor}",
                        separate.Gradients[repeat][tensor], fused.Gradients[repeat][tensor],
                        rows, width: 512, passCount: repeat + 1, gamma: tensor == 2);
            }
        }
    }

    [Fact]
    public void UnsupportedWidthLeavesSeparateGradientScanUnchanged()
    {
        RequireXmx();
        Snapshot separate = Run(513, 65, aliasBranch: true, dropout: .125f,
            fused: false);
        Snapshot enabled = Run(513, 65, aliasBranch: true, dropout: .125f,
            fused: true);

        Assert.Contains("gradient_rows", enabled.Kernels);
        Assert.DoesNotContain("norm_parameter_4row_partials_256row", enabled.Kernels);
        Assert.Equal(separate.RetainedBytes, enabled.RetainedBytes);
        for (int repeat = 0; repeat < separate.Outputs.Length; repeat++)
        {
            Assert.Equal(separate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                enabled.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            for (int tensor = 0; tensor < separate.Gradients[repeat].Length; tensor++)
                Assert.Equal(separate.Gradients[repeat][tensor].Select(BitConverter.SingleToInt32Bits),
                    enabled.Gradients[repeat][tensor].Select(BitConverter.SingleToInt32Bits));
        }
    }

    [Fact]
    public void FusedParameterOptionDoesNotChangeMix8_32()
    {
        RequireXmx();
        Snapshot separate = Run(512, 512, aliasBranch: false, dropout: .125f,
            fused: false, precision: TensorPrecisionMode.Mix8_32);
        Snapshot enabled = Run(512, 512, aliasBranch: false, dropout: .125f,
            fused: true, precision: TensorPrecisionMode.Mix8_32);

        Assert.DoesNotContain("norm_dx_row_sg16_w64_parallel_residual_parameter_bf16",
            enabled.Kernels);
        Assert.DoesNotContain("norm_parameter_4row_partials_256row", enabled.Kernels);
        Assert.Equal(separate.RetainedBytes, enabled.RetainedBytes);
        for (int repeat = 0; repeat < separate.Outputs.Length; repeat++)
        {
            Assert.Equal(separate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                enabled.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            for (int tensor = 0; tensor < separate.Gradients[repeat].Length; tensor++)
                Assert.Equal(separate.Gradients[repeat][tensor].Select(BitConverter.SingleToInt32Bits),
                    enabled.Gradients[repeat][tensor].Select(BitConverter.SingleToInt32Bits));
        }
    }

    private static Snapshot Run(int rows, int width, bool aliasBranch, float dropout,
        bool fused, TensorPrecisionMode precision = TensorPrecisionMode.Mix8_16)
    {
        using var scope = Tensor.BeginArcExecution(precision: precision,
            options: new ArcExecutionOptions
            {
                Mix8_16Bf16Activations = true,
                Mix8_16DirectBf16NormOutput = true,
                Mix8_16FusedNormResidualBackward = true,
                Mix8_16ParallelNormReduction = true,
                Mix8_16FusedNormParameterGradients = fused,
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

        Tensor residual = Make(rows * width, [rows, width], .37f, .02f,
            precision == TensorPrecisionMode.Mix8_16
                ? TensorDType.BFloat16 : TensorDType.Bfp8);
        Tensor branch = aliasBranch ? residual : Make(rows * width, [rows, width],
            .23f, -.04f, TensorDType.Bfp8);
        Tensor gamma = Make(width, [width], .16f, 1f, TensorDType.Bfp8);
        Tensor beta = Make(width, [width], .02f, 0f, TensorDType.Bfp8);
        float[] seed = Enumerable.Range(0, rows * width)
            .Select(i => MathF.Cos(i * .0167f) * .023f).ToArray();
        var outputs = new float[2][];
        var gradients = new float[2][][];
        long retained = -1;
        for (int repeat = 0; repeat < 2; repeat++)
        {
            Tensor value = residual.AddDropoutLayerNormLastDim(branch, gamma, beta,
                dropout, new Random(123 + repeat));
            outputs[repeat] = value.Data.ToArray();
            value.BackwardAndRelease(seed);
            Tensor.ArcLane.Synchronize();
            gradients[repeat] =
                [residual.Grad.ToArray(), branch.Grad.ToArray(),
                 gamma.Grad.ToArray(), beta.Grad.ToArray()];
            if (retained >= 0) Assert.Equal(retained, Tensor.ArcLane.AllocatedBytes);
            retained = Tensor.ArcLane.AllocatedBytes;
        }
        Tensor.ArcLane.CheckNumericStatus();
        return new(outputs, gradients, Tensor.ArcLane.KernelTimings.Keys.ToArray(),
            retained);
    }

    private static void AssertParameterGradientClose(string label, float[] expected,
        float[] actual, int rows, int width, int passCount, bool gamma)
    {
        Assert.Equal(expected.Length, actual.Length);
        // The seed magnitude is at most .023. BF16 publication may round it
        // upward by one relative step. A standardized row value is bounded by
        // sqrt(width); the extra unit covers FP32 mean/variance rounding.
        double contributionBound = .023d * (1d + Math.ScaleB(1d, -7))
            * (gamma ? Math.Sqrt(width) + 1d : 1d);
        // Four FP32 operations per contribution is a conservative budget for
        // the changed four-row/256-row addition tree and gamma products. Each
        // pass contributes rows terms. This is a comparison budget for two
        // valid reduction orders, not a guarantee for arbitrary inputs.
        double fp32Budget = passCount * rows * 4d * Math.ScaleB(1d, -23)
            * contributionBound;
        double squaredError = 0, squaredReference = 0;
        double largestRatio = 0;
        int worstIndex = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            double error = Math.Abs((double)expected[i] - actual[i]);
            squaredError += error * error;
            squaredReference += (double)expected[i] * expected[i];
            float magnitude = MathF.Max(MathF.Abs(expected[i]), MathF.Abs(actual[i]));
            int exponent = (BitConverter.SingleToInt32Bits(magnitude) >> 23) & 255;
            float ulp = exponent == 0 ? MathF.ScaleB(1f, -133)
                : MathF.ScaleB(1f, exponent - 127 - 7);
            // First-pass and repeated accumulation each publish retained BF16
            // gradients, allowing up to three one-ULP boundary crossings.
            double ratio = error / (3d * ulp + fp32Budget);
            if (ratio > largestRatio) { largestRatio = ratio; worstIndex = i; }
        }
        double relativeRms = Math.Sqrt(squaredError / Math.Max(squaredReference, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"{label}: relative RMS={relativeRms:G6}, max BF16 allowance ratio="
            + $"{largestRatio:G6} at {worstIndex}, FP32 budget={fp32Budget:G9}");
        Assert.InRange(relativeRms, 0d, 5e-4d);
        Assert.True(largestRatio <= 1d,
            $"{label}: max BF16 allowance ratio {largestRatio:G6} at {worstIndex}.");
    }

    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }

    private sealed record Snapshot(float[][] Outputs, float[][][] Gradients,
        string[] Kernels, long RetainedBytes);
}
