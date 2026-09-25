using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816NormBoundaryTests
{
    [Theory]
    [InlineData(512, 512, false)]
    [InlineData(129, 64, true)]
    [InlineData(129, 65, false)]
    public void BackwardScatterAndDirectBf16OutputPreservePublishedBits(
        int rows, int width, bool aliasBranch)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");

        (int[][] Outputs, int[][] Gradients, string[] Kernels, long RetainedBytes)
            Run(bool fusedBackward, bool directOutput)
        {
            using var scope = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16Bf16Activations = true,
                    Mix8_16FusedNormResidualBackward = fusedBackward,
                    Mix8_16DirectBf16NormOutput = directOutput,
                    Mix8_16ParallelNormReduction = false,
                    FusedPackedResidualNorm = false,
                    BlockResidualNorm = false,
                    OrderedTiledNorm = true,
                    PackedNormInput = true,
                    FusedNormGradient = true,
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
                TensorDType.BFloat16);
            Tensor branch = aliasBranch ? residual : Make(rows * width, [rows, width],
                .23f, -.04f, TensorDType.Bfp8);
            Tensor gamma = Make(width, [width], .16f, 1f, TensorDType.Bfp8);
            Tensor beta = Make(width, [width], .02f, 0f, TensorDType.Bfp8);
            float[] upstream = Enumerable.Range(0, rows * width)
                .Select(i => MathF.Cos(i * .0167f) * .023f).ToArray();
            int[][] outputs = new int[2][];
            long retainedBytes = -1;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Tensor value = residual.AddDropoutLayerNormLastDim(branch, gamma, beta,
                    .125f, new Random(123 + repeat));
                Assert.Equal(TensorDType.BFloat16, value.DType);
                Assert.Equal(rows * width * sizeof(ushort), value.StorageByteLength);
                outputs[repeat] = value.Data.ToArray()
                    .Select(BitConverter.SingleToInt32Bits).ToArray();
                value.BackwardAndRelease(upstream);
                Tensor.ArcLane.Synchronize();
                if (retainedBytes >= 0)
                    Assert.Equal(retainedBytes, Tensor.ArcLane.AllocatedBytes);
                retainedBytes = Tensor.ArcLane.AllocatedBytes;
            }
            Tensor.ArcLane.CheckNumericStatus();
            return (outputs,
                [residual.Grad.Select(BitConverter.SingleToInt32Bits).ToArray(),
                 branch.Grad.Select(BitConverter.SingleToInt32Bits).ToArray(),
                 gamma.Grad.Select(BitConverter.SingleToInt32Bits).ToArray(),
                 beta.Grad.Select(BitConverter.SingleToInt32Bits).ToArray()],
                Tensor.ArcLane.KernelTimings.Keys.ToArray(), retainedBytes);
        }

        var baseline = Run(false, false);
        foreach ((bool fusedBackward, bool directOutput) in
            new[] { (true, false), (false, true), (true, true) })
        {
            var candidate = Run(fusedBackward, directOutput);
            for (int repeat = 0; repeat < baseline.Outputs.Length; repeat++)
                Assert.Equal(baseline.Outputs[repeat], candidate.Outputs[repeat]);
            for (int index = 0; index < baseline.Gradients.Length; index++)
                Assert.Equal(baseline.Gradients[index], candidate.Gradients[index]);
            Assert.Equal(baseline.RetainedBytes, candidate.RetainedBytes);
            Assert.Equal(fusedBackward,
                candidate.Kernels.Contains("norm_dx_row_sg16_w64_residual_accumulate_bf16"));
            Assert.Equal(directOutput,
                candidate.Kernels.Contains("norm_row_sg16_w64_direct_bf16"));
            Assert.Equal(!directOutput, candidate.Kernels.Contains("resident_bf16"));
        }
    }
}
