using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcFusedPackedResidualNormTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, .1f, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, .1f, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, .1f, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, .75f, true)]
    public void PackedResidualFusionPreservesOutputAndAccumulatedGradients(
        TensorPrecisionMode precision, float dropout, bool aliasBranch)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "The fused path requires SG16 Intel XMX.");

        (float[] Output, float[][] Gradients, long Peak, string[] Kernels) Run(bool fused)
        {
            using var scope = Tensor.BeginArcExecution(precision: precision, options: new()
            {
                FusedPackedResidualNorm = fused,
                BlockResidualNorm = false,
                OrderedTiledNorm = true,
                PackedNormInput = true,
                ParallelReductions = true
            });
            const int rows = 512, width = 512;
            Tensor Make(int count, int[] shape, float amplitude, float offset)
            {
                var tensor = new Tensor(Enumerable.Range(0, count)
                    .Select(i => MathF.Sin(i * .0137f) * amplitude + offset).ToArray(), shape);
                tensor.ConvertStorageInPlace(precision.ToStorageDType(),
                    precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
                return tensor;
            }

            var x = Make(rows * width, [rows, width], .37f, .02f);
            var branch = aliasBranch ? x : Make(rows * width, [rows, width], .23f, -.04f);
            var gamma = Make(width, [width], .16f, 1f);
            var beta = Make(width, [width], .02f, 0f);
            float[] upstream = Enumerable.Range(0, rows * width)
                .Select(i => MathF.Cos(i * .0167f) * .023f).ToArray();
            float[] output = [];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var y = x.AddDropoutLayerNormLastDim(branch, gamma, beta, dropout, new Random(123 + repeat));
                if (repeat == 0) output = y.Data.ToArray();
                y.BackwardAndRelease(upstream);
            }
            Tensor.ArcLane.Synchronize();
            return (output,
                [x.Grad.ToArray(), branch.Grad.ToArray(), gamma.Grad.ToArray(), beta.Grad.ToArray()],
                Tensor.ArcLane.PeakAllocatedBytes,
                Tensor.ArcLane.KernelTimings.Keys.ToArray());
        }

        var expected = Run(false);
        var actual = Run(true);
        Assert.Contains("norm_packed_residual_row_sg16_w64", actual.Kernels);
        Assert.DoesNotContain("norm_packed_residual_input", actual.Kernels);
        Assert.Equal(expected.Output, actual.Output);
        for (int i = 0; i < expected.Gradients.Length; i++)
            Assert.Equal(expected.Gradients[i], actual.Gradients[i]);
    }

    [Fact]
    public void TailRowsAndSpecialFp32ValuesKeepPublishedBits()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "The fused path requires SG16 Intel XMX.");

        const int rows = 513, width = 512;
        float negativeZero = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
        float[] values = Enumerable.Range(0, rows * width)
            .Select(i => i == width + 7 ? float.NaN
                : i % 97 == 0 ? negativeZero : MathF.Sin(i * .011f) * .23f).ToArray();
        float[] branchValues = Enumerable.Range(0, rows * width)
            .Select(i => i % 73 == 0 ? negativeZero : MathF.Cos(i * .013f) * .17f).ToArray();
        int[] Run(bool fused)
        {
            using var scope = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Float32,
                options: new() { FusedPackedResidualNorm = fused, BlockResidualNorm = false,
                    OrderedTiledNorm = true, PackedNormInput = true, ParallelReductions = true });
            var x = new Tensor((float[])values.Clone(), [rows, width]);
            var branch = new Tensor((float[])branchValues.Clone(), [rows, width]);
            var gamma = new Tensor(Enumerable.Repeat(1f, width).ToArray(), [width]);
            var beta = new Tensor(new float[width], [width]);
            var y = x.AddDropoutLayerNormLastDim(branch, gamma, beta, .125f, new Random(96));
            return y.Data.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray();
        }

        Assert.Equal(Run(false), Run(true));
    }
}
