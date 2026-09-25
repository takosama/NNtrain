using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcFusedLossHeadLogitRoundTests
{
    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }

    [Theory]
    [InlineData(137, 133, 65)]
    [InlineData(257, 129, 512)]
    [InlineData(129, 65, 2113)]
    public void PackedGemmEpilogueMatchesPostBiasRoundBitwise(int rows, int columns, int reduction)
    {
        RequireXmx();
        using var lane = new ArcExecutionLane();
        float[] source = Enumerable.Range(0, rows * reduction)
            .Select(i => MathF.Sin(i * .013f) * .031f).ToArray();
        float[] weights = Enumerable.Range(0, columns * reduction)
            .Select(i => MathF.Cos(i * .017f) * .023f).ToArray();
        float[] bias = Enumerable.Range(0, columns)
            .Select(i => MathF.Sin(i * .31f) * .11f).ToArray();
        // Infinity and NaN must pass through the same BF16 rounding rule; the
        // signed-zero column also checks the GEMM's post-bias ordering.
        bias[0] = float.PositiveInfinity;
        bias[1] = float.NegativeInfinity;
        bias[2] = BitConverter.Int32BitsToSingle(unchecked((int)0x7fc12345));
        bias[3] = -0f;
        using var input = lane.Upload(source);
        using var weight = lane.Upload(weights);
        using var biases = lane.Upload(bias);
        using var a = new ArcXmxStorageOperand(lane, input, null, TensorDType.Float32, source.Length);
        using var b = new ArcXmxStorageOperand(lane, weight, null, TensorDType.Float32, weights.Length);
        using var packedA = a.PackA(rows, reduction, false);
        using var packedB = b.PackB(columns, reduction, true);
        using var reference = lane.Allocate(rows * columns);
        using var fused = lane.Allocate(rows * columns);
        ArcXmxStorageOperand.GemmPanels(lane, packedA, packedB, reference,
            rows, columns, reduction, tb: true, bias: biases);
        lane.Run("round_bf16_values", rows * columns, 0, reference, rows * columns);
        ArcXmxStorageOperand.GemmPanels(lane, packedA, packedB, fused,
            rows, columns, reduction, tb: true, bias: biases, roundBf16Output: true);
        float[] expected = new float[rows * columns], actual = new float[rows * columns];
        lane.Read(reference, expected);
        lane.Read(fused, actual);
        Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void LossAndAccumulatedGradientsMatchWithoutSeparateRoundKernel(TensorPrecisionMode precision)
    {
        RequireXmx();
        const int rows = 640, width = 65, vocabulary = 133;
        Snapshot Run(bool fused)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                DirectXmxMatrices = true, PackedMatrixStorage = true, InlineMatrixGradient = true,
                MixedBackwardMatrixOperands = true, LossChunkRows = 512,
                FusedLossHeadLogitRound = fused });
            Tensor input = Make([rows, width], 1, precision);
            Tensor weight = Make([vocabulary, width], 2, precision);
            Tensor bias = Make([vocabulary], 3, precision);
            float[] losses = new float[2];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                int[] labels = Enumerable.Range(0, rows)
                    .Select(i => i % 17 == 0 ? -1 : (i * 13 + repeat) % vocabulary).ToArray();
                Tensor loss = input.ArcLinearCrossEntropy(weight, bias, labels, -1);
                losses[repeat] = loss.item();
                loss.BackwardAndRelease([repeat == 0 ? .375f : .625f]);
            }
            Tensor.ArcLane.Synchronize();
            bool separateRoundRan = Tensor.ArcLane.KernelTimings.ContainsKey("round_bf16_values");
            return new(losses, [input.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()], separateRoundRan);
        }

        Snapshot expected = Run(false), actual = Run(true);
        Assert.True(expected.SeparateRoundRan);
        Assert.False(actual.SeparateRoundRan);
        Assert.Equal(expected.Losses, actual.Losses);
        for (int i = 0; i < expected.Gradients.Length; i++)
            Assert.Equal(expected.Gradients[i], actual.Gradients[i]);
    }

    private static Tensor Make(int[] shape, int phase, TensorPrecisionMode precision)
    {
        int count = shape.Aggregate(1, (a, b) => checked(a * b));
        var tensor = new Tensor(Enumerable.Range(0, count)
            .Select(i => MathF.Sin(i * .013f + phase) * .031f).ToArray(), shape);
        tensor.ConvertStorageInPlace(precision.ToStorageDType(),
            precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
        return tensor;
    }

    private sealed record Snapshot(float[] Losses, float[][] Gradients, bool SeparateRoundRan);
}
