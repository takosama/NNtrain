using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcFusedDualGradientPackTests
{
    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }

    [Theory]
    [InlineData(137, 133)]
    [InlineData(512, 257)]
    [InlineData(513, 65)]
    public void BothPanelsMatchIndependentFp32PacksBitwise(int rows, int cols)
    {
        RequireXmx();
        using var lane = new ArcExecutionLane();
        float[] source = Enumerable.Range(0, rows * cols)
            .Select(i => MathF.Sin(i * .041f) * .031f).ToArray();
        float[] special = [0f, -0f, float.PositiveInfinity, float.NegativeInfinity,
            BitConverter.Int32BitsToSingle(unchecked((int)0x7f800001)),
            BitConverter.Int32BitsToSingle(unchecked((int)0xff800001))];
        for (int i = 0; i < special.Length; i++) source[i * cols + (i * 17) % cols] = special[i];
        using var gradient = lane.Upload(source);
        using var operand = new ArcXmxStorageOperand(lane, gradient, null, TensorDType.Float32, source.Length);
        using var expectedNormal = operand.PackA(rows, cols, false);
        using var expectedTransposed = operand.PackA(cols, rows, true);
        long downloads = lane.D2HBytes;
        var dual = ArcXmxStorageOperand.PackDualGradientA(lane, gradient, rows, cols);
        using var actualNormal = dual.Normal;
        using var actualTransposed = dual.Transposed;
        Assert.Equal(downloads, lane.D2HBytes);
        ushort[] normal = new ushort[((rows + 7) / 8) * ((cols + 15) / 16) * 128];
        ushort[] transposed = new ushort[((cols + 7) / 8) * ((rows + 15) / 16) * 128];
        ushort[] referenceNormal = new ushort[normal.Length];
        ushort[] referenceTransposed = new ushort[transposed.Length];
        lane.ReadRaw(actualNormal, normal);
        lane.ReadRaw(actualTransposed, transposed);
        lane.ReadRaw(expectedNormal, referenceNormal);
        lane.ReadRaw(expectedTransposed, referenceTransposed);
        Assert.Equal(referenceNormal, normal);
        Assert.Equal(referenceTransposed, transposed);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix16_32)]
    [InlineData(TensorPrecisionMode.Mix8_32)]
    public void LossHeadBackwardAndAccumulationMatchIndependentPanels(TensorPrecisionMode precision)
    {
        RequireXmx();
        const int rows = 640, width = 65, vocabulary = 133;
        (float[] Loss, float[][] Gradients) Run(bool fused)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                DirectXmxMatrices = true, PackedMatrixStorage = true, InlineMatrixGradient = true,
                MixedBackwardMatrixOperands = true, LossChunkRows = 512,
                FusedDualGradientPack = fused });
            var x = Make([rows, width], 1, precision);
            var weight = Make([vocabulary, width], 2, precision);
            var bias = Make([vocabulary], 3, precision);
            float[] losses = new float[2];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                int[] labels = Enumerable.Range(0, rows)
                    .Select(i => i % 17 == 0 ? -1 : (i * 13 + repeat) % vocabulary).ToArray();
                Tensor loss = x.ArcLinearCrossEntropy(weight, bias, labels, -1);
                losses[repeat] = loss.item();
                loss.BackwardAndRelease([repeat == 0 ? .375f : .625f]);
            }
            return (losses, [x.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()]);
        }

        var expected = Run(false);
        var actual = Run(true);
        Assert.Equal(expected.Loss, actual.Loss);
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
}
