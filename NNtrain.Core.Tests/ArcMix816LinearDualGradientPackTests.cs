using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816LinearDualGradientPackTests
{
    [Fact]
    public void NormalAndTransposedPanelsMatchSeparatePackingBitwise()
    {
        RequireXmx();
        const int rows = 513, columns = 128;
        float[] values = Enumerable.Range(0, rows * columns)
            .Select(i => MathF.Sin(i * .019f) * .031f).ToArray();
        values[0] = -0f;
        values[columns + 17] = float.PositiveInfinity;
        values[2 * columns + 31] = float.NaN;
        using var lane = new ArcExecutionLane();
        using var source = lane.Upload(values);
        using var operand = new ArcXmxStorageOperand(lane, source, null,
            TensorDType.Float32, values.Length);
        using var separateNormal = operand.PackA(rows, columns, false);
        using var separateTransposed = operand.PackA(columns, rows, true);
        var dual = ArcXmxStorageOperand.PackDualGradientA(lane, source, rows, columns);
        using var normal = dual.Normal;
        using var transposed = dual.Transposed;

        ushort[] Read(ArcExecutionLane.ArcBuffer panel)
        {
            var bits = new ushort[checked((int)(panel.ByteLength / sizeof(ushort)))];
            lane.ReadRaw(panel, bits);
            return bits;
        }
        Assert.Equal(Read(separateNormal), Read(normal));
        Assert.Equal(Read(separateTransposed), Read(transposed));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix8_16, 513, false, true)]
    [InlineData(TensorPrecisionMode.Mix8_16, 127, false, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, 513, true, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, 513, false, false)]
    public void TwoAccumulationsAndFallbackMatchSeparatePacksBitwise(
        TensorPrecisionMode precision, int rows, bool relu, bool expectedDual)
    {
        RequireXmx();
        const int columns = 128, width = 128;
        (float[][] Outputs, float[][] Gradients) Run(bool dual)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: new ArcExecutionOptions
                {
                    Mix8_16LinearDualGradientPack = dual,
                    Mix8_16ReluDualGradientPack = false,
                    Mix8_16BiasOnlyGradientReduction = false,
                    ExpandedXmxTiles = true,
                    FusedPackedReluBackward = true,
                });
            Tensor Make(int[] shape, int phase)
            {
                var tensor = new Tensor(Enumerable.Range(0,
                        shape.Aggregate(1, (a, b) => checked(a * b)))
                    .Select(i => MathF.Sin(i * .013f + phase) * .031f).ToArray(), shape);
                tensor.ConvertStorageInPlace(TensorDType.Bfp8,
                    Bfp8QuantizationDescriptor.Block(32));
                return tensor;
            }
            var x = Make([rows, width], 1);
            var weight = Make([columns, width], 2);
            var bias = Make([columns], 3);
            var outputs = new float[2][];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var y = x.LinearLastDim(weight, bias, applyRelu: relu);
                outputs[repeat] = y.Data.ToArray();
                float[] seed = Enumerable.Range(0, rows * columns)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .003f).ToArray();
                y.BackwardAndRelease(seed);
            }
            Tensor.ArcLane.Synchronize();
            Tensor.ArcLane.CheckNumericStatus();
            Assert.Equal(dual && expectedDual,
                Tensor.ArcLane.KernelTimings.ContainsKey("xmx_storage_pack_a_dual_f32"));
            return (outputs,
                [x.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()]);
        }

        var baseline = Run(false);
        var candidate = Run(true);
        for (int i = 0; i < 2; i++)
            Assert.Equal(baseline.Outputs[i].Select(BitConverter.SingleToInt32Bits),
                candidate.Outputs[i].Select(BitConverter.SingleToInt32Bits));
        for (int i = 0; i < baseline.Gradients.Length; i++)
            Assert.Equal(baseline.Gradients[i].Select(BitConverter.SingleToInt32Bits),
                candidate.Gradients[i].Select(BitConverter.SingleToInt32Bits));
    }

    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }
}
