using NNtrain;
using NNtrain.Arc;
using Xunit;
using ArcBuffer = NNtrain.Arc.ArcExecutionLane.ArcBuffer;

public sealed class ArcMix816ReluDualGradientPackTests
{
    [Theory]
    [InlineData(false, 513, 129)]
    [InlineData(true, 513, 129)]
    [InlineData(false, 520, 136)]
    [InlineData(true, 521, 65)]
    public void GatedDualPanelsAndBiasPartialsMatchSeparatePacksBitwise(
        bool bfp8Gate, int rows, int cols)
    {
        RequireXmx();
        using var lane = new ArcExecutionLane();
        int count = checked(rows * cols), groups = (rows + 255) / 256;
        float[] gradients = Enumerable.Range(0, count)
            .Select(i => MathF.Cos(i * .019f) * .013f).ToArray();
        gradients[0] = -0f;
        gradients[cols + 1] = .01865f;
        using var dy = lane.Upload(gradients);

        const int blockSize = 32;
        ushort[] bf16Gate = Enumerable.Range(0, count)
            .Select(i => (ushort)(i % 5 switch
            {
                0 => 0xbf80, // negative
                1 => 0x0000, // positive zero
                2 => 0x3f80, // positive
                3 => 0x8000, // negative zero
                _ => 0x7fc0, // NaN leaves the gate open
            })).ToArray();
        sbyte[] bfp8Values = Enumerable.Range(0, count)
            .Select(i => (sbyte)(i % 5 switch { 0 => -7, 1 => 0, 2 => 6, 3 => 0, _ => 3 }))
            .ToArray();
        using var gateValues = bfp8Gate ? lane.UploadRaw(bfp8Values) : lane.UploadRaw(bf16Gate);
        using var scales = bfp8Gate ? lane.Upload(Enumerable.Repeat(.125f,
            (count + blockSize - 1) / blockSize).ToArray()) : null;
        using var gate = new ArcXmxStorageOperand(lane, gateValues, scales,
            bfp8Gate ? TensorDType.Bfp8 : TensorDType.BFloat16, count,
            bfp8Gate ? blockSize : 1);

        int normalElements = ((rows + 7) / 8) * ((cols + 15) / 16) * 128;
        int transposedElements = ((cols + 7) / 8) * ((rows + 15) / 16) * 128;
        using var expectedNormal = lane.AllocateBytes(normalElements * sizeof(ushort));
        using var expectedTransposed = lane.AllocateBytes(transposedElements * sizeof(ushort));
        using var expectedPartials = lane.Allocate(groups * cols);
        using var actualPartials = lane.Allocate(groups * cols);
        ArcBuffer gateScales = scales ?? gateValues;
        lane.Run2D("linear_relu_grad_pack_a_bias",
            ((cols + 31L) / 32) * 32, groups * 8L, 32, 8,
            dy, gateValues, gateScales, expectedNormal, expectedPartials,
            rows, cols, 0, gate.BlockSize, bfp8Gate ? 1 : 0, 0);
        lane.Run2D("linear_relu_grad_pack_a_transpose",
            ((cols + 31L) / 32) * 256, (rows + 31L) / 32, 256, 1,
            dy, gateValues, gateScales, expectedTransposed,
            cols, rows, 0, gate.BlockSize, bfp8Gate ? 1 : 0);
        long before = lane.D2HBytes;
        var dual = ArcXmxStorageOperand.PackReluDualGradientA(
            lane, dy, gate, actualPartials, rows, cols);
        using var actualNormal = dual.Normal;
        using var actualTransposed = dual.Transposed;
        Assert.Equal(before, lane.D2HBytes);

        ushort[] ReadPanel(ArcBuffer buffer, int elements)
        {
            var result = new ushort[elements];
            lane.ReadRaw(buffer, result);
            return result;
        }
        float[] ReadPartials(ArcBuffer buffer)
        {
            var result = new float[groups * cols];
            lane.Read(buffer, result);
            return result;
        }
        Assert.Equal(ReadPanel(expectedNormal, normalElements),
            ReadPanel(actualNormal, normalElements));
        Assert.Equal(ReadPanel(expectedTransposed, transposedElements),
            ReadPanel(actualTransposed, transposedElements));
        Assert.Equal(ReadPartials(expectedPartials).Select(BitConverter.SingleToInt32Bits),
            ReadPartials(actualPartials).Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix8_16, 513, true, true)]
    [InlineData(TensorPrecisionMode.Mix8_16, 129, true, false)]
    [InlineData(TensorPrecisionMode.Mix8_16, 513, false, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, 513, true, false)]
    public void TwoBackwardAccumulationsMatchSeparateReluPacksBitwise(
        TensorPrecisionMode precision, int rows, bool parallelReductions, bool eligible)
    {
        RequireXmx();
        const int inputWidth = 65, outputWidth = 129;
        (float[][] Outputs, float[][] Gradients) Run(bool dual)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: new ArcExecutionOptions
                {
                    PackedReluBackward = true,
                    FusedPackedReluBackward = true,
                    FusedReluPackBias = true,
                    ParallelReductions = parallelReductions,
                    Mix8_16ReluDualGradientPack = dual,
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
            var input = Make([rows, inputWidth], 1);
            var weight = Make([outputWidth, inputWidth], 2);
            var bias = Make([outputWidth], 3);
            var outputs = new float[2][];
            var lane = Tensor.ArcLane;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var result = input.LinearLastDim(weight, bias, applyRelu: true);
                outputs[repeat] = result.Data.ToArray();
                float[] seed = Enumerable.Range(0, rows * outputWidth)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .003f).ToArray();
                long beforeBackwardD2H = lane.D2HBytes;
                result.BackwardAndRelease(seed);
                lane.Synchronize();
                Assert.Equal(beforeBackwardD2H, lane.D2HBytes);
            }
            lane.CheckNumericStatus();
            Assert.Equal(dual && eligible,
                lane.KernelTimings.ContainsKey("linear_relu_grad_pack_a_bias_dual"));
            return (outputs,
                [input.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()]);
        }

        var baseline = Run(false);
        var candidate = Run(true);
        for (int i = 0; i < baseline.Outputs.Length; i++)
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
