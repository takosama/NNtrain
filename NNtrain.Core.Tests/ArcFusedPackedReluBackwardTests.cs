using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcFusedPackedReluBackwardTests
{
    private const long ForcedPanelBudget = 1L * 1024 * 1024;

    public static IEnumerable<object[]> Cases()
    {
        foreach (var precision in new[] { TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
            foreach (int rows in new[] { 129, 513, 2051, 4101 })
                yield return [precision, rows, rows != 2051, 32];
        yield return [TensorPrecisionMode.Mix8_32, 513, true, 128];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void FusedPanelsMatchEncodedGradientBitwiseAcrossTwoAccumulations(
        TensorPrecisionMode precision, int rows, bool parallel, int block)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16, "SG16 XMX is required.");
        const int inputWidth = 65, outputWidth = 129;

        float[][] Run(bool fused, bool packBias)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: new() { PackedReluBackward = true, FusedPackedReluBackward = fused,
                    FusedReluPackBias = packBias, ParallelReductions = parallel });
            Tensor Make(int[] shape, int seed)
            {
                var data = Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b))
                    .Select(i => MathF.Sin(i * .031f + seed) * .037f).ToArray();
                var tensor = new Tensor(data, shape);
                tensor.ConvertStorageInPlace(precision.ToStorageDType(),
                    precision == TensorPrecisionMode.Mix8_32
                        ? Bfp8QuantizationDescriptor.Block(block) : null);
                return tensor;
            }
            var input = Make([rows, inputWidth], 1);
            var weight = Make([outputWidth, inputWidth], 2);
            var bias = Make([outputWidth], 3);
            var lane = Tensor.ArcLane;
            float[] outputValues = [];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var output = input.LinearLastDim(weight, bias, applyRelu: true);
                outputValues = output.Data.ToArray();
                long beforeBackwardD2H = lane.D2HBytes;
                output.BackwardAndRelease(Enumerable.Range(0, rows * outputWidth)
                    .Select(i => MathF.Cos(i * .013f + repeat) * .01f).ToArray());
                lane.Synchronize();
                Assert.Equal(beforeBackwardD2H, lane.D2HBytes);
                Assert.Equal(0, lane.RetiredBytes);
            }
            bool packedBias = fused && packBias && parallel && rows >= 512;
            Assert.Equal(fused && !packedBias,
                lane.KernelTimings.ContainsKey("linear_relu_grad_pack_a_vec4"));
            Assert.Equal(packedBias,
                lane.KernelTimings.ContainsKey("linear_relu_grad_pack_a_bias"));
            Assert.Equal(fused, lane.KernelTimings.ContainsKey("linear_relu_grad_pack_a_transpose"));
            Assert.Equal(!fused, lane.KernelTimings.ContainsKey("linear_relu_grad_packed"));
            return [outputValues, input.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()];
        }

        var baseline = Run(false, false);
        var candidate = Run(true, false);
        for (int i = 0; i < baseline.Length; i++)
            Assert.Equal(baseline[i].Select(BitConverter.SingleToInt32Bits),
                candidate[i].Select(BitConverter.SingleToInt32Bits));
        if (parallel && rows >= 512)
        {
            var packedBias = Run(true, true);
            for (int i = 0; i < baseline.Length; i++)
                Assert.Equal(baseline[i].Select(BitConverter.SingleToInt32Bits),
                    packedBias[i].Select(BitConverter.SingleToInt32Bits));
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Mix16_32, 32)]
    [InlineData(TensorPrecisionMode.Mix8_32, 32)]
    [InlineData(TensorPrecisionMode.Mix8_32, 128)]
    public void ForcedStreamedPanelsMatchEncodedGradientBitwise(
        TensorPrecisionMode precision, int block)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16, "SG16 XMX is required.");
        using var execution = Tensor.BeginArcExecution(precision: precision,
            options: new() { PackedReluBackward = true, FusedPackedReluBackward = true,
                StreamedXmxMatrices = true, ParallelWeightGradients = true });
        const int rows = 4101, inputWidth = 65, outputWidth = 129;
        var descriptor = precision == TensorPrecisionMode.Mix8_32
            ? Bfp8QuantizationDescriptor.Block(block) : null;
        Tensor Make(float[] values, int[] shape)
        {
            var tensor = new Tensor(values, shape);
            tensor.ConvertStorageInPlace(precision.ToStorageDType(), descriptor);
            return tensor;
        }
        var gate = Make(Enumerable.Range(0, rows * outputWidth)
            .Select(i => i % 7 < 3 ? 0f : MathF.Abs(MathF.Sin(i * .019f)) * .1f)
            .ToArray(), [rows, outputWidth]);
        var input = Make(Enumerable.Range(0, rows * inputWidth)
            .Select(i => MathF.Sin(i * .009f) * .02f).ToArray(), [rows, inputWidth]);
        var weight = Make(Enumerable.Range(0, outputWidth * inputWidth)
            .Select(i => MathF.Cos(i * .021f) * .02f).ToArray(), [outputWidth, inputWidth]);
        var gradientValues = Enumerable.Range(0, rows * outputWidth)
            .Select(i => MathF.Cos(i * .013f) * .01f).ToArray();
        var lane = Tensor.ArcLane;
        using var gateStorage = gate.ArcMatrixOperand();
        using var inputStorage = input.ArcMatrixOperand();
        using var weightStorage = weight.ArcMatrixOperand();
        using var dy = lane.Upload(gradientValues);
        using var encoded = lane.AllocateBytes(checked(rows * outputWidth * 2));
        lane.Run("linear_relu_grad_packed", rows * outputWidth, 0,
            dy, gateStorage.Value, gateStorage.Scales ?? gateStorage.Value,
            encoded, rows * outputWidth, gateStorage.BlockSize,
            gateStorage.DType == TensorDType.Bfp8 ? 1 : 0);
        using var baselineOperand = new ArcXmxStorageOperand(lane, encoded, null,
            TensorDType.BFloat16, rows * outputWidth);
        using var baselineDx = lane.Upload(new float[rows * inputWidth]);
        using var candidateDx = lane.Upload(new float[rows * inputWidth]);
        using var baselineDw = lane.Upload(new float[outputWidth * inputWidth]);
        using var candidateDw = lane.Upload(new float[outputWidth * inputWidth]);
        int groups = (rows + 255) / 256;
        using var baselineBiasParts = lane.Allocate(groups * outputWidth);
        using var candidateBiasParts = lane.Allocate(groups * outputWidth);
        using var baselineDb = lane.Upload(new float[outputWidth]);
        using var candidateDb = lane.Upload(new float[outputWidth]);
        lane.Run2D("gradient_rows_packed_bf16", ((outputWidth + 31L) / 32) * 32,
            groups * 8L, 32, 8, encoded, baselineBiasParts, rows, outputWidth);
        Assert.True(ArcXmxStorageOperand.TryGemmStreamed(lane, baselineOperand, weightStorage,
            baselineDx, rows, inputWidth, outputWidth, accumulate: true,
            panelBudget: ForcedPanelBudget));
        ArcXmxStorageOperand.GemmReluGradient(lane, dy, gateStorage, weightStorage,
            candidateDx, rows, inputWidth, outputWidth, transpose: false,
            accumulate: true, panelBudget: ForcedPanelBudget,
            biasPartials: candidateBiasParts);
        Assert.True(ArcXmxStorageOperand.TryGemmStreamed(lane, baselineOperand, inputStorage,
            baselineDw, outputWidth, inputWidth, rows, ta: true, accumulate: true,
            panelBudget: ForcedPanelBudget));
        ArcXmxStorageOperand.GemmReluGradient(lane, dy, gateStorage, inputStorage,
            candidateDw, outputWidth, inputWidth, rows, transpose: true,
            accumulate: true, panelBudget: ForcedPanelBudget);
        lane.Run("gradient_rows_finish", outputWidth, 0, baselineBiasParts,
            baselineDb, baselineDb, groups, outputWidth, 0);
        lane.Run("gradient_rows_finish", outputWidth, 0, candidateBiasParts,
            candidateDb, candidateDb, groups, outputWidth, 0);
        foreach (var (expected, actual, length) in new[] {
            (baselineDx, candidateDx, rows * inputWidth),
            (baselineDw, candidateDw, outputWidth * inputWidth),
            (baselineDb, candidateDb, outputWidth) })
        {
            var expectedValues = new float[length];
            var actualValues = new float[length];
            lane.Read(expected, expectedValues);
            lane.Read(actual, actualValues);
            Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits),
                actualValues.Select(BitConverter.SingleToInt32Bits));
        }
    }
}
