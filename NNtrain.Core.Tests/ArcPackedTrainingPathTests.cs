using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcPackedTrainingPathTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var precision in new[] { TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
        foreach (bool relu in new[] { false, true })
        {
            // 512+128 is wholly eligible; 512+512+7 must fall back as a whole.
            yield return [precision, relu, 640, false, true];
            yield return [precision, relu, 1031, false, true];
        }
        yield return [TensorPrecisionMode.Mix16_32, true, 640, true, true];
        yield return [TensorPrecisionMode.Mix8_32, true, 640, true, true];
        yield return [TensorPrecisionMode.Mix16_32, true, 640, false, false];
        yield return [TensorPrecisionMode.Float32, true, 640, false, true];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PackedStorageAndInlineGradientMatchEncodedLinearAndLossHead(
        TensorPrecisionMode precision, bool relu, int rows, bool allIgnored, bool mixedBackward)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
        const int width = 65, hidden = 129, vocabulary = 133;
        Snapshot Run(bool packed)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                DirectXmxMatrices = true, PackedMatrixStorage = packed, InlineMatrixGradient = packed,
                MixedBackwardMatrixOperands = mixedBackward, LossChunkRows = 512, DetailedProfiling = true });
            var x = Make([rows, width], 1, precision);
            var linearWeight = Make([hidden, width], 2, precision);
            var linearBias = Make([hidden], 3, precision);
            var headWeight = Make([vocabulary, hidden], 4, precision);
            var headBias = Make([vocabulary], 5, precision);
            var losses = new float[2];
            var lane = Tensor.ArcLane;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                lane.DetailedProfiler!.Phase = "linear-forward";
                Tensor hiddenValues = x.LinearLastDim(linearWeight, linearBias, applyRelu: relu);
                int[] labels = Enumerable.Range(0, rows)
                    .Select(i => allIgnored || i % 17 == 0 ? -1 : (i * 13 + repeat) % vocabulary).ToArray();
                lane.DetailedProfiler.Phase = "loss-forward";
                Tensor loss = hiddenValues.ArcLinearCrossEntropy(headWeight, headBias, labels, -1);
                losses[repeat] = loss.item();
                Assert.Equal(TensorDType.Float32, loss.DType);
                long downloads = lane.D2HBytes;
                lane.DetailedProfiler.Phase = "backward";
                loss.BackwardAndRelease([repeat == 0 ? .375f : .625f]);
                Assert.Equal(downloads, lane.D2HBytes);
            }
            lane.Synchronize();
            bool eligibleHead = packed && mixedBackward && precision != TensorPrecisionMode.Float32
                && (rows % 512 == 0 || rows % 512 >= 128);
            long headStorageLaunches = lane.DetailedProfiler!.Snapshot()
                .Where(e => e.Kind == "gpu-kernel" && e.Detail.StartsWith("loss-forward/xmx_storage_pack_", StringComparison.Ordinal)
                    && (e.Detail.EndsWith("_bf16", StringComparison.Ordinal) || e.Detail.EndsWith("_bfp8", StringComparison.Ordinal)))
                .Sum(e => e.Count);
            if (eligibleHead) Assert.True(headStorageLaunches > 0, "The eligible loss head never used packed resident storage.");
            else Assert.Equal(0, headStorageLaunches);
            if (packed) Assert.DoesNotContain("matrix_gradient_bf16", lane.KernelTimings.Keys);
            var gradients = new[] { x, linearWeight, linearBias, headWeight, headBias }
                .Select(t => t.Grad.ToArray()).ToArray();
            Assert.All(gradients.SelectMany(g => g), value => Assert.True(float.IsFinite(value)));
            if (allIgnored)
            {
                Assert.All(losses, value => Assert.Equal(0f, value));
                Assert.All(gradients.SelectMany(g => g), value => Assert.Equal(0f, value));
            }
            return new(losses, gradients);
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Losses, actual.Losses);
        for (int parameter = 0; parameter < expected.Gradients.Length; parameter++)
            Assert.Equal(expected.Gradients[parameter], actual.Gradients[parameter]);
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

    private sealed record Snapshot(float[] Losses, float[][] Gradients);
}
