using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcPackedReluBackwardTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var precision in new[] { TensorPrecisionMode.Float32, TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
            foreach (int rows in new[] { 129, 513, 2051, 4101 })
                yield return [precision, rows, rows != 2051, true];
        yield return [TensorPrecisionMode.Mix8_32, 513, true, false];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void PackedGatePreservesOutputAndTwoAccumulations(TensorPrecisionMode precision, int rows, bool parallel, bool xmx)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16, "SG16 XMX required.");
        const int ni = 65, no = 129;
        float[][] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() { PackedReluBackward = enabled, ParallelReductions = parallel, XmxMatrices = xmx });
            Tensor Make(int[] shape, int seed)
            {
                var t = new Tensor(Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(i => MathF.Sin(i * .031f + seed) * .037f).ToArray(), shape);
                t.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null); return t;
            }
            var input = Make([rows, ni], 1); var weight = Make([no, ni], 2); var bias = Make([no], 3);
            float[] values = []; var lane = Tensor.ArcLane; long retained = -1;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var output = input.LinearLastDim(weight, bias, applyRelu: true); values = output.Data.ToArray();
                long d2h = lane.D2HBytes;
                output.BackwardAndRelease(Enumerable.Range(0, rows * no).Select(i => MathF.Cos(i * .013f + repeat) * .01f).ToArray());
                lane.Synchronize(); Assert.Equal(d2h, lane.D2HBytes); Assert.Equal(0, lane.RetiredBytes);
                if (repeat == 0) retained = lane.AllocatedBytes; else Assert.Equal(retained, lane.AllocatedBytes);
            }
            bool selected = enabled && xmx && precision is TensorPrecisionMode.Mix16_32 or TensorPrecisionMode.Mix8_32;
            Assert.Equal(selected, lane.KernelTimings.ContainsKey("linear_relu_grad_packed"));
            return [values, input.Grad.ToArray(), weight.Grad.ToArray(), bias.Grad.ToArray()];
        }
        var expected = Run(false); var actual = Run(true);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i].Select(BitConverter.SingleToInt32Bits), actual[i].Select(BitConverter.SingleToInt32Bits));
    }
}
