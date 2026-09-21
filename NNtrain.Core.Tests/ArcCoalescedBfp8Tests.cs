using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcCoalescedBfp8Tests
{
    [Theory]
    [InlineData(32, true)]
    [InlineData(128, true)]
    [InlineData(64, true)]
    [InlineData(32, false)]
    public void PublicationDispatchPreservesStorageAndCapabilityFallback(int block, bool xmx)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16, "SG16 XMX required.");
        float[] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32,
                options: new() { CoalescedBfp8Publication = enabled, XmxMatrices = xmx });
            var x = new Tensor(Enumerable.Range(0, 8193).Select(i => (i % 17 - 8) * .0019f).ToArray(), [8193]);
            x.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(block));
            long d2h = Tensor.ArcLane.D2HBytes;
            var y = x + x;
            Tensor.ArcLane.Synchronize();
            Assert.Equal(d2h, Tensor.ArcLane.D2HBytes);
            bool selected = enabled && xmx && block is 32 or 128;
            string kernel = !selected ? "resident_bfp8" : block == 32 ? "resident_bfp8_quad4_32" : "resident_bfp8_sg16_128";
            Assert.Contains(kernel, Tensor.ArcLane.KernelTimings.Keys);
            Assert.Equal(TensorDType.Bfp8, y.DType);
            return y.Data.ToArray();
        }
        Assert.Equal(Run(false).Select(BitConverter.SingleToInt32Bits), Run(true).Select(BitConverter.SingleToInt32Bits));
    }

    [Theory]
    [InlineData(32, 1)]
    [InlineData(32, 33)]
    [InlineData(32, 8191)]
    [InlineData(128, 31)]
    [InlineData(128, 257)]
    [InlineData(128, 8193)]
    public void CoalescedCodecMatchesPayloadScaleAndErrorFlag(int block, int count)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16, "SG16 XMX required.");
        using var lane = new ArcExecutionLane(); int groups = (count + block - 1) / block;
        foreach (int mode in new[] { 0, 1, 2, 3 })
        {
            float[] values = Enumerable.Range(0, count).Select(i => mode switch
            {
                0 => MathF.ScaleB((i % 31 - 15) * .0019f, i % 23 - 11),
                1 => 0f,
                2 => float.Epsilon,
                _ => i % 17 switch { 0 => float.NaN, 1 => float.PositiveInfinity, 2 => float.NegativeInfinity, 3 => float.MaxValue, _ => -0f }
            }).ToArray();
            using var source = lane.Upload(values); sbyte[]? expected = null; int[]? scalesExpected = null; int? flagExpected = null;
            foreach (bool candidate in new[] { false, true })
            {
                using var output = lane.AllocateBytes(count); using var scales = lane.Allocate(groups); using var status = lane.UploadRaw(new int[1]);
                long d2h = lane.D2HBytes, h2d = lane.H2DBytes;
                lane.Run(!candidate ? "resident_bfp8" : block == 32 ? "resident_bfp8_quad4_32" : "resident_bfp8_sg16_128",
                    groups * (candidate ? block == 32 ? 4L : 16L : 1L), candidate ? 256 : 0, source, output, scales, status, count, block);
                Assert.Equal(d2h, lane.D2HBytes); Assert.Equal(h2d, lane.H2DBytes);
                sbyte[] bytes = new sbyte[count]; float[] scale = new float[groups]; int[] flags = new int[1];
                lane.ReadRaw(output, bytes); lane.Read(scales, scale); lane.ReadRaw(status, flags);
                int[] bits = scale.Select(BitConverter.SingleToInt32Bits).ToArray();
                if (!candidate) { expected = bytes; scalesExpected = bits; flagExpected = flags[0]; }
                else { Assert.Equal(expected, bytes); Assert.Equal(scalesExpected, bits); Assert.Equal(flagExpected, flags[0]); }
            }
        }
    }
}
