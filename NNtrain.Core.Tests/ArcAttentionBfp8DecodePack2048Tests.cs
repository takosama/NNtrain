using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcAttentionBfp8DecodePack2048Tests
{
    [Fact]
    public void FusedDecodeAndQkPanelsMatchSeparateKernelsBitForBit()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");

        const int batch = 2, sequence = 2048, heads = 3, width = heads * 32, block = 32;
        int n = checked(batch * sequence * 3 * width);
        int panelElements = checked(batch * heads * sequence * 32);
        var payload = new sbyte[n];
        for (int i = 0; i < n; i++) payload[i] = (sbyte)((i * 19 % 247) - 123);
        var scales = new float[n / block];
        for (int i = 0; i < scales.Length; i++) scales[i] = MathF.ScaleB(1f + i % 7 * .125f, i % 9 - 8);
        scales[3] = float.NaN;
        scales[7] = float.PositiveInfinity;
        scales[11] = -0f;

        using var lane = new ArcExecutionLane();
        using var source = lane.UploadRaw(payload);
        using var scale = lane.Upload(scales);
        using var separate = lane.Allocate(n);
        using var combined = lane.Allocate(n);
        using var separateQ = lane.AllocateBytes(panelElements * 2);
        using var separateK = lane.AllocateBytes(panelElements * 2);
        using var combinedQ = lane.AllocateBytes(panelElements * 2);
        using var combinedK = lane.AllocateBytes(panelElements * 2);

        lane.Run("decode_bfp8", n, 0, source, scale, separate, n, block, 1);
        lane.Run("attention_qk_pack_combined_candidate", panelElements / 2L, 256,
            separate, separateQ, separateK, sequence, width, heads, 0, batch * heads);
        lane.Run("attention_bfp8_decode_qk_2048_candidate", n / 2, 256,
            source, scale, combined, combinedQ, combinedK, batch, sequence, width, heads, block);

        var expectedValues = new float[n]; var actualValues = new float[n];
        var expectedQ = new ushort[panelElements]; var actualQ = new ushort[panelElements];
        var expectedK = new ushort[panelElements]; var actualK = new ushort[panelElements];
        lane.Read(separate, expectedValues); lane.Read(combined, actualValues);
        lane.ReadRaw(separateQ, expectedQ); lane.ReadRaw(combinedQ, actualQ);
        lane.ReadRaw(separateK, expectedK); lane.ReadRaw(combinedK, actualK);

        Assert.Equal(expectedValues.Select(BitConverter.SingleToInt32Bits),
            actualValues.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(expectedQ, actualQ);
        Assert.Equal(expectedK, actualK);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedDecodePreservesAttentionForwardAndGradient(bool causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");

        const int batch = 1, sequence = 2048, width = 64, heads = 2;
        var values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .017f) * .15f).ToArray();
        var seed = Enumerable.Range(0, batch * sequence * width)
            .Select(i => MathF.Cos(i * .021f) * .007f).ToArray();

        (float[] Output, float[] Gradient) Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32,
                options: new() { FusedAttentionQkvDecodePack2048 = enabled,
                    BulkAttentionQkPanels = true, AttentionWorkspaceMiB = 16,
                    PipelineEventCollection = true });
            var lane = Tensor.ArcLane;
            var input = new Tensor((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
            var output = input.FusedMultiHeadAttention(heads, causal);
            float[] result = output.Data.ToArray();
            long downloads = lane.D2HBytes;
            output.BackwardAndRelease(seed);
            Assert.Equal(downloads, lane.D2HBytes);
            Assert.Equal(enabled, lane.KernelTimings.ContainsKey("attention_bfp8_decode_qk_2048_candidate"));
            Assert.Equal(!enabled, lane.KernelTimings.ContainsKey("attention_qk_pack_combined_candidate"));
            return (result, input.Grad.ToArray());
        }

        var reference = Run(false); var candidate = Run(true);
        Assert.Equal(reference.Output.Select(BitConverter.SingleToInt32Bits),
            candidate.Output.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(reference.Gradient.Select(BitConverter.SingleToInt32Bits),
            candidate.Gradient.Select(BitConverter.SingleToInt32Bits));
    }
}
