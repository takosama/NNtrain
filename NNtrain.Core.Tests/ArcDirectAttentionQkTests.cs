using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcDirectAttentionQkTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var precision in new[] { TensorPrecisionMode.Float32, TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
        foreach (bool causal in new[] { false, true })
        {
            // T137 exercises the last row/column tiles. T1024 with H3/B3 and an
            // 8MiB score budget visits one head per tile across batch boundaries.
            yield return [precision, causal, 137, 32, true, true, true];
            yield return [precision, causal, 1024, 32, true, true, true];
        }
        // Dispatch options must be honored rather than silently overridden.
        yield return [TensorPrecisionMode.Mix16_32, true, 137, 32, false, true, true];
        yield return [TensorPrecisionMode.Mix16_32, true, 137, 32, true, false, true];
        yield return [TensorPrecisionMode.Mix16_32, true, 137, 32, true, true, false];
        yield return [TensorPrecisionMode.Mix8_32, true, 65, 32, true, true, true];
        yield return [TensorPrecisionMode.Mix8_32, true, 137, 17, true, true, true];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DirectQkPreservesExactAttentionOutputTwoGradientsAndResidency(TensorPrecisionMode precision,
        bool causal, int sequence, int d, bool bounds, bool xmx, bool panel)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
        const int batch = 3, heads = 3;
        int width = heads * d;
        var values = Enumerable.Range(0, checked(batch * sequence * width * 3))
            .Select(i => i % 101 == 0 ? -0f : MathF.Sin(i * .013f) * .1f).ToArray();
        Snapshot Run(bool direct)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                DirectAttentionQk = direct, CausalAttentionBounds = bounds, XmxMatrices = xmx,
                PanelAttention = panel, AttentionWorkspaceMiB = 8, BulkAttentionQkPanels = false });
            var lane = Tensor.ArcLane;
            var input = new Tensor((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32
                ? Bfp8QuantizationDescriptor.Block(32) : null);
            var outputs = new float[2][];
            var gradients = new float[2][];
            long retained = -1;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                Tensor output = input.FusedMultiHeadAttention(heads, causal);
                Assert.Equal(downloads, lane.D2HBytes);
                if (repeat != 0) Assert.Equal(uploads, lane.H2DBytes);
                outputs[repeat] = output.Data.ToArray();
                float[] seed = Enumerable.Range(0, batch * sequence * width)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .01f).ToArray();
                downloads = lane.D2HBytes;
                output.BackwardAndRelease(seed);
                lane.Synchronize();
                Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(0, lane.RetiredBytes);
                if (repeat == 0) retained = lane.AllocatedBytes;
                else Assert.Equal(retained, lane.AllocatedBytes);
                gradients[repeat] = input.Grad.ToArray();
            }
            bool eligible = direct && precision != TensorPrecisionMode.Float32 && xmx && panel && sequence >= 128 && d == 32;
            Assert.Equal(eligible, lane.KernelTimings.ContainsKey("attention_qk_pack_combined_candidate"));
            Assert.Equal(eligible && causal && bounds, lane.KernelTimings.ContainsKey("attention_qk_direct_256x32_candidate"));
            Assert.Equal(eligible && !(causal && bounds), lane.KernelTimings.ContainsKey("attention_qk_direct_128x32_candidate"));
            return new(outputs, gradients);
        }
        var reference = Run(false); var actual = Run(true);
        for (int repeat = 0; repeat < 2; repeat++)
        {
            Assert.Equal(reference.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                actual.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(reference.Gradients[repeat].Select(BitConverter.SingleToInt32Bits),
                actual.Gradients[repeat].Select(BitConverter.SingleToInt32Bits));
        }
    }

    private sealed record Snapshot(float[][] Outputs, float[][] Gradients);
}
