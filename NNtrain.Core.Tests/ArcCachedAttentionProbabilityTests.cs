using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcCachedAttentionProbabilityTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (var precision in new[] { TensorPrecisionMode.Float32, TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
        {
            foreach (bool causal in new[] { false, true })
            foreach (int sequence in new[] { 512, 1024 })
                yield return [precision, causal, sequence, true, false, true];
            // An unsupported sequence must retain the general row kernel.
            yield return [precision, true, 137, true, false, true];
        }
        // Preserve the non-bounded causal mask and explicit subgroup selection.
        yield return [TensorPrecisionMode.Mix16_32, true, 512, false, false, true];
        yield return [TensorPrecisionMode.Mix16_32, true, 512, true, true, true];
        // An explicit subgroup request must also keep its existing fallback
        // when XMX is disabled, not silently select the new row cache instead.
        yield return [TensorPrecisionMode.Float32, true, 512, true, true, false];
    }

    [Fact]
    public void ReferenceOptionsKeepProbabilityCachingDisabled()
    {
        Assert.True(new ArcExecutionOptions().CachedAttentionProbabilities);
        Assert.False(ArcExecutionOptions.Reference.CachedAttentionProbabilities);
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void CachedProbabilitiesPreserveExactOutputTwoGradientsAndResidency(TensorPrecisionMode precision,
        bool causal, int sequence, bool bounds, bool subgroup, bool xmx)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "The measured SG16 Intel XMX device is required.");
        const int batch = 2, heads = 3, d = 32, width = heads * d;
        var values = Enumerable.Range(0, checked(batch * sequence * width * 3))
            .Select(i => i % 101 == 0 ? -0f : MathF.Sin(i * .013f) * .1f).ToArray();
        Snapshot Run(bool cached)
        {
            // At T512, the 8MiB budget visits four then two heads, crossing a
            // batch boundary. At T1024 it visits each head separately.
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                CachedAttentionProbabilities = cached, CausalAttentionBounds = bounds,
                SubgroupAttentionReduction = subgroup, XmxMatrices = xmx, AttentionWorkspaceMiB = 8 });
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
            bool eligible = cached && !subgroup && sequence is 512 or 1024;
            Assert.Equal(eligible, lane.KernelTimings.Keys.Any(k => k.StartsWith("attention_probabilities_register_", StringComparison.Ordinal)));
            if (eligible) Assert.True(lane.KernelTimings.ContainsKey($"attention_probabilities_register_{sequence}"));
            Assert.Equal(subgroup && xmx, lane.KernelTimings.ContainsKey("attention_probabilities_subgroup_candidate"));
            Assert.Equal(!eligible && !(subgroup && xmx), lane.KernelTimings.ContainsKey("attention_probabilities"));
            // The negligible derivative experiment is deliberately not adopted.
            Assert.DoesNotContain(lane.KernelTimings.Keys, k => k.StartsWith("attention_derivatives_register_", StringComparison.Ordinal));
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
