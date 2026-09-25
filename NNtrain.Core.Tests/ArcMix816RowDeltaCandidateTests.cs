using NNtrain;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;
using Xunit;

public sealed class ArcMix816RowDeltaCandidateTests
{
    [Fact]
    public void NoGradAttentionDoesNotSaveRawOutputForRowDelta()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "The candidate requires an SG16 XMX device with SLM block I/O.");
        const int sequence = 2048, width = 64, heads = 2;
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
            options: new ArcExecutionOptions
            {
                Mix8_16Bf16QkvActivations = true,
                Mix8_16Bf16Activations = true,
                Mix8_16PackedAttentionBackward = true,
                Mix8_16AttentionRowDelta = true,
                Mix8_16FusedAttentionDpDs = true,
                // Keep this no-grad regression on its original PV route.
                Mix8_16NativeExpAttention = false,
                Mix8_16DpDsK32 = false,
                Mix8_16DkvPackedSlm = false,
                Mix8_16PvPackedSlm = false,
                Mix8_16DqPackedSlm = false,
            });
        Tensor input = new(Enumerable.Range(0, sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray(), [1, sequence, width * 3]);
        input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
        using (AutogradContext.NoGrad())
        {
            Tensor output = input.FusedMultiHeadAttention(heads, causal: true);
            Assert.All(output.Data, value => Assert.True(float.IsFinite(value)));
        }
        Tensor.ArcLane.Synchronize();
        Assert.True(Tensor.ArcLane.KernelTimings.ContainsKey("attention_mix8_16_bf16_pv_d32_2048"));
        Assert.False(Tensor.ArcLane.KernelTimings.ContainsKey("attention_mix8_16_store_raw_bf16_2048"));
        Assert.False(Tensor.ArcLane.KernelTimings.ContainsKey("attention_mix8_16_row_delta_bf16_2048"));
        Assert.False(Tensor.ArcLane.KernelTimings.ContainsKey("attention_mix8_16_bf16_dpds_d32_2048"));
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void TransformerAttentionRowDeltaBoundsAccumulatedGradientAndReleasesRawOutput(
        bool fused, bool bf16Activations)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "The candidate requires an SG16 XMX device with SLM block I/O.");
        const int batch = 1, sequence = 2048, width = 64, heads = 2;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray();

        (float[][] Outputs, float[][] Gradients, long RetainedBytes) Run(bool rowDelta)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16Bf16QkvActivations = true,
                    Mix8_16Bf16Activations = bf16Activations,
                    Mix8_16PackedAttentionBackward = true,
                    Mix8_16AttentionRowDelta = rowDelta,
                    Mix8_16FusedAttentionDpDs = rowDelta && fused,
                    // Compare row-delta and BF16 activation choices without
                    // changing the established attention kernels underneath.
                    Mix8_16NativeExpAttention = false,
                    Mix8_16DpDsK32 = false,
                    Mix8_16DkvPackedSlm = false,
                    Mix8_16PvPackedSlm = false,
                    Mix8_16DqPackedSlm = false,
                    AttentionWorkspaceMiB = 64,
                });
            Tensor input = new((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
            var outputs = new float[2][];
            var gradients = new float[2][];
            long retainedBytes = -1;
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                Tensor output = input.FusedMultiHeadAttention(heads, causal: true);
                outputs[repeat] = output.Data.ToArray();
                float[] seed = Enumerable.Range(0, batch * sequence * width)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .01f).ToArray();
                output.BackwardAndRelease(seed);
                Tensor.ArcLane.Synchronize();
                gradients[repeat] = input.Grad.ToArray();
                if (retainedBytes >= 0)
                    Assert.Equal(retainedBytes, Tensor.ArcLane.AllocatedBytes);
                retainedBytes = Tensor.ArcLane.AllocatedBytes;
            }
            Assert.Equal(rowDelta, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_store_raw_bf16_2048"));
            Assert.Equal(rowDelta, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_row_delta_bf16_2048"));
            Assert.Equal(rowDelta && !fused, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_row_derivatives_packed_2048"));
            Assert.Equal(rowDelta && fused, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_bf16_dpds_d32_2048"));
            Assert.Equal(!rowDelta, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_derivatives_prefix8_pack_bf16_2048"));
            return (outputs, gradients, retainedBytes);
        }

        var baseline = Run(rowDelta: false);
        var candidate = Run(rowDelta: true);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            Assert.Equal(baseline.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            double error = 0, reference = 0;
            for (int i = 0; i < baseline.Gradients[repeat].Length; ++i)
            {
                float expected = baseline.Gradients[repeat][i];
                float actual = candidate.Gradients[repeat][i];
                Assert.True(float.IsFinite(actual), $"Gradient {i} is non-finite.");
                error += (double)(actual - expected) * (actual - expected);
                reference += (double)expected * expected;
            }
            double rms = Math.Sqrt(error / reference);
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"accumulation {repeat + 1}: row-delta gradient relative RMS = {rms:G6}");
            Assert.True(rms < .002, $"Row-delta accumulated gradient RMS {rms}.");
        }
        TestContext.Current.TestOutputHelper?.WriteLine(
            $"retained baseline/candidate bytes: {baseline.RetainedBytes}/{candidate.RetainedBytes}");
    }

    [Fact]
    public void Bf16RawOutputDeltaBoundsFp32AndPackedScoreDerivatives()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "The candidate requires an SG16 XMX device with SLM block I/O.");

        const int sequence = 2048, heads = 3, width = heads * 32, first = 1, count = 1;
        const int scores = count * sequence * sequence;
        var probabilities = new float[scores];
        for (int query = 0; query < sequence; ++query)
        {
            float denominator = 0f;
            for (int key = 0; key <= query; ++key)
            {
                float value = MathF.Exp((key - query) * .02f
                    + MathF.Sin(key * .041f) * .13f);
                probabilities[query * sequence + key] = value;
                denominator += value;
            }
            for (int key = 0; key <= query; ++key)
                probabilities[query * sequence + key] /= denominator;
        }
        ushort[] qkvValues = Enumerable.Range(0, sequence * 3 * width)
            .Select(i => Bf16(MathF.Sin(i * .013f) * .09f)).ToArray();
        float[] dyValues = Enumerable.Range(0, sequence * width)
            .Select(i => MathF.Cos(i * .017f) * .012f).ToArray();

        using var lane = new ArcExecutionLane();
        using var qkv = lane.UploadRaw(qkvValues);
        using var dy = lane.Upload(dyValues);
        using var raw = lane.Upload(new float[sequence * width]);
        using var saved = lane.AllocateBytes(sequence * width * sizeof(ushort));
        using var delta = lane.Allocate(heads * sequence);
        using var pForward = lane.Upload(probabilities);
        lane.Run3D("attention_mix8_16_bf16_pv_d32_2048", 16,
            sequence / 64L * 16, count, 16, 16, 1,
            pForward, qkv, raw, sequence, width, heads, first);
        lane.Run("attention_mix8_16_store_raw_bf16_2048", sequence * (long)width,
            256, raw, saved, sequence * width);
        lane.Run("attention_mix8_16_row_delta_bf16_2048", count * (long)sequence,
            256, dy, saved, delta, sequence, width, heads, first, count);

        using var dP = lane.Allocate(scores);
        lane.Run3D("attention_mix8_16_bf16_dp_d32_2048", sequence / 64L * 16,
            sequence / 64L * 16, count, 16, 16, 1,
            dy, qkv, dP, sequence, width, heads, first);
        using var dSReference = lane.Allocate(scores);
        using var dSRowDelta = lane.Allocate(scores);
        using var dPPackedReference = lane.Allocate(scores);
        using var dPPackedRowDelta = lane.Allocate(scores);
        foreach (ArcBuffer target in new[] { dSReference, dSRowDelta, dPPackedReference, dPPackedRowDelta })
            lane.CopyBytes(dP, target, 0, 0, scores * sizeof(float));
        using var pReference = lane.Upload(probabilities);
        using var pRowDelta = lane.Upload(probabilities);
        using var pPackedReference = lane.Upload(probabilities);
        using var pPackedRowDelta = lane.Upload(probabilities);
        using var pFused = lane.Upload(probabilities);
        using var pFusedXmx = lane.Upload(probabilities);
        lane.Run("attention_derivatives_prefix8_2048", count * sequence * 64L, 64,
            pReference, dSReference, sequence, width, heads, 2);
        lane.Run("attention_mix8_16_row_derivatives_fp32_2048", scores, 256,
            pRowDelta, dSRowDelta, delta, sequence, width, heads, first, count, 2);
        lane.Run("attention_derivatives_prefix8_pack_bf16_2048", count * sequence * 64L, 64,
            pPackedReference, dPPackedReference, sequence, width, heads, 2);
        lane.Run("attention_mix8_16_row_derivatives_packed_2048", scores, 256,
            pPackedRowDelta, dPPackedRowDelta, delta, sequence, width, heads, first, count, 2);
        lane.Run3D("attention_mix8_16_bf16_dpds_d32_2048", sequence / 64L * 16,
            sequence / 64L * 16, count, 16, 16, 1,
            dy, qkv, pFused, delta, sequence, width, heads, first);
        lane.Run3D("attention_mix8_16_bf16_dpds_xmx_d32_2048", sequence / 64L * 16,
            sequence / 64L * 8, count, 16, 8, 1,
            dy, qkv, pFusedXmx, delta, sequence, width, heads, first);
        lane.Synchronize();

        var expectedFp32 = new float[scores];
        var actualFp32 = new float[scores];
        var expectedPacked = new uint[scores];
        var actualPacked = new uint[scores];
        var fusedPacked = new uint[scores];
        var fusedXmxPacked = new uint[scores];
        lane.Read(dSReference, expectedFp32);
        lane.Read(dSRowDelta, actualFp32);
        lane.ReadRaw(pPackedReference, expectedPacked);
        lane.ReadRaw(pPackedRowDelta, actualPacked);
        lane.ReadRaw(pFused, fusedPacked);
        lane.ReadRaw(pFusedXmx, fusedXmxPacked);
        double fp32Error = 0, fp32Reference = 0, packedError = 0, packedReference = 0;
        for (int query = 0; query < sequence; ++query)
        for (int key = 0; key <= query; ++key)
        {
            int index = query * sequence + key;
            float expected = expectedFp32[index], actual = actualFp32[index];
            Assert.True(float.IsFinite(actual), $"Non-finite FP32 derivative at {query},{key}.");
            fp32Error += (double)(actual - expected) * (actual - expected);
            fp32Reference += (double)expected * expected;
            Assert.Equal(expectedPacked[index] & 0xffffu, actualPacked[index] & 0xffffu);
            float packedExpected = BitConverter.UInt32BitsToSingle(expectedPacked[index] & 0xffff0000u);
            float packedActual = BitConverter.UInt32BitsToSingle(actualPacked[index] & 0xffff0000u);
            Assert.True(float.IsFinite(packedActual), $"Non-finite packed derivative at {query},{key}.");
            packedError += (double)(packedActual - packedExpected) * (packedActual - packedExpected);
            packedReference += (double)packedExpected * packedExpected;
        }
        double fp32Rms = Math.Sqrt(fp32Error / fp32Reference);
        double packedRms = Math.Sqrt(packedError / packedReference);
        double xmxError = 0, xmxReference = 0;
        for (int i = 0; i < scores; ++i)
            Assert.True(actualPacked[i] == fusedPacked[i],
                $"Fused dP/dS differs from separate row derivative at score {i}: "
                + $"0x{actualPacked[i]:x8} vs 0x{fusedPacked[i]:x8}.");
        for (int query = 0; query < sequence; ++query)
        for (int key = 0; key <= query; ++key)
        {
            int index = query * sequence + key;
            Assert.Equal(fusedPacked[index] & 0xffffu, fusedXmxPacked[index] & 0xffffu);
            float expected = BitConverter.UInt32BitsToSingle(fusedPacked[index] & 0xffff0000u);
            float actual = BitConverter.UInt32BitsToSingle(fusedXmxPacked[index] & 0xffff0000u);
            Assert.True(float.IsFinite(actual), $"Non-finite XMX dS at {query},{key}.");
            xmxError += (double)(actual - expected) * (actual - expected);
            xmxReference += (double)expected * expected;
        }
        double xmxRms = Math.Sqrt(xmxError / xmxReference);
        TestContext.Current.TestOutputHelper?.WriteLine($"FP32 dS relative RMS: {fp32Rms:G6}");
        TestContext.Current.TestOutputHelper?.WriteLine($"packed dS relative RMS: {packedRms:G6}");
        TestContext.Current.TestOutputHelper?.WriteLine($"XMX fused dS relative RMS: {xmxRms:G6}");
        foreach (string kernel in new[] { "attention_mix8_16_store_raw_bf16_2048",
            "attention_mix8_16_row_delta_bf16_2048", "attention_derivatives_prefix8_2048",
            "attention_mix8_16_row_derivatives_fp32_2048",
            "attention_derivatives_prefix8_pack_bf16_2048",
            "attention_mix8_16_row_derivatives_packed_2048",
            "attention_mix8_16_bf16_dp_d32_2048", "attention_mix8_16_bf16_dpds_d32_2048",
            "attention_mix8_16_bf16_dpds_xmx_d32_2048" })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"{kernel}: {lane.KernelTimings.GetValueOrDefault(kernel):F4} GPU ms");
        Assert.True(fp32Rms < .03, $"Row-delta FP32 derivative RMS {fp32Rms}.");
        Assert.True(packedRms < .03, $"Row-delta packed derivative RMS {packedRms}.");
        Assert.True(xmxRms < .03, $"XMX fused dP/dS RMS {xmxRms}.");
    }

    private static ushort Bf16(float value)
    {
        uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(value));
        if ((bits & 0x7f800000u) != 0x7f800000u)
            bits += 0x7fffu + ((bits >> 16) & 1u);
        return (ushort)(bits >> 16);
    }
}
