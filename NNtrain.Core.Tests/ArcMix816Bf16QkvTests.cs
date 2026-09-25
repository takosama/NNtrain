using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816Bf16QkvTests
{
    [Fact]
    public void PhysicalBf16QkvRetainsForwardAndAccumulatedGradient()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 1, sequence = 2048, width = 64, heads = 2;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray();

        (float[][] Outputs, float[][] Gradients, Dictionary<string, double> KernelMs) Run(bool physicalBf16, bool packed)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16PackedAttentionBackward = packed,
                    Mix8_16Bf16QkvActivations = physicalBf16,
                    Mix8_16Bf16Activations = false,
                    Mix8_16AttentionRowDelta = false,
                    Mix8_16FusedAttentionDpDs = false,
                    // Isolate the physical-QKV and packed-backward comparison
                    // from newer attention dispatch defaults.
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
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                long downloads = Tensor.ArcLane.D2HBytes;
                Tensor output = input.FusedMultiHeadAttention(heads, causal: true);
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
                outputs[repeat] = output.Data.ToArray();
                float[] seed = Enumerable.Range(0, batch * sequence * width)
                    .Select(i => MathF.Cos(i * .017f + repeat) * .01f).ToArray();
                downloads = Tensor.ArcLane.D2HBytes;
                output.BackwardAndRelease(seed);
                Tensor.ArcLane.Synchronize();
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
                gradients[repeat] = input.Grad.ToArray();
            }
            foreach (string kernel in new[] { "attention_bfp8_decode_bf16_qk_2048",
                "attention_mix8_16_bf16_pv_d32_2048", "attention_mix8_16_bf16_dp_d32_2048" })
                Assert.Equal(physicalBf16, Tensor.ArcLane.KernelTimings.ContainsKey(kernel));
            Assert.Equal(physicalBf16 && !packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_bf16_dq_d32_2048"));
            Assert.Equal(physicalBf16 && !packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_bf16_dkv_d32_2048"));
            Assert.Equal(physicalBf16 && packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_bf16_dq_packed_2048"));
            Assert.Equal(physicalBf16 && packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_mix8_16_bf16_dkv_packed_2048"));
            return (outputs, gradients, new Dictionary<string, double>(Tensor.ArcLane.KernelTimings));
        }

        var baseline = Run(physicalBf16: false, packed: false);
        var candidate = Run(physicalBf16: true, packed: false);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            Assert.Equal(baseline.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            float numerator = 0f, denominator = 0f;
            for (int i = 0; i < baseline.Gradients[repeat].Length; ++i)
            {
                float expected = baseline.Gradients[repeat][i];
                float actual = candidate.Gradients[repeat][i];
                Assert.True(float.IsFinite(actual), $"Gradient {i} is non-finite.");
                numerator += (actual - expected) * (actual - expected);
                denominator += expected * expected;
            }
            float relativeRms = MathF.Sqrt(numerator / denominator);
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"accumulation {repeat + 1}: relative gradient RMS = {relativeRms:G6}");
            Assert.True(relativeRms < .005f,
                $"Physical BF16 QKV relative gradient RMS = {relativeRms}.");
        }
        foreach (string kernel in new[] { "attention_bfp8_decode_qk_2048_candidate",
            "attention_fp32_pv_d32_aligned", "attention_fp32_dp_d32_aligned",
            "attention_fp32_dq_d32_aligned", "attention_dkv_block_slm_causal" })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"baseline {kernel}: {baseline.KernelMs.GetValueOrDefault(kernel):F4} GPU ms");
        foreach (string kernel in new[] { "attention_bfp8_decode_bf16_qk_2048",
            "attention_mix8_16_bf16_pv_d32_2048", "attention_mix8_16_bf16_dp_d32_2048",
            "attention_mix8_16_bf16_dq_d32_2048", "attention_mix8_16_bf16_dkv_d32_2048" })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"candidate {kernel}: {candidate.KernelMs.GetValueOrDefault(kernel):F4} GPU ms");

        var packedBaseline = Run(physicalBf16: false, packed: true);
        var combined = Run(physicalBf16: true, packed: true);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            Assert.Equal(packedBaseline.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                combined.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(packedBaseline.Gradients[repeat].Select(BitConverter.SingleToInt32Bits),
                combined.Gradients[repeat].Select(BitConverter.SingleToInt32Bits));
        }
        foreach (string kernel in new[] { "attention_bfp8_decode_bf16_qk_2048",
            "attention_mix8_16_bf16_pv_d32_2048", "attention_mix8_16_bf16_dp_d32_2048",
            "attention_mix8_16_bf16_dq_packed_2048", "attention_mix8_16_bf16_dkv_packed_2048" })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"combined {kernel}: {combined.KernelMs.GetValueOrDefault(kernel):F4} GPU ms");
    }
}
