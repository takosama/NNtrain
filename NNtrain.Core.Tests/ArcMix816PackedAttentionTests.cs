using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816PackedAttentionTests
{
    [Fact]
    public void PackedBackwardKeepsForwardAndBoundsTwoAccumulatedGradients()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 1, sequence = 2048, width = 64, heads = 2;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray();

        (float[][] Outputs, float[][] Gradients, Dictionary<string, double> KernelMs) Run(bool packed)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16PackedAttentionBackward = packed,
                    Mix8_16Bf16QkvActivations = false,
                    Mix8_16Bf16Activations = false,
                    Mix8_16AttentionRowDelta = false,
                    Mix8_16FusedAttentionDpDs = false,
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
            Assert.Equal(packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_derivatives_prefix8_pack_bf16_2048"));
            Assert.Equal(packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_dq_packed_bf16_2048"));
            Assert.Equal(packed, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_dkv_packed_bf16_2048_causal"));
            return (outputs, gradients, new Dictionary<string, double>(Tensor.ArcLane.KernelTimings));
        }

        var baseline = Run(packed: false);
        var candidate = Run(packed: true);
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
            Assert.True(relativeRms < .01f,
                $"Packed attention relative gradient RMS = {relativeRms}.");
        }
        foreach (string stage in new[] { "attention_derivatives_prefix8_2048",
            "attention_fp32_dq_d32_aligned", "attention_dkv_block_slm_causal" })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"baseline {stage}: {baseline.KernelMs.GetValueOrDefault(stage):F4} GPU ms");
        foreach (string stage in new[] { "attention_derivatives_prefix8_pack_bf16_2048",
            "attention_dq_packed_bf16_2048", "attention_dkv_packed_bf16_2048_causal" })
            TestContext.Current.TestOutputHelper?.WriteLine(
                $"candidate {stage}: {candidate.KernelMs.GetValueOrDefault(stage):F4} GPU ms");
    }
}
