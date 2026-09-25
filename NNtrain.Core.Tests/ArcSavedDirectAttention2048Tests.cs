using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcSavedDirectAttention2048Tests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false, 1, 64, 2)]
    [InlineData(TensorPrecisionMode.Float32, true, 1, 64, 2)]
    [InlineData(TensorPrecisionMode.Mix8_32, false, 1, 64, 2)]
    [InlineData(TensorPrecisionMode.Mix8_32, true, 1, 64, 2)]
    [InlineData(TensorPrecisionMode.Mix8_32, true, 1, 512, 16)]
    [InlineData(TensorPrecisionMode.Mix8_32, true, 16, 512, 16)]
    public void SavedProbabilityDispatchPreservesForwardAndAccumulatedGradientBits(
        TensorPrecisionMode precision, bool causal, int batch, int width, int heads)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int sequence = 2048;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .1f).ToArray();

        (float[][] Outputs, float[][] Gradients) Run(
            bool direct, bool prefix4 = false, bool prefix8 = false, bool prefix16 = false)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new()
            {
                SavedAttentionProbabilitiesDirect2048 = direct,
                AttentionDerivativePrefix4T2048 = prefix4,
                AttentionDerivativePrefix8T2048 = prefix8,
                AttentionDerivativePrefix16T2048 = prefix16,
                AttentionWorkspaceMiB = 64,
            });
            Tensor input = new((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(precision.ToStorageDType(),
                precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            var outputs = new float[2][];
            var gradients = new float[2][];
            for (int repeat = 0; repeat < 2; ++repeat)
            {
                long downloads = Tensor.ArcLane.D2HBytes;
                Tensor output = input.FusedMultiHeadAttention(heads, causal);
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
            Assert.Equal(direct, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_probabilities_saved_direct_2048"));
            Assert.Equal(prefix4 && !prefix8 && !prefix16, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_derivatives_prefix4_2048"));
            Assert.Equal(prefix8 && !prefix16, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_derivatives_prefix8_2048"));
            Assert.Equal(prefix16, Tensor.ArcLane.KernelTimings.ContainsKey(
                "attention_derivatives_prefix16_2048"));
            return (outputs, gradients);
        }

        var baseline = Run(false);
        var candidate = Run(true);
        var derivativeCandidate = Run(true, prefix4: true);
        var prefix8Candidate = Run(true, prefix4: true, prefix8: true);
        var prefix16Candidate = Run(true, prefix4: true, prefix8: true, prefix16: true);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            Assert.Equal(baseline.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(baseline.Gradients[repeat].Select(BitConverter.SingleToInt32Bits),
                candidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                derivativeCandidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(candidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits),
                derivativeCandidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(derivativeCandidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                prefix8Candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(derivativeCandidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits),
                prefix8Candidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(prefix8Candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits),
                prefix16Candidate.Outputs[repeat].Select(BitConverter.SingleToInt32Bits));
            Assert.Equal(prefix8Candidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits),
                prefix16Candidate.Gradients[repeat].Select(BitConverter.SingleToInt32Bits));
        }
    }
}
