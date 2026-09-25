using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816XmxAttentionTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Bf16XmxProductsKeepFiniteForwardAndAccumulatedGradients(bool causal, bool slm)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 1, sequence = 2048, width = 64, heads = 2;
        float[] values = Enumerable.Range(0, batch * sequence * width * 3)
            .Select(i => MathF.Sin(i * .013f) * .2f).ToArray();

        (float[][] Outputs, float[][] Gradients) Run(bool xmx)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16XmxAttentionProducts = xmx,
                    Mix8_16XmxAttentionSlm = slm,
                    Mix8_16Bf16Activations = false,
                    Mix8_16AttentionRowDelta = false,
                    Mix8_16FusedAttentionDpDs = false,
                });
            Tensor input = new((float[])values.Clone(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
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
            string prefix = slm ? "attention_products_slm" : "attention_products_direct";
            Assert.Equal(xmx, Tensor.ArcLane.KernelTimings.ContainsKey(prefix + "_n32_bf16"));
            Assert.Equal(xmx, Tensor.ArcLane.KernelTimings.ContainsKey(prefix + "_n64_bf16"));
            return (outputs, gradients);
        }

        var baseline = Run(false);
        var candidate = Run(true);
        for (int repeat = 0; repeat < 2; ++repeat)
        {
            Check(baseline.Outputs[repeat], candidate.Outputs[repeat], .02, $"output {repeat}");
            Check(baseline.Gradients[repeat], candidate.Gradients[repeat], .02, $"gradient {repeat}");
        }
    }

    private static void Check(float[] expected, float[] actual, double tolerance, string label)
    {
        double error = 0, scale = 0;
        for (int i = 0; i < expected.Length; ++i)
        {
            Assert.True(float.IsFinite(actual[i]), $"{label}[{i}] is not finite.");
            double difference = actual[i] - expected[i];
            error += difference * difference;
            scale += (double)expected[i] * expected[i];
        }
        double relativeRms = Math.Sqrt(error / Math.Max(scale, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine($"{label}: relative RMS={relativeRms:G6}");
        Assert.True(relativeRms < tolerance, $"{label}: relative RMS {relativeRms} >= {tolerance}");
    }
}
