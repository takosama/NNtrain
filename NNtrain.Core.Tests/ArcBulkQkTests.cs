using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcBulkQkTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Mix16_32, true, 1024)]
    [InlineData(TensorPrecisionMode.Mix8_32, true, 1024)]
    [InlineData(TensorPrecisionMode.Mix16_32, false, 137)]
    [InlineData(TensorPrecisionMode.Mix8_32, false, 137)]
    [InlineData(TensorPrecisionMode.Float32, true, 128)]
    public void PhaseLocalBulkPanelsPreserveExactValuesAndAccumulation(TensorPrecisionMode precision, bool causal, int sequence)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 2, heads = 5, width = heads * 32;
        float[][] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() { BulkAttentionQkPanels = enabled,
                PipelineEventCollection = true, AttentionWorkspaceMiB = 8 });
            var input = new Tensor(Enumerable.Range(0, batch * sequence * width * 3).Select(i => MathF.Sin(i * .013f) * .1f).ToArray(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Block(32) : null);
            float[] values = [];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                var output = input.FusedMultiHeadAttention(heads, causal); values = output.Data.ToArray();
                long d2h = Tensor.ArcLane.D2HBytes;
                output.BackwardAndRelease(Enumerable.Range(0, batch * sequence * width).Select(i => MathF.Cos(i * .017f + repeat) * .01f).ToArray());
                Assert.Equal(d2h, Tensor.ArcLane.D2HBytes);
            }
            Tensor.ArcLane.Synchronize();
            bool eligible = enabled && precision != TensorPrecisionMode.Float32;
            Assert.Equal(eligible, Tensor.ArcLane.KernelTimings.Keys.Any(k =>
                k.StartsWith("attention_qk_direct_", StringComparison.Ordinal)
                && k.EndsWith("_offset", StringComparison.Ordinal)));
            return [values, input.Grad.ToArray()];
        }
        var expected = Run(false); var actual = Run(true);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i].Select(BitConverter.SingleToInt32Bits), actual[i].Select(BitConverter.SingleToInt32Bits));
    }
}
