using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcCacheSizedAttentionTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void SmallerBackwardTilePreservesExactOutputAndGradient(TensorPrecisionMode precision, bool causal)
        => Compare(precision, causal, false);

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void BlockIoPreservesExactOutputAndAccumulatedGradient(TensorPrecisionMode precision, bool causal)
        => Compare(precision, causal, true);

    private static void Compare(TensorPrecisionMode precision, bool causal, bool blockIo)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 2, heads = 5, sequence = 1024, width = heads * 32;
        float[][] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                CacheSizedAttentionBackward = blockIo || enabled, BlockIoAttention = blockIo && enabled, AttentionWorkspaceMiB = 64 });
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
            return [values, input.Grad.ToArray()];
        }
        var expected = Run(false); var actual = Run(true);
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i].Select(BitConverter.SingleToInt32Bits), actual[i].Select(BitConverter.SingleToInt32Bits));
    }
}
