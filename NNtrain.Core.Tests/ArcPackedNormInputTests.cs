using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcPackedNormInputTests
{
    [Theory]
    [InlineData(TensorPrecisionMode.Float32, 0f)]
    [InlineData(TensorPrecisionMode.Float32, .1f)]
    [InlineData(TensorPrecisionMode.Mix16_32, .1f)]
    [InlineData(TensorPrecisionMode.Mix8_32, 0f)]
    [InlineData(TensorPrecisionMode.Mix8_32, .1f)]
    [InlineData(TensorPrecisionMode.Mix8_32, .75f)]
    public void PackedDecodeResidualMatchesMaterializedInputsExactly(TensorPrecisionMode precision, float dropout)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        (float[] Y, float[][] Grad) Run(bool packed)
        {
            using var scope = Tensor.BeginArcExecution(precision: precision,
                options: new() { PackedNormInput = packed, BlockResidualNorm = false });
            const int rows = 17, width = 37;
            Tensor Create(int count, int[] shape, float amplitude, float offset) {
                var t = new Tensor(Enumerable.Range(0, count).Select(i => MathF.Sin(i * .137f) * amplitude + offset).ToArray(), shape);
                t.ConvertStorageInPlace(precision.ToStorageDType(), precision == TensorPrecisionMode.Mix8_32
                    ? Bfp8QuantizationDescriptor.Mix8_32 : null);
                return t;
            }
            var x = Create(rows * width, [rows, width], .3f, 0);
            var branch = Create(rows * width, [rows, width], .13f, .1f);
            var gamma = Create(width, [width], .2f, 1f);
            var beta = Create(width, [width], .01f, 0);
            var rng = new Random(123);
            Tensor y = x.AddDropoutLayerNormLastDim(branch, gamma, beta, dropout, rng);
            float[] values = y.Data.ToArray();
            long downloads = Tensor.ArcLane.D2HBytes;
            y.BackwardAndRelease(Enumerable.Range(0, rows * width).Select(i => MathF.Cos(i * .017f) * .01f).ToArray());
            Tensor.ArcLane.Synchronize();
            Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
            return (values, new[] { x, branch, gamma, beta }.Select(t => t.Grad.ToArray()).ToArray());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Y, actual.Y);
        for (int i = 0; i < expected.Grad.Length; i++) Assert.Equal(expected.Grad[i], actual.Grad[i]);
    }
}
