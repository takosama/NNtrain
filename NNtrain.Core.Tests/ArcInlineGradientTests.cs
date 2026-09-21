using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcInlineGradientTests
{
    public static IEnumerable<object[]> LinearCases()
    {
        foreach (TensorPrecisionMode precision in new[] {
            TensorPrecisionMode.Float32, TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
        foreach (bool relu in new[] { false, true })
        foreach (bool parallel in new[] { false, true })
            yield return [precision, relu, parallel, parallel ? 513 : 2051, true];
        foreach (TensorPrecisionMode precision in new[] {
            TensorPrecisionMode.Mix16_32, TensorPrecisionMode.Mix8_32 })
        {
            // Exercise dW's parallel split-K, row tails, and non-XMX fallback.
            yield return [precision, true, true, 4101, true];
            yield return [precision, true, true, 513, false];
        }
    }

    [Theory]
    [MemberData(nameof(LinearCases))]
    public void InlineLinearMatchesEncodedGradientExactly(
        TensorPrecisionMode precision, bool relu, bool parallel, int rows, bool xmx)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int ni = 65, no = 71;
        Snapshot Run(bool inline)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                InlineMatrixGradient = inline, ParallelReductions = parallel, XmxMatrices = xmx,
                DirectXmxMatrices = false });
            var x = Make([rows, ni], 1, precision);
            var w = Make([no, ni], 2, precision);
            var b = Make([no], 3, precision);
            float[] output = [];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Tensor y = x.LinearLastDim(w, b, applyRelu: relu);
                output = y.Data.ToArray();
                var seed = Enumerable.Range(0, rows * no)
                    .Select(i => MathF.Cos(i * .019f + repeat) * .013f).ToArray();
                long downloads = Tensor.ArcLane.D2HBytes;
                y.BackwardAndRelease(seed);
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
            }
            var result = new Snapshot(output, x.Grad.ToArray(), w.Grad.ToArray(), b.Grad.ToArray());
            Assert.Equal(precision != TensorPrecisionMode.Float32 && !inline,
                Tensor.ArcLane.KernelTimings.ContainsKey("matrix_gradient_bf16"));
            if (inline && precision != TensorPrecisionMode.Float32 && !parallel)
                Assert.Contains("linear_db_bf16_chunk", Tensor.ArcLane.KernelTimings.Keys);
            return result;
        }
        AssertExact(Run(false), Run(true));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Float32, false)]
    [InlineData(TensorPrecisionMode.Float32, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Mix16_32, true)]
    [InlineData(TensorPrecisionMode.Mix8_32, false)]
    [InlineData(TensorPrecisionMode.Mix8_32, true)]
    public void InlineChunkedLossMatchesEncodedGradientWithIgnoredTargetsAndAccumulation(
        TensorPrecisionMode precision, bool parallel)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int rows = 1031, width = 65, vocabulary = 71;
        Snapshot Run(bool inline)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision, options: new() {
                InlineMatrixGradient = inline, ParallelReductions = parallel, LossChunkRows = 512,
                DirectXmxMatrices = false });
            var x = Make([rows, width], 1, precision);
            var w = Make([vocabulary, width], 2, precision);
            var b = Make([vocabulary], 3, precision);
            var losses = new float[2];
            for (int repeat = 0; repeat < 2; repeat++)
            {
                int[] labels = Enumerable.Range(0, rows)
                    .Select(i => i % 17 == 0 ? -1 : (i * 13 + repeat) % vocabulary).ToArray();
                Tensor loss = x.ArcLinearCrossEntropy(w, b, labels, -1);
                losses[repeat] = loss.item();
                long downloads = Tensor.ArcLane.D2HBytes;
                loss.BackwardAndRelease([repeat == 0 ? .375f : .625f]);
                Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
            }
            var result = new Snapshot(losses, x.Grad.ToArray(), w.Grad.ToArray(), b.Grad.ToArray());
            Assert.Equal(precision != TensorPrecisionMode.Float32 && !inline,
                Tensor.ArcLane.KernelTimings.ContainsKey("matrix_gradient_bf16"));
            // The final seven-row tile must still take the serial bias path.
            if (inline && precision != TensorPrecisionMode.Float32)
                Assert.Contains("linear_db_bf16_chunk", Tensor.ArcLane.KernelTimings.Keys);
            return result;
        }
        AssertExact(Run(false), Run(true));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InlineLinearPreservesNonFiniteSeedClassification(bool parallel)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int rows = 513, width = 65;
        Snapshot Run(bool inline)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix16_32,
                options: new() { InlineMatrixGradient = inline, ParallelReductions = parallel,
                    DirectXmxMatrices = false });
            var x = Make([rows, width], 1, TensorPrecisionMode.Mix16_32);
            var w = Make([width, width], 2, TensorPrecisionMode.Mix16_32);
            var b = Make([width], 3, TensorPrecisionMode.Mix16_32);
            Tensor y = x.LinearLastDim(w, b, applyRelu: false);
            float[] output = y.Data.ToArray();
            float[] seed = Enumerable.Repeat(.001f, rows * width).ToArray();
            seed[0] = BitConverter.Int32BitsToSingle(0x7f800001);
            seed[width + 1] = float.PositiveInfinity;
            seed[2 * width + 2] = float.NegativeInfinity;
            y.BackwardAndRelease(seed);
            return new(output, x.Grad.ToArray(), w.Grad.ToArray(), b.Grad.ToArray());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Output, actual.Output);
        AssertClassifications(expected.InputGradient, actual.InputGradient);
        AssertClassifications(expected.WeightGradient, actual.WeightGradient);
        AssertClassifications(expected.BiasGradient, actual.BiasGradient);
        Assert.True(float.IsNaN(actual.BiasGradient[0]));
        Assert.Equal(float.PositiveInfinity, actual.BiasGradient[1]);
        Assert.Equal(float.NegativeInfinity, actual.BiasGradient[2]);
    }

    [Fact]
    public void RawXmxOperandKeepsLowPayloadNanAsNan()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx, "Intel XMX is required.");
        const int m = 129, n = 65, k = 33;
        using var lane = new ArcExecutionLane(options: new() { DirectXmxMatrices = false });
        float[] values = Enumerable.Repeat(.125f, m * k).ToArray();
        // Do not pass this through Backward's host += seed, which would already
        // quiet the signaling NaN and conceal a bad BF16 load conversion.
        values[0] = BitConverter.Int32BitsToSingle(0x7f800001);
        using var raw = lane.Upload(values);
        using var rounded = lane.Allocate(values.Length);
        using var b = lane.Upload(Enumerable.Repeat(.25f, n * k).ToArray());
        using var oldResult = lane.Allocate(m * n);
        using var newResult = lane.Allocate(m * n);
        lane.Run("matrix_gradient_bf16", values.Length, 0, raw, raw, rounded, values.Length, 0);
        ArcMuonMath.Gemm(lane, rounded, b, oldResult, m, n, k, bf16: 3);
        ArcMuonMath.Gemm(lane, raw, b, newResult, m, n, k, bf16: 3);
        var expected = new float[m * n]; var actual = new float[m * n];
        lane.Read(oldResult, expected); lane.Read(newResult, actual);
        AssertClassifications(expected, actual);
        for (int col = 0; col < n; col++) Assert.True(float.IsNaN(actual[col]));
    }

    private static Tensor Make(int[] shape, int phase, TensorPrecisionMode precision)
    {
        int length = shape.Aggregate(1, (a, b) => checked(a * b));
        var result = new Tensor(Enumerable.Range(0, length)
            .Select(i => MathF.Sin(i * .013f + phase) * .031f).ToArray(), shape);
        result.ConvertStorageInPlace(precision.ToStorageDType(),
            precision == TensorPrecisionMode.Mix8_32 ? Bfp8QuantizationDescriptor.Mix8_32 : null);
        return result;
    }

    private static void AssertExact(Snapshot expected, Snapshot actual)
    {
        Assert.Equal(expected.Output, actual.Output);
        Assert.Equal(expected.InputGradient, actual.InputGradient);
        Assert.Equal(expected.WeightGradient, actual.WeightGradient);
        Assert.Equal(expected.BiasGradient, actual.BiasGradient);
    }

    private static void AssertClassifications(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            if (float.IsNaN(expected[i])) Assert.True(float.IsNaN(actual[i]));
            else Assert.Equal(expected[i], actual[i]);
        }
    }

    private sealed record Snapshot(float[] Output, float[] InputGradient,
        float[] WeightGradient, float[] BiasGradient);
}
