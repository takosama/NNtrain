using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcMix816Int8LinearTests
{
    [Theory]
    [InlineData(256, 64, 64, false)]
    [InlineData(513, 96, 128, true)]
    [InlineData(8201, 32, 32, false)] // Crosses the bounded A-panel row slice.
    public void Int8XmxForwardMatchesDecodedBfp8AndKeepsBackwardFinite(
        int rows, int columns, int k, bool relu)
    {
        RequireXmx();
        float[] sourceX = Enumerable.Range(0, rows * k)
            .Select(i => MathF.Sin(i * .013f) * .17f).ToArray();
        float[] sourceW = Enumerable.Range(0, columns * k)
            .Select(i => MathF.Cos(i * .019f) * .11f).ToArray();
        float[] sourceB = Enumerable.Range(0, columns)
            .Select(i => (i % 7 - 3) * .0017f).ToArray();

        (float[] Output, float[] InputGradient, float[] WeightGradient) Run(bool int8)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    Mix8_16Int8Linear = int8,
                    Mix8_16Bf16Activations = false,
                });
            Tensor Make(float[] values, int[] shape)
            {
                var t = new Tensor((float[])values.Clone(), shape);
                t.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
                return t;
            }
            var x = Make(sourceX, [rows, k]);
            var w = Make(sourceW, [columns, k]);
            var b = Make(sourceB, [columns]);
            long downloads = Tensor.ArcLane.D2HBytes;
            var y = x.LinearLastDim(w, b, applyRelu: relu);
            Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
            float[] output = y.Data.ToArray();
            Assert.Equal(int8, Tensor.ArcLane.KernelTimings.ContainsKey("xmx_i8_bfp8_linear_16x32"));
            float[] seed = Enumerable.Range(0, rows * columns)
                .Select(i => MathF.Cos(i * .023f) * .003f).ToArray();
            downloads = Tensor.ArcLane.D2HBytes;
            y.BackwardAndRelease(seed);
            Tensor.ArcLane.Synchronize();
            Assert.Equal(downloads, Tensor.ArcLane.D2HBytes);
            Tensor.ArcLane.CheckNumericStatus();
            return (output, x.Grad.ToArray(), w.Grad.ToArray());
        }

        var reference = Run(false);
        var candidate = Run(true);
        CheckRelativeRms(reference.Output, candidate.Output, .005, "output");
        CheckRelativeRms(reference.InputGradient, candidate.InputGradient, .03, "dX");
        CheckRelativeRms(reference.WeightGradient, candidate.WeightGradient, .03, "dW");
    }

    [Fact]
    public void Int8OptionDoesNotChangeMix8_32Dispatch()
    {
        RequireXmx();
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32,
            options: new ArcExecutionOptions
            {
                Mix8_16Int8Linear = true,
                Mix8_16Bf16Activations = false,
            });
        var x = new Tensor(Enumerable.Repeat(.125f, 256 * 64).ToArray(), [256, 64]);
        var w = new Tensor(Enumerable.Repeat(.0625f, 64 * 64).ToArray(), [64, 64]);
        var b = new Tensor(new float[64], [64]);
        foreach (Tensor t in new[] { x, w, b })
            t.ConvertStorageInPlace(TensorDType.Bfp8, Bfp8QuantizationDescriptor.Block(32));
        var y = x.LinearLastDim(w, b, applyRelu: false);
        Assert.DoesNotContain("xmx_i8_bfp8_linear_16x32", Tensor.ArcLane.KernelTimings.Keys);
        Assert.All(y.Data.ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    private static void CheckRelativeRms(float[] expected, float[] actual, double tolerance, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        double error = 0, scale = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{label}[{i}] is non-finite.");
            double difference = (double)actual[i] - expected[i];
            error += difference * difference;
            scale += (double)expected[i] * expected[i];
        }
        double relativeRms = Math.Sqrt(error / Math.Max(scale, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine($"{label}: relative RMS={relativeRms:G6}");
        Assert.True(relativeRms < tolerance,
            $"{label}: relative RMS {relativeRms:G6} exceeds {tolerance:G6}.");
    }

    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }
}
