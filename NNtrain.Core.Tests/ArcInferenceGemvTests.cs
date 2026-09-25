using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcInferenceGemvTests
{
    private const string Kernel = "arc_inference_gemv_packed";

    [Theory]
    [InlineData(512, 512, TensorDType.Float32, TensorPrecisionMode.Float32, false)]
    [InlineData(1536, 512, TensorDType.BFloat16, TensorPrecisionMode.Mix16_32, true)]
    [InlineData(11500, 512, TensorDType.Bfp8, TensorPrecisionMode.Mix8_16, false)]
    [InlineData(65, 79, TensorDType.Bfp8, TensorPrecisionMode.Mix8_32, true)]
    public void ResidentSingleTokenLinearReadsPackedWeightAndPreservesPrediction(
        int outputWidth, int inputWidth, TensorDType storage,
        TensorPrecisionMode precision, bool relu)
    {
        RequireSg16Arc();
        float[] inputs = Values(inputWidth, 17, .08f);
        float[] weights = Values(checked(outputWidth * inputWidth), 29, .01f);
        float[] biases = Values(outputWidth, 41, .015f);
        biases[23] = 0.75f; // The winning token has an intentional margin.

        float[] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: new ArcExecutionOptions
                {
                    InferenceGemv = enabled,
                    InferenceFusedGemv = false,
                });
            Tensor x = Make(inputs, [1, inputWidth], storage);
            Tensor w = Make(weights, [outputWidth, inputWidth], storage);
            Tensor b = Make(biases, [outputWidth], storage);
            float[] values;
            using (AutogradContext.NoGrad())
            {
                Tensor y = x.LinearLastDim(w, b, applyRelu: relu);
                values = y.Data.ToArray();
            }
            Assert.Equal(enabled, Tensor.ArcLane.KernelTimings.ContainsKey(Kernel));
            Tensor.ArcLane.CheckNumericStatus();
            return values;
        }

        float[] reference = Run(false);
        float[] candidate = Run(true);
        CheckRelativeRms(reference, candidate, storage switch
        {
            TensorDType.Float32 => 1e-4,
            TensorDType.BFloat16 => 2e-3,
            _ => 2e-2,
        });
        Assert.Equal(Array.IndexOf(reference, reference.Max()),
            Array.IndexOf(candidate, candidate.Max()));
        Assert.Equal(23, Array.IndexOf(candidate, candidate.Max()));
    }

    [Fact]
    public void TensorParallelPartialKeepsFloat32WithoutBias()
    {
        RequireSg16Arc();
        const int k = 256, n = 512;
        float[] inputs = Values(k, 3, .1f);
        float[] weights = Values(n * k, 11, .05f);

        float[] Run(bool enabled)
        {
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    InferenceGemv = enabled,
                    InferenceFusedGemv = false,
                });
            Tensor x = Make(inputs, [1, k], TensorDType.Bfp8);
            Tensor w = Make(weights, [n, k], TensorDType.Bfp8);
            float[] values;
            using (AutogradContext.NoGrad())
            {
                Tensor partial = x.ArcInferenceLinearPartial(w);
                Assert.Equal(TensorDType.Float32, partial.DType);
                values = partial.Data.ToArray();
            }
            Assert.Equal(enabled, Tensor.ArcLane.KernelTimings.ContainsKey(Kernel));
            Tensor.ArcLane.CheckNumericStatus();
            return values;
        }

        CheckRelativeRms(Run(false), Run(true), 2e-3);
    }

    [Theory]
    [InlineData(1, false)] // Training can have a one-row projection.
    [InlineData(2, true)] // Inference prefill has more than one row.
    public void NonInferenceOrMultipleRowsKeepExistingDispatch(int rows, bool noGrad)
    {
        RequireSg16Arc();
        const int k = 79, n = 65;
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Float32,
            options: new ArcExecutionOptions
            {
                InferenceGemv = true,
                InferenceFusedGemv = false,
            });
        Tensor x = Make(Values(rows * k, 7, .08f), [rows, k], TensorDType.Float32);
        Tensor w = Make(Values(n * k, 19, .04f), [n, k], TensorDType.Float32);
        Tensor b = Make(Values(n, 31, .01f), [n], TensorDType.Float32);
        using var noGradScope = noGrad ? AutogradContext.NoGrad() : null;
        Tensor y = x.LinearLastDim(w, b, applyRelu: false);
        Assert.All(y.Data.ToArray(), value => Assert.True(float.IsFinite(value)));
        Assert.DoesNotContain(Kernel, Tensor.ArcLane.KernelTimings.Keys);
    }

    private static void RequireSg16Arc()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx
            || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel Arc is required.");
    }

    private static Tensor Make(float[] source, int[] shape, TensorDType storage)
    {
        var value = new Tensor((float[])source.Clone(), shape);
        if (storage == TensorDType.Bfp8)
            value.ConvertStorageInPlace(storage, Bfp8QuantizationDescriptor.Block(32));
        else if (storage == TensorDType.BFloat16)
            value.ConvertStorageInPlace(storage);
        return value;
    }

    private static float[] Values(int count, int seed, float scale)
    {
        var values = new float[count];
        for (int i = 0; i < count; i++)
            values[i] = MathF.Sin(i * .013f + seed * .19f) * scale;
        return values;
    }

    private static void CheckRelativeRms(float[] reference, float[] actual, double tolerance)
    {
        Assert.Equal(reference.Length, actual.Length);
        double error = 0, signal = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"non-finite GEMV output at {i}");
            double delta = (double)actual[i] - reference[i];
            error += delta * delta;
            signal += (double)reference[i] * reference[i];
        }
        double relativeRms = Math.Sqrt(error / Math.Max(signal, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine($"GEMV relative RMS: {relativeRms:G6}");
        Assert.True(relativeRms <= tolerance,
            $"GEMV relative RMS {relativeRms:G6} exceeds {tolerance:G6}.");
    }
}
