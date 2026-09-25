using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcInferenceFusedGemvTests
{
    private const string PackedKernel = "arc_inference_gemv_packed";
    private const string FusedKernel = "arc_inference_gemv_fused_packed";

    [Theory]
    [InlineData(1536, TensorDType.BFloat16, TensorDType.Bfp8,
        TensorDType.Bfp8, TensorPrecisionMode.Mix8_16, false)]
    [InlineData(11500, TensorDType.Bfp8, TensorDType.Bfp8,
        TensorDType.Bfp8, TensorPrecisionMode.Mix8_16, true)]
    [InlineData(512, TensorDType.Float32, TensorDType.Bfp8,
        TensorDType.Bfp8, TensorPrecisionMode.Mix8_32, false)]
    [InlineData(512, TensorDType.Bfp8, TensorDType.Float32,
        TensorDType.Float32, TensorPrecisionMode.Mix8_32, false)]
    public void FusedPackedOperandsMatchExistingGemv(
        int outputWidth, TensorDType inputStorage, TensorDType weightStorage,
        TensorDType biasStorage, TensorPrecisionMode precision, bool relu)
    {
        RequireSg16Arc();
        const int k = 512;
        float[] inputs = Values(k, 13, .08f);
        float[] weights = Values(checked(outputWidth * k), 29, .012f);
        float[] biases = Values(outputWidth, 41, .018f);
        biases[23] = .9f;

        Snapshot Run(bool fused)
        {
            using var execution = Tensor.BeginArcExecution(precision: precision,
                options: new ArcExecutionOptions
                {
                    InferenceGemv = true,
                    InferenceFusedGemv = fused,
                    Mix8_16Bf16Activations = true,
                });
            Tensor x = Make(inputs, [1, k], inputStorage);
            Tensor w = Make(weights, [outputWidth, k], weightStorage);
            Tensor b = Make(biases, [outputWidth], biasStorage);
            float[] values;
            TensorDType outputStorage;
            long launches;
            using (AutogradContext.NoGrad())
            {
                long before = Tensor.ArcLane.KernelLaunchCount;
                Tensor y = x.LinearLastDim(w, b, applyRelu: relu);
                outputStorage = y.DType;
                values = y.Data.ToArray();
                launches = Tensor.ArcLane.KernelLaunchCount - before;
            }
            Tensor.ArcLane.CheckNumericStatus();
            return new(values, outputStorage,
                Tensor.ArcLane.KernelTimings.Keys.ToArray(), launches);
        }

        Snapshot reference = Run(false);
        Snapshot candidate = Run(true);
        Assert.Contains(PackedKernel, reference.Kernels);
        Assert.DoesNotContain(FusedKernel, reference.Kernels);
        Assert.Contains(FusedKernel, candidate.Kernels);
        Assert.DoesNotContain(PackedKernel, candidate.Kernels);
        Assert.Equal(reference.OutputStorage, candidate.OutputStorage);
        Assert.Equal(23, Array.IndexOf(candidate.Values, candidate.Values.Max()));
        Assert.Equal(Array.IndexOf(reference.Values, reference.Values.Max()),
            Array.IndexOf(candidate.Values, candidate.Values.Max()));
        CheckRelativeRms(reference.Values, candidate.Values,
            candidate.OutputStorage == TensorDType.Float32 ? 2e-4 : 2e-3);

        // The direct BF16 case removes input decode, bias decode, and publish.
        if (precision == TensorPrecisionMode.Mix8_16)
        {
            Assert.Equal(TensorDType.BFloat16, candidate.OutputStorage);
            Assert.DoesNotContain("resident_bf16", candidate.Kernels);
            Assert.True(reference.Launches - candidate.Launches >= 3,
                $"Expected at least three fewer launches, old={reference.Launches}, "
                + $"fused={candidate.Launches}.");
        }
    }

    [Fact]
    public void TensorParallelPartialRemainsFloat32AndSkipsInputDecode()
    {
        RequireSg16Arc();
        const int k = 256, n = 512;
        float[] inputs = Values(k, 5, .1f);
        float[] weights = Values(n * k, 17, .03f);

        Snapshot Run(bool fused)
        {
            using var execution = Tensor.BeginArcExecution(
                precision: TensorPrecisionMode.Mix8_16,
                options: new ArcExecutionOptions
                {
                    InferenceGemv = true,
                    InferenceFusedGemv = fused,
                    Mix8_16Bf16Activations = true,
                });
            Tensor x = Make(inputs, [1, k], TensorDType.Bfp8);
            Tensor w = Make(weights, [n, k], TensorDType.Bfp8);
            float[] values;
            TensorDType outputStorage;
            long launches;
            using (AutogradContext.NoGrad())
            {
                long before = Tensor.ArcLane.KernelLaunchCount;
                Tensor partial = x.ArcInferenceLinearPartial(w);
                outputStorage = partial.DType;
                values = partial.Data.ToArray();
                launches = Tensor.ArcLane.KernelLaunchCount - before;
            }
            Tensor.ArcLane.CheckNumericStatus();
            return new(values, outputStorage,
                Tensor.ArcLane.KernelTimings.Keys.ToArray(), launches);
        }

        Snapshot reference = Run(false), candidate = Run(true);
        Assert.Equal(TensorDType.Float32, candidate.OutputStorage);
        Assert.Contains(PackedKernel, reference.Kernels);
        Assert.Contains(FusedKernel, candidate.Kernels);
        Assert.DoesNotContain("decode_bfp8", candidate.Kernels);
        Assert.True(candidate.Launches < reference.Launches);
        CheckRelativeRms(reference.Values, candidate.Values, 2e-4);
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(2, true)]
    public void RecordingOrPrefillDoesNotUseFusedGemv(int rows, bool noGrad)
    {
        RequireSg16Arc();
        const int k = 79, n = 65;
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Float32,
            options: new ArcExecutionOptions
            {
                InferenceGemv = true,
                InferenceFusedGemv = true,
            });
        Tensor x = Make(Values(rows * k, 3, .05f),
            [rows, k], TensorDType.Float32);
        Tensor w = Make(Values(n * k, 7, .02f),
            [n, k], TensorDType.Float32);
        Tensor b = Make(Values(n, 11, .01f), [n], TensorDType.Float32);
        using var noGradScope = noGrad ? AutogradContext.NoGrad() : null;
        Tensor y = x.LinearLastDim(w, b, applyRelu: false);
        Assert.All(y.Data.ToArray(), value => Assert.True(float.IsFinite(value)));
        Assert.DoesNotContain(FusedKernel, Tensor.ArcLane.KernelTimings.Keys);
        Assert.DoesNotContain(PackedKernel, Tensor.ArcLane.KernelTimings.Keys);
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

    private static void CheckRelativeRms(float[] reference, float[] candidate, double tolerance)
    {
        Assert.Equal(reference.Length, candidate.Length);
        double error = 0, signal = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            Assert.True(float.IsFinite(reference[i]) && float.IsFinite(candidate[i]));
            double delta = (double)candidate[i] - reference[i];
            error += delta * delta;
            signal += (double)reference[i] * reference[i];
        }
        double relativeRms = Math.Sqrt(error / Math.Max(signal, 1e-30));
        TestContext.Current.TestOutputHelper?.WriteLine($"Fused GEMV RMS={relativeRms:G6}");
        Assert.True(relativeRms <= tolerance,
            $"Fused GEMV relative RMS {relativeRms:G6} exceeds {tolerance:G6}.");
    }

    private sealed record Snapshot(
        float[] Values, TensorDType OutputStorage, string[] Kernels, long Launches);
}
