using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Quantization;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class Bfp8QuantizationTests
{
    [Fact]
    public void ExistingEnumValuesRemainStableAndBfp8IsAppended()
    {
        Assert.Equal(0, (int)TensorDType.Float32);
        Assert.Equal(1, (int)TensorDType.Float16);
        Assert.Equal(2, (int)TensorDType.Float8E4M3Fn);
        Assert.Equal(3, (int)TensorDType.Float8E5M2);
        Assert.Equal(4, (int)TensorDType.Float4);
        Assert.Equal(5, (int)TensorDType.Float2);
        Assert.Equal(6, (int)TensorDType.Ternary1Bit58);
        Assert.Equal(7, (int)TensorDType.BFloat16);
        Assert.Equal(8, (int)TensorDType.Bfp8);
        Assert.Equal(3, (int)TensorPrecisionMode.Bfp8);
        Assert.Equal(4, (int)TensorPrecisionMode.Mix8_32);
    }

    [Fact]
    public void TensorWideStorageUsesOneScale()
    {
        float[] source = [-4f, -1f, 0f, 0.75f, 3f];
        Tensor tensor = Tensor.FromBfp8(
            source,
            [source.Length],
            Bfp8QuantizationDescriptor.TensorWide);

        Assert.Equal(TensorDType.Bfp8, tensor.DType);
        Assert.Equal(TensorDType.Bfp8, tensor.ComputeDType);
        Assert.Equal(TensorDType.Float32, tensor.AccumulationDType);
        Assert.Equal(Bfp8ScaleGranularity.Tensor, tensor.Bfp8Quantization!.Granularity);
        TensorQuantizationMetadata metadata = Assert.IsType<TensorQuantizationMetadata>(
            tensor.StorageDescriptor.EffectiveMetadata.Quantization);
        float scale = Assert.Single(metadata.Scales);
        Assert.Equal(source.Length, metadata.BlockSize);
        Assert.Equal(source.Length + sizeof(float), tensor.StorageByteLength);
        AssertClose(source, tensor.Data, scale * 0.501f);

        Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
            source,
            Bfp8QuantizationDescriptor.TensorWide);
        Assert.Contains(encoded.Payload.ToArray(), value => value < 0);
        Assert.Contains(encoded.Payload.ToArray(), value => value > 0);
    }

    [Fact]
    public void Mix8StorageUsesExactlyBlock128ScalesAndFloat32StablePolicy()
    {
        float[] source = Enumerable.Range(0, 300)
            .Select(index => MathF.Sin(index * 0.17f) * (1f + index / 50f))
            .ToArray();
        Tensor tensor = Tensor.FromBfp8(
            source,
            [source.Length],
            Bfp8QuantizationDescriptor.Mix8_32);

        TensorQuantizationMetadata metadata = tensor.StorageDescriptor
            .EffectiveMetadata.Quantization!;
        Assert.Equal(128, metadata.BlockSize);
        Assert.Equal(3, metadata.Scales.Length);
        Assert.Equal(source.Length + 3 * sizeof(float), tensor.StorageByteLength);

        PrecisionPolicy policy = PrecisionPolicy.Mix8_32;
        Assert.Equal(NumericFormat.Bfp8, policy.ParameterStorage);
        Assert.Equal(NumericFormat.Bfp8, policy.MatrixOperand);
        Assert.Equal(
            GemmExecutionFormat.BFloat16,
            policy.GemmExecutionFormats);
        Assert.Equal(NumericFormat.Float32, policy.Accumulation);
        Assert.Equal(NumericFormat.Float32, policy.Reduction);
        Assert.Equal(NumericFormat.Float32, policy.Normalization);
        Assert.Equal(NumericFormat.Float32, policy.Loss);
        Assert.Equal(NumericFormat.Float32, policy.Gradient);
        Assert.Equal(NumericFormat.Float32, policy.MasterWeight);
        Assert.Equal(NumericFormat.Float32, policy.OptimizerState);

        Assert.Equal(
            GemmExecutionFormat.Int8 | GemmExecutionFormat.BFloat16,
            PrecisionPolicy.Bfp8.GemmExecutionFormats);
        Assert.Equal(
            PrecisionPolicy.Mix16_32,
            PrecisionPolicy.Parse("fp16_32"));
    }

    [Fact]
    public void ZeroBlocksHaveCanonicalPositiveUnitScales()
    {
        Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
            new float[257],
            Bfp8QuantizationDescriptor.Mix8_32);

        Assert.Equal(3, encoded.Scales.Length);
        Assert.All(encoded.Scales.ToArray(), scale => Assert.Equal(1f, scale));
        Assert.All(encoded.Payload.ToArray(), value => Assert.Equal((sbyte)0, value));
    }

    [Fact]
    public void InPlaceConversionKeepsTensorIdentityAndGradient()
    {
        var tensor = new Tensor([1f, -2f, 3f, -4f], [4]);
        tensor.Sum().Backward();
        float[] gradient = tensor.Grad.ToArray();

        Tensor identity = tensor;
        tensor.ConvertStorageInPlace(
            TensorDType.Bfp8,
            Bfp8QuantizationDescriptor.Mix8_32);

        Assert.Same(identity, tensor);
        Assert.Equal(TensorDType.Bfp8, tensor.DType);
        Assert.Equal(gradient, tensor.Grad);
        Assert.Equal(Bfp8ScaleGranularity.Block, tensor.Bfp8Quantization!.Granularity);
    }

    [Fact]
    public void GemmPreflightNeverReturnsCpuFallback()
    {
        var capabilities = new CudaKernelCapabilities(
            8,
            6,
            CudaKernelFeature.TensorCores
                | CudaKernelFeature.BFloat16
                | CudaKernelFeature.Bfp8Quantization
                | CudaKernelFeature.Int8TensorCores);

        CudaBfp8GemmPlan aligned = CudaBfp8GemmDispatch.Preflight(
            capabilities,
            64,
            128,
            256,
            128,
            CudaBfp8ScaleGranularity.TensorWide);
        CudaBfp8GemmPlan tail = CudaBfp8GemmDispatch.Preflight(
            capabilities,
            63,
            127,
            255,
            128,
            CudaBfp8ScaleGranularity.TensorWide);

        Assert.Equal(
            CudaBfp8GemmBackend.CublasLtInt8TensorCore,
            aligned.Backend);
        Assert.Equal(
            CudaBfp8GemmBackend.BFloat16Dequantize,
            tail.Backend);
    }

    [Fact]
    public void GemmPreflightRejectsMissingCudaCapability()
    {
        var capabilities = new CudaKernelCapabilities(
            7,
            5,
            CudaKernelFeature.None);

        NotSupportedException exception = Assert.Throws<NotSupportedException>(
            () => CudaBfp8GemmDispatch.Preflight(
                capabilities, 64, 64, 64, 128));
        Assert.Contains("CPU fallback is forbidden", exception.Message);
    }

    [Theory]
    [InlineData("bfp8", TensorPrecisionMode.Bfp8)]
    [InlineData("mix8_32", TensorPrecisionMode.Mix8_32)]
    public void PrecisionNamesRoundTrip(
        string name,
        TensorPrecisionMode expected)
    {
        TensorPrecisionMode parsed = TensorPrecisionModeNames.Parse(name);
        Assert.Equal(expected, parsed);
        Assert.Equal(TensorDType.Bfp8, parsed.ToStorageDType());
        Assert.Equal(name, TensorPrecisionModeNames.Format(parsed));
    }

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                tolerance);
        }
    }
}
