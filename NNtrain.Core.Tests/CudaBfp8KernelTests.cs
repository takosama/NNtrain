using NNtrain;
using NNtrain.Cuda.Execution;
using Xunit;

public sealed class CudaBfp8KernelTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NativeQuantizeAndDequantizeMatchesCpuReference(bool block128)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        Bfp8QuantizationDescriptor descriptor = block128
            ? Bfp8QuantizationDescriptor.Mix8_32
            : Bfp8QuantizationDescriptor.TensorWide;
        float[] source = Enumerable.Range(0, 515)
            .Select(index =>
                MathF.Sin(index * 0.071f) * (0.5f + (index % 137) * 0.03f))
            .ToArray();
        Bfp8EncodedStorage reference = Bfp8QuantizationCodec.Default.Encode(
            source,
            descriptor);
        NativeCudaDevice device = ForgetMemoryV2Cuda.GetAccelerator(0);

        using NativeCudaBuffer<float> input = device.Allocate1D(source);
        using NativeCudaBuffer<sbyte> payload = device.Allocate1D<sbyte>(source.Length);
        using NativeCudaBuffer<float> scales = device.Allocate1D<float>(
            descriptor.GetScaleCount(source.Length));
        using NativeCudaBuffer<float> decoded = device.Allocate1D<float>(source.Length);
        CudaBfp8Native.QuantizeFloat32(
            0, input, payload, scales, descriptor);
        CudaBfp8Native.DequantizeFloat32(
            0, payload, scales, decoded, descriptor);
        device.Synchronize();

        var actualPayload = new sbyte[source.Length];
        var actualScales = new float[scales.Length];
        var actualDecoded = new float[source.Length];
        payload.CopyToCPU(actualPayload);
        scales.CopyToCPU(actualScales);
        decoded.CopyToCPU(actualDecoded);

        Assert.Equal(reference.Payload.ToArray(), actualPayload);
        Assert.Equal(reference.Scales.Length, actualScales.Length);
        for (int index = 0; index < actualScales.Length; index++)
        {
            Assert.InRange(
                MathF.Abs(reference.Scales.Span[index] - actualScales[index]),
                0f,
                1e-6f);
        }
        AssertClose(reference, actualDecoded);
    }

    [Fact]
    public void NativeBFloat16FallbackStaysOnCuda()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] source = Enumerable.Range(0, 259)
            .Select(index => MathF.Cos(index * 0.19f) * 3.25f)
            .ToArray();
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;
        NativeCudaDevice device = ForgetMemoryV2Cuda.GetAccelerator(0);
        using NativeCudaBuffer<float> input = device.Allocate1D(source);
        using NativeCudaBuffer<sbyte> payload = device.Allocate1D<sbyte>(source.Length);
        using NativeCudaBuffer<float> scales = device.Allocate1D<float>(
            descriptor.GetScaleCount(source.Length));
        using NativeCudaBuffer<ushort> decoded = device.Allocate1D<ushort>(source.Length);

        CudaBfp8Native.QuantizeFloat32(
            0, input, payload, scales, descriptor);
        CudaBfp8Native.DequantizeBFloat16(
            0, payload, scales, decoded, descriptor);
        device.Synchronize();

        var encodedBfloat16 = new ushort[source.Length];
        decoded.CopyToCPU(encodedBfloat16);
        Assert.All(
            encodedBfloat16,
            value => Assert.True(float.IsFinite(
                TensorStorageCodec.DecodeBFloat16(value))));
    }

    [Fact]
    public void TensorKeepsPayloadAndScaleResidentOnSm86Path()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        float[] source = Enumerable.Range(0, 300)
            .Select(index => (index - 150) * 0.03125f)
            .ToArray();
        Tensor tensor = Tensor.FromBfp8(
            source,
            [source.Length],
            Bfp8QuantizationDescriptor.Mix8_32);
        tensor.to(new TorchDevice(TensorDevice.Cuda, 0));

        CudaBfp8BufferView resident = tensor.EnsureCudaBfp8Buffer(0);
        CudaKernelCapabilities capabilities = CudaBfp8Native.GetCapabilities(0);
        Assert.True(capabilities.Supports(CudaKernelFeature.Bfp8Quantization));
        Assert.Equal(source.Length, resident.Payload.Length);
        Assert.Equal(3, resident.Scales.Length);
        Assert.Equal(128, resident.Descriptor.BlockSize);

        tensor.InvalidateCudaBuffers();
    }

    private static void AssertClose(
        Bfp8EncodedStorage reference,
        IReadOnlyList<float> actual)
    {
        var expected = new float[reference.Count];
        Bfp8QuantizationCodec.Default.Decode(
            reference.Payload.Span,
            reference.Scales.Span,
            reference.Descriptor,
            expected);
        for (int index = 0; index < expected.Length; index++)
        {
            Assert.InRange(
                MathF.Abs(expected[index] - actual[index]),
                0f,
                1e-6f);
        }
    }
}
