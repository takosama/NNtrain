using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
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
    public void NativeBFloat16Block128RoundTripMatchesCpuCodesAndBits()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        const int length = 515;
        Bfp8QuantizationDescriptor descriptor =
            Bfp8QuantizationDescriptor.Mix8_32;
        ushort[] sourceBits = Enumerable.Range(0, length)
            .Select(index => TensorStorageCodec.EncodeBFloat16(
                MathF.Sin(index * 0.037f) *
                (0.25f + (index % 131) * 0.021f)))
            .ToArray();
        float[] source = sourceBits
            .Select(TensorStorageCodec.DecodeBFloat16)
            .ToArray();
        Bfp8EncodedStorage reference = Bfp8QuantizationCodec.Default.Encode(
            source,
            descriptor);
        var referenceDecoded = new float[length];
        Bfp8QuantizationCodec.Default.Decode(
            reference.Payload.Span,
            reference.Scales.Span,
            descriptor,
            referenceDecoded);
        ushort[] expectedBits = referenceDecoded
            .Select(TensorStorageCodec.EncodeBFloat16)
            .ToArray();

        NativeCudaDevice device = ForgetMemoryV2Cuda.GetAccelerator(0);
        using NativeCudaBuffer<ushort> input = device.Allocate1D(sourceBits);
        using NativeCudaBuffer<sbyte> payload =
            device.Allocate1D<sbyte>(length);
        using NativeCudaBuffer<float> scales = device.Allocate1D<float>(
            descriptor.GetScaleCount(length));
        using NativeCudaBuffer<ushort> decoded =
            device.Allocate1D<ushort>(length);
        CudaBfp8Native.QuantizeBFloat16(
            0, input, payload, scales, descriptor);
        CudaBfp8Native.DequantizeBFloat16(
            0, payload, scales, decoded, descriptor);
        device.Synchronize();

        var actualPayload = new sbyte[length];
        var actualScales = new float[scales.Length];
        var actualBits = new ushort[length];
        payload.CopyToCPU(actualPayload);
        scales.CopyToCPU(actualScales);
        decoded.CopyToCPU(actualBits);
        Assert.Equal(reference.Payload.ToArray(), actualPayload);
        Assert.Equal(reference.Scales.ToArray(), actualScales);
        Assert.Equal(expectedBits, actualBits);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TensorWideLargeTailUsesExactMultiBlockQuantization(
        bool bfloat16Source)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        // More than 1024 * 256 values exercises the capped reduction grid's
        // grid-stride tail. Put the unique maximum at the final element so a
        // missed tail changes both the scale and payload deterministically.
        const int length = 300_137;
        float[] source = Enumerable.Range(0, length)
            .Select(static index =>
                MathF.Sin(index * 0.00391f) * (0.5f + index % 31 * 0.02f))
            .ToArray();
        source[^1] = 19.875f;
        NativeCudaDevice device = ForgetMemoryV2Cuda.GetAccelerator(0);
        using NativeCudaBuffer<sbyte> payload =
            device.Allocate1D<sbyte>(length);
        using NativeCudaBuffer<float> scales = device.Allocate1D<float>(1);

        float[] referenceSource;
        if (bfloat16Source)
        {
            ushort[] bits = source
                .Select(TensorStorageCodec.EncodeBFloat16)
                .ToArray();
            referenceSource = bits
                .Select(TensorStorageCodec.DecodeBFloat16)
                .ToArray();
            using NativeCudaBuffer<ushort> input = device.Allocate1D(bits);
            CudaBfp8Native.QuantizeBFloat16(
                0,
                input,
                payload,
                scales,
                Bfp8QuantizationDescriptor.TensorWide);
        }
        else
        {
            referenceSource = source;
            using NativeCudaBuffer<float> input = device.Allocate1D(source);
            CudaBfp8Native.QuantizeFloat32(
                0,
                input,
                payload,
                scales,
                Bfp8QuantizationDescriptor.TensorWide);
        }
        device.Synchronize();

        Bfp8EncodedStorage expected = Bfp8QuantizationCodec.Default.Encode(
            referenceSource,
            Bfp8QuantizationDescriptor.TensorWide);
        var actualPayload = new sbyte[length];
        var actualScale = new float[1];
        payload.CopyToCPU(actualPayload);
        scales.CopyToCPU(actualScale);
        Assert.Equal(expected.Payload.ToArray(), actualPayload);
        Assert.Equal(expected.Scales.Span[0], actualScale[0]);
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

    [Theory]
    [InlineData(1)]
    [InlineData(257)]
    [InlineData(515)]
    public void PureGradientPublishMakesBfp8TheOnlyNumericAuthority(int length)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            float[] source = Enumerable.Range(0, length)
                .Select(index => index == length - 1
                    ? 9.8125f
                    : MathF.Sin(index * 0.113f) * 2.137f)
                .ToArray();
            Tensor tensor = Tensor.FromBfp8(
                new float[length],
                [length],
                Bfp8QuantizationDescriptor.TensorWide);
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            tensor.SetCudaGradient(source, 0);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;

            CudaBfp8BufferView published = tensor.PublishCudaBfp8Gradient(0);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            NativeCudaTransferTelemetry transfers =
                NativeCudaRuntime.TransferTelemetry - before;
            Assert.Equal(0, transfers.HostToDeviceBytes);
            Assert.Equal(sizeof(int), transfers.DeviceToHostBytes);
            Assert.True(tensor.HasAuthoritativeCudaBfp8Gradient);
            Assert.Equal(Bfp8ScaleGranularity.Tensor,
                published.Descriptor.Granularity);

            Bfp8EncodedStorage expected = Bfp8QuantizationCodec.Default.Encode(
                source,
                Bfp8QuantizationDescriptor.TensorWide);
            var payload = new sbyte[length];
            var scales = new float[1];
            published.Payload.CopyToCPU(payload);
            published.Scales.CopyToCPU(scales);
            Assert.Equal(expected.Payload.ToArray(), payload);
            Assert.InRange(
                MathF.Abs(expected.Scales.Span[0] - scales[0]),
                0f,
                1e-7f);

            // EnsureCudaGradientBuffer may expose a Float32 decode cache, but
            // never the higher-precision pre-publish accumulator.
            NativeCudaBuffer<float> decodedDevice =
                tensor.EnsureCudaGradientBuffer(0);
            var decoded = new float[length];
            decodedDevice.CopyToCPU(decoded);
            var expectedDecoded = new float[length];
            Bfp8QuantizationCodec.Default.Decode(
                expected.Payload.Span,
                expected.Scales.Span,
                expected.Descriptor,
                expectedDecoded);
            Assert.Equal(expectedDecoded, decoded);
            if (length > 1)
            {
                Assert.Contains(
                    Enumerable.Range(0, length),
                    index => decoded[index] != source[index]);
            }

            tensor.InvalidateCudaBuffers();
        });
    }

    [Fact]
    public void PureGradientAllZeroUsesCanonicalPositiveScale()
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            Tensor tensor = Tensor.FromBfp8(
                new float[513],
                [513],
                Bfp8QuantizationDescriptor.TensorWide);
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            tensor.SetCudaGradient(new float[513], 0);
            CudaBfp8BufferView published = tensor.PublishCudaBfp8Gradient(0);
            ForgetMemoryV2Cuda.GetAccelerator(0).Synchronize();

            var payload = new sbyte[513];
            var scale = new float[1];
            published.Payload.CopyToCPU(payload);
            published.Scales.CopyToCPU(scale);
            Assert.All(payload, value => Assert.Equal((sbyte)0, value));
            Assert.Equal(1f, scale[0]);
            tensor.InvalidateCudaBuffers();
        });
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void PureGradientRejectsNonFiniteValuesBeforeUpload(float value)
    {
        if (!Tensor.IsCudaAvailable())
            return;

        WithCuda(() =>
        {
            Tensor tensor = Tensor.FromBfp8(
                [0f, 0f],
                [2],
                Bfp8QuantizationDescriptor.TensorWide);
            tensor.to(new TorchDevice(TensorDevice.Cuda, 0));
            ArgumentException exception = Assert.Throws<ArgumentException>(
                () => tensor.SetCudaGradient([0f, value], 0));
            Assert.Contains("finite", exception.Message);
            tensor.InvalidateCudaBuffers();
        });
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

    private static void WithCuda(Action action)
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [0];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            using IDisposable precision =
                TensorExecutionContext.PushPrecisionPolicy(
                    PrecisionPolicy.Bfp8);
            action();
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }
}
