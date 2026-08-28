using System.Collections.Concurrent;
using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Interop;

namespace NNtrain;

/// <summary>
/// Managed validation/launch boundary for resident BFP8 CUDA storage. Every
/// path remains on the selected device; lack of capability is a preflight
/// error rather than a CPU fallback.
/// </summary>
internal static class CudaBfp8Native
{
    private static readonly ConcurrentDictionary<int, CudaKernelCapabilities>
        CapabilityCache = new();

    internal static void QuantizeFloat32(
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        nint stream = 0)
    {
        ValidateBuffers(
            deviceIndex, source, payload, scales, descriptor);
        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8QuantizeFloat32(
                deviceIndex,
                source.NativePtr,
                payload.NativePtr,
                scales.NativePtr,
                source.Length,
                descriptor.GetEffectiveBlockSize(source.Length),
                stream),
            "CUDA BFP8 quantize(float32)");
    }

    internal static void DequantizeFloat32(
        int deviceIndex,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        NativeCudaBuffer<float> destination,
        Bfp8QuantizationDescriptor descriptor,
        nint stream = 0)
    {
        ValidateBuffers(
            deviceIndex, destination, payload, scales, descriptor);
        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8DequantizeFloat32(
                deviceIndex,
                payload.NativePtr,
                scales.NativePtr,
                destination.NativePtr,
                destination.Length,
                descriptor.GetEffectiveBlockSize(destination.Length),
                stream),
            "CUDA BFP8 dequantize(float32)");
    }

    internal static void QuantizeFloat32Roundtrip(
        int deviceIndex,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        nint stream = 0)
    {
        ValidateBuffers(
            deviceIndex, source, payload, scales, descriptor);
        if (descriptor.Granularity != Bfp8ScaleGranularity.Block)
        {
            throw new ArgumentException(
                "Roundtrip quantization requires block-scaled BFP8.",
                nameof(descriptor));
        }
        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8QuantizeFloat32Roundtrip(
                deviceIndex,
                source.NativePtr,
                payload.NativePtr,
                scales.NativePtr,
                source.Length,
                descriptor.GetEffectiveBlockSize(source.Length),
                stream),
            "CUDA BFP8 quantize/roundtrip(float32)");
    }

    internal static void DequantizeBFloat16(
        int deviceIndex,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        NativeCudaBuffer<ushort> destination,
        Bfp8QuantizationDescriptor descriptor,
        nint stream = 0)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (destination.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "BFP8 destination must reside on the requested CUDA device.",
                nameof(destination));
        }
        ValidateEncodedBuffers(
            deviceIndex,
            payload,
            scales,
            descriptor,
            destination.Length);
        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        RequireCapability(deviceIndex, CudaKernelFeature.BFloat16);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8DequantizeBFloat16(
                deviceIndex,
                payload.NativePtr,
                scales.NativePtr,
                destination.NativePtr,
                destination.Length,
                descriptor.GetEffectiveBlockSize(destination.Length),
                stream),
            "CUDA BFP8 dequantize(bfloat16)");
    }

    internal static void QuantizeBFloat16(
        int deviceIndex,
        NativeCudaBuffer<ushort> source,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        nint stream = 0)
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateEncodedBuffers(
            deviceIndex,
            payload,
            scales,
            descriptor,
            source.Length);
        if (source.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "BFP8 BF16 source must reside on the requested CUDA device.",
                nameof(source));
        }
        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        RequireCapability(deviceIndex, CudaKernelFeature.BFloat16);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8QuantizeBFloat16(
                deviceIndex,
                source.NativePtr,
                payload.NativePtr,
                scales.NativePtr,
                source.Length,
                descriptor.GetEffectiveBlockSize(source.Length),
                stream),
            "CUDA BFP8 quantize(bfloat16)");
    }

    internal static void RequantizeInt32(
        int deviceIndex,
        NativeCudaBuffer<int> source,
        CudaBfp8BufferView left,
        CudaBfp8BufferView right,
        CudaBfp8BufferView? bias,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor outputDescriptor,
        int outputWidth,
        bool applyRelu,
        nint stream = 0)
    {
        ArgumentNullException.ThrowIfNull(source);
        ValidateEncodedBuffers(
            deviceIndex,
            payload,
            scales,
            outputDescriptor,
            source.Length);
        ValidateTensorWideScale(deviceIndex, left, nameof(left));
        ValidateTensorWideScale(deviceIndex, right, nameof(right));
        if (source.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "Int32 GEMM output must reside on the requested CUDA device.",
                nameof(source));
        }
        if (outputWidth <= 0 || source.Length % outputWidth != 0)
            throw new ArgumentOutOfRangeException(nameof(outputWidth));

        nint biasPayload = 0;
        nint biasScales = 0;
        int biasBlockSize = 1;
        if (bias is { } biasView)
        {
            ValidateTensorWideScale(deviceIndex, biasView, nameof(bias));
            if (biasView.Payload.Length != outputWidth)
            {
                throw new ArgumentException(
                    "BFP8 bias length must match the GEMM output width.",
                    nameof(bias));
            }
            biasPayload = biasView.Payload.NativePtr;
            biasScales = biasView.Scales.NativePtr;
            biasBlockSize = biasView.Descriptor.GetEffectiveBlockSize(
                biasView.Payload.Length);
        }

        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        RequireCapability(deviceIndex, CudaKernelFeature.Int8TensorCores);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8RequantizeInt32(
                deviceIndex,
                source.NativePtr,
                left.Scales.NativePtr,
                right.Scales.NativePtr,
                biasPayload,
                biasScales,
                payload.NativePtr,
                scales.NativePtr,
                source.Length,
                outputWidth,
                outputDescriptor.GetEffectiveBlockSize(source.Length),
                biasBlockSize,
                applyRelu,
                stream),
            "CUDA BFP8 requantize(int32)");
    }

    internal static void TransposeInt8RowToColumn(
        int deviceIndex,
        NativeCudaBuffer<sbyte> source,
        NativeCudaBuffer<sbyte> destination,
        int rows,
        int columns,
        nint stream = 0)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        int length = checked(rows * columns);
        if (source.Length != length || destination.Length != length)
        {
            throw new ArgumentException(
                "BFP8 Int8 layout transform buffer length mismatch.");
        }
        if (source.Device.Index != deviceIndex
            || destination.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "BFP8 Int8 layout transform buffers must reside on the " +
                "requested CUDA device.");
        }
        RequireCapability(deviceIndex, CudaKernelFeature.Bfp8Quantization);
        RequireCapability(deviceIndex, CudaKernelFeature.Int8TensorCores);
        NativeCudaRuntime.Check(
            CudaNativeGateway.Bfp8TransposeInt8(
                deviceIndex,
                source.NativePtr,
                destination.NativePtr,
                rows,
                columns,
                stream),
            "CUDA BFP8 transpose(int8)");
    }

    internal static CudaKernelCapabilities GetCapabilities(int deviceIndex)
        => CapabilityCache.GetOrAdd(
            deviceIndex,
            static device => NativeCudaRuntime.GetKernelCapabilities(device));

    private static void RequireCapability(
        int deviceIndex,
        CudaKernelFeature feature)
    {
        CudaKernelCapabilities capabilities = GetCapabilities(deviceIndex);
        if (!capabilities.Supports(feature))
        {
            throw new NotSupportedException(
                $"CUDA device {deviceIndex} does not support {feature}. " +
                "BFP8 CPU fallback is forbidden.");
        }
    }

    private static void ValidateBuffers(
        int deviceIndex,
        NativeCudaBuffer<float> values,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(values);
        ValidateEncodedBuffers(
            deviceIndex,
            payload,
            scales,
            descriptor,
            values.Length);
        if (values.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "BFP8 values must reside on the requested CUDA device.",
                nameof(values));
        }
    }

    private static void ValidateEncodedBuffers(
        int deviceIndex,
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        int valueCount)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(scales);
        ArgumentNullException.ThrowIfNull(descriptor);
        if (valueCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(valueCount));
        if (payload.Length != valueCount)
            throw new ArgumentException("BFP8 payload length mismatch.", nameof(payload));
        if (scales.Length != descriptor.GetScaleCount(valueCount))
            throw new ArgumentException("BFP8 scale count mismatch.", nameof(scales));
        if (payload.Device.Index != deviceIndex
            || scales.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "BFP8 payload and scales must reside on the requested CUDA device.");
        }
    }

    private static void ValidateTensorWideScale(
        int deviceIndex,
        CudaBfp8BufferView view,
        string parameterName)
    {
        if (view.Descriptor.Granularity != Bfp8ScaleGranularity.Tensor
            || view.Scales.Length != 1
            || view.Payload.Device.Index != deviceIndex
            || view.Scales.Device.Index != deviceIndex)
        {
            throw new ArgumentException(
                "The cuBLASLt Int8 route requires one resident tensor-wide scale.",
                parameterName);
        }
    }
}
