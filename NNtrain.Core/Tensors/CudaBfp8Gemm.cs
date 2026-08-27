using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Quantization;

namespace NNtrain;

internal readonly record struct CudaBfp8GemmTelemetrySnapshot(
    long Int8TensorCoreExecutions,
    long BFloat16FallbackExecutions,
    long BFloat16DecodeCacheMisses,
    long Int8LayoutTransformCacheMisses,
    CudaBfp8GemmBackend? LastBackend)
{
    public static CudaBfp8GemmTelemetrySnapshot operator -(
        CudaBfp8GemmTelemetrySnapshot left,
        CudaBfp8GemmTelemetrySnapshot right)
        => new(
            left.Int8TensorCoreExecutions - right.Int8TensorCoreExecutions,
            left.BFloat16FallbackExecutions -
                right.BFloat16FallbackExecutions,
            left.BFloat16DecodeCacheMisses -
                right.BFloat16DecodeCacheMisses,
            left.Int8LayoutTransformCacheMisses -
                right.Int8LayoutTransformCacheMisses,
            left.LastBackend);
}

internal static class CudaBfp8GemmTelemetry
{
    private static long _int8TensorCoreExecutions;
    private static long _bfloat16FallbackExecutions;
    private static long _bfloat16DecodeCacheMisses;
    private static long _int8LayoutTransformCacheMisses;
    private static int _lastBackend = -1;

    internal static CudaBfp8GemmTelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _int8TensorCoreExecutions),
        Interlocked.Read(ref _bfloat16FallbackExecutions),
        Interlocked.Read(ref _bfloat16DecodeCacheMisses),
        Interlocked.Read(ref _int8LayoutTransformCacheMisses),
        Volatile.Read(ref _lastBackend) is int backend && backend >= 0
            ? (CudaBfp8GemmBackend)backend
            : null);

    internal static void Record(CudaBfp8GemmBackend backend)
    {
        if (backend == CudaBfp8GemmBackend.CublasLtInt8TensorCore)
            Interlocked.Increment(ref _int8TensorCoreExecutions);
        else
            Interlocked.Increment(ref _bfloat16FallbackExecutions);
        Volatile.Write(ref _lastBackend, (int)backend);
    }

    internal static void RecordBFloat16DecodeCacheMiss()
        => Interlocked.Increment(ref _bfloat16DecodeCacheMisses);

    internal static void RecordInt8LayoutTransformCacheMiss()
        => Interlocked.Increment(ref _int8LayoutTransformCacheMisses);
}

internal sealed class CudaBfp8OwnedBuffers : IDisposable
{
    private NativeCudaBuffer<sbyte>? _payload;
    private NativeCudaBuffer<float>? _scales;
    private int _disposed;

    internal CudaBfp8OwnedBuffers(
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor)
    {
        _payload = payload ?? throw new ArgumentNullException(nameof(payload));
        _scales = scales ?? throw new ArgumentNullException(nameof(scales));
        Descriptor = descriptor
            ?? throw new ArgumentNullException(nameof(descriptor));
    }

    internal Bfp8QuantizationDescriptor Descriptor { get; }

    internal NativeCudaBuffer<sbyte> Payload => Volatile.Read(ref _payload)
        ?? throw new ObjectDisposedException(nameof(CudaBfp8OwnedBuffers));

    internal NativeCudaBuffer<float> Scales => Volatile.Read(ref _scales)
        ?? throw new ObjectDisposedException(nameof(CudaBfp8OwnedBuffers));

    internal (
        NativeCudaBuffer<sbyte> Payload,
        NativeCudaBuffer<float> Scales) Detach()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        NativeCudaBuffer<sbyte> payload = Interlocked.Exchange(
                ref _payload,
                null)
            ?? throw new InvalidOperationException(
                "CUDA BFP8 result buffers were already detached.");
        NativeCudaBuffer<float> scales = Interlocked.Exchange(
                ref _scales,
                null)
            ?? throw new InvalidOperationException(
                "CUDA BFP8 result buffers were already detached.");
        return (payload, scales);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try
        {
            Interlocked.Exchange(ref _payload, null)?.Dispose();
        }
        finally
        {
            Interlocked.Exchange(ref _scales, null)?.Dispose();
        }
    }
}

/// <summary>
/// Resident BFP8 Linear/MatMul implementation. There is intentionally no host
/// fallback: unsupported Int8 shapes dequantize to a versioned BF16 cache and
/// execute the existing BF16 Tensor Core GEMM.
/// </summary>
internal static class CudaBfp8Gemm
{
    internal static CudaBfp8OwnedBuffers MatMulForward(
        Tensor left,
        Tensor right,
        Bfp8QuantizationDescriptor outputDescriptor,
        int batch,
        int m,
        int k,
        int n)
    {
        if (batch != 1)
        {
            // cuBLASLt batch strides are not part of the phase-one Int8 ABI;
            // batched inputs still remain resident on the BF16 route.
            return MatMulForwardBFloat16(
                left, right, outputDescriptor, batch, m, k, n);
        }

        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaBfp8BufferView leftView = left.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8BufferView rightView = right.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8GemmPlan plan = CreatePlan(
            deviceIndex,
            m,
            n,
            k,
            outputDescriptor,
            leftView,
            rightView);
        if (plan.Backend == CudaBfp8GemmBackend.BFloat16Dequantize)
        {
            return MatMulForwardBFloat16(
                left, right, outputDescriptor, batch, m, k, n);
        }

        int outputLength = checked(m * n);
        CudaBfp8OwnedBuffers result = AllocateOutput(
            accelerator,
            outputLength,
            outputDescriptor);
        NativeCudaBuffer<int> exact =
            Tensor.RentCudaIntBuffer(deviceIndex, outputLength);
        try
        {
            NativeCudaBuffer<sbyte> rightColumnMajor =
                right.EnsureCudaBfp8ColumnMajorPayload(
                    k,
                    n,
                    deviceIndex);
            if (!CudaBlasLtInt8.TryMatMul(
                accelerator,
                deviceIndex,
                leftView.Payload,
                rightColumnMajor,
                exact,
                m,
                k,
                n))
            {
                result.Dispose();
                return MatMulForwardBFloat16(
                    left, right, outputDescriptor, batch, m, k, n);
            }
            QuantizeExactInt32(
                deviceIndex,
                accelerator,
                exact,
                leftView,
                rightView,
                bias: null,
                result,
                outputLength,
                n,
                applyRelu: false);
            CudaBfp8GemmTelemetry.Record(plan.Backend);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
        finally
        {
            Tensor.ReturnCudaIntBuffer(accelerator, exact);
        }
    }

    internal static CudaBfp8OwnedBuffers LinearForward(
        Tensor input,
        Tensor weight,
        Tensor bias,
        Bfp8QuantizationDescriptor outputDescriptor,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaBfp8BufferView inputView = input.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8BufferView weightView = weight.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8BufferView biasView = bias.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8GemmPlan plan = CreatePlan(
            deviceIndex,
            rows,
            outputWidth,
            inputWidth,
            outputDescriptor,
            inputView,
            weightView,
            biasView);
        if (plan.Backend == CudaBfp8GemmBackend.BFloat16Dequantize)
        {
            return LinearForwardBFloat16(
                input,
                weight,
                bias,
                outputDescriptor,
                rows,
                inputWidth,
                outputWidth,
                applyRelu);
        }

        int outputLength = checked(rows * outputWidth);
        CudaBfp8OwnedBuffers result = AllocateOutput(
            accelerator,
            outputLength,
            outputDescriptor);
        NativeCudaBuffer<int> exact =
            Tensor.RentCudaIntBuffer(deviceIndex, outputLength);
        try
        {
            if (!CudaBlasLtInt8.TryLinear(
                accelerator,
                deviceIndex,
                inputView.Payload,
                weightView.Payload,
                exact,
                rows,
                inputWidth,
                outputWidth))
            {
                result.Dispose();
                return LinearForwardBFloat16(
                    input,
                    weight,
                    bias,
                    outputDescriptor,
                    rows,
                    inputWidth,
                    outputWidth,
                    applyRelu);
            }
            // Linear computes input * weight^T. The exact dot product scale is
            // therefore inputScale*weightScale; bias and ReLU are fused into
            // the resident requantization kernel.
            QuantizeExactInt32(
                deviceIndex,
                accelerator,
                exact,
                inputView,
                weightView,
                biasView,
                result,
                outputLength,
                outputWidth,
                applyRelu);
            CudaBfp8GemmTelemetry.Record(plan.Backend);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
        finally
        {
            Tensor.ReturnCudaIntBuffer(accelerator, exact);
        }
    }

    internal static void MatMulBackward(
        Tensor left,
        Tensor right,
        Tensor output,
        int batch,
        int m,
        int k,
        int n)
    {
        ThrowIfPureGradientPublishUnsupported(output.Bfp8Quantization);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int length = checked(batch * m * n);
        NativeCudaBuffer<ushort> encodedGradient =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        try
        {
            CudaTensorNative.EncodeBFloat16(
                deviceIndex,
                output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                encodedGradient.NativePtr,
                length);
            CudaBlas.MatMulBackwardBFloat16(
                accelerator,
                deviceIndex,
                left.EnsureCudaBfp8BFloat16Buffer(deviceIndex),
                right.EnsureCudaBfp8BFloat16Buffer(deviceIndex),
                encodedGradient,
                left.EnsureCudaGradientBuffer(deviceIndex),
                right.EnsureCudaGradientBuffer(deviceIndex),
                batch,
                m,
                k,
                n);
            left.MarkCudaGradientMutated(deviceIndex);
            right.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient);
        }
    }

    internal static void LinearBackward(
        Tensor input,
        Tensor weight,
        Tensor bias,
        Tensor output,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        ThrowIfPureGradientPublishUnsupported(output.Bfp8Quantization);
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int length = checked(rows * outputWidth);
        NativeCudaBuffer<ushort> encodedGradient =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        try
        {
            CudaTensorNative.LinearEncodeBFloat16(
                deviceIndex,
                output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                output.EnsureCudaBfp8BFloat16Buffer(deviceIndex).NativePtr,
                encodedGradient.NativePtr,
                length,
                applyRelu);
            CudaBlas.LinearBackwardInputBFloat16(
                accelerator,
                deviceIndex,
                encodedGradient,
                weight.EnsureCudaBfp8BFloat16Buffer(deviceIndex),
                input.EnsureCudaGradientBuffer(deviceIndex),
                rows,
                inputWidth,
                outputWidth);
            CudaBlas.LinearBackwardWeightBFloat16(
                accelerator,
                deviceIndex,
                input.EnsureCudaBfp8BFloat16Buffer(deviceIndex),
                encodedGradient,
                weight.EnsureCudaGradientBuffer(deviceIndex),
                rows,
                inputWidth,
                outputWidth);
            CudaTensorNative.LinearBiasBackward(
                deviceIndex,
                encodedGradient.NativePtr,
                bias.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                rows,
                outputWidth,
                bfloat16: true);
            input.MarkCudaGradientMutated(deviceIndex);
            weight.MarkCudaGradientMutated(deviceIndex);
            bias.MarkCudaGradientMutated(deviceIndex);
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, encodedGradient);
        }
    }

    private static CudaBfp8OwnedBuffers MatMulForwardBFloat16(
        Tensor left,
        Tensor right,
        Bfp8QuantizationDescriptor outputDescriptor,
        int batch,
        int m,
        int k,
        int n)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int outputLength = checked(batch * m * n);
        NativeCudaBuffer<ushort> temporary =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, outputLength);
        CudaBfp8OwnedBuffers result = AllocateOutput(
            accelerator,
            outputLength,
            outputDescriptor);
        try
        {
            CudaBlas.MatMulForwardBFloat16(
                accelerator,
                deviceIndex,
                left.EnsureCudaBfp8BFloat16Buffer(deviceIndex),
                right.EnsureCudaBfp8BFloat16Buffer(deviceIndex),
                temporary,
                batch,
                m,
                k,
                n);
            QuantizeBFloat16(
                deviceIndex, accelerator, temporary, result, outputDescriptor);
            CudaBfp8GemmTelemetry.Record(
                CudaBfp8GemmBackend.BFloat16Dequantize);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, temporary);
        }
    }

    private static CudaBfp8OwnedBuffers LinearForwardBFloat16(
        Tensor input,
        Tensor weight,
        Tensor bias,
        Bfp8QuantizationDescriptor outputDescriptor,
        int rows,
        int inputWidth,
        int outputWidth,
        bool applyRelu)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int outputLength = checked(rows * outputWidth);
        NativeCudaBuffer<ushort> temporary =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, outputLength);
        CudaBfp8OwnedBuffers result = AllocateOutput(
            accelerator,
            outputLength,
            outputDescriptor);
        try
        {
            NativeCudaBuffer<ushort> inputBuffer =
                input.EnsureCudaBfp8BFloat16Buffer(deviceIndex);
            NativeCudaBuffer<ushort> weightBuffer =
                weight.EnsureCudaBfp8BFloat16Buffer(deviceIndex);
            NativeCudaBuffer<ushort> biasBuffer =
                bias.EnsureCudaBfp8BFloat16Buffer(deviceIndex);
            if (!CudaBlasLt.TryLinearForwardBFloat16(
                accelerator,
                deviceIndex,
                inputBuffer,
                weightBuffer,
                biasBuffer,
                temporary,
                rows,
                inputWidth,
                outputWidth,
                applyRelu))
            {
                CudaBlas.LinearForwardBFloat16(
                    accelerator,
                    deviceIndex,
                    inputBuffer,
                    weightBuffer,
                    temporary,
                    rows,
                    inputWidth,
                    outputWidth);
                CudaTensorNative.LinearBias(
                    deviceIndex,
                    temporary.NativePtr,
                    biasBuffer.NativePtr,
                    outputLength,
                    outputWidth,
                    applyRelu,
                    bfloat16: true);
            }
            QuantizeBFloat16(
                deviceIndex, accelerator, temporary, result, outputDescriptor);
            CudaBfp8GemmTelemetry.Record(
                CudaBfp8GemmBackend.BFloat16Dequantize);
            return result;
        }
        catch
        {
            result.Dispose();
            throw;
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, temporary);
        }
    }

    private static CudaBfp8GemmPlan CreatePlan(
        int deviceIndex,
        int m,
        int n,
        int k,
        Bfp8QuantizationDescriptor outputDescriptor,
        params CudaBfp8BufferView[] operands)
    {
        bool tensorWide = outputDescriptor.Granularity ==
            Bfp8ScaleGranularity.Tensor;
        foreach (CudaBfp8BufferView operand in operands)
        {
            tensorWide &= operand.Descriptor.Granularity ==
                    Bfp8ScaleGranularity.Tensor
                && operand.Scales.Length == 1;
        }
        CudaKernelCapabilities capabilities =
            CudaBfp8Native.GetCapabilities(deviceIndex);
        return CudaBfp8GemmDispatch.Preflight(
            capabilities,
            m,
            n,
            k,
            outputDescriptor.GetEffectiveBlockSize(checked(m * n)),
            tensorWide
                ? CudaBfp8ScaleGranularity.TensorWide
                : CudaBfp8ScaleGranularity.Block);
    }

    private static CudaBfp8OwnedBuffers AllocateOutput(
        NativeCudaDevice accelerator,
        int length,
        Bfp8QuantizationDescriptor descriptor)
    {
        NativeCudaBuffer<sbyte>? payload = null;
        NativeCudaBuffer<float>? scales = null;
        try
        {
            payload = accelerator.Allocate1D<sbyte>(length);
            scales = accelerator.Allocate1D<float>(
                descriptor.GetScaleCount(length));
            return new CudaBfp8OwnedBuffers(payload, scales, descriptor);
        }
        catch
        {
            payload?.Dispose();
            scales?.Dispose();
            throw;
        }
    }

    private static void QuantizeBFloat16(
        int deviceIndex,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> source,
        CudaBfp8OwnedBuffers destination,
        Bfp8QuantizationDescriptor descriptor)
    {
        CudaBfp8Native.QuantizeBFloat16(
            deviceIndex,
            source,
            destination.Payload,
            destination.Scales,
            descriptor,
            accelerator.DefaultStream);
    }

    private static void QuantizeExactInt32(
        int deviceIndex,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<int> source,
        CudaBfp8BufferView left,
        CudaBfp8BufferView right,
        CudaBfp8BufferView? bias,
        CudaBfp8OwnedBuffers destination,
        int length,
        int outputWidth,
        bool applyRelu)
    {
        _ = length;
        CudaBfp8Native.RequantizeInt32(
            deviceIndex,
            source,
            left,
            right,
            bias,
            destination.Payload,
            destination.Scales,
            destination.Descriptor,
            outputWidth,
            applyRelu,
            accelerator.DefaultStream);
    }

    private static void ThrowIfPureGradientPublishUnsupported(
        Bfp8QuantizationDescriptor? descriptor)
    {
        if (descriptor?.Granularity != Bfp8ScaleGranularity.Tensor)
            return;
        throw new NotSupportedException(
            "Pure BFP8 backward accumulated gradients in FP32, but publishing " +
            "them as resident tensor-wide BFP8 and consuming BFP8 optimizer " +
            "state is not implemented yet. Use mix8_32; silent FP32 gradient " +
            "publication is forbidden.");
    }
}
