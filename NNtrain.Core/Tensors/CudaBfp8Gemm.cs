using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Cuda.Quantization;

namespace NNtrain;

internal readonly record struct CudaBfp8GemmTelemetrySnapshot(
    long Int8TensorCoreExecutions,
    long BFloat16FallbackExecutions,
    long DirectBFloat16LossHeadExecutions,
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
            left.DirectBFloat16LossHeadExecutions -
                right.DirectBFloat16LossHeadExecutions,
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
    private static long _directBFloat16LossHeadExecutions;
    private static long _bfloat16DecodeCacheMisses;
    private static long _int8LayoutTransformCacheMisses;
    private static int _lastBackend = -1;

    internal static CudaBfp8GemmTelemetrySnapshot Snapshot => new(
        Interlocked.Read(ref _int8TensorCoreExecutions),
        Interlocked.Read(ref _bfloat16FallbackExecutions),
        Interlocked.Read(ref _directBFloat16LossHeadExecutions),
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

    internal static void RecordDirectBFloat16LossHead()
        => Interlocked.Increment(ref _directBFloat16LossHeadExecutions);

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

    internal static CudaBfp8OwnedBuffers Allocate(
        NativeCudaDevice accelerator,
        int length,
        Bfp8QuantizationDescriptor descriptor)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(descriptor);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        NativeCudaBuffer<sbyte>? payload = null;
        NativeCudaBuffer<float>? scales = null;
        try
        {
            payload = accelerator.Allocate1D<sbyte>(
                length,
                CudaMemoryKind.Transient);
            scales = accelerator.Allocate1D<float>(
                descriptor.GetScaleCount(length),
                CudaMemoryKind.Transient);
            return new CudaBfp8OwnedBuffers(payload, scales, descriptor);
        }
        catch (Exception allocationFailure)
        {
            List<Exception>? cleanupFailures = null;
            TryDispose(payload, ref cleanupFailures);
            TryDispose(scales, ref cleanupFailures);
            if (cleanupFailures is not null)
            {
                cleanupFailures.Insert(0, allocationFailure);
                throw new AggregateException(
                    "CUDA BFP8 output allocation and rollback failed.",
                    cleanupFailures);
            }
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(allocationFailure)
                .Throw();
            throw;
        }
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
        List<Exception>? failures = null;
        TryDispose(Interlocked.Exchange(ref _payload, null), ref failures);
        TryDispose(Interlocked.Exchange(ref _scales, null), ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                "CUDA BFP8 payload/scale cleanup failed.", failures);
        }
    }

    private static void TryDispose(
        IDisposable? resource,
        ref List<Exception>? failures)
    {
        if (resource is null)
            return;
        try
        {
            resource.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
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

    internal static CudaBfp8OwnedBuffers MatMulTransposedRightForward(
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
        CudaBfp8BufferView leftView = left.EnsureCudaBfp8Buffer(deviceIndex);
        CudaBfp8BufferView rightView = right.EnsureCudaBfp8Buffer(deviceIndex);

        if (batch == 1)
        {
            CudaBfp8GemmPlan plan = CreatePlan(
                deviceIndex,
                m,
                n,
                k,
                outputDescriptor,
                leftView,
                rightView);
            if (plan.Backend == CudaBfp8GemmBackend.CublasLtInt8TensorCore)
            {
                int outputLength = checked(m * n);
                CudaBfp8OwnedBuffers result = AllocateOutput(
                    accelerator,
                    outputLength,
                    outputDescriptor);
                NativeCudaBuffer<int> exact =
                    Tensor.RentCudaIntBuffer(deviceIndex, outputLength);
                try
                {
                    if (CudaBlasLtInt8.TryLinear(
                            accelerator,
                            deviceIndex,
                            leftView.Payload,
                            rightView.Payload,
                            exact,
                            m,
                            k,
                            n))
                    {
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
                result.Dispose();
            }
        }

        int length = checked(batch * m * n);
        NativeCudaBuffer<ushort> temporary =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        CudaBfp8OwnedBuffers fallback = AllocateOutput(
            accelerator,
            length,
            outputDescriptor);
        using CudaBfp8BFloat16Lease leftDecode =
            left.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease rightDecode =
            right.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        try
        {
            CudaBlas.MatMulTransposedRightForwardBFloat16(
                accelerator,
                deviceIndex,
                leftDecode.Buffer,
                rightDecode.Buffer,
                temporary,
                batch,
                m,
                k,
                n);
            QuantizeBFloat16(
                deviceIndex,
                accelerator,
                temporary,
                fallback,
                outputDescriptor);
            CudaBfp8GemmTelemetry.Record(
                CudaBfp8GemmBackend.BFloat16Dequantize);
            return fallback;
        }
        catch
        {
            fallback.Dispose();
            throw;
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, temporary);
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
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int length = checked(batch * m * n);
        NativeCudaBuffer<ushort> encodedGradient =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        using CudaBfp8BFloat16Lease leftDecode =
            left.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease rightDecode =
            right.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
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
                leftDecode.Buffer,
                rightDecode.Buffer,
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
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int length = checked(rows * outputWidth);
        // The direct loss-head path no longer needs its BF16 logits after
        // cross-entropy backward has consumed them. Reuse that 400+ MiB
        // allocation in-place for the encoded logits gradient instead of
        // renting an equally large second buffer.
        NativeCudaBuffer<ushort>? rentedGradient = null;
        bool encodedGradientReady = output.TryGetCudaBFloat16GradientBuffer(
            deviceIndex,
            out NativeCudaBuffer<ushort>? directGradient);
        NativeCudaBuffer<ushort> encodedGradient = encodedGradientReady
            ? directGradient!
            : rentedGradient =
                Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        using CudaBfp8BFloat16Lease inputDecode =
            input.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease weightDecode =
            weight.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        try
        {
            if (!encodedGradientReady)
            {
                nint outputGradient = output
                    .EnsureCudaGradientBuffer(deviceIndex).NativePtr;
                if (applyRelu)
                {
                    CudaBfp8BufferView outputView =
                        output.EnsureCudaBfp8Buffer(deviceIndex);
                    CudaTensorNative.LinearEncodeBfp8Relu(
                        deviceIndex,
                        outputGradient,
                        outputView.Payload.NativePtr,
                        encodedGradient.NativePtr,
                        length);
                }
                else
                {
                    CudaTensorNative.LinearEncodeBFloat16(
                        deviceIndex,
                        outputGradient,
                        nint.Zero,
                        encodedGradient.NativePtr,
                        length,
                        relu: false);
                }
            }
            CudaBlas.LinearBackwardInputBFloat16(
                accelerator,
                deviceIndex,
                encodedGradient,
                weightDecode.Buffer,
                input.EnsureCudaGradientBuffer(deviceIndex),
                rows,
                inputWidth,
                outputWidth);
            CudaBlas.LinearBackwardWeightBFloat16(
                accelerator,
                deviceIndex,
                inputDecode.Buffer,
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
            if (rentedGradient is not null)
            {
                Tensor.ReturnCudaBFloat16Buffer(
                    accelerator,
                    rentedGradient);
            }
        }
    }

    internal static void MatMulTransposedRightBackward(
        Tensor left,
        Tensor right,
        Tensor output,
        int batch,
        int m,
        int k,
        int n)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int length = checked(batch * m * n);
        NativeCudaBuffer<ushort> encodedGradient =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, length);
        using CudaBfp8BFloat16Lease leftDecode =
            left.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease rightDecode =
            right.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        try
        {
            CudaTensorNative.EncodeBFloat16(
                deviceIndex,
                output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                encodedGradient.NativePtr,
                length);
            CudaBlas.MatMulTransposedRightBackwardBFloat16(
                accelerator,
                deviceIndex,
                leftDecode.Buffer,
                rightDecode.Buffer,
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
        using CudaBfp8BFloat16Lease leftDecode =
            left.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease rightDecode =
            right.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        try
        {
            CudaBlas.MatMulForwardBFloat16(
                accelerator,
                deviceIndex,
                leftDecode.Buffer,
                rightDecode.Buffer,
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
        using CudaBfp8BFloat16Lease inputDecode =
            input.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease weightDecode =
            weight.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease biasDecode =
            bias.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        try
        {
            NativeCudaBuffer<ushort> inputBuffer = inputDecode.Buffer;
            NativeCudaBuffer<ushort> weightBuffer = weightDecode.Buffer;
            NativeCudaBuffer<ushort> biasBuffer = biasDecode.Buffer;
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

    /// <summary>
    /// Produces the BF16 accumulator output directly for a consumer such as
    /// cross entropy. This deliberately skips the otherwise redundant
    /// BF16 -&gt; block-BFP8 -&gt; BF16 round trip at the language-model head.
    /// Ownership of the returned transient buffer transfers to the caller.
    /// </summary>
    internal static NativeCudaBuffer<ushort> LinearForwardBFloat16Output(
        Tensor input,
        Tensor weight,
        Tensor bias,
        int rows,
        int inputWidth,
        int outputWidth)
    {
        int deviceIndex = Tensor.CudaDeviceIndex;
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        int outputLength = checked(rows * outputWidth);
        NativeCudaBuffer<ushort> output =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, outputLength);
        using CudaBfp8BFloat16Lease inputDecode =
            input.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease weightDecode =
            weight.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        using CudaBfp8BFloat16Lease biasDecode =
            bias.AcquireCudaBfp8BFloat16Buffer(deviceIndex);
        try
        {
            if (!CudaBlasLt.TryLinearForwardBFloat16(
                accelerator,
                deviceIndex,
                inputDecode.Buffer,
                weightDecode.Buffer,
                biasDecode.Buffer,
                output,
                rows,
                inputWidth,
                outputWidth,
                applyRelu: false))
            {
                CudaBlas.LinearForwardBFloat16(
                    accelerator,
                    deviceIndex,
                    inputDecode.Buffer,
                    weightDecode.Buffer,
                    output,
                    rows,
                    inputWidth,
                    outputWidth);
                CudaTensorNative.LinearBias(
                    deviceIndex,
                    output.NativePtr,
                    biasDecode.Buffer.NativePtr,
                    outputLength,
                    outputWidth,
                    relu: false,
                    bfloat16: true);
            }
            CudaBfp8GemmTelemetry.Record(
                CudaBfp8GemmBackend.BFloat16Dequantize);
            CudaBfp8GemmTelemetry.RecordDirectBFloat16LossHead();
            return output;
        }
        catch
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, output);
            throw;
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
        => CudaBfp8OwnedBuffers.Allocate(
            accelerator, length, descriptor);

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

}
