using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static class CudaFlashAttention
{
    private static int _availability;
    private static int _tensorCoreAvailability;
    internal static bool NativeBackendActive => Volatile.Read(ref _availability) > 0;
    internal static bool TensorCoreBackendActive
        => Volatile.Read(ref _tensorCoreAvailability) > 0;
    internal static bool TryForward(NativeCudaDevice accelerator,
        NativeCudaBuffer<float> projected,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> softmaxLogSumExp, int batch,
        int sequence, int modelWidth, int heads, bool causal)
    {
        if (CudaDispatchPolicy.Current.DisableNativeFlashAttention
            || Volatile.Read(ref _availability) < 0
            || modelWidth / heads > 128)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = CudaNativeGateway.FlashAttentionForward(
                accelerator.Index,
                projected.NativePtr, output.NativePtr,
                softmaxLogSumExp.NativePtr, batch, sequence, modelWidth,
                heads, causal ? 1 : 0, stream);
            if (status != 0)
                throw new InvalidOperationException($"FlashAttention CUDA error {status}.");
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (DllNotFoundException) { Volatile.Write(ref _availability, -1); return false; }
        catch (EntryPointNotFoundException) { Volatile.Write(ref _availability, -1); return false; }
    }

    internal static void Backward(NativeCudaDevice accelerator,
        NativeCudaBuffer<float> projected,
        NativeCudaBuffer<float> output,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> softmaxLogSumExp,
        NativeCudaBuffer<float> projectedGradient, int batch,
        int sequence, int modelWidth, int heads, bool causal)
    {
        accelerator.Bind();
        nint stream = accelerator.DefaultStream;
        int status = CudaNativeGateway.FlashAttentionBackward(
            accelerator.Index,
            projected.NativePtr, output.NativePtr,
            outputGradient.NativePtr, softmaxLogSumExp.NativePtr,
            projectedGradient.NativePtr, batch, sequence,
            modelWidth, heads, causal ? 1 : 0, stream);
        if (status != 0)
            throw new InvalidOperationException($"FlashAttention backward CUDA error {status}.");
    }

    internal static bool TryForwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> softmaxLogSumExp,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        bool causal,
        out bool tensorCore)
    {
        tensorCore = false;
        CudaDispatchPolicy dispatch = CudaDispatchPolicy.Current;
        if (dispatch.DisableNativeFlashAttention
            || Volatile.Read(ref _availability) < 0
            || modelWidth / heads > 128)
        {
            return false;
        }
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            if (!dispatch.DisableTensorCoreFlashAttention
                && Volatile.Read(ref _tensorCoreAvailability) >= 0)
            {
                try
                {
                    bool synchronousLoad =
                        dispatch.DisableAsyncFlashAttention;
                    int tensorCoreStatus = synchronousLoad
                        ? CudaNativeGateway
                            .FlashAttentionForwardBFloat16TensorCoreSync(
                            accelerator.Index,
                            projected.NativePtr,
                            output.NativePtr,
                            softmaxLogSumExp.NativePtr,
                            batch,
                            sequence,
                            modelWidth,
                            heads,
                            causal ? 1 : 0,
                            stream)
                        : CudaNativeGateway
                            .FlashAttentionForwardBFloat16TensorCore(
                            accelerator.Index,
                            projected.NativePtr,
                            output.NativePtr,
                            softmaxLogSumExp.NativePtr,
                            batch,
                            sequence,
                            modelWidth,
                            heads,
                            causal ? 1 : 0,
                            stream);
                    if (tensorCoreStatus != 0)
                    {
                        throw new InvalidOperationException(
                            $"BF16 Tensor Core FlashAttention CUDA error " +
                            $"{tensorCoreStatus}.");
                    }
                    Volatile.Write(ref _tensorCoreAvailability, 1);
                    Volatile.Write(ref _availability, 1);
                    tensorCore = true;
                    return true;
                }
                catch (EntryPointNotFoundException)
                {
                    Volatile.Write(ref _tensorCoreAvailability, -1);
                }
            }
            int status = CudaNativeGateway.FlashAttentionForwardBFloat16(
                accelerator.Index,
                projected.NativePtr,
                output.NativePtr,
                softmaxLogSumExp.NativePtr,
                batch,
                sequence,
                modelWidth,
                heads,
                causal ? 1 : 0,
                stream);
            if (status != 0)
                throw new InvalidOperationException(
                    $"BF16 FlashAttention CUDA error {status}.");
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (DllNotFoundException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static void BackwardBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float>? outputGradient,
        NativeCudaBuffer<ushort>? outputGradientBFloat16,
        NativeCudaBuffer<float> softmaxLogSumExp,
        NativeCudaBuffer<float>? rowDelta,
        NativeCudaBuffer<float>? projectedGradient,
        NativeCudaBuffer<ushort>? projectedGradientBFloat16,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        bool causal,
        bool tensorCore)
    {
        accelerator.Bind();
        nint stream = accelerator.DefaultStream;
        int status;
        if (tensorCore)
        {
            if (rowDelta is null)
            {
                throw new InvalidOperationException(
                    "Tensor Core FlashAttention backward requires row workspace.");
            }
            CudaDispatchPolicy dispatch = CudaDispatchPolicy.Current;
            bool parallelDkv = !dispatch.DisableParallelAttentionDkv;
            bool asyncBackwardLoads = !dispatch.DisableAsyncAttentionBackward;
            if (projectedGradientBFloat16 is not null
                && outputGradientBFloat16 is not null)
            {
                status = asyncBackwardLoads
                    ? CudaNativeGateway
                        .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradient(
                        accelerator.Index,
                        projected.NativePtr,
                        output.NativePtr,
                        outputGradientBFloat16.NativePtr,
                        softmaxLogSumExp.NativePtr,
                        rowDelta.NativePtr,
                        projectedGradientBFloat16.NativePtr,
                        batch,
                        sequence,
                        modelWidth,
                        heads,
                        causal ? 1 : 0,
                        stream)
                    : CudaNativeGateway
                        .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradientSync(
                        accelerator.Index,
                        projected.NativePtr,
                        output.NativePtr,
                        outputGradientBFloat16.NativePtr,
                        softmaxLogSumExp.NativePtr,
                        rowDelta.NativePtr,
                        projectedGradientBFloat16.NativePtr,
                        batch,
                        sequence,
                        modelWidth,
                        heads,
                        causal ? 1 : 0,
                        stream);
            }
            else if (projectedGradientBFloat16 is not null)
            {
                if (outputGradient is null)
                {
                    throw new InvalidOperationException(
                        "FlashAttention backward requires output gradient storage.");
                }
                status = CudaNativeGateway
                    .FlashAttentionBackwardBFloat16TensorCoreBFloat16Gradient(
                    accelerator.Index,
                    projected.NativePtr,
                    output.NativePtr,
                    outputGradient.NativePtr,
                    softmaxLogSumExp.NativePtr,
                    rowDelta.NativePtr,
                    projectedGradientBFloat16.NativePtr,
                    batch,
                    sequence,
                    modelWidth,
                    heads,
                    causal ? 1 : 0,
                    stream);
            }
            else if (projectedGradient is null)
            {
                throw new InvalidOperationException(
                    "FlashAttention backward requires gradient storage.");
            }
            else
            {
                if (outputGradient is null)
                {
                    throw new InvalidOperationException(
                        "FlashAttention backward requires FP32 output gradient storage.");
                }
                status = parallelDkv
                    ? CudaNativeGateway
                        .FlashAttentionBackwardBFloat16TensorCoreParallelDkv(
                        accelerator.Index,
                        projected.NativePtr,
                        output.NativePtr,
                        outputGradient.NativePtr,
                        softmaxLogSumExp.NativePtr,
                        rowDelta.NativePtr,
                        projectedGradient.NativePtr,
                        batch,
                        sequence,
                        modelWidth,
                        heads,
                        causal ? 1 : 0,
                        stream)
                    : CudaNativeGateway
                        .FlashAttentionBackwardBFloat16TensorCore(
                        accelerator.Index,
                        projected.NativePtr,
                        output.NativePtr,
                        outputGradient.NativePtr,
                        softmaxLogSumExp.NativePtr,
                        rowDelta.NativePtr,
                        projectedGradient.NativePtr,
                        batch,
                        sequence,
                        modelWidth,
                        heads,
                        causal ? 1 : 0,
                        stream);
            }
        }
        else
        {
            if (projectedGradient is null || outputGradient is null)
            {
                throw new InvalidOperationException(
                    "FlashAttention backward requires FP32 gradient storage.");
            }
            status = CudaNativeGateway.FlashAttentionBackwardBFloat16(
                accelerator.Index,
                projected.NativePtr,
                output.NativePtr,
                outputGradient.NativePtr,
                softmaxLogSumExp.NativePtr,
                projectedGradient.NativePtr,
                batch,
                sequence,
                modelWidth,
                heads,
                causal ? 1 : 0,
                stream);
        }
        if (status != 0)
            throw new InvalidOperationException(
                $"BF16 FlashAttention backward CUDA error {status}.");
    }

    internal static bool TryBackwardBFloat16Bfp8Output(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        CudaBfp8BufferView output,
        NativeCudaBuffer<float> outputGradient,
        NativeCudaBuffer<float> softmaxLogSumExp,
        NativeCudaBuffer<float> rowDelta,
        NativeCudaBuffer<float> projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        bool causal,
        bool tensorCore)
    {
        CudaDispatchPolicy dispatch = CudaDispatchPolicy.Current;
        int outputBlockSize = output.Descriptor.GetEffectiveBlockSize(
            output.Payload.Length);
        if (!tensorCore
            || dispatch.DisableDirectBfp8AttentionOutput
            || dispatch.DisableParallelAttentionDkv
            || dispatch.DisableAsyncAttentionBackward
            || modelWidth / heads != 32
            || output.Descriptor.Granularity != Bfp8ScaleGranularity.Block
            || outputBlockSize != 32
            || CudaNativeGateway.AbiVersion.Minor
                < CudaAbiVersion.DirectBfp8AttentionOutputMinor)
        {
            return false;
        }

        accelerator.Bind();
        int status = CudaNativeGateway
            .FlashAttentionBackwardBFloat16TensorCoreParallelDkvBfp8Output(
                accelerator.Index,
                projected.NativePtr,
                output.Payload.NativePtr,
                output.Scales.NativePtr,
                outputBlockSize,
                outputGradient.NativePtr,
                softmaxLogSumExp.NativePtr,
                rowDelta.NativePtr,
                projectedGradient.NativePtr,
                batch,
                sequence,
                modelWidth,
                heads,
                causal ? 1 : 0,
                accelerator.DefaultStream);
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"Direct BFP8-output FlashAttention backward CUDA error " +
                $"{status}.");
        }
        return true;
    }

    internal static void IncrementalBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> keyCache,
        NativeCudaBuffer<ushort> valueCache,
        NativeCudaBuffer<ushort> output,
        int position,
        int cacheCapacity,
        int modelWidth,
        int heads)
    {
        accelerator.Bind();
        int status = CudaNativeGateway.FlashAttentionIncrementalBFloat16(
            accelerator.Index,
            projected.NativePtr,
            keyCache.NativePtr,
            valueCache.NativePtr,
            output.NativePtr,
            position,
            cacheCapacity,
            modelWidth,
            heads,
            accelerator.DefaultStream);
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"BF16 incremental attention CUDA error {status}.");
        }
    }

    internal static void PrefillCacheBFloat16(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> keyCache,
        NativeCudaBuffer<ushort> valueCache,
        int sequence,
        int cacheCapacity,
        int modelWidth)
    {
        accelerator.Bind();
        int status = CudaNativeGateway.FlashAttentionPrefillCacheBFloat16(
            accelerator.Index,
            projected.NativePtr,
            keyCache.NativePtr,
            valueCache.NativePtr,
            sequence,
            cacheCapacity,
            modelWidth,
            accelerator.DefaultStream);
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"BF16 attention cache prefill CUDA error {status}.");
        }
    }

}
