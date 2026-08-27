using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaFlashAttention
{
    private const string Library = "NNtrain.CudaKernels";
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
        if (Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_NATIVE_FLASH_ATTENTION") == "1"
            || Volatile.Read(ref _availability) < 0
            || modelWidth / heads > 128)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = Forward(projected.NativePtr, output.NativePtr,
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
        int status = BackwardNative(projected.NativePtr, output.NativePtr,
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
        if (Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_NATIVE_FLASH_ATTENTION") == "1"
            || Volatile.Read(ref _availability) < 0
            || modelWidth / heads > 128)
        {
            return false;
        }
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            if (!string.Equals(
                    Environment.GetEnvironmentVariable(
                        "NNTRAIN_DISABLE_TENSOR_CORE_FLASH_ATTENTION"),
                    "1",
                    StringComparison.Ordinal)
                && Volatile.Read(ref _tensorCoreAvailability) >= 0)
            {
                try
                {
                    bool synchronousLoad = string.Equals(
                        Environment.GetEnvironmentVariable(
                            "NNTRAIN_DISABLE_ASYNC_FLASH_ATTENTION"),
                        "1",
                        StringComparison.Ordinal);
                    int tensorCoreStatus = synchronousLoad
                        ? ForwardBFloat16TensorCoreSync(
                            projected.NativePtr,
                            output.NativePtr,
                            softmaxLogSumExp.NativePtr,
                            batch,
                            sequence,
                            modelWidth,
                            heads,
                            causal ? 1 : 0,
                            stream)
                        : ForwardBFloat16TensorCore(
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
            int status = ForwardBFloat16(
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
            bool parallelDkv = !string.Equals(
                Environment.GetEnvironmentVariable(
                    "NNTRAIN_DISABLE_PARALLEL_ATTENTION_DKV"),
                "1",
                StringComparison.Ordinal);
            bool asyncBackwardLoads = !string.Equals(
                Environment.GetEnvironmentVariable(
                    "NNTRAIN_DISABLE_ASYNC_ATTENTION_BACKWARD"),
                "1",
                StringComparison.Ordinal);
            if (projectedGradientBFloat16 is not null
                && outputGradientBFloat16 is not null)
            {
                status = asyncBackwardLoads
                    ? BackwardNativeBFloat16TensorCoreBFloat16IoGradient(
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
                    : BackwardNativeBFloat16TensorCoreBFloat16IoGradientSync(
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
                status = BackwardNativeBFloat16TensorCoreBFloat16Gradient(
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
                    ? BackwardNativeBFloat16TensorCoreParallelDkv(
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
                    : BackwardNativeBFloat16TensorCore(
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
            status = BackwardNativeBFloat16(
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
        int status = IncrementalBFloat16Native(
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
        int status = PrefillCacheBFloat16Native(
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

    [DllImport(Library, EntryPoint = "nntrain_flash_attention_forward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int Forward(nint projected, nint output,
        nint softmaxLogSumExp, int batch, int sequence, int modelWidth,
        int heads, int causal, nint stream);
    [DllImport(Library, EntryPoint = "nntrain_flash_attention_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNative(nint projected, nint output,
        nint outputGradient, nint softmaxLogSumExp, nint projectedGradient,
        int batch, int sequence, int modelWidth,
        int heads, int causal, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_flash_attention_forward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardBFloat16(
        nint projected,
        nint output,
        nint softmaxLogSumExp,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(Library, EntryPoint = "nntrain_flash_attention_backward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNativeBFloat16(
        nint projected,
        nint output,
        nint outputGradient,
        nint softmaxLogSumExp,
        nint projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_forward_bf16_tensor_core",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardBFloat16TensorCore(
        nint projected,
        nint output,
        nint softmaxLogSumExp,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_forward_bf16_tensor_core_sync",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardBFloat16TensorCoreSync(
        nint projected,
        nint output,
        nint softmaxLogSumExp,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNativeBFloat16TensorCore(
        nint projected,
        nint output,
        nint outputGradient,
        nint softmaxLogSumExp,
        nint rowDelta,
        nint projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_parallel_dkv",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNativeBFloat16TensorCoreParallelDkv(
        nint projected,
        nint output,
        nint outputGradient,
        nint softmaxLogSumExp,
        nint rowDelta,
        nint projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_bf16_gradient",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNativeBFloat16TensorCoreBFloat16Gradient(
        nint projected,
        nint output,
        nint outputGradient,
        nint softmaxLogSumExp,
        nint rowDelta,
        nint projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNativeBFloat16TensorCoreBFloat16IoGradient(
        nint projected,
        nint output,
        nint outputGradient,
        nint softmaxLogSumExp,
        nint rowDelta,
        nint projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient_sync",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int
        BackwardNativeBFloat16TensorCoreBFloat16IoGradientSync(
            nint projected,
            nint output,
            nint outputGradient,
            nint softmaxLogSumExp,
            nint rowDelta,
            nint projectedGradient,
            int batch,
            int sequence,
            int modelWidth,
            int heads,
            int causal,
            nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_incremental_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int IncrementalBFloat16Native(
        nint projected,
        nint keyCache,
        nint valueCache,
        nint output,
        int position,
        int cacheCapacity,
        int modelWidth,
        int heads,
        nint stream);

    [DllImport(
        Library,
        EntryPoint = "nntrain_flash_attention_prefill_cache_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PrefillCacheBFloat16Native(
        nint projected,
        nint keyCache,
        nint valueCache,
        int sequence,
        int cacheCapacity,
        int modelWidth,
        nint stream);

}
