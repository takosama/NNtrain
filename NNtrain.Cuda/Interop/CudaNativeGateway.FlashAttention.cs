using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>Versioned FlashAttention training and cache entry points.</summary>
public static partial class CudaNativeGateway
{
    public static int FlashAttentionForward(
        int device, nint projected, nint output, nint softmaxLogSumExp,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
    {
        EnsureTrainingKernelAbi("CUDA FlashAttention");
        return Complete(
            FlashAttentionNativeMethods.Forward(
                projected, output, softmaxLogSumExp, batch, sequence,
                modelWidth, heads, causal, stream),
            CudaNativeOperation.FlashAttentionForward,
            device);
    }

    public static int FlashAttentionBackward(
        int device, nint projected, nint output, nint outputGradient,
        nint softmaxLogSumExp, nint projectedGradient, int batch,
        int sequence, int modelWidth, int heads, int causal, nint stream)
    {
        EnsureTrainingKernelAbi("CUDA FlashAttention backward");
        return Complete(
            FlashAttentionNativeMethods.Backward(
                projected, output, outputGradient, softmaxLogSumExp,
                projectedGradient, batch, sequence, modelWidth, heads,
                causal, stream),
            CudaNativeOperation.FlashAttentionBackward,
            device);
    }

    public static int FlashAttentionForwardBFloat16(
        int device, nint projected, nint output, nint softmaxLogSumExp,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
    {
        EnsureTrainingKernelAbi("BF16 CUDA FlashAttention");
        return Complete(
            FlashAttentionNativeMethods.ForwardBFloat16(
                projected, output, softmaxLogSumExp, batch, sequence,
                modelWidth, heads, causal, stream),
            CudaNativeOperation.FlashAttentionForwardBFloat16,
            device);
    }

    public static int FlashAttentionBackwardBFloat16(
        int device, nint projected, nint output, nint outputGradient,
        nint softmaxLogSumExp, nint projectedGradient, int batch,
        int sequence, int modelWidth, int heads, int causal, nint stream)
    {
        EnsureTrainingKernelAbi("BF16 CUDA FlashAttention backward");
        return Complete(
            FlashAttentionNativeMethods.BackwardBFloat16(
                projected, output, outputGradient, softmaxLogSumExp,
                projectedGradient, batch, sequence, modelWidth, heads,
                causal, stream),
            CudaNativeOperation.FlashAttentionBackwardBFloat16,
            device);
    }

    public static int FlashAttentionForwardBFloat16TensorCore(
        int device, nint projected, nint output, nint softmaxLogSumExp,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
        => FlashAttentionTensorCoreForward(
            device, projected, output, softmaxLogSumExp, batch, sequence,
            modelWidth, heads, causal, stream, synchronousLoads: false);

    public static int FlashAttentionForwardBFloat16TensorCoreSync(
        int device, nint projected, nint output, nint softmaxLogSumExp,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
        => FlashAttentionTensorCoreForward(
            device, projected, output, softmaxLogSumExp, batch, sequence,
            modelWidth, heads, causal, stream, synchronousLoads: true);

    public static int FlashAttentionBackwardBFloat16TensorCore(
        int device, nint projected, nint output, nint outputGradient,
        nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
        => FlashAttentionTensorCoreBackward(
            device, projected, output, outputGradient, softmaxLogSumExp,
            rowDelta, projectedGradient, batch, sequence, modelWidth, heads,
            causal, stream,
            FlashAttentionBackwardVariant.Float32Gradient);

    public static int FlashAttentionBackwardBFloat16TensorCoreParallelDkv(
        int device, nint projected, nint output, nint outputGradient,
        nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
        => FlashAttentionTensorCoreBackward(
            device, projected, output, outputGradient, softmaxLogSumExp,
            rowDelta, projectedGradient, batch, sequence, modelWidth, heads,
            causal, stream,
            FlashAttentionBackwardVariant.ParallelDkv);

    public static int FlashAttentionBackwardBFloat16TensorCoreBFloat16Gradient(
        int device, nint projected, nint output, nint outputGradient,
        nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream)
        => FlashAttentionTensorCoreBackward(
            device, projected, output, outputGradient, softmaxLogSumExp,
            rowDelta, projectedGradient, batch, sequence, modelWidth, heads,
            causal, stream,
            FlashAttentionBackwardVariant.BFloat16Gradient);

    public static int
        FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradient(
            int device, nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
            int batch, int sequence, int modelWidth, int heads, int causal,
            nint stream)
        => FlashAttentionTensorCoreBackward(
            device, projected, output, outputGradient, softmaxLogSumExp,
            rowDelta, projectedGradient, batch, sequence, modelWidth, heads,
            causal, stream,
            FlashAttentionBackwardVariant.BFloat16IoGradient);

    public static int
        FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradientSync(
            int device, nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
            int batch, int sequence, int modelWidth, int heads, int causal,
            nint stream)
        => FlashAttentionTensorCoreBackward(
            device, projected, output, outputGradient, softmaxLogSumExp,
            rowDelta, projectedGradient, batch, sequence, modelWidth, heads,
            causal, stream,
            FlashAttentionBackwardVariant.BFloat16IoGradientSync);

    public static int FlashAttentionIncrementalBFloat16(
        int device, nint projected, nint keyCache, nint valueCache,
        nint output, int position, int cacheCapacity, int modelWidth,
        int heads, nint stream)
    {
        EnsureTrainingKernelAbi("BF16 CUDA incremental FlashAttention");
        return Complete(
            FlashAttentionNativeMethods.IncrementalBFloat16(
                projected, keyCache, valueCache, output, position,
                cacheCapacity, modelWidth, heads, stream),
            CudaNativeOperation.FlashAttentionIncrementalBFloat16,
            device);
    }

    public static int FlashAttentionPrefillCacheBFloat16(
        int device, nint projected, nint keyCache, nint valueCache,
        int sequence, int cacheCapacity, int modelWidth, nint stream)
    {
        EnsureTrainingKernelAbi("BF16 CUDA FlashAttention cache prefill");
        return Complete(
            FlashAttentionNativeMethods.PrefillCacheBFloat16(
                projected, keyCache, valueCache, sequence, cacheCapacity,
                modelWidth, stream),
            CudaNativeOperation.FlashAttentionPrefillCacheBFloat16,
            device);
    }

    private static int FlashAttentionTensorCoreForward(
        int device, nint projected, nint output, nint softmaxLogSumExp,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream, bool synchronousLoads)
    {
        EnsureTrainingKernelAbi("Tensor Core BF16 CUDA FlashAttention");
        int status = synchronousLoads
            ? FlashAttentionNativeMethods.ForwardBFloat16TensorCoreSync(
                projected, output, softmaxLogSumExp, batch, sequence,
                modelWidth, heads, causal, stream)
            : FlashAttentionNativeMethods.ForwardBFloat16TensorCore(
                projected, output, softmaxLogSumExp, batch, sequence,
                modelWidth, heads, causal, stream);
        return Complete(
            status,
            synchronousLoads
                ? CudaNativeOperation.FlashAttentionForwardBFloat16TensorCoreSync
                : CudaNativeOperation.FlashAttentionForwardBFloat16TensorCore,
            device);
    }

    private static int FlashAttentionTensorCoreBackward(
        int device, nint projected, nint output, nint outputGradient,
        nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
        int batch, int sequence, int modelWidth, int heads, int causal,
        nint stream, FlashAttentionBackwardVariant variant)
    {
        EnsureTrainingKernelAbi(
            "Tensor Core BF16 CUDA FlashAttention backward");
        int status = variant switch
        {
            FlashAttentionBackwardVariant.Float32Gradient =>
                FlashAttentionNativeMethods.BackwardBFloat16TensorCore(
                    projected, output, outputGradient, softmaxLogSumExp,
                    rowDelta, projectedGradient, batch, sequence, modelWidth,
                    heads, causal, stream),
            FlashAttentionBackwardVariant.ParallelDkv =>
                FlashAttentionNativeMethods
                    .BackwardBFloat16TensorCoreParallelDkv(
                        projected, output, outputGradient, softmaxLogSumExp,
                        rowDelta, projectedGradient, batch, sequence,
                        modelWidth, heads, causal, stream),
            FlashAttentionBackwardVariant.BFloat16Gradient =>
                FlashAttentionNativeMethods
                    .BackwardBFloat16TensorCoreBFloat16Gradient(
                        projected, output, outputGradient, softmaxLogSumExp,
                        rowDelta, projectedGradient, batch, sequence,
                        modelWidth, heads, causal, stream),
            FlashAttentionBackwardVariant.BFloat16IoGradient =>
                FlashAttentionNativeMethods
                    .BackwardBFloat16TensorCoreBFloat16IoGradient(
                        projected, output, outputGradient, softmaxLogSumExp,
                        rowDelta, projectedGradient, batch, sequence,
                        modelWidth, heads, causal, stream),
            FlashAttentionBackwardVariant.BFloat16IoGradientSync =>
                FlashAttentionNativeMethods
                    .BackwardBFloat16TensorCoreBFloat16IoGradientSync(
                        projected, output, outputGradient, softmaxLogSumExp,
                        rowDelta, projectedGradient, batch, sequence,
                        modelWidth, heads, causal, stream),
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };
        CudaNativeOperation operation = variant switch
        {
            FlashAttentionBackwardVariant.Float32Gradient =>
                CudaNativeOperation.FlashAttentionBackwardBFloat16TensorCore,
            FlashAttentionBackwardVariant.ParallelDkv => CudaNativeOperation
                .FlashAttentionBackwardBFloat16TensorCoreParallelDkv,
            FlashAttentionBackwardVariant.BFloat16Gradient =>
                CudaNativeOperation
                    .FlashAttentionBackwardBFloat16TensorCoreBFloat16Gradient,
            FlashAttentionBackwardVariant.BFloat16IoGradient =>
                CudaNativeOperation
                    .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradient,
            FlashAttentionBackwardVariant.BFloat16IoGradientSync =>
                CudaNativeOperation
                    .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradientSync,
            _ => throw new ArgumentOutOfRangeException(nameof(variant)),
        };
        return Complete(status, operation, device);
    }

    private enum FlashAttentionBackwardVariant
    {
        Float32Gradient,
        ParallelDkv,
        BFloat16Gradient,
        BFloat16IoGradient,
        BFloat16IoGradientSync,
    }

    private static class FlashAttentionNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_flash_attention_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Forward(
            nint projected, nint output, nint softmaxLogSumExp, int batch,
            int sequence, int modelWidth, int heads, int causal, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_flash_attention_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Backward(
            nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint projectedGradient, int batch,
            int sequence, int modelWidth, int heads, int causal, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_forward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForwardBFloat16(
            nint projected, nint output, nint softmaxLogSumExp, int batch,
            int sequence, int modelWidth, int heads, int causal, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_backward_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardBFloat16(
            nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint projectedGradient, int batch,
            int sequence, int modelWidth, int heads, int causal, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_forward_bf16_tensor_core",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForwardBFloat16TensorCore(
            nint projected, nint output, nint softmaxLogSumExp, int batch,
            int sequence, int modelWidth, int heads, int causal, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_forward_bf16_tensor_core_sync",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForwardBFloat16TensorCoreSync(
            nint projected, nint output, nint softmaxLogSumExp, int batch,
            int sequence, int modelWidth, int heads, int causal, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardBFloat16TensorCore(
            nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
            int batch, int sequence, int modelWidth, int heads, int causal,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_parallel_dkv",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardBFloat16TensorCoreParallelDkv(
            nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
            int batch, int sequence, int modelWidth, int heads, int causal,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_bf16_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardBFloat16TensorCoreBFloat16Gradient(
            nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
            int batch, int sequence, int modelWidth, int heads, int causal,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardBFloat16TensorCoreBFloat16IoGradient(
            nint projected, nint output, nint outputGradient,
            nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
            int batch, int sequence, int modelWidth, int heads, int causal,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_flash_attention_backward_bf16_tensor_core_bf16_io_gradient_sync",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int
            BackwardBFloat16TensorCoreBFloat16IoGradientSync(
                nint projected, nint output, nint outputGradient,
                nint softmaxLogSumExp, nint rowDelta, nint projectedGradient,
                int batch, int sequence, int modelWidth, int heads, int causal,
                nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_incremental_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int IncrementalBFloat16(
            nint projected, nint keyCache, nint valueCache, nint output,
            int position, int cacheCapacity, int modelWidth, int heads,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_flash_attention_prefill_cache_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PrefillCacheBFloat16(
            nint projected, nint keyCache, nint valueCache, int sequence,
            int cacheCapacity, int modelWidth, nint stream);
    }
}
