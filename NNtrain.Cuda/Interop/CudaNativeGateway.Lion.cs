using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Stream-explicit fused multi-tensor Lion entry points. The public ABI uses
/// existing optimizer capability bits so this additive gateway remains
/// compatible with ABI 1.x runtimes while still producing an immutable native
/// failure snapshot through <see cref="CudaNativeGateway"/>.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int LionMultiTensorFloat32(
        int device,
        nint chunks,
        int chunkCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        bool decay1D,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.FusedFirstOrderOptimizerMinor,
            "fused float32/mix16_32 Lion optimizer");
        return Complete(
            LionNativeMethods.Float32(
                device, chunks, chunkCount, beta1, beta2, learningRate,
                weightDecay, decay1D ? 1 : 0, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int LionMultiTensorBFloat16(
        int device,
        nint chunks,
        int chunkCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        bool decay1D,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.FusedFirstOrderOptimizerMinor,
            "fused pure-BF16 Lion optimizer");
        return Complete(
            LionNativeMethods.BFloat16(
                device, chunks, chunkCount, beta1, beta2, learningRate,
                weightDecay, decay1D ? 1 : 0, stream),
            CudaNativeOperation.OptimizerBFloat16,
            device);
    }

    public static int LionMultiTensorMix8(
        int device,
        nint blocks,
        int blockCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        bool decay1D,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.FusedFirstOrderOptimizerMinor,
            "fused mix8_32 Lion optimizer");
        return Complete(
            LionNativeMethods.Mix8(
                device, blocks, blockCount, beta1, beta2, learningRate,
                weightDecay, decay1D ? 1 : 0, finiteStatus, stream),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    public static int LionMultiTensorBfp8(
        int device,
        nint tensors,
        int tensorCount,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        bool decay1D,
        nint reduction,
        int maximumChunks,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.FusedFirstOrderOptimizerMinor,
            "fused pure-BFP8 Lion optimizer");
        return Complete(
            LionNativeMethods.Bfp8(
                device, tensors, tensorCount, beta1, beta2, learningRate,
                weightDecay, decay1D ? 1 : 0, reduction, maximumChunks,
                finiteStatus, stream),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    private static class LionNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_lion_multi_tensor_f32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Float32(
            int device,
            nint chunks,
            int chunkCount,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            int decay1D,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_lion_multi_tensor_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BFloat16(
            int device,
            nint chunks,
            int chunkCount,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            int decay1D,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_lion_multi_tensor_mix8",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Mix8(
            int device,
            nint blocks,
            int blockCount,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            int decay1D,
            nint finiteStatus,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_lion_multi_tensor_bfp8",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8(
            int device,
            nint tensors,
            int tensorCount,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            int decay1D,
            nint reduction,
            int maximumChunks,
            nint finiteStatus,
            nint stream);
    }
}
