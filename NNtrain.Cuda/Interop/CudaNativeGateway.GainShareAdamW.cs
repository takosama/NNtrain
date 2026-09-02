using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>Versioned launch boundary for GainShareAdamW CUDA kernels.</summary>
public static partial class CudaNativeGateway
{
    public static int GainSharePrepareFloat32(
        int device, nint gradient, nint first, nint second, nint direction,
        nint groupStats, int groupIndex, int length, float beta1,
        float beta2, float inverseBiasCorrection1,
        float inverseBiasCorrection2, float epsilon, nint finiteStatus,
        nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.PrepareFloat32(
                gradient, first, second, direction, groupStats, groupIndex,
                length, beta1, beta2, inverseBiasCorrection1,
                inverseBiasCorrection2, epsilon, finiteStatus, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int GainSharePrepareBFloat16(
        int device, nint gradient, nint first, nint second, nint direction,
        nint groupStats, int groupIndex, int length, float beta1,
        float beta2, float inverseBiasCorrection1,
        float inverseBiasCorrection2, float epsilon, nint finiteStatus,
        nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.PrepareBFloat16(
                gradient, first, second, direction, groupStats, groupIndex,
                length, beta1, beta2, inverseBiasCorrection1,
                inverseBiasCorrection2, epsilon, finiteStatus, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int GainShareMomentsFloat32(
        int device, nint gradient, nint first, nint second, int length,
        float beta1, float beta2, nint finiteStatus, nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.MomentsFloat32(
                gradient, first, second, length, beta1, beta2,
                finiteStatus, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int GainShareDirectionFloat32(
        int device, nint gradient, nint first, nint second, nint direction,
        nint groupStats, int groupIndex, int length,
        float inverseBiasCorrection1, float inverseBiasCorrection2,
        float epsilon, nint finiteStatus, nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.DirectionFloat32(
                gradient, first, second, direction, groupStats, groupIndex,
                length, inverseBiasCorrection1, inverseBiasCorrection2,
                epsilon, finiteStatus, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int GainSharePrepareBfp8MultiTensor(
        int device, nint tensors, int tensorCount, int maximumChunks,
        nint reduction, nint groupStats, float beta1, float beta2,
        float inverseBiasCorrection1, float inverseBiasCorrection2,
        float epsilon, nint finiteStatus, nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.PrepareBfp8MultiTensor(
                tensors, tensorCount, maximumChunks, reduction, groupStats,
                beta1, beta2, inverseBiasCorrection1,
                inverseBiasCorrection2, epsilon, finiteStatus, stream),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    public static int GainShareApplyBfp8MultiTensor(
        int device, nint tensors, int tensorCount, int maximumChunks,
        nint reduction, nint scales, float learningRate, float weightDecay,
        bool decay1D, nint finiteStatus, nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.ApplyBfp8MultiTensor(
                tensors, tensorCount, maximumChunks, reduction, scales,
                learningRate, weightDecay, decay1D ? 1 : 0,
                finiteStatus, stream),
            CudaNativeOperation.OptimizerBfp8,
            device);
    }

    public static int GainShareComputeScales(
        int device, nint groupStats, nint alignmentEma, nint scales,
        int groupCount, float rho, float gamma, float minScale,
        float maxScale, float epsilon, nint finiteStatus, nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.ComputeScales(
                groupStats, alignmentEma, scales, groupCount, rho, gamma,
                minScale, maxScale, epsilon, finiteStatus, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int GainShareApplyFloat32(
        int device, nint data, nint direction, nint scales,
        nint bfloat16Output, int groupIndex, int length, float learningRate,
        float weightDecay, bool applyWeightDecay, nint finiteStatus,
        nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.ApplyFloat32(
                data, direction, scales, bfloat16Output, groupIndex, length,
                learningRate, weightDecay, applyWeightDecay ? 1 : 0,
                finiteStatus, stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    public static int GainShareApplyBFloat16(
        int device, nint data, nint direction, nint scales, int groupIndex,
        int length, float learningRate, float weightDecay,
        bool applyWeightDecay, nint finiteStatus, nint stream)
    {
        EnsureGainShareAbi();
        return Complete(
            GainShareNativeMethods.ApplyBFloat16(
                data, direction, scales, groupIndex, length, learningRate,
                weightDecay, applyWeightDecay ? 1 : 0, finiteStatus,
                stream),
            CudaNativeOperation.Optimizer,
            device);
    }

    private static void EnsureGainShareAbi()
        => EnsureMinimumAbiMinor(
            CudaAbiVersion.FusedFirstOrderOptimizerMinor,
            "fused resident GainShareAdamW optimizer");

    private static class GainShareNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_prepare_fp32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PrepareFloat32(
            nint gradient, nint first, nint second, nint direction,
            nint groupStats, int groupIndex, int length, float beta1,
            float beta2, float inverseBiasCorrection1,
            float inverseBiasCorrection2, float epsilon, nint finiteStatus,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_prepare_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PrepareBFloat16(
            nint gradient, nint first, nint second, nint direction,
            nint groupStats, int groupIndex, int length, float beta1,
            float beta2, float inverseBiasCorrection1,
            float inverseBiasCorrection2, float epsilon, nint finiteStatus,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_moments_fp32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MomentsFloat32(
            nint gradient, nint first, nint second, int length, float beta1,
            float beta2, nint finiteStatus, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_direction_fp32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DirectionFloat32(
            nint gradient, nint first, nint second, nint direction,
            nint groupStats, int groupIndex, int length,
            float inverseBiasCorrection1, float inverseBiasCorrection2,
            float epsilon, nint finiteStatus, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_gainshare_prepare_bfp8_multi_tensor",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int PrepareBfp8MultiTensor(
            nint tensors, int tensorCount, int maximumChunks,
            nint reduction, nint groupStats, float beta1, float beta2,
            float inverseBiasCorrection1, float inverseBiasCorrection2,
            float epsilon, nint finiteStatus, nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_gainshare_apply_bfp8_multi_tensor",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ApplyBfp8MultiTensor(
            nint tensors, int tensorCount, int maximumChunks,
            nint reduction, nint scales, float learningRate,
            float weightDecay, int decay1D, nint finiteStatus, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_compute_scales",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ComputeScales(
            nint groupStats, nint alignmentEma, nint scales, int groupCount,
            float rho, float gamma, float minScale, float maxScale,
            float epsilon, nint finiteStatus, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_apply_fp32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ApplyFloat32(
            nint data, nint direction, nint scales, nint bfloat16Output,
            int groupIndex, int length, float learningRate,
            float weightDecay, int applyWeightDecay, nint finiteStatus,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_gainshare_apply_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ApplyBFloat16(
            nint data, nint direction, nint scales, int groupIndex,
            int length, float learningRate, float weightDecay,
            int applyWeightDecay, nint finiteStatus, nint stream);
    }
}
