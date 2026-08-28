using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// ABI 1.18 optimizer entry points whose persistent parameter, gradient and
/// moment storage is physically BF16. FP32 is confined to kernel-local
/// accumulation and the small NekoMuon reduction/control buffers.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int OptimizerAdamWPureBFloat16(
        int device, nint data, nint gradient, nint first, nint second,
        int length, float beta1, float beta2, float learningRate,
        float weightDecay, float updateScale, float scaledEpsilon,
        bool applyWeightDecay)
    {
        EnsurePureBFloat16OptimizerAbi();
        return Complete(
            PureBFloat16OptimizerNativeMethods.AdamW(
                data, gradient, first, second, length, beta1, beta2,
                learningRate, weightDecay, updateScale, scaledEpsilon,
                applyWeightDecay ? 1 : 0),
            CudaNativeOperation.OptimizerBFloat16,
            device);
    }

    public static int OptimizerNekoMuonInitializeBFloat16(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseFastCorrection, float inverseNorm)
    {
        EnsurePureBFloat16OptimizerAbi();
        return Complete(
            PureBFloat16OptimizerNativeMethods.NekoInitialize(
                source, destination, length, originalRows, originalColumns,
                transpose ? 1 : 0, inverseFastCorrection, inverseNorm),
            CudaNativeOperation.OptimizerNekoMuonBFloat16,
            device);
    }

    public static int OptimizerNekoMuonInitializeBFloat16FromDeviceStats(
        int device, nint source, nint destination, int length,
        int originalRows, int originalColumns, bool transpose,
        float inverseFastCorrection, nint stats, float epsilon,
        nint finiteStatus)
    {
        EnsurePureBFloat16OptimizerAbi();
        return Complete(
            PureBFloat16OptimizerNativeMethods.NekoInitializeDeviceStats(
                source, destination, length, originalRows, originalColumns,
                transpose ? 1 : 0, inverseFastCorrection, stats, epsilon,
                finiteStatus),
            CudaNativeOperation.OptimizerNekoMuonBFloat16,
            device);
    }

    public static int OptimizerNekoMuonApplyBFloat16(
        int device, nint data, nint update, int length, float learningRate,
        float finalScale, float weightDecay, bool applyWeightDecay)
    {
        EnsurePureBFloat16OptimizerAbi();
        return Complete(
            PureBFloat16OptimizerNativeMethods.NekoApply(
                data, update, length, learningRate, finalScale, weightDecay,
                applyWeightDecay ? 1 : 0),
            CudaNativeOperation.OptimizerNekoMuonBFloat16,
            device);
    }

    public static int NekoMuonMomentsStatsBFloat16Compact(
        int device, nint gradient, nint fast, nint slow, nint stats,
        int length, float betaFast, float betaSlow, float fastCorrection,
        float slowCorrection, nint stream)
    {
        EnsurePureBFloat16OptimizerAbi();
        return Complete(
            PureBFloat16OptimizerNativeMethods.NekoMomentsStats(
                gradient, fast, slow, stats, length, betaFast, betaSlow,
                fastCorrection, slowCorrection, stream),
            CudaNativeOperation.NekoMuonMomentsStatsCompact,
            device);
    }

    public static int NekoMuonMomentsStatsBFloat16CompactFinite(
        int device, nint gradient, nint fast, nint slow, nint stats,
        int length, float betaFast, float betaSlow, float fastCorrection,
        float slowCorrection, nint finiteStatus, nint stream)
    {
        EnsurePureBFloat16OptimizerAbi();
        return Complete(
            PureBFloat16OptimizerNativeMethods.NekoMomentsStatsFinite(
                gradient, fast, slow, stats, length, betaFast, betaSlow,
                fastCorrection, slowCorrection, finiteStatus, stream),
            CudaNativeOperation.NekoMuonMomentsStatsCompactFinite,
            device);
    }

    private static void EnsurePureBFloat16OptimizerAbi()
        => EnsureMinimumAbiMinor(
            CudaAbiVersion.PureBFloat16OptimizerMinor,
            "pure-BF16 CUDA optimizer state and parameter updates");

    private static class PureBFloat16OptimizerNativeMethods
    {
        [DllImport(LibraryName,
            EntryPoint = "nntrain_optimizer_adamw_pure_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamW(
            nint data, nint gradient, nint first, nint second, int length,
            float beta1, float beta2, float learningRate, float weightDecay,
            float updateScale, float scaledEpsilon, int applyWeightDecay);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_optimizer_neko_initialize_bf16_corrected",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoInitialize(
            nint source, nint destination, int length, int originalRows,
            int originalColumns, int transpose, float inverseFastCorrection,
            float inverseNorm);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_optimizer_neko_initialize_bf16_device_stats",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoInitializeDeviceStats(
            nint source, nint destination, int length, int originalRows,
            int originalColumns, int transpose, float inverseFastCorrection,
            nint stats, float epsilon, nint finiteStatus);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_optimizer_neko_apply_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoApply(
            nint data, nint update, int length, float learningRate,
            float finalScale, float weightDecay, int applyWeightDecay);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_nekomuon_moments_stats_bf16_compact",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoMomentsStats(
            nint gradient, nint fast, nint slow, nint stats, int length,
            float betaFast, float betaSlow, float fastCorrection,
            float slowCorrection, nint stream);

        [DllImport(LibraryName,
            EntryPoint =
                "nntrain_nekomuon_moments_stats_bf16_compact_finite",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoMomentsStatsFinite(
            nint gradient, nint fast, nint slow, nint stats, int length,
            float betaFast, float betaSlow, float fastCorrection,
            float slowCorrection, nint finiteStatus, nint stream);
    }
}
