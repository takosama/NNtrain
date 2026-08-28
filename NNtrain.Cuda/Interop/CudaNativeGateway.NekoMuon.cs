using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for the block-reduced NekoMuon moment/statistics
/// kernels. The ABI 1.7 variant folds finite detection into the existing
/// pass so mixed optimizers do not launch one scan per state tensor.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int NekoMuonMomentsStatsCompact(
        int device,
        nint gradient,
        nint fast,
        nint slow,
        nint stats,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.TrainingKernelGatewayMinor,
            "block-reduced CUDA NekoMuon statistics");
        return Complete(
            NekoMuonNativeMethods.MomentsStatsCompact(
                gradient,
                fast,
                slow,
                stats,
                length,
                betaFast,
                betaSlow,
                fastCorrection,
                slowCorrection,
                stream),
            CudaNativeOperation.NekoMuonMomentsStatsCompact,
            device);
    }

    public static int NekoMuonMomentsStatsCompactFinite(
        int device,
        nint gradient,
        nint fast,
        nint slow,
        nint stats,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.NekoMuonFiniteStatusMinor,
            "finite-aware block-reduced CUDA NekoMuon statistics");
        return Complete(
            NekoMuonNativeMethods.MomentsStatsCompactFinite(
                gradient,
                fast,
                slow,
                stats,
                length,
                betaFast,
                betaSlow,
                fastCorrection,
                slowCorrection,
                finiteStatus,
                stream),
            CudaNativeOperation.NekoMuonMomentsStatsCompactFinite,
            device);
    }

    private static class NekoMuonNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_nekomuon_moments_stats_compact",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MomentsStatsCompact(
            nint gradient,
            nint fast,
            nint slow,
            nint stats,
            int length,
            float betaFast,
            float betaSlow,
            float fastCorrection,
            float slowCorrection,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_nekomuon_moments_stats_compact_finite",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MomentsStatsCompactFinite(
            nint gradient,
            nint fast,
            nint slow,
            nint stats,
            int length,
            float betaFast,
            float betaSlow,
            float fastCorrection,
            float slowCorrection,
            nint finiteStatus,
            nint stream);
    }
}
