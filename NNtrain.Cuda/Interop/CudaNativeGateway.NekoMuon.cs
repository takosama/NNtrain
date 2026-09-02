using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for the block-reduced NekoMuon moment/statistics
/// kernels. The ABI 1.7 variant folds finite detection into the existing
/// pass so mixed optimizers do not launch one scan per state tensor.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int MuonMomentsStatsCompact(
        int device, nint gradient, nint fast, nint direction, nint stats,
        int length, float beta, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.OrdinaryMuonNesterovMinor,
            "reference Nesterov CUDA Muon statistics");
        return Complete(NekoMuonNativeMethods.MuonMomentsStatsCompact(
                gradient, fast, direction, stats, length, beta, stream),
            CudaNativeOperation.NekoMuonMomentsStatsCompact, device);
    }

    public static int MuonMomentsStatsCompactFinite(
        int device, nint gradient, nint fast, nint direction, nint stats,
        int length, float beta, nint finiteStatus, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.OrdinaryMuonNesterovMinor,
            "finite-aware reference Nesterov CUDA Muon statistics");
        return Complete(NekoMuonNativeMethods.MuonMomentsStatsCompactFinite(
                gradient, fast, direction, stats, length, beta,
                finiteStatus, stream),
            CudaNativeOperation.NekoMuonMomentsStatsCompactFinite, device);
    }

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
        [DllImport(LibraryName,
            EntryPoint = "nntrain_muon_moments_stats_compact",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MuonMomentsStatsCompact(
            nint gradient, nint fast, nint direction, nint stats,
            int length, float beta, nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_muon_moments_stats_compact_finite",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MuonMomentsStatsCompactFinite(
            nint gradient, nint fast, nint direction, nint stats,
            int length, float beta, nint finiteStatus, nint stream);

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
