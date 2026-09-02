using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static class CudaNekoMuon
{
    private static int _availability;

    internal static bool TryMomentsAndStatsCompact(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> gradient,
        NativeCudaBuffer<float> fast,
        NativeCudaBuffer<float> slow,
        NativeCudaBuffer<float> stats,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        bool nesterov = false)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = nesterov
                ? CudaNativeGateway.MuonMomentsStatsCompact(
                    accelerator.Index, gradient.NativePtr, fast.NativePtr,
                    slow.NativePtr, stats.NativePtr, length, betaFast, stream)
                : CudaNativeGateway.NekoMuonMomentsStatsCompact(
                accelerator.Index,
                gradient.NativePtr,
                fast.NativePtr,
                slow.NativePtr,
                stats.NativePtr,
                length,
                betaFast,
                betaSlow,
                fastCorrection,
                slowCorrection,
                stream);
            if (status != 0)
            {
                throw new InvalidOperationException(
                    $"NekoMuon CUDA statistics error {status}.");
            }
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (Exception exception) when (
            exception is DllNotFoundException
                or EntryPointNotFoundException
                or BadImageFormatException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryMomentsAndStatsCompactFinite(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> gradient,
        NativeCudaBuffer<float> fast,
        NativeCudaBuffer<float> slow,
        NativeCudaBuffer<float> stats,
        NativeCudaBuffer<int> finiteStatus,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        bool nesterov = false)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = nesterov
                ? CudaNativeGateway.MuonMomentsStatsCompactFinite(
                    accelerator.Index, gradient.NativePtr, fast.NativePtr,
                    slow.NativePtr, stats.NativePtr, length, betaFast,
                    finiteStatus.NativePtr, stream)
                : CudaNativeGateway.NekoMuonMomentsStatsCompactFinite(
                    accelerator.Index,
                    gradient.NativePtr,
                    fast.NativePtr,
                    slow.NativePtr,
                    stats.NativePtr,
                    length,
                    betaFast,
                    betaSlow,
                    fastCorrection,
                    slowCorrection,
                    finiteStatus.NativePtr,
                    stream);
            if (status != 0)
            {
                throw new InvalidOperationException(
                    $"Finite-aware NekoMuon CUDA statistics error {status}.");
            }
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (Exception exception) when (
            exception is DllNotFoundException
                or EntryPointNotFoundException
                or BadImageFormatException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    internal static bool TryMomentsAndStatsBFloat16Compact(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> gradient,
        NativeCudaBuffer<ushort> fast,
        NativeCudaBuffer<ushort> slow,
        NativeCudaBuffer<float> stats,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        NativeCudaBuffer<int>? finiteStatus = null,
        bool nesterov = false)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = nesterov
                ? finiteStatus is null
                    ? CudaNativeGateway.MuonMomentsStatsBFloat16Compact(
                        accelerator.Index, gradient.NativePtr, fast.NativePtr,
                        slow.NativePtr, stats.NativePtr, length, betaFast, stream)
                    : CudaNativeGateway.MuonMomentsStatsBFloat16CompactFinite(
                        accelerator.Index, gradient.NativePtr, fast.NativePtr,
                        slow.NativePtr, stats.NativePtr, length, betaFast,
                        finiteStatus.NativePtr, stream)
                : finiteStatus is null
                ? CudaNativeGateway.NekoMuonMomentsStatsBFloat16Compact(
                    accelerator.Index,
                    gradient.NativePtr,
                    fast.NativePtr,
                    slow.NativePtr,
                    stats.NativePtr,
                    length,
                    betaFast,
                    betaSlow,
                    fastCorrection,
                    slowCorrection,
                    stream)
                : CudaNativeGateway
                    .NekoMuonMomentsStatsBFloat16CompactFinite(
                        accelerator.Index,
                        gradient.NativePtr,
                        fast.NativePtr,
                        slow.NativePtr,
                        stats.NativePtr,
                        length,
                        betaFast,
                        betaSlow,
                        fastCorrection,
                        slowCorrection,
                        finiteStatus.NativePtr,
                        stream);
            if (status != 0)
            {
                throw new InvalidOperationException(
                    $"Pure BF16 NekoMuon CUDA statistics error {status}.");
            }
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (Exception exception) when (
            exception is DllNotFoundException
                or EntryPointNotFoundException
                or BadImageFormatException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }
}
