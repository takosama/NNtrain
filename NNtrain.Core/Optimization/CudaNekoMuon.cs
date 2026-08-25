using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaNekoMuon
{
    private const string Library = "NNtrain.CudaKernels";
    private static int _availability;

    internal static bool TryMomentsAndStats(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> gradient,
        NativeCudaBuffer<float> fast,
        NativeCudaBuffer<float> slow,
        NativeCudaBuffer<float> fastHat,
        NativeCudaBuffer<float> slowHat,
        NativeCudaBuffer<float> stats,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection)
    {
        if (Volatile.Read(ref _availability) < 0)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = MomentsAndStats(
                gradient.NativePtr,
                fast.NativePtr,
                slow.NativePtr,
                fastHat.NativePtr,
                slowHat.NativePtr,
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

    [DllImport(Library, EntryPoint = "nntrain_nekomuon_moments_stats",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int MomentsAndStats(
        nint gradient,
        nint fast,
        nint slow,
        nint fastHat,
        nint slowHat,
        nint stats,
        int length,
        float betaFast,
        float betaSlow,
        float fastCorrection,
        float slowCorrection,
        nint stream);
}
