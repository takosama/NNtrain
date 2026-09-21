using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{
    public static int Bfp8Elementwise(int device, nint left, nint leftScales,
        nint right, nint rightScales, nint output, nint outputScales,
        int length, int leftBlock, int rightBlock, int outputBlock, int operation,
        uint seed, uint threshold, float scale, nint counter, ulong operationSeed, nint stream)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.DirectBfp8ElementwiseMinor, "direct BFP8 elementwise");
        return Complete(Bfp8ElementwiseNativeMethods.Bfp8Elementwise(device, left, leftScales,
            right, rightScales, output, outputScales, length, leftBlock, rightBlock,
            outputBlock, operation, seed, threshold, scale, counter, operationSeed, stream),
            CudaNativeOperation.Bfp8Elementwise, device);
    }

    private static class Bfp8ElementwiseNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_bfp8_elementwise", CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8Elementwise(int device, nint left, nint leftScales,
            nint right, nint rightScales, nint output, nint outputScales,
            int length, int leftBlock, int rightBlock, int outputBlock, int operation,
            uint seed, uint threshold, float scale, nint counter, ulong operationSeed, nint stream);
    }
}
