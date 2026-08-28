using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// One stable CUDA vocabulary candidate. Equal values are ordered by the
/// smaller logical token index.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CudaTopKCandidate
{
    public int Index;
    public float Value;
}

public static partial class CudaNativeGateway
{
    public static int TensorTopKFloat32(
        int device,
        nint values,
        int offset,
        int count,
        int k,
        nint workspace,
        int reductionBlocks,
        nint output)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.TensorTopKMinor,
            "two-stage CUDA vocabulary top-K");
        return Complete(
            TopKNativeMethods.Float32(
                values,
                offset,
                count,
                k,
                workspace,
                reductionBlocks,
                output),
            CudaNativeOperation.TensorTopK,
            device);
    }

    public static int TensorTopKBFloat16(
        int device,
        nint values,
        int offset,
        int count,
        int k,
        nint workspace,
        int reductionBlocks,
        nint output)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.TensorTopKMinor,
            "two-stage CUDA vocabulary top-K");
        return Complete(
            TopKNativeMethods.BFloat16(
                values,
                offset,
                count,
                k,
                workspace,
                reductionBlocks,
                output),
            CudaNativeOperation.TensorTopK,
            device);
    }

    private static class TopKNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_topk_float",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Float32(
            nint values,
            int offset,
            int count,
            int k,
            nint workspace,
            int reductionBlocks,
            nint output);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_topk_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BFloat16(
            nint values,
            int offset,
            int count,
            int k,
            nint workspace,
            int reductionBlocks,
            nint output);
    }
}
