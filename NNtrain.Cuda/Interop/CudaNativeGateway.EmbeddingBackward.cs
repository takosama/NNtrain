using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned owner-reduced embedding-gradient primitives. The caller owns a
/// reusable device workspace; no allocation or host readback occurs in the
/// native launch.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int EmbeddingBackwardReduced(
        int device,
        nint indices,
        nint outputGradient,
        nint tableGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int width)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.ReducedEmbeddingBackwardMinor,
            "owner-reduced CUDA embedding backward");
        return Complete(
            EmbeddingBackwardNativeMethods.EmbeddingBackwardReduced(
                indices,
                outputGradient,
                tableGradient,
                workspace,
                workspaceInts,
                length,
                width),
            CudaNativeOperation.EmbeddingBackwardReduced,
            device);
    }

    public static int EmbeddingPositionsBackwardReduced(
        int device,
        nint indices,
        nint outputGradient,
        nint tokenGradient,
        nint positionGradient,
        nint workspace,
        int workspaceInts,
        int length,
        int sequence,
        int width)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.ReducedEmbeddingBackwardMinor,
            "owner-reduced CUDA token/position embedding backward");
        return Complete(
            EmbeddingBackwardNativeMethods.EmbeddingPositionsBackwardReduced(
                indices,
                outputGradient,
                tokenGradient,
                positionGradient,
                workspace,
                workspaceInts,
                length,
                sequence,
                width),
            CudaNativeOperation.EmbeddingPositionsBackwardReduced,
            device);
    }

    private static class EmbeddingBackwardNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_embedding_backward_reduced",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingBackwardReduced(
            nint indices,
            nint outputGradient,
            nint tableGradient,
            nint workspace,
            int workspaceInts,
            int length,
            int width);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_tensor_embedding_positions_backward_reduced",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingPositionsBackwardReduced(
            nint indices,
            nint outputGradient,
            nint tokenGradient,
            nint positionGradient,
            nint workspace,
            int workspaceInts,
            int length,
            int sequence,
            int width);
    }
}
