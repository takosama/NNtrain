using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned declarations for resident signed-Int8 embedding lookup. These
/// entry points accept an explicit stream and never materialize a decoded
/// embedding table.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int Bfp8EmbeddingForward(
        int device,
        nint tablePayload,
        nint tableScales,
        int tableLength,
        int tableBlockSize,
        nint indices,
        int indexCount,
        int width,
        nint outputPayload,
        nint outputScales,
        int outputBlockSize,
        int outputScaleCount,
        nint workspace,
        int workspaceLength,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.Bfp8EmbeddingMinor,
            "resident BFP8 embedding lookup");
        return Complete(
            Bfp8EmbeddingNativeMethods.EmbeddingForward(
                device,
                tablePayload,
                tableScales,
                tableLength,
                tableBlockSize,
                indices,
                indexCount,
                width,
                outputPayload,
                outputScales,
                outputBlockSize,
                outputScaleCount,
                workspace,
                workspaceLength,
                stream),
            CudaNativeOperation.Bfp8Embedding,
            device);
    }

    public static int Bfp8EmbeddingPositionsForward(
        int device,
        nint tokenPayload,
        nint tokenScales,
        int tokenLength,
        int tokenBlockSize,
        nint positionPayload,
        nint positionScales,
        int positionLength,
        int positionBlockSize,
        nint indices,
        int indexCount,
        int sequenceLength,
        int width,
        nint outputPayload,
        nint outputScales,
        int outputBlockSize,
        int outputScaleCount,
        nint workspace,
        int workspaceLength,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.Bfp8EmbeddingMinor,
            "resident BFP8 embedding and position lookup");
        return Complete(
            Bfp8EmbeddingNativeMethods.EmbeddingPositionsForward(
                device,
                tokenPayload,
                tokenScales,
                tokenLength,
                tokenBlockSize,
                positionPayload,
                positionScales,
                positionLength,
                positionBlockSize,
                indices,
                indexCount,
                sequenceLength,
                width,
                outputPayload,
                outputScales,
                outputBlockSize,
                outputScaleCount,
                workspace,
                workspaceLength,
                stream),
            CudaNativeOperation.Bfp8EmbeddingPositions,
            device);
    }

    private static class Bfp8EmbeddingNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_bfp8_embedding_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingForward(
            int device,
            nint tablePayload,
            nint tableScales,
            int tableLength,
            int tableBlockSize,
            nint indices,
            int indexCount,
            int width,
            nint outputPayload,
            nint outputScales,
            int outputBlockSize,
            int outputScaleCount,
            nint workspace,
            int workspaceLength,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_bfp8_embedding_positions_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EmbeddingPositionsForward(
            int device,
            nint tokenPayload,
            nint tokenScales,
            int tokenLength,
            int tokenBlockSize,
            nint positionPayload,
            nint positionScales,
            int positionLength,
            int positionBlockSize,
            nint indices,
            int indexCount,
            int sequenceLength,
            int width,
            nint outputPayload,
            nint outputScales,
            int outputBlockSize,
            int outputScaleCount,
            nint workspace,
            int workspaceLength,
            nint stream);
    }
}
