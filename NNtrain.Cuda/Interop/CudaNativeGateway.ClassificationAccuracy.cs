using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for classification accuracy reduction. The native
/// operation never exposes logits to the host; it writes one Int32 count.
/// </summary>
public static partial class CudaNativeGateway
{
    public static int ClassificationCorrectFloat32(
        int device,
        nint logits,
        nint targets,
        nint correctCount,
        int sampleCount,
        int classCount,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.ClassificationAccuracyMinor,
            "CUDA classification accuracy reduction");
        return Complete(
            ClassificationAccuracyNativeMethods.CorrectFloat32(
                device,
                logits,
                targets,
                correctCount,
                sampleCount,
                classCount,
                stream),
            CudaNativeOperation.ClassificationCorrectCount,
            device);
    }

    public static int ClassificationCorrectBFloat16(
        int device,
        nint logits,
        nint targets,
        nint correctCount,
        int sampleCount,
        int classCount,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.ClassificationAccuracyMinor,
            "CUDA BF16 classification accuracy reduction");
        return Complete(
            ClassificationAccuracyNativeMethods.CorrectBFloat16(
                device,
                logits,
                targets,
                correctCount,
                sampleCount,
                classCount,
                stream),
            CudaNativeOperation.ClassificationCorrectCount,
            device);
    }

    public static int ClassificationCorrectBfp8(
        int device,
        nint logitsPayload,
        nint logitsScales,
        int logitsBlockSize,
        nint targets,
        nint correctCount,
        int sampleCount,
        int classCount,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.ClassificationAccuracyMinor,
            "CUDA BFP8 classification accuracy reduction");
        return Complete(
            ClassificationAccuracyNativeMethods.CorrectBfp8(
                device,
                logitsPayload,
                logitsScales,
                logitsBlockSize,
                targets,
                correctCount,
                sampleCount,
                classCount,
                stream),
            CudaNativeOperation.ClassificationCorrectCount,
            device);
    }

    private static class ClassificationAccuracyNativeMethods
    {
        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_classification_correct_f32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CorrectFloat32(
            int device,
            nint logits,
            nint targets,
            nint correctCount,
            int sampleCount,
            int classCount,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_classification_correct_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CorrectBFloat16(
            int device,
            nint logits,
            nint targets,
            nint correctCount,
            int sampleCount,
            int classCount,
            nint stream);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_classification_correct_bfp8",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CorrectBfp8(
            int device,
            nint logitsPayload,
            nint logitsScales,
            int logitsBlockSize,
            nint targets,
            nint correctCount,
            int sampleCount,
            int classCount,
            nint stream);
    }
}
