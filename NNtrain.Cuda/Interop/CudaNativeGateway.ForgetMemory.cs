using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>Versioned ForgetMemory training entry points.</summary>
public static partial class CudaNativeGateway
{
    public static int ForgetMemoryForward(
        int device, nint projected, nint projectedBFloat16, nint output,
        nint outputBFloat16, nint states, nint state, int batch, int sequence,
        int projectionWidth, int keyWidth, int valueWidth,
        float retentionFloor, int memoryVariant, int bfloat16)
    {
        EnsureTrainingKernelAbi("CUDA ForgetMemory");
        return Complete(
            ForgetMemoryNativeMethods.Forward(
                projected, projectedBFloat16, output, outputBFloat16, states,
                state, batch, sequence, projectionWidth, keyWidth, valueWidth,
                retentionFloor, memoryVariant, bfloat16),
            CudaNativeOperation.ForgetMemoryForward,
            device);
    }

    public static int ForgetMemoryBackward(
        int device, nint projected, nint projectedBFloat16,
        nint projectedGradient, nint outputGradient, nint states,
        nint stateGradient, nint previousGradient, int batch, int sequence,
        int projectionWidth, int keyWidth, int valueWidth,
        float retentionFloor, int memoryVariant, int bfloat16)
    {
        EnsureTrainingKernelAbi("CUDA ForgetMemory backward");
        return Complete(
            ForgetMemoryNativeMethods.Backward(
                projected, projectedBFloat16, projectedGradient,
                outputGradient, states, stateGradient, previousGradient,
                batch, sequence, projectionWidth, keyWidth, valueWidth,
                retentionFloor, memoryVariant, bfloat16),
            CudaNativeOperation.ForgetMemoryBackward,
            device);
    }

    public static int ForgetMemoryForwardBFloat16TensorCore(
        int device, nint projected, nint output, nint states, nint state,
        int batch, int sequence, int projectionWidth, int keyWidth,
        int valueWidth, float retentionFloor, int memoryVariant, nint stream)
    {
        EnsureTrainingKernelAbi("Tensor Core BF16 CUDA ForgetMemory");
        return Complete(
            ForgetMemoryNativeMethods.ForwardBFloat16TensorCore(
                projected, output, states, state, batch, sequence,
                projectionWidth, keyWidth, valueWidth, retentionFloor,
                memoryVariant, stream),
            CudaNativeOperation.ForgetMemoryForwardBFloat16TensorCore,
            device);
    }

    private static class ForgetMemoryNativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_forget_forward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Forward(
            nint projected, nint projectedBFloat16, nint output,
            nint outputBFloat16, nint states, nint state, int batch,
            int sequence, int projectionWidth, int keyWidth, int valueWidth,
            float retentionFloor, int memoryVariant, int bfloat16);

        [DllImport(LibraryName, EntryPoint = "nntrain_forget_backward",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Backward(
            nint projected, nint projectedBFloat16, nint projectedGradient,
            nint outputGradient, nint states, nint stateGradient,
            nint previousGradient, int batch, int sequence,
            int projectionWidth, int keyWidth, int valueWidth,
            float retentionFloor, int memoryVariant, int bfloat16);

        [DllImport(
            LibraryName,
            EntryPoint = "nntrain_forget_memory_forward_bf16_tensor_core",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ForwardBFloat16TensorCore(
            nint projected, nint output, nint states, nint state, int batch,
            int sequence, int projectionWidth, int keyWidth, int valueWidth,
            float retentionFloor, int memoryVariant, nint stream);
    }
}
