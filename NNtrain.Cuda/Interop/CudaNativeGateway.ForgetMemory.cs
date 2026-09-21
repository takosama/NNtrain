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
        if (memoryVariant == 2) EnsureDrnRetentionFloorAbi(retentionFloor);
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
        if (memoryVariant == 2) EnsureDrnRetentionFloorAbi(retentionFloor);
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
        if (memoryVariant == 2) EnsureDrnRetentionFloorAbi(retentionFloor);
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
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_backward_prepared_with_floor",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardPreparedWithFloor(nint projected, nint projectedBFloat16,
            nint projectedGradient, nint outputGradient, nint states, nint stateGradient,
            nint previousGradient, nint prepared, int batch, int sequence,
            int projectionWidth, int keyWidth, int valueWidth, int bfloat16, float retentionFloor);
        [DllImport(LibraryName, EntryPoint = "nntrain_drn_backward_prepared",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int BackwardPrepared(nint projected, nint projectedBFloat16,
            nint projectedGradient, nint outputGradient, nint states, nint stateGradient,
            nint previousGradient, nint prepared, int batch, int sequence,
            int projectionWidth, int keyWidth, int valueWidth, int bfloat16);
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

    public static int DrnBackwardPrepared(int device, nint projected, nint projectedBFloat16,
        nint projectedGradient, nint outputGradient, nint states, nint stateGradient,
        nint previousGradient, nint prepared, int batch, int sequence,
        int projectionWidth, int keyWidth, int valueWidth, int bfloat16)
    {
        EnsureMinimumAbiMinor(CudaAbiVersion.DrnPreparedBackwardMinor, "DRN prepared backward");
        return Complete(ForgetMemoryNativeMethods.BackwardPrepared(projected, projectedBFloat16,
            projectedGradient, outputGradient, states, stateGradient, previousGradient,
            prepared, batch, sequence, projectionWidth, keyWidth, valueWidth, bfloat16),
            CudaNativeOperation.ForgetMemoryBackward, device);
    }

    public static int DrnBackwardPrepared(int device, nint projected, nint projectedBFloat16,
        nint projectedGradient, nint outputGradient, nint states, nint stateGradient,
        nint previousGradient, nint prepared, int batch, int sequence,
        int projectionWidth, int keyWidth, int valueWidth, int bfloat16, float retentionFloor)
    {
        EnsureDrnRetentionFloorAbi(retentionFloor);
        return Complete(ForgetMemoryNativeMethods.BackwardPreparedWithFloor(projected, projectedBFloat16,
            projectedGradient, outputGradient, states, stateGradient, previousGradient,
            prepared, batch, sequence, projectionWidth, keyWidth, valueWidth, bfloat16, retentionFloor),
            CudaNativeOperation.ForgetMemoryBackward, device);
    }

    private static void EnsureDrnRetentionFloorAbi(float retentionFloor)
    {
        if (!float.IsFinite(retentionFloor) || retentionFloor < 0f || retentionFloor >= 1f)
            throw new ArgumentOutOfRangeException(nameof(retentionFloor));
        EnsureMinimumAbiMinor(CudaAbiVersion.DrnRetentionFloorMinor,
            "DRN retention floor (rebuild NNtrain.CudaKernels.dll)");
    }
}
