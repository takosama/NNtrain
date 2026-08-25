using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaForgetMemoryNative
{
    private const string Library = "NNtrain.CudaKernels.dll";

    internal static void Forward(NativeCudaDevice accelerator, nint projected,
        nint projectedBFloat16, nint output, nint outputBFloat16, nint states,
        nint state, int batch, int sequence, int projectionWidth,
        int keyWidth, int valueWidth, float retentionFloor, int memoryVariant,
        bool bfloat16)
    {
        Prepare(accelerator);
        Check(ForwardNative(projected, projectedBFloat16, output,
            outputBFloat16, states, state, batch, sequence, projectionWidth,
            keyWidth, valueWidth, retentionFloor, memoryVariant,
            bfloat16 ? 1 : 0), "ForgetMemory forward");
    }

    internal static void Backward(NativeCudaDevice accelerator, nint projected,
        nint projectedBFloat16, nint projectedGradient, nint outputGradient,
        nint states, nint stateGradient, nint previousGradient, int batch,
        int sequence, int projectionWidth, int keyWidth, int valueWidth,
        float retentionFloor, int memoryVariant, bool bfloat16)
    {
        Prepare(accelerator);
        Check(BackwardNative(projected, projectedBFloat16,
            projectedGradient, outputGradient, states, stateGradient,
            previousGradient, batch, sequence, projectionWidth, keyWidth,
            valueWidth, retentionFloor, memoryVariant, bfloat16 ? 1 : 0),
            "ForgetMemory backward");
    }

    private static void Prepare(NativeCudaDevice accelerator)
    {
        accelerator.Bind();
        NativeCudaRuntime.Check(
            NativeCudaRuntime.UseExternalStreamNative(
                accelerator.DefaultStream),
            "select CUDA stream");
    }

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

    [DllImport(Library, EntryPoint = "nntrain_forget_forward",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardNative(nint projected,
        nint projectedBFloat16, nint output, nint outputBFloat16, nint states,
        nint state, int batch, int sequence, int projectionWidth,
        int keyWidth, int valueWidth, float retentionFloor, int memoryVariant,
        int bfloat16);

    [DllImport(Library, EntryPoint = "nntrain_forget_backward",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNative(nint projected,
        nint projectedBFloat16, nint projectedGradient, nint outputGradient,
        nint states, nint stateGradient, nint previousGradient, int batch,
        int sequence, int projectionWidth, int keyWidth, int valueWidth,
        float retentionFloor, int memoryVariant, int bfloat16);
}
