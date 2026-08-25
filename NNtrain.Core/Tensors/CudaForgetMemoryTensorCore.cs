using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaForgetMemoryTensorCore
{
    private const string Library = "NNtrain.CudaKernels";
    private static int _availability;

    internal static bool BackendActive => Volatile.Read(ref _availability) > 0;

    internal static bool TryForward(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> projected,
        NativeCudaBuffer<ushort> output,
        NativeCudaBuffer<float> states,
        NativeCudaBuffer<float> state,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        int memoryVariant)
    {
        if (Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_TENSOR_CORE_FORGET_MEMORY") == "1"
            || Volatile.Read(ref _availability) < 0
            || keyWidth <= 0
            || keyWidth > 128
            || valueWidth <= 0
            || valueWidth > 128
            || keyWidth % 16 != 0
            || valueWidth % 16 != 0)
        {
            return false;
        }
        try
        {
            accelerator.Bind();
            nint stream = accelerator.DefaultStream;
            int status = ForwardNative(
                projected.NativePtr,
                output.NativePtr,
                states.NativePtr,
                state.NativePtr,
                batch,
                sequence,
                projectionWidth,
                keyWidth,
                valueWidth,
                retentionFloor,
                memoryVariant,
                stream);
            if (status != 0)
            {
                throw new InvalidOperationException(
                    $"ForgetMemory Tensor Core CUDA error {status}.");
            }
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (DllNotFoundException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
        catch (EntryPointNotFoundException)
        {
            Volatile.Write(ref _availability, -1);
            return false;
        }
    }

    [DllImport(
        Library,
        EntryPoint = "nntrain_forget_memory_forward_bf16_tensor_core",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardNative(
        nint projected,
        nint output,
        nint states,
        nint state,
        int batch,
        int sequence,
        int projectionWidth,
        int keyWidth,
        int valueWidth,
        float retentionFloor,
        int memoryVariant,
        nint stream);
}
