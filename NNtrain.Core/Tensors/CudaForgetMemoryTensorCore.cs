using NNtrain.Cuda.Interop;

namespace NNtrain;

internal static class CudaForgetMemoryTensorCore
{
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
        if (CudaDispatchPolicy.Current.DisableTensorCoreForgetMemory
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
            int status = CudaNativeGateway
                .ForgetMemoryForwardBFloat16TensorCore(
                accelerator.Index,
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

}
