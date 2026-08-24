using System.Runtime.InteropServices;
using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

internal static class CudaFlashAttention
{
    private const string Library = "NNtrain.CudaKernels";
    private static int _availability;
    internal static bool NativeBackendActive => Volatile.Read(ref _availability) > 0;
    internal static bool TryForward(CudaAccelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> projected,
        MemoryBuffer1D<float, Stride1D.Dense> output, int batch,
        int sequence, int modelWidth, int heads, bool causal)
    {
        if (Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_NATIVE_FLASH_ATTENTION") == "1"
            || Volatile.Read(ref _availability) < 0
            || modelWidth / heads > 128)
            return false;
        try
        {
            accelerator.Bind();
            nint stream = ((CudaStream)accelerator.DefaultStream).StreamPtr;
            int status = Forward(projected.NativePtr, output.NativePtr, batch,
                sequence, modelWidth, heads, causal ? 1 : 0, stream);
            if (status != 0)
                throw new InvalidOperationException($"FlashAttention CUDA error {status}.");
            Volatile.Write(ref _availability, 1);
            return true;
        }
        catch (DllNotFoundException) { Volatile.Write(ref _availability, -1); return false; }
        catch (EntryPointNotFoundException) { Volatile.Write(ref _availability, -1); return false; }
    }

    internal static void Backward(CudaAccelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> projected,
        MemoryBuffer1D<float, Stride1D.Dense> output,
        MemoryBuffer1D<float, Stride1D.Dense> outputGradient,
        MemoryBuffer1D<float, Stride1D.Dense> projectedGradient, int batch,
        int sequence, int modelWidth, int heads, bool causal)
    {
        accelerator.Bind();
        nint stream = ((CudaStream)accelerator.DefaultStream).StreamPtr;
        int status = BackwardNative(projected.NativePtr, output.NativePtr,
            outputGradient.NativePtr, projectedGradient.NativePtr, batch,
            sequence, modelWidth, heads, causal ? 1 : 0, stream);
        if (status != 0)
            throw new InvalidOperationException($"FlashAttention backward CUDA error {status}.");
    }

    internal static bool TryForwardBFloat16(
        CudaAccelerator accelerator,
        MemoryBuffer1D<ushort, Stride1D.Dense> projected,
        MemoryBuffer1D<ushort, Stride1D.Dense> output,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        bool causal)
    {
        if (Environment.GetEnvironmentVariable(
                "NNTRAIN_DISABLE_NATIVE_FLASH_ATTENTION") == "1"
            || Volatile.Read(ref _availability) < 0
            || modelWidth / heads > 128)
        {
            return false;
        }
        try
        {
            accelerator.Bind();
            nint stream = ((CudaStream)accelerator.DefaultStream).StreamPtr;
            int status = ForwardBFloat16(
                projected.NativePtr,
                output.NativePtr,
                batch,
                sequence,
                modelWidth,
                heads,
                causal ? 1 : 0,
                stream);
            if (status != 0)
                throw new InvalidOperationException(
                    $"BF16 FlashAttention CUDA error {status}.");
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

    internal static void BackwardBFloat16(
        CudaAccelerator accelerator,
        MemoryBuffer1D<ushort, Stride1D.Dense> projected,
        MemoryBuffer1D<ushort, Stride1D.Dense> output,
        MemoryBuffer1D<float, Stride1D.Dense> outputGradient,
        MemoryBuffer1D<float, Stride1D.Dense> projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        bool causal)
    {
        accelerator.Bind();
        nint stream = ((CudaStream)accelerator.DefaultStream).StreamPtr;
        int status = BackwardNativeBFloat16(
            projected.NativePtr,
            output.NativePtr,
            outputGradient.NativePtr,
            projectedGradient.NativePtr,
            batch,
            sequence,
            modelWidth,
            heads,
            causal ? 1 : 0,
            stream);
        if (status != 0)
            throw new InvalidOperationException(
                $"BF16 FlashAttention backward CUDA error {status}.");
    }

    [DllImport(Library, EntryPoint = "nntrain_flash_attention_forward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int Forward(nint projected, nint output, int batch,
        int sequence, int modelWidth, int heads, int causal, nint stream);
    [DllImport(Library, EntryPoint = "nntrain_flash_attention_backward", CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNative(nint projected, nint output,
        nint outputGradient, nint projectedGradient, int batch, int sequence,
        int modelWidth, int heads, int causal, nint stream);

    [DllImport(Library, EntryPoint = "nntrain_flash_attention_forward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ForwardBFloat16(
        nint projected,
        nint output,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

    [DllImport(Library, EntryPoint = "nntrain_flash_attention_backward_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int BackwardNativeBFloat16(
        nint projected,
        nint output,
        nint outputGradient,
        nint projectedGradient,
        int batch,
        int sequence,
        int modelWidth,
        int heads,
        int causal,
        nint stream);

}
