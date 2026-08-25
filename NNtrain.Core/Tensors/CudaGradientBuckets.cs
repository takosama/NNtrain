using System.Runtime.InteropServices;

namespace NNtrain;

internal static class CudaGradientBuckets
{
    private const string Library = "NNtrain.CudaKernels";

    internal static nint CreateCommunicationStream(
        NativeCudaDevice accelerator,
        int device)
    {
        accelerator.Bind();
        ThrowIfFailed(CreateCommunicationStreamNative(device, out nint stream));
        return stream;
    }

    internal static nint CreateReadyEvent(
        NativeCudaDevice accelerator,
        int device)
    {
        accelerator.Bind();
        ThrowIfFailed(CreateReadyEventNative(device, out nint readyEvent));
        return readyEvent;
    }

    internal static void Pack(
        int device,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> source,
        NativeCudaBuffer<ushort> destination,
        int destinationOffset,
        int length)
    {
        accelerator.Bind();
        ThrowIfFailed(PackNative(
            device,
            source.NativePtr,
            destination.NativePtr,
            destinationOffset,
            length,
            accelerator.DefaultStream));
    }

    internal static void RecordReady(
        int device,
        NativeCudaDevice accelerator,
        nint readyEvent)
    {
        accelerator.Bind();
        ThrowIfFailed(RecordReadyNative(
            device,
            readyEvent,
            accelerator.DefaultStream));
    }

    internal static void Exchange(
        NativeCudaDevice destinationAccelerator,
        int destinationDevice,
        int sourceDevice,
        NativeCudaBuffer<ushort> local,
        NativeCudaBuffer<ushort> remoteSource,
        NativeCudaBuffer<ushort> remoteStaging,
        NativeCudaBuffer<float> reduced,
        int length,
        nint communicationStream,
        nint localReadyEvent,
        nint remoteReadyEvent)
    {
        destinationAccelerator.Bind();
        ThrowIfFailed(ExchangeNative(
            destinationDevice,
            sourceDevice,
            local.NativePtr,
            remoteSource.NativePtr,
            remoteStaging.NativePtr,
            reduced.NativePtr,
            length,
            communicationStream,
            localReadyEvent,
            remoteReadyEvent));
    }

    internal static void Unpack(
        NativeCudaDevice accelerator,
        int device,
        NativeCudaBuffer<float> source,
        int sourceOffset,
        NativeCudaBuffer<float> destination,
        int length,
        nint communicationStream)
    {
        accelerator.Bind();
        ThrowIfFailed(UnpackNative(
            device,
            source.NativePtr,
            sourceOffset,
            destination.NativePtr,
            length,
            communicationStream));
    }

    internal static void Synchronize(
        NativeCudaDevice accelerator,
        int device,
        nint communicationStream)
    {
        accelerator.Bind();
        ThrowIfFailed(SynchronizeNative(device, communicationStream));
    }

    internal static void DestroyEvent(
        NativeCudaDevice accelerator,
        int device,
        nint readyEvent)
    {
        if (readyEvent != 0)
        {
            accelerator.Bind();
            _ = DestroyEventNative(device, readyEvent);
        }
    }

    internal static void DestroyCommunicationStream(
        NativeCudaDevice accelerator,
        int device,
        nint communicationStream)
    {
        if (communicationStream != 0)
        {
            accelerator.Bind();
            _ = DestroyCommunicationStreamNative(device, communicationStream);
        }
    }

    private static void ThrowIfFailed(int status)
    {
        if (status != 0)
        {
            throw new InvalidOperationException(
                $"CUDA BF16 gradient bucket error {status}.");
        }
    }

    [DllImport(Library, EntryPoint = "nntrain_gradient_comm_create",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CreateCommunicationStreamNative(
        int device, out nint stream);

    [DllImport(Library, EntryPoint = "nntrain_gradient_event_create",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int CreateReadyEventNative(
        int device, out nint readyEvent);

    [DllImport(Library, EntryPoint = "nntrain_gradient_pack_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int PackNative(
        int device, nint source, nint destination, int destinationOffset,
        int length, nint computeStream);

    [DllImport(Library, EntryPoint = "nntrain_gradient_record_ready",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int RecordReadyNative(
        int device, nint readyEvent, nint computeStream);

    [DllImport(Library, EntryPoint = "nntrain_gradient_exchange_bf16",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int ExchangeNative(
        int destinationDevice, int sourceDevice, nint local,
        nint remoteSource, nint remoteStaging, nint reduced, int length,
        nint communicationStream, nint localReadyEvent,
        nint remoteReadyEvent);

    [DllImport(Library, EntryPoint = "nntrain_gradient_unpack_float",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int UnpackNative(
        int device, nint source, int sourceOffset, nint destination,
        int length, nint communicationStream);

    [DllImport(Library, EntryPoint = "nntrain_gradient_comm_synchronize",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int SynchronizeNative(
        int device, nint communicationStream);

    [DllImport(Library, EntryPoint = "nntrain_gradient_event_destroy",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int DestroyEventNative(int device, nint readyEvent);

    [DllImport(Library, EntryPoint = "nntrain_gradient_comm_destroy",
        CallingConvention = CallingConvention.Cdecl)]
    private static extern int DestroyCommunicationStreamNative(
        int device, nint communicationStream);
}
