using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{

    public static int UseExternalStream(nint stream)
    {
        EnsureCompatibleAbi();
        unchecked
        {
            _threadUseExternalStreamCallCount++;
        }
        int status = NativeMethods.UseExternalStream(stream);
        _capturedFailure = null;
        if (status == 0)
        {
            PublishExternalStream(stream);
        }
        else
        {
            InvalidateExternalStream();
        }
        return status;
    }

    public static int StreamCreate(int device, out nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.StreamCreate(device, out stream),
            CudaNativeOperation.StreamCreate,
            device);
    }

    public static int StreamDestroy(int device, nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.StreamDestroy(device, stream),
            CudaNativeOperation.StreamDestroy,
            device);
    }

    public static int StreamSynchronize(int device, nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.StreamSynchronize(device, stream),
            CudaNativeOperation.StreamSynchronize,
            device);
    }

    public static int EventCreate(int device, out nint cudaEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.EventCreate(device, out cudaEvent),
            CudaNativeOperation.EventCreate,
            device);
    }

    public static int EventDestroy(int device, nint cudaEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.EventDestroy(device, cudaEvent),
            CudaNativeOperation.EventDestroy,
            device);
    }

    public static int EventRecord(
        int device,
        nint cudaEvent,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.EventRecord(device, cudaEvent, stream),
            CudaNativeOperation.EventRecord,
            device);
    }

    public static int EventQuery(int device, nint cudaEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.EventQuery(device, cudaEvent),
            CudaNativeOperation.EventQuery,
            device);
    }

    public static int EventSynchronize(int device, nint cudaEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.EventSynchronize(device, cudaEvent),
            CudaNativeOperation.EventSynchronize,
            device);
    }

    private static partial class NativeMethods
    {

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_use_external_stream",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UseExternalStream(nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_stream_create",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int StreamCreate(int device, out nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_stream_destroy",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int StreamDestroy(int device, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_stream_synchronize",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int StreamSynchronize(int device, nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_event_create",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EventCreate(int device, out nint cudaEvent);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_event_destroy",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EventDestroy(int device, nint cudaEvent);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_event_record",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EventRecord(
            int device,
            nint cudaEvent,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_event_query",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EventQuery(int device, nint cudaEvent);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_event_synchronize",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int EventSynchronize(
            int device,
            nint cudaEvent);
    }
}
