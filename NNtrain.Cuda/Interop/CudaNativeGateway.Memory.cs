using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

public static partial class CudaNativeGateway
{

    public static int MemoryInfo(
        int device,
        out nuint freeBytes,
        out nuint totalBytes)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.MemoryInfo(device, out freeBytes, out totalBytes),
            CudaNativeOperation.MemoryInfo,
            device);
    }

    public static int Allocate(
        int device,
        nuint bytes,
        out nint pointer)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Allocate(device, bytes, out pointer),
            CudaNativeOperation.Allocate,
            device);
    }

    public static int Free(int device, nint pointer)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Free(device, pointer),
            CudaNativeOperation.Free,
            device);
    }

    public static int Memset(
        int device,
        nint destination,
        int value,
        nuint bytes)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Memset(device, destination, value, bytes),
            CudaNativeOperation.Memset,
            device);
    }

    public static int MemsetAsync(
        int device,
        nint destination,
        int value,
        nuint bytes,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.StreamAwareMemoryMinor,
            "stream-aware CUDA memory operations");
        return Complete(
            NativeMethods.MemsetAsync(
                device,
                destination,
                value,
                bytes,
                stream),
            CudaNativeOperation.MemsetAsync,
            device);
    }

    public static int CopyHostToDevice(
        int device,
        nint destination,
        nint source,
        nuint bytes)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.CopyHostToDevice(
                device,
                destination,
                source,
                bytes),
            CudaNativeOperation.CopyHostToDevice,
            device);
    }

    /// <summary>
    /// Enqueues a host-to-device copy. The source storage must remain valid
    /// until the supplied stream has completed.
    /// </summary>
    public static int CopyHostToDeviceAsync(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.StreamAwareMemoryMinor,
            "stream-aware CUDA memory operations");
        return Complete(
            NativeMethods.CopyHostToDeviceAsync(
                device,
                destination,
                source,
                bytes,
                stream),
            CudaNativeOperation.CopyHostToDeviceAsync,
            device);
    }

    public static int CopyDeviceToHost(
        int device,
        nint destination,
        nint source,
        nuint bytes)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.CopyDeviceToHost(
                device,
                destination,
                source,
                bytes),
            CudaNativeOperation.CopyDeviceToHost,
            device);
    }

    /// <summary>
    /// Enqueues a device-to-host copy. The destination storage must remain
    /// valid until the supplied stream has completed.
    /// </summary>
    public static int CopyDeviceToHostAsync(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.StreamAwareMemoryMinor,
            "stream-aware CUDA memory operations");
        return Complete(
            NativeMethods.CopyDeviceToHostAsync(
                device,
                destination,
                source,
                bytes,
                stream),
            CudaNativeOperation.CopyDeviceToHostAsync,
            device);
    }

    public static int HostAllocate(nuint bytes, out nint pointer)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.HostAllocate(bytes, out pointer),
            CudaNativeOperation.HostAllocate,
            device: -1);
    }

    public static int HostFree(nint pointer)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.HostFree(pointer),
            CudaNativeOperation.HostFree,
            device: -1);
    }

    public static int CopyDeviceToHostAsyncRecord(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream,
        nint cudaEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.CopyDeviceToHostAsyncRecord(
                device,
                destination,
                source,
                bytes,
                stream,
                cudaEvent),
            CudaNativeOperation.CopyDeviceToHostAsync,
            device);
    }

    public static int CopyHostToDeviceAsyncRecord(
        int device,
        nint destination,
        nint source,
        nuint bytes,
        nint stream,
        nint cudaEvent)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.CopyHostToDeviceAsyncRecord(
                device,
                destination,
                source,
                bytes,
                stream,
                cudaEvent),
            CudaNativeOperation.CopyHostToDeviceAsync,
            device);
    }

    public static int CopyDeviceToDevice(
        int destinationDevice,
        nint destination,
        int sourceDevice,
        nint source,
        nuint bytes)
    {
        EnsureCompatibleAbi();
        int status = NativeMethods.CopyDeviceToDevice(
            destinationDevice,
            destination,
            sourceDevice,
            source,
            bytes);
        // The synchronous peer-copy branch does not call select_device;
        // same-device copies do. Preserve the exact native TLS mirror.
        return destinationDevice == sourceDevice
            ? CompleteSelectingDevice(
                status,
                CudaNativeOperation.CopyDeviceToDevice,
                destinationDevice)
            : Complete(
                status,
                CudaNativeOperation.CopyDeviceToDevice,
                destinationDevice);
    }

    /// <summary>
    /// Enqueues an intra-device or peer copy on a stream owned by the
    /// destination device.
    /// </summary>
    public static int CopyDeviceToDeviceAsync(
        int destinationDevice,
        nint destination,
        int sourceDevice,
        nint source,
        nuint bytes,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.StreamAwareMemoryMinor,
            "stream-aware CUDA memory operations");
        return Complete(
            NativeMethods.CopyDeviceToDeviceAsync(
                destinationDevice,
                destination,
                sourceDevice,
                source,
                bytes,
                stream),
            CudaNativeOperation.CopyDeviceToDeviceAsync,
            destinationDevice);
    }

    private static partial class NativeMethods
    {

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_mem_info",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MemoryInfo(
            int device,
            out nuint freeBytes,
            out nuint totalBytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_malloc",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Allocate(
            int device,
            nuint bytes,
            out nint pointer);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_free",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Free(int device, nint pointer);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_memset",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Memset(
            int device,
            nint destination,
            int value,
            nuint bytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_memset_async",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int MemsetAsync(
            int device,
            nint destination,
            int value,
            nuint bytes,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_h2d",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyHostToDevice(
            int device,
            nint destination,
            nint source,
            nuint bytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_h2d_async",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyHostToDeviceAsync(
            int device,
            nint destination,
            nint source,
            nuint bytes,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_d2h",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToHost(
            int device,
            nint destination,
            nint source,
            nuint bytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_d2h_async",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToHostAsync(
            int device,
            nint destination,
            nint source,
            nuint bytes,
            nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_host_alloc",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HostAllocate(
            nuint bytes,
            out nint pointer);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_host_free",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HostFree(nint pointer);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_copy_d2h_async_record",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToHostAsyncRecord(
            int device,
            nint destination,
            nint source,
            nuint bytes,
            nint stream,
            nint cudaEvent);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_copy_h2d_async_record",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyHostToDeviceAsyncRecord(
            int device,
            nint destination,
            nint source,
            nuint bytes,
            nint stream,
            nint cudaEvent);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_d2d",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToDevice(
            int destinationDevice,
            nint destination,
            int sourceDevice,
            nint source,
            nuint bytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_d2d_async",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToDeviceAsync(
            int destinationDevice,
            nint destination,
            int sourceDevice,
            nint source,
            nuint bytes,
            nint stream);
    }
}
