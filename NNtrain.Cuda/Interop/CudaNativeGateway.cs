using System.Runtime.InteropServices;
using System.Text;
using NNtrain.Cuda.Execution;

namespace NNtrain.Cuda.Interop;

public readonly record struct CudaAbiVersion(int Major, int Minor)
{
    public const int SupportedMajor = 1;

    public uint Packed =>
        ((uint)(ushort)Major << 16) | (ushort)Minor;

    public static CudaAbiVersion FromPacked(uint packed)
        => new((int)(packed >> 16), (int)(packed & 0xffff));

    public override string ToString() => $"{Major}.{Minor}";
}

public sealed class CudaNativeAbiMismatchException : InvalidOperationException
{
    public CudaNativeAbiMismatchException(string message)
        : base(message)
    {
    }

    public CudaNativeAbiMismatchException(string message, Exception innerException)
        : base(message, innerException)
    {
    }
}

public enum CudaNativeOperation : uint
{
    None = 0,
    DeviceCount = 1,
    DeviceName = 2,
    SetDevice = 3,
    Synchronize = 4,
    MemoryInfo = 5,
    Allocate = 6,
    Free = 7,
    Memset = 8,
    CopyHostToDevice = 9,
    CopyDeviceToHost = 10,
    HostAllocate = 11,
    HostFree = 12,
    StreamCreate = 13,
    StreamDestroy = 14,
    StreamSynchronize = 15,
    EventCreate = 16,
    EventDestroy = 17,
    EventRecord = 18,
    EventQuery = 19,
    EventSynchronize = 20,
    CopyDeviceToHostAsync = 21,
    CopyHostToDeviceAsync = 22,
    CopyDeviceToDevice = 23,
    PeerAccess = 24,
    Capabilities = 25,
    Bfp8Quantize = 26,
    Bfp8DequantizeFloat32 = 27,
    Bfp8DequantizeBFloat16 = 28,
    Bfp8QuantizeBFloat16 = 29,
    Bfp8RequantizeInt32 = 30,
    Bfp8TransposeInt8 = 31,
}

/// <summary>
/// Immutable copy of the process-wide native failure record. Sequence makes
/// replacement by a concurrent device call observable; no native pointer or
/// shared mutable structure escapes the gateway.
/// </summary>
public readonly record struct CudaNativeErrorInfo(
    CudaAbiVersion AbiVersion,
    ulong Sequence,
    int Status,
    int DeviceIndex,
    CudaNativeOperation Operation);

/// <summary>
/// Versioned gateway for the CUDA runtime bridge. Runtime, memory, stream,
/// event, and copy entry points are declared only in this type.
/// </summary>
public static class CudaNativeGateway
{
    public const string LibraryName = "NNtrain.CudaKernels.dll";

    private const ulong KnownCapabilityMask =
        (ulong)(CudaKernelFeature.TensorCores |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.FlashAttention |
            CudaKernelFeature.FusedLayerNorm |
            CudaKernelFeature.ForgetMemory |
            CudaKernelFeature.BlockReducedMuon |
            CudaKernelFeature.AsynchronousGradientReduction |
            CudaKernelFeature.CudaGraphs |
            CudaKernelFeature.Bfp8Quantization |
            CudaKernelFeature.Int8TensorCores);

    private static readonly Lazy<CudaAbiVersion> CompatibleAbi = new(
        LoadAndValidateAbi,
        LazyThreadSafetyMode.ExecutionAndPublication);

    // A native failure is copied immediately after the returning P/Invoke.
    // This per-managed-caller slot never references the process-wide native
    // record and exists only until NativeCudaException consumes it.
    [ThreadStatic]
    private static CudaNativeErrorInfo? _capturedFailure;

    public static CudaAbiVersion AbiVersion => CompatibleAbi.Value;

    public static void EnsureCompatibleAbi() => _ = CompatibleAbi.Value;

    public static CudaNativeErrorInfo? TakeCapturedFailure(int status)
    {
        CudaNativeErrorInfo? snapshot = _capturedFailure;
        _capturedFailure = null;
        return snapshot is { } error && error.Status == status
            ? error
            : null;
    }

    public static bool TryGetLastError(out CudaNativeErrorInfo error)
    {
        EnsureCompatibleAbi();
        return TryReadLastError(out error);
    }

    public static int DeviceCount(out int count)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.DeviceCount(out count),
            CudaNativeOperation.DeviceCount,
            device: -1);
    }

    public static int DeviceName(
        int device,
        StringBuilder destination,
        int capacity)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.DeviceName(device, destination, capacity),
            CudaNativeOperation.DeviceName,
            device);
    }

    public static int SetDevice(int device)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.SetDevice(device),
            CudaNativeOperation.SetDevice,
            device);
    }

    public static int UseExternalStream(nint stream)
    {
        EnsureCompatibleAbi();
        int status = NativeMethods.UseExternalStream(stream);
        _capturedFailure = null;
        return status;
    }

    public static int Synchronize(int device)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Synchronize(device),
            CudaNativeOperation.Synchronize,
            device);
    }

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
        return Complete(
            NativeMethods.CopyDeviceToDevice(
                destinationDevice,
                destination,
                sourceDevice,
                source,
                bytes),
            CudaNativeOperation.CopyDeviceToDevice,
            destinationDevice);
    }

    public static int CanAccessPeer(
        int device,
        int peerDevice,
        out int canAccess)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.CanAccessPeer(device, peerDevice, out canAccess),
            CudaNativeOperation.PeerAccess,
            device);
    }

    public static int KernelCapabilities(
        int device,
        out CudaKernelCapabilities capabilities)
    {
        EnsureCompatibleAbi();
        int status = Complete(
            NativeMethods.CapabilityBitmap(
                device,
                out ulong bitmap,
                out int major,
                out int minor),
            CudaNativeOperation.Capabilities,
            device);
        capabilities = new CudaKernelCapabilities(
            major,
            minor,
            (CudaKernelFeature)(bitmap & KnownCapabilityMask));
        return status;
    }

    public static int Bfp8QuantizeFloat32(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8QuantizeFloat32(
                device,
                source,
                payload,
                scales,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int Bfp8DequantizeFloat32(
        int device,
        nint payload,
        nint scales,
        nint destination,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8DequantizeFloat32(
                device,
                payload,
                scales,
                destination,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8DequantizeFloat32,
            device);
    }

    public static int Bfp8DequantizeBFloat16(
        int device,
        nint payload,
        nint scales,
        nint destination,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8DequantizeBFloat16(
                device,
                payload,
                scales,
                destination,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8DequantizeBFloat16,
            device);
    }

    public static int Bfp8QuantizeBFloat16(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8QuantizeBFloat16(
                device,
                source,
                payload,
                scales,
                length,
                blockSize,
                stream),
            CudaNativeOperation.Bfp8QuantizeBFloat16,
            device);
    }

    public static int Bfp8RequantizeInt32(
        int device,
        nint source,
        nint leftScales,
        nint rightScales,
        nint biasPayload,
        nint biasScales,
        nint payload,
        nint scales,
        int length,
        int outputWidth,
        int blockSize,
        int biasBlockSize,
        bool applyRelu,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8RequantizeInt32(
                device,
                source,
                leftScales,
                rightScales,
                biasPayload,
                biasScales,
                payload,
                scales,
                length,
                outputWidth,
                blockSize,
                biasBlockSize,
                applyRelu ? 1 : 0,
                stream),
            CudaNativeOperation.Bfp8RequantizeInt32,
            device);
    }

    public static int Bfp8TransposeInt8(
        int device,
        nint source,
        nint destination,
        int rows,
        int columns,
        nint stream)
    {
        EnsureCompatibleAbi();
        return Complete(
            NativeMethods.Bfp8TransposeInt8(
                device,
                source,
                destination,
                rows,
                columns,
                stream),
            CudaNativeOperation.Bfp8TransposeInt8,
            device);
    }

    public static string ErrorString(int status)
    {
        EnsureCompatibleAbi();
        nint pointer = NativeMethods.ErrorString(status);
        return pointer == nint.Zero
            ? "unknown CUDA error"
            : Marshal.PtrToStringAnsi(pointer) ?? "unknown CUDA error";
    }

    private static int Complete(
        int status,
        CudaNativeOperation operation,
        int device)
    {
        if (status == 0)
        {
            _capturedFailure = null;
            return status;
        }

        _capturedFailure = TryReadErrorSnapshot(
                status,
                device,
                operation,
                out CudaNativeErrorInfo error) &&
            error.Status == status
                ? error
                : null;
        return status;
    }

    private static bool TryReadErrorSnapshot(
        int status,
        int device,
        CudaNativeOperation operation,
        out CudaNativeErrorInfo error)
    {
        int snapshotStatus = NativeMethods.ErrorSnapshot(
            status,
            device,
            (uint)operation,
            out NativeErrorInfo native,
            (nuint)Marshal.SizeOf<NativeErrorInfo>());
        return TryConvertNativeError(snapshotStatus, native, out error);
    }

    private static bool TryReadLastError(out CudaNativeErrorInfo error)
    {
        int status = NativeMethods.LastError(
            out NativeErrorInfo native,
            (nuint)Marshal.SizeOf<NativeErrorInfo>());
        return TryConvertNativeError(status, native, out error);
    }

    private static bool TryConvertNativeError(
        int status,
        NativeErrorInfo native,
        out CudaNativeErrorInfo error)
    {
        if (status != 0 ||
            native.StructSize < (uint)Marshal.SizeOf<NativeErrorInfo>())
        {
            error = default;
            return false;
        }

        error = new CudaNativeErrorInfo(
            CudaAbiVersion.FromPacked(native.AbiVersion),
            native.Sequence,
            native.Status,
            native.DeviceIndex,
            (CudaNativeOperation)native.Operation);
        return true;
    }

    private static CudaAbiVersion LoadAndValidateAbi()
    {
        uint packed;
        try
        {
            packed = NativeMethods.AbiVersion();
        }
        catch (EntryPointNotFoundException exception)
        {
            throw new CudaNativeAbiMismatchException(
                $"{LibraryName} does not expose nntrain_abi_version. " +
                $"ABI major {CudaAbiVersion.SupportedMajor} is required.",
                exception);
        }

        CudaAbiVersion version = CudaAbiVersion.FromPacked(packed);
        if (version.Major != CudaAbiVersion.SupportedMajor)
        {
            throw new CudaNativeAbiMismatchException(
                $"CUDA native ABI major mismatch: managed runtime requires " +
                $"{CudaAbiVersion.SupportedMajor}.x but {LibraryName} reports " +
                $"{version}.");
        }

        return version;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct NativeErrorInfo
    {
        internal uint StructSize;
        internal uint AbiVersion;
        internal ulong Sequence;
        internal int Status;
        internal int DeviceIndex;
        internal uint Operation;
        internal uint Reserved;
    }

    private static class NativeMethods
    {
        [DllImport(LibraryName, EntryPoint = "nntrain_abi_version",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern uint AbiVersion();

        [DllImport(LibraryName, EntryPoint = "nntrain_last_error",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int LastError(
            out NativeErrorInfo error,
            nuint errorSize);

        [DllImport(LibraryName, EntryPoint = "nntrain_error_snapshot",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int ErrorSnapshot(
            int status,
            int device,
            uint operation,
            out NativeErrorInfo error,
            nuint errorSize);

        [DllImport(LibraryName, EntryPoint = "nntrain_capability_bitmap",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CapabilityBitmap(
            int device,
            out ulong bitmap,
            out int computeCapabilityMajor,
            out int computeCapabilityMinor);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_device_count",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DeviceCount(out int count);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_error_string",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern nint ErrorString(int status);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_device_name",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int DeviceName(
            int device,
            StringBuilder destination,
            int capacity);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_set_device",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int SetDevice(int device);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_use_external_stream",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int UseExternalStream(nint stream);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_synchronize",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Synchronize(int device);

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

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_h2d",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyHostToDevice(
            int device,
            nint destination,
            nint source,
            nuint bytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_d2h",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToHost(
            int device,
            nint destination,
            nint source,
            nuint bytes);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_host_alloc",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HostAllocate(
            nuint bytes,
            out nint pointer);

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_host_free",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int HostFree(nint pointer);

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

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_can_access_peer",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CanAccessPeer(
            int device,
            int peerDevice,
            out int canAccess);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_quantize_f32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8QuantizeFloat32(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_dequantize_f32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8DequantizeFloat32(
            int device,
            nint payload,
            nint scales,
            nint destination,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_dequantize_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8DequantizeBFloat16(
            int device,
            nint payload,
            nint scales,
            nint destination,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_quantize_bf16",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8QuantizeBFloat16(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_requantize_i32",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8RequantizeInt32(
            int device,
            nint source,
            nint leftScales,
            nint rightScales,
            nint biasPayload,
            nint biasScales,
            nint payload,
            nint scales,
            int length,
            int outputWidth,
            int blockSize,
            int biasBlockSize,
            int applyRelu,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_bfp8_transpose_i8",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8TransposeInt8(
            int device,
            nint source,
            nint destination,
            int rows,
            int columns,
            nint stream);
    }
}
