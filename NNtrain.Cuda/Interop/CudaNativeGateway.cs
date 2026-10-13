using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Interop;

/// <summary>
/// Versioned gateway for the CUDA runtime bridge. Runtime, memory, stream,
/// event, and copy entry points are declared only in this type.
/// </summary>
public static partial class CudaNativeGateway
{
    public const string LibraryName = "NNtrain.CudaKernels.dll";

    private static readonly Lazy<CudaAbiVersion> CompatibleAbi = new(
        LoadAndValidateAbi,
        LazyThreadSafetyMode.ExecutionAndPublication);

    // A native failure is copied immediately after the returning P/Invoke.
    // Legacy kernels that predate the native error ring receive an immutable
    // managed snapshot with sequence zero. This per-managed-caller slot never
    // references process-wide mutable state and exists only until
    // NativeCudaException consumes it.
    [ThreadStatic]
    private static CudaNativeErrorInfo? _capturedFailure;

    [ThreadStatic]
    private static long _threadContextGeneration;
    [ThreadStatic]
    private static bool _threadDeviceKnown;
    [ThreadStatic]
    private static int _threadSelectedDevice;
    [ThreadStatic]
    private static bool _threadExternalStreamKnown;
    [ThreadStatic]
    private static nint _threadExternalStream;
    [ThreadStatic]
    private static long _threadSetDeviceCallCount;
    [ThreadStatic]
    private static long _threadUseExternalStreamCallCount;

    public static CudaAbiVersion AbiVersion => CompatibleAbi.Value;

    public static CudaNativeThreadContextSnapshot CurrentThreadContext => new(
        _threadContextGeneration,
        _threadDeviceKnown,
        _threadSelectedDevice,
        _threadExternalStreamKnown,
        _threadExternalStream,
        _threadSetDeviceCallCount,
        _threadUseExternalStreamCallCount);

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
            if (OperationSelectsDevice(operation))
                PublishDeviceSelection(device);
            return status;
        }

        if (OperationSelectsDevice(operation))
            InvalidateDeviceSelection();
        _capturedFailure = CaptureFailure(status, device, operation);
        return status;
    }

    private static int CompleteSelectingDevice(
        int status,
        CudaNativeOperation operation,
        int device)
    {
        int completed = Complete(status, operation, device);
        if (OperationSelectsDevice(operation))
            return completed;
        if (completed == 0)
            PublishDeviceSelection(device);
        else
            InvalidateDeviceSelection();
        return completed;
    }

    private static bool OperationSelectsDevice(CudaNativeOperation operation)
        => operation is
            CudaNativeOperation.SetDevice or
            CudaNativeOperation.Synchronize or
            CudaNativeOperation.MemoryInfo or
            CudaNativeOperation.Allocate or
            CudaNativeOperation.Free or
            CudaNativeOperation.Memset or
            CudaNativeOperation.MemsetAsync or
            CudaNativeOperation.CopyHostToDevice or
            CudaNativeOperation.CopyDeviceToHost or
            CudaNativeOperation.CopyDeviceToHostAsync or
            CudaNativeOperation.CopyHostToDeviceAsync or
            CudaNativeOperation.CopyDeviceToDeviceAsync or
            CudaNativeOperation.StreamCreate or
            CudaNativeOperation.StreamDestroy or
            CudaNativeOperation.StreamSynchronize or
            CudaNativeOperation.EventCreate or
            CudaNativeOperation.EventDestroy or
            CudaNativeOperation.EventRecord or
            CudaNativeOperation.EventQuery or
            CudaNativeOperation.EventSynchronize or
            CudaNativeOperation.Capabilities or
            CudaNativeOperation.Bfp8Quantize or
            CudaNativeOperation.Bfp8DequantizeFloat32 or
            CudaNativeOperation.Bfp8DequantizeBFloat16 or
            CudaNativeOperation.Bfp8QuantizeBFloat16 or
            CudaNativeOperation.Bfp8RequantizeInt32 or
            CudaNativeOperation.Bfp8TransposeInt8 or
            CudaNativeOperation.Bfp8Embedding or
            CudaNativeOperation.Bfp8EmbeddingPositions or
            CudaNativeOperation.Bfp8GradientQuantize or
            CudaNativeOperation.Bfp8GradientReduce or
            CudaNativeOperation.Bfp8GradientBroadcast or
            CudaNativeOperation.Bfp8GradientQuantizeAccumulate or
            CudaNativeOperation.Bfp8GradientSquaredSum or
            CudaNativeOperation.Bfp8GradientScale or
            CudaNativeOperation.GraphBeginCapture or
            CudaNativeOperation.GraphEndCapture or
            CudaNativeOperation.GraphInstantiate or
            CudaNativeOperation.GraphLaunch or
            CudaNativeOperation.GraphDestroy or
            CudaNativeOperation.GraphExecutableDestroy or
            CudaNativeOperation.GraphRngStep or
            CudaNativeOperation.GraphCounterSet or
            CudaNativeOperation.GraphCounterAdvance or
            CudaNativeOperation.GraphDropoutForward or
            CudaNativeOperation.GraphAddDropoutForward or
            CudaNativeOperation.GraphDropoutBackward or
            CudaNativeOperation.GraphAddDropoutBackward or
            CudaNativeOperation.ClassificationCorrectCount;

    private static void PublishDeviceSelection(int device)
    {
        if (_threadDeviceKnown && _threadSelectedDevice == device)
            return;
        _threadSelectedDevice = device;
        _threadDeviceKnown = true;
        AdvanceThreadContextGeneration();
    }

    private static void PublishExternalStream(nint stream)
    {
        if (_threadExternalStreamKnown && _threadExternalStream == stream)
            return;
        _threadExternalStream = stream;
        _threadExternalStreamKnown = true;
        AdvanceThreadContextGeneration();
    }

    private static void InvalidateDeviceSelection()
    {
        _threadDeviceKnown = false;
        AdvanceThreadContextGeneration();
    }

    private static void InvalidateExternalStream()
    {
        _threadExternalStreamKnown = false;
        AdvanceThreadContextGeneration();
    }

    private static void AdvanceThreadContextGeneration()
    {
        unchecked
        {
            _threadContextGeneration++;
        }
    }

    private static CudaNativeErrorInfo CaptureFailure(
        int status,
        int device,
        CudaNativeOperation operation)
    {
        if (TryReadErrorSnapshot(
                status,
                device,
                operation,
                out CudaNativeErrorInfo error) &&
            error.Status == status)
        {
            return error;
        }

        // Older tensor/optimizer/collective exports return CUDA status codes
        // directly and therefore have no entry in the ABI 1.x native ring.
        // Preserve the same exception contract without a second native call.
        return new CudaNativeErrorInfo(
            AbiVersion,
            Sequence: 0,
            Status: status,
            DeviceIndex: device,
            Operation: operation);
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

    private static void EnsureMinimumAbiMinor(
        int minimumMinor,
        string feature)
    {
        CudaAbiVersion version = AbiVersion;
        if (version.Minor < minimumMinor)
        {
            throw new CudaNativeAbiMismatchException(
                $"{feature} requires CUDA native ABI " +
                $"{CudaAbiVersion.SupportedMajor}.{minimumMinor} or newer, " +
                $"but {LibraryName} reports {version}.");
        }
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

    private static partial class NativeMethods
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

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_error_string",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern nint ErrorString(int status);
    }
}
