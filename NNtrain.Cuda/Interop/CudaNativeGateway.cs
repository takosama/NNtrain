using System.Runtime.InteropServices;
using System.Text;
using NNtrain.Cuda.Execution;

namespace NNtrain.Cuda.Interop;

public readonly record struct CudaAbiVersion(int Major, int Minor)
{
    public const int SupportedMajor = 1;
    public const int StreamAwareMemoryMinor = 4;
    public const int Bfp8EmbeddingMinor = 5;
    public const int TrainingKernelGatewayMinor = 5;
    public const int Bfp8ScaleAwareGradientMinor = 6;
    public const int NekoMuonFiniteStatusMinor = 7;
    public const int CudaOutputGradientSeedMinor = 8;
    public const int ReducedEmbeddingBackwardMinor = 9;
    public const int TensorTopKMinor = 10;
    public const int CudaGraphMinor = 11;
    public const int CudaGraphDropoutMinor = 12;
    public const int PublicTensorOpsMinor = 14;
    public const int PureBFloat16GradientMinor = 15;
    public const int ClassificationAccuracyMinor = 16;
    public const int GraphFusedLayerNormMinor = 17;
    public const int PureBFloat16OptimizerMinor = 18;
    public const int ExternalGradientReadyEventMinor = 19;
    public const int BlockBfp8GradientTransportMinor = 20;
    public const int LayerNormOneScanMinor = 21;
    public const int DirectBfp8LayerNormMinor = 22;
    public const int BlockBfp8OptimizerStateMinor = 23;

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
    MemsetAsync = 32,
    CopyDeviceToDeviceAsync = 33,
    Bfp8Embedding = 34,
    Bfp8EmbeddingPositions = 35,
    LayerNormForward = 36,
    LayerNormForwardBFloat16 = 37,
    LayerNormBackward = 38,
    LayerNormBackwardBFloat16 = 39,
    ResidualDropoutLayerNormForward = 40,
    ResidualDropoutLayerNormForwardBFloat16 = 41,
    ResidualDropoutLayerNormBackward = 42,
    ResidualDropoutLayerNormBackwardBFloat16 = 43,
    ResidualDropoutLayerNormBackwardBFloat16BranchGradient = 44,
    ResidualDropoutLayerNormBackwardBFloat16IoGradient = 45,
    FlashAttentionForward = 46,
    FlashAttentionBackward = 47,
    FlashAttentionForwardBFloat16 = 48,
    FlashAttentionBackwardBFloat16 = 49,
    FlashAttentionForwardBFloat16TensorCore = 50,
    FlashAttentionForwardBFloat16TensorCoreSync = 51,
    FlashAttentionBackwardBFloat16TensorCore = 52,
    FlashAttentionBackwardBFloat16TensorCoreParallelDkv = 53,
    FlashAttentionBackwardBFloat16TensorCoreBFloat16Gradient = 54,
    FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradient = 55,
    FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradientSync = 56,
    FlashAttentionIncrementalBFloat16 = 57,
    FlashAttentionPrefillCacheBFloat16 = 58,
    ForgetMemoryForward = 59,
    ForgetMemoryBackward = 60,
    ForgetMemoryForwardBFloat16TensorCore = 61,
    Bfp8GradientQuantize = 62,
    Bfp8GradientReduce = 63,
    Bfp8GradientBroadcast = 64,
    Bfp8GradientQuantizeAccumulate = 65,
    Bfp8GradientSquaredSum = 66,
    Bfp8GradientScale = 67,
    NekoMuonMomentsStatsCompact = 68,
    NekoMuonMomentsStatsCompactFinite = 69,
    TensorAccumulateScalar = 70,
    EmbeddingBackwardReduced = 71,
    EmbeddingPositionsBackwardReduced = 72,
    TensorTopK = 73,
    GraphBeginCapture = 74,
    GraphEndCapture = 75,
    GraphInstantiate = 76,
    GraphLaunch = 77,
    GraphDestroy = 78,
    GraphExecutableDestroy = 79,
    GraphRngStep = 80,
    GraphCounterSet = 81,
    GraphCounterAdvance = 82,
    GraphDropoutForward = 83,
    GraphAddDropoutForward = 84,
    GraphDropoutBackward = 85,
    GraphAddDropoutBackward = 86,
    PublicTensorOps = 87,
    EmbeddingBackwardBFloat16Gradient = 88,
    EmbeddingPositionsBackwardBFloat16Gradient = 89,
    DropoutBackwardBFloat16Gradient = 90,
    AddDropoutBackwardBFloat16Gradient = 91,
    LinearBiasBackwardBFloat16Gradient = 92,
    BFloat16GradientSquaredSum = 93,
    BFloat16GradientScale = 94,
    ClassificationCorrectCount = 95,
    TensorPrimitiveFloat32 = 96,
    TensorPrimitiveBFloat16 = 97,
    Optimizer = 98,
    OptimizerBFloat16 = 99,
    OptimizerBfp8 = 100,
    OptimizerNekoMuon = 101,
    OptimizerNekoMuonBFloat16 = 102,
    GradientCollective = 103,
    GradientCollectiveBFloat16 = 104,
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
/// Managed mirror of CUDA's native thread-local device and tensor-kernel
/// stream selection. Generation changes whenever a gateway call changes or
/// invalidates either selection, allowing cached callers to detect stale
/// bindings without another native transition.
/// </summary>
public readonly record struct CudaNativeThreadContextSnapshot(
    long Generation,
    bool HasSelectedDevice,
    int SelectedDevice,
    bool HasExternalStream,
    nint ExternalStream,
    long SetDeviceCallCount,
    long UseExternalStreamCallCount);

/// <summary>
/// Versioned gateway for the CUDA runtime bridge. Runtime, memory, stream,
/// event, and copy entry points are declared only in this type.
/// </summary>
public static partial class CudaNativeGateway
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
        unchecked
        {
            _threadSetDeviceCallCount++;
        }
        return Complete(
            NativeMethods.SetDevice(device),
            CudaNativeOperation.SetDevice,
            device);
    }

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

    /// <summary>
    /// Returns the minimum native capability bits advertised by a device
    /// before a training-kernel operation should be selected.
    /// </summary>
    public static CudaKernelFeature RequiredFeatures(
        CudaNativeOperation operation) => operation switch
    {
        CudaNativeOperation.LayerNormForward or
        CudaNativeOperation.LayerNormBackward or
        CudaNativeOperation.ResidualDropoutLayerNormForward or
        CudaNativeOperation.ResidualDropoutLayerNormBackward =>
            CudaKernelFeature.FusedLayerNorm,

        CudaNativeOperation.LayerNormForwardBFloat16 or
        CudaNativeOperation.LayerNormBackwardBFloat16 or
        CudaNativeOperation.ResidualDropoutLayerNormForwardBFloat16 or
        CudaNativeOperation.ResidualDropoutLayerNormBackwardBFloat16 or
        CudaNativeOperation
            .ResidualDropoutLayerNormBackwardBFloat16BranchGradient or
        CudaNativeOperation
            .ResidualDropoutLayerNormBackwardBFloat16IoGradient =>
            CudaKernelFeature.FusedLayerNorm |
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.FlashAttentionForward or
        CudaNativeOperation.FlashAttentionBackward =>
            CudaKernelFeature.FlashAttention,

        CudaNativeOperation.FlashAttentionForwardBFloat16 or
        CudaNativeOperation.FlashAttentionBackwardBFloat16 or
        CudaNativeOperation.FlashAttentionIncrementalBFloat16 or
        CudaNativeOperation.FlashAttentionPrefillCacheBFloat16 =>
            CudaKernelFeature.FlashAttention |
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.FlashAttentionForwardBFloat16TensorCore or
        CudaNativeOperation.FlashAttentionForwardBFloat16TensorCoreSync or
        CudaNativeOperation.FlashAttentionBackwardBFloat16TensorCore or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreParallelDkv or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBFloat16Gradient or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradient or
        CudaNativeOperation
            .FlashAttentionBackwardBFloat16TensorCoreBFloat16IoGradientSync =>
            CudaKernelFeature.FlashAttention |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.TensorCores,

        CudaNativeOperation.ForgetMemoryForward or
        CudaNativeOperation.ForgetMemoryBackward =>
            CudaKernelFeature.ForgetMemory,

        CudaNativeOperation.ForgetMemoryForwardBFloat16TensorCore =>
            CudaKernelFeature.ForgetMemory |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.TensorCores,

        CudaNativeOperation.Bfp8GradientQuantize or
        CudaNativeOperation.Bfp8GradientQuantizeAccumulate or
        CudaNativeOperation.Bfp8GradientSquaredSum or
        CudaNativeOperation.Bfp8GradientScale =>
            CudaKernelFeature.Bfp8Quantization,

        CudaNativeOperation.Bfp8GradientReduce or
        CudaNativeOperation.Bfp8GradientBroadcast =>
            CudaKernelFeature.Bfp8Quantization |
            CudaKernelFeature.AsynchronousGradientReduction,

        CudaNativeOperation.NekoMuonMomentsStatsCompact or
        CudaNativeOperation.NekoMuonMomentsStatsCompactFinite =>
            CudaKernelFeature.BlockReducedMuon,

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
        CudaNativeOperation.GraphAddDropoutBackward =>
            CudaKernelFeature.CudaGraphs,

        CudaNativeOperation.EmbeddingBackwardBFloat16Gradient or
        CudaNativeOperation.EmbeddingPositionsBackwardBFloat16Gradient or
        CudaNativeOperation.DropoutBackwardBFloat16Gradient or
        CudaNativeOperation.AddDropoutBackwardBFloat16Gradient or
        CudaNativeOperation.LinearBiasBackwardBFloat16Gradient or
        CudaNativeOperation.BFloat16GradientSquaredSum or
        CudaNativeOperation.BFloat16GradientScale =>
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.TensorPrimitiveBFloat16 or
        CudaNativeOperation.OptimizerBFloat16 =>
            CudaKernelFeature.BFloat16,

        CudaNativeOperation.OptimizerBfp8 =>
            CudaKernelFeature.Bfp8Quantization,

        CudaNativeOperation.OptimizerNekoMuon =>
            CudaKernelFeature.BlockReducedMuon,

        CudaNativeOperation.OptimizerNekoMuonBFloat16 =>
            CudaKernelFeature.BlockReducedMuon |
            CudaKernelFeature.BFloat16 |
            CudaKernelFeature.TensorCores,

        CudaNativeOperation.GradientCollective =>
            CudaKernelFeature.AsynchronousGradientReduction,

        CudaNativeOperation.GradientCollectiveBFloat16 =>
            CudaKernelFeature.AsynchronousGradientReduction |
            CudaKernelFeature.BFloat16,

        _ => CudaKernelFeature.None,
    };

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

    public static int Bfp8QuantizeFloat32Roundtrip(
        int device,
        nint source,
        nint payload,
        nint scales,
        int length,
        int blockSize,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8OptimizerStateMinor,
            "block-BFP8 optimizer state roundtrip");
        return Complete(
            NativeMethods.Bfp8QuantizeFloat32Roundtrip(
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

    public static int AdamWBlockBfp8State(
        int device,
        nint data,
        nint gradient,
        nint firstPayload,
        nint firstScales,
        nint secondPayload,
        nint secondScales,
        int length,
        float beta1,
        float beta2,
        float learningRate,
        float weightDecay,
        float updateScale,
        float scaledEpsilon,
        bool applyWeightDecay,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8OptimizerStateMinor,
            "fused block-BFP8 AdamW state");
        return Complete(
            NativeMethods.AdamWBlockBfp8State(
                device,
                data,
                gradient,
                firstPayload,
                firstScales,
                secondPayload,
                secondScales,
                length,
                beta1,
                beta2,
                learningRate,
                weightDecay,
                updateScale,
                scaledEpsilon,
                applyWeightDecay ? 1 : 0,
                finiteStatus,
                stream),
            CudaNativeOperation.Bfp8Quantize,
            device);
    }

    public static int NekoMuonBlockBfp8Moments(
        int device,
        nint gradient,
        nint fastPayload,
        nint fastScales,
        nint slowPayload,
        nint slowScales,
        nint fastRoundtrip,
        nint slowRoundtrip,
        int length,
        float betaFast,
        float betaSlow,
        nint finiteStatus,
        nint stream)
    {
        EnsureMinimumAbiMinor(
            CudaAbiVersion.BlockBfp8OptimizerStateMinor,
            "fused block-BFP8 NekoMuon moments");
        return Complete(
            NativeMethods.NekoMuonBlockBfp8Moments(
                device,
                gradient,
                fastPayload,
                fastScales,
                slowPayload,
                slowScales,
                fastRoundtrip,
                slowRoundtrip,
                length,
                betaFast,
                betaSlow,
                finiteStatus,
                stream),
            CudaNativeOperation.Bfp8Quantize,
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

        [DllImport(LibraryName, EntryPoint = "nntrain_cuda_copy_d2d_async",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int CopyDeviceToDeviceAsync(
            int destinationDevice,
            nint destination,
            int sourceDevice,
            nint source,
            nuint bytes,
            nint stream);

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
            EntryPoint = "nntrain_cuda_bfp8_quantize_f32_roundtrip",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int Bfp8QuantizeFloat32Roundtrip(
            int device,
            nint source,
            nint payload,
            nint scales,
            int length,
            int blockSize,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_adamw_block_bfp8_state",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int AdamWBlockBfp8State(
            int device,
            nint data,
            nint gradient,
            nint firstPayload,
            nint firstScales,
            nint secondPayload,
            nint secondScales,
            int length,
            float beta1,
            float beta2,
            float learningRate,
            float weightDecay,
            float updateScale,
            float scaledEpsilon,
            int applyWeightDecay,
            nint finiteStatus,
            nint stream);

        [DllImport(LibraryName,
            EntryPoint = "nntrain_cuda_nekomuon_block_bfp8_moments",
            CallingConvention = CallingConvention.Cdecl)]
        internal static extern int NekoMuonBlockBfp8Moments(
            int device,
            nint gradient,
            nint fastPayload,
            nint fastScales,
            nint slowPayload,
            nint slowScales,
            nint fastRoundtrip,
            nint slowRoundtrip,
            int length,
            float betaFast,
            float betaSlow,
            nint finiteStatus,
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
