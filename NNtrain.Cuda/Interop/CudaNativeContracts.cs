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
    public const int FusedFirstOrderOptimizerMinor = 24;
    public const int DirectBfp8LayerNormBlock32x384Minor = 25;
    public const int DirectBfp8FfnGradientMinor = 26;
    public const int DirectBfp8LayerNormBlock32x512Minor = 27;
    public const int DirectBfp8AttentionOutputMinor = 28;
    public const int OrdinaryMuonNesterovMinor = 29;
    public const int Mix8DiagnosticsMinor = 30;
    public const int DeviceAdaptiveNekoMuonMinor = 31;
    public const int DirectBfp8ElementwiseMinor = 32;
    public const int DrnPreparedBackwardMinor = 33;
    public const int DrnChunkBackwardMinor = 34;
    public const int DrnRetentionFloorMinor = 35;

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
    FlashAttentionBackwardBFloat16TensorCoreBfp8Output = 105,
    Bfp8Elementwise = 106,
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
