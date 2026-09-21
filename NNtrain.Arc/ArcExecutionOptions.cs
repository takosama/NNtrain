namespace NNtrain.Arc;

/// <summary>Immutable switches allow numerical/performance A/B checks in separate sessions.</summary>
public sealed record ArcExecutionOptions
{
    public bool ResidentTensors { get; init; } = true;
    public bool CoalescedAttention { get; init; } = true;
    public bool BatchedAttention { get; init; } = true;
    public bool XmxMatrices { get; init; } = true;
    public ArcXmxGemmMode XmxGemmMode { get; init; } = ArcXmxGemmMode.Auto;
    public bool ParallelReductions { get; init; } = true;
    public bool WideMatrices { get; init; } = true;
    public bool DeepMatrices { get; init; } = true;
    public bool CausalAttentionBounds { get; init; } = true;
    public bool ParallelWeightGradients { get; init; } = true;
    public int AttentionWorkspaceMiB { get; init; } = 64;
    public bool DetailedProfiling { get; init; } = false;
    public bool CompactAttentionTiles { get; init; } = true;
    public bool UnrolledAttentionTiles { get; init; } = true;
    public bool PanelAttention { get; init; } = true;
    public bool MixedBackwardMatrixOperands { get; init; } = true;
    public bool InlineMatrixGradient { get; init; } = true;
    public bool FusedAttentionDkv { get; init; } = true;
    public bool FusedAttentionDq { get; init; } = false;
    public bool FusedAttentionPv { get; init; } = false;
    // Exact-order fused normalization is available for A/B, but is slower on
    // B580 than the separate resident kernels. Do not trade accuracy for fusion.
    public bool BlockResidualNorm { get; init; } = false;
    public bool FusedNormGradient { get; init; } = true;
    public bool DirectXmxMatrices { get; init; } = true;
    public bool PackedMatrixStorage { get; init; } = true;
    public bool SubgroupAttentionReduction { get; init; } = false;
    public bool TunedFp32Attention { get; init; } = true;
    public int QueuedKernelLimit { get; init; } = 128;
    public long DeferredReleaseBytes { get; init; } = 128L * 1024 * 1024;
    public bool BatchDispatch { get; init; } = true;
    public int LossChunkRows { get; init; } = 512;
    public bool TiledMatrices { get; init; } = true;
    public bool StreamingAttention { get; init; } = true;
    public bool ResidentMuonIterations { get; init; } = true;
    public bool ChunkedLossHead { get; init; } = true;
    public bool PackedUploads { get; init; } = true;
    public bool FusedNormalization { get; init; } = true;
    public long BufferPoolBytes { get; init; } = 512L * 1024 * 1024;
    public bool LruBufferPool { get; init; } = true;
    public bool ReuseRetiredBuffers { get; init; } = true;
    public static ArcExecutionOptions Reference { get; } = new() {
        ResidentTensors = false, TiledMatrices = false, StreamingAttention = false, ResidentMuonIterations = false, ChunkedLossHead = false, PackedUploads = false, FusedNormalization = false, BufferPoolBytes = 0, BatchDispatch = false, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false,
        InlineMatrixGradient = false, FusedAttentionDkv = false, FusedAttentionDq = false, FusedAttentionPv = false,
        BlockResidualNorm = false, FusedNormGradient = false, DirectXmxMatrices = false, PackedMatrixStorage = false,
        SubgroupAttentionReduction = false, TunedFp32Attention = false, LruBufferPool = false, ReuseRetiredBuffers = false };
}
