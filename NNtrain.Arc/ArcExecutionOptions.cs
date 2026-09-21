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
    public int AttentionDkvRows { get; init; } = 16;
    public bool FusedAttentionDq { get; init; } = false;
    public bool FusedAttentionPv { get; init; } = false;
    // Exact-order fused normalization is available for A/B, but is slower on
    // B580 than the separate resident kernels. Do not trade accuracy for fusion.
    public bool BlockResidualNorm { get; init; } = false;
    public bool FusedNormGradient { get; init; } = true;
    public bool DirectXmxMatrices { get; init; } = true;
    public bool PackedMatrixStorage { get; init; } = true;
    public bool SubgroupAttentionReduction { get; init; } = false;
    // Preserve the original 64-lane reduction order while keeping each row's
    // scores/probabilities in registers on the measured SG16 512/1024 paths.
    public bool CachedAttentionProbabilities { get; init; } = true;
    public bool CacheSizedAttentionBackward { get; init; } = true;
    public bool TunedFp32Attention { get; init; } = true;
    public int QueuedKernelLimit { get; init; } = 128;
    public bool PipelineEventCollection { get; init; } = true;
    public bool PowerOfTwoPackScales { get; init; } = true;
    public bool BulkAttentionQkPanels { get; init; } = true;
    // Large SG16 projections publish block32 BFP8 directly from DPAS registers.
    // Small/unsupported shapes and non-BFP8 outputs retain the old path.
    public bool FusedBfp8Linear { get; init; } = true;
    public bool ExpandedXmxTiles { get; init; } = true;
    // SG16 D32 uses coalesced SLM block reads, preserving FP32 FMA order.
    public bool BlockIoAttention { get; init; } = true;
    // Opt-in only: the 512 MiB experiment did not improve full-step throughput.
    public int MatrixPanelCacheMiB { get; init; } = 0;
    // Opt-in microbench kernels are excluded from production compilation.
    public bool ExperimentalOptimizationKernels { get; init; } = false;
    // Keep uploads visible to queue ordering/profiling even on a pool miss.
    // COPY_HOST_PTR otherwise hides its device transfer inside clCreateBuffer.
    public bool ExplicitHostUploads { get; init; } = true;
    public long DeferredReleaseBytes { get; init; } = 128L * 1024 * 1024;
    public bool BatchDispatch { get; init; } = true;
    public int LossChunkRows { get; init; } = 512;
    public bool TiledMatrices { get; init; } = true;
    public bool StreamingAttention { get; init; } = true;
    // Independently switchable stages for numerical and end-to-end A/B checks.
    public bool FlashAttention { get; init; } = false;
    public bool FlashAttentionXmxProducts { get; init; } = false;
    public bool FlashAttentionAsyncCopy { get; init; } = false;
    public bool FlashAttentionLargeRegisters { get; init; } = true;
    public bool XmxAttentionProducts { get; init; } = false;
    public bool DirectXmxAttentionProducts { get; init; } = true;
    public bool ResidentMuonIterations { get; init; } = true;
    public bool ChunkedLossHead { get; init; } = true;
    public bool PackedUploads { get; init; } = true;
    public bool FusedNormalization { get; init; } = true;
    public long BufferPoolBytes { get; init; } = 4L * 1024 * 1024 * 1024;
    public bool LruBufferPool { get; init; } = true;
    public bool ReuseRetiredBuffers { get; init; } = true;
    // Zero uses 90% of reported device memory. Only idle cached buffers are
    // trimmed; live model/activation owners are never silently discarded.
    public long PhysicalBufferBudgetBytes { get; init; } = 0;
    // Numerically verified B580 paths; switches remain available for A/B checks.
    public bool StreamedXmxMatrices { get; init; } = true;
    // B64/T1024 FFN and QKV dW need 96MiB for their original 2048-wide
    // partials. Other shapes keep the same bounded, full-K admission test.
    public int StreamedWeightGradientWorkspaceMiB { get; init; } = 96;
    public bool PackedReluBackward { get; init; } = true;
    public bool CoalescedBfp8Publication { get; init; } = true;
    public bool DirectAttentionQk { get; init; } = true;
    public bool TransformerCheckpointing { get; init; } = false;
    // Recompute a prefix only: later blocks retain their normal saved values.
    // This bounds memory without paying for unnecessary recomputation.
    public int TransformerCheckpointLayers { get; init; } = int.MaxValue;
    // Recompute only the two FFN projections; a containing full-block
    // checkpoint takes precedence, so the same block is never nested.
    public bool TransformerFfnCheckpointing { get; init; } = false;
    public bool AutomaticTransformerMemoryPlan { get; init; } = true;
    public bool OrderedTiledNorm { get; init; } = true;
    public bool PackedNormInput { get; init; } = true;
    public static ArcExecutionOptions Reference { get; } = new() {
        ResidentTensors = false, TiledMatrices = false, StreamingAttention = false, ResidentMuonIterations = false, ChunkedLossHead = false, PackedUploads = false, FusedNormalization = false, BufferPoolBytes = 0, BatchDispatch = false, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false,
        InlineMatrixGradient = false, FusedAttentionDkv = false, FusedAttentionDq = false, FusedAttentionPv = false,
        BlockResidualNorm = false, FusedNormGradient = false, DirectXmxMatrices = false, PackedMatrixStorage = false, DirectAttentionQk = false,
        SubgroupAttentionReduction = false, CachedAttentionProbabilities = false, CacheSizedAttentionBackward = false,
        PackedReluBackward = false, CoalescedBfp8Publication = false, StreamedWeightGradientWorkspaceMiB = 64,
        TunedFp32Attention = false, LruBufferPool = false, ReuseRetiredBuffers = false,
        TransformerCheckpointing = false, TransformerFfnCheckpointing = false, AutomaticTransformerMemoryPlan = false,
        PipelineEventCollection = false, PowerOfTwoPackScales = false, BulkAttentionQkPanels = false,
        FusedBfp8Linear = false, ExpandedXmxTiles = false,
        BlockIoAttention = false, MatrixPanelCacheMiB = 0,
        StreamedXmxMatrices = false, OrderedTiledNorm = false, PackedNormInput = false };
}
