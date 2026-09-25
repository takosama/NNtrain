namespace NNtrain.Arc;

/// <summary>Immutable switches allow numerical/performance A/B checks in separate sessions.</summary>
public sealed record ArcExecutionOptions
{
    // Measured one-sequence generation paths; matrix/normalization dispatch
    // remains unchanged while recording gradients. Each switch permits A/B.
    public bool InferenceKvCache { get; init; } = true;
    public bool InferenceGemv { get; init; } = true;
    public bool InferencePackedEmbedding { get; init; } = true;
    public bool InferenceSmallRowNorm { get; init; } = true;
    public bool InferenceFusedGemv { get; init; } = true;
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
    // T2048 candidates are enabled independently for measured full-step A/B.
    public bool CachedAttentionProbabilities2048 { get; init; } = false;
    // Measured B580 T2048 backward replay: reuse saved row statistics without
    // the generic reduction's unused local-memory barriers.
    public bool SavedAttentionProbabilitiesDirect2048 { get; init; } = true;
    public bool AttentionDqM32T2048 { get; init; } = false;
    public bool CachedAttentionDerivatives2048 { get; init; } = false;
    // B580 T2048/D32: bitwise-verified register reuse saved 116 ms/update
    // in a same-binary OFF/ON/ON/OFF full-training comparison.
    public bool AttentionDerivativePrefix4T2048 { get; init; } = true;
    // B580 T2048/D32: an extended same-binary A/B saved 97 ms/update over
    // prefix4, with bitwise attention gradients and unchanged memory/transfer.
    public bool AttentionDerivativePrefix8T2048 { get; init; } = true;
    // B580 mix8_16 T2048/D32: pair physical BF16 P/dS in the existing
    // backward score buffer for the dQ and dK/dV consumers.
    public bool Mix8_16PackedAttentionBackward { get; init; } = true;
    // B580 T2048/D32: retain decoded BFP8 QKV activations as physical BF16
    // for ordered FP32 attention products. Works with packed P/dS backward.
    public bool Mix8_16Bf16QkvActivations { get; init; } = true;
    // Publish eligible non-weight Arc operation results as BF16. Two full-step
    // B580 comparisons were slightly faster; the memory planner accounts for
    // the wider activations. Fused GEMM epilogues keep their fast BFP8 output.
    public bool Mix8_16Bf16Activations { get; init; } = true;
    // B580 full-update comparisons: remove duplicate gradient packing and
    // normalization boundary passes without changing published values.
    public bool Mix8_16LinearDualGradientPack { get; init; } = true;
    public bool Mix8_16ReluDualGradientPack { get; init; } = true;
    public bool Mix8_16FusedNormResidualBackward { get; init; } = true;
    public bool Mix8_16DirectBf16NormOutput { get; init; } = true;
    // SG16 parallel sums for normalization statistics and input gradients.
    // Changes FP32 reduction order before BF16 publication under the speed policy.
    public bool Mix8_16ParallelNormReduction { get; init; } = true;
    // Cache rounded BF16 logits only while bounded device headroom permits it.
    public bool Mix8_16CachedLossLogits { get; init; } = true;
    public bool Mix8_16BiasOnlyGradientReduction { get; init; } = true;
    // Speed-policy candidate: single BF16 XMX attention operands without
    // residual products used by the legacy FP32-emulating path.
    public bool Mix8_16XmxAttentionProducts { get; init; } = false;
    public bool Mix8_16XmxAttentionSlm { get; init; } = false;
    // T2048/D32 training: save BF16 O for delta=dY dot O and fuse dP+dS.
    // Avoids the row reduction and a full FP32 derivative workspace.
    public bool Mix8_16AttentionRowDelta { get; init; } = true;
    public bool Mix8_16FusedAttentionDpDs { get; init; } = true;
    public bool Mix8_16FusedAttentionDpDsXmx { get; init; } = false;
    // Measured T2048/D32 mix8_16 paths. Native exp follows the speed policy;
    // the packed SLM and K32 tiles preserve the existing FP32 FMA order.
    public bool Mix8_16NativeExpAttention { get; init; } = true;
    public bool Mix8_16DkvPackedSlm { get; init; } = true;
    public bool Mix8_16DpDsK32 { get; init; } = true;
    public bool Mix8_16PvM128 { get; init; } = false;
    public bool Mix8_16DqM128 { get; init; } = false;
    public bool Mix8_16PvPackedSlm { get; init; } = true;
    public bool Mix8_16DqPackedSlm { get; init; } = true;
    public bool Mix8_16DqPackedSlmBoth { get; init; } = false;
    public bool Mix8_16FusedNormParameterGradients { get; init; } = true;
    // Benchmark-only: the ~0.17% whole-update pilot did not justify a costly
    // extended acceptance run. Keep the measured prefix8 route as default.
    public bool AttentionDerivativePrefix16T2048 { get; init; } = false;
    public bool FusedAttentionDpDs2048 { get; init; } = false;
    // 0 keeps the established dK/dV SLM layout; 32/33 select query-major pitches.
    public int DkvQueryMajorPitch2048 { get; init; } = 0;
    public bool CacheSizedAttentionBackward { get; init; } = true;
    public bool TunedFp32Attention { get; init; } = true;
    public int QueuedKernelLimit { get; init; } = 128;
    public bool PipelineEventCollection { get; init; } = true;
    public bool PowerOfTwoPackScales { get; init; } = true;
    public bool BulkAttentionQkPanels { get; init; } = true;
    // Large SG16 projections publish block32 BFP8 directly from DPAS registers.
    // Small/unsupported shapes and non-BFP8 outputs retain the old path.
    public bool FusedBfp8Linear { get; init; } = true;
    // Signed INT8 XMX forward Linear for block32 BFP8 operands. The measured
    // B580 training shape is faster with BF16 weight GEMM, so retain that
    // selection by default; this switch permits a same-binary INT8 A/B.
    public bool Mix8_16Int8Linear { get; init; } = false;
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
    public int LossLogitsWorkspaceMiB { get; init; } = 32;
    public int LossPanelWorkspaceMiB { get; init; } = 96;
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
    // Opt-in shape-specific streamed GEMM tiles; keep the measured 16x32
    // production path until an end-to-end update gate establishes a win.
    public bool ExpandedStreamedXmxTiles { get; init; } = false;
    // B64/T1024 FFN and QKV dW need 96MiB for their original 2048-wide
    // partials. Other shapes keep the same bounded, full-K admission test.
    public int StreamedWeightGradientWorkspaceMiB { get; init; } = 96;
    public bool PackedReluBackward { get; init; } = true;
    // The paired panel+bias path is measurably faster than separate ReLU
    // encoding/packing on B580 T2048 while retaining the old rounding tree.
    public bool FusedPackedReluBackward { get; init; } = true;
    public bool FusedReluPackBias { get; init; } = true;
    // B580 T2048 full-step A/B: fuse the packed QKV decode with Q/K panel
    // publication; all downstream FP32 consumers still receive decoded QKV.
    public bool FusedAttentionQkvDecodePack2048 { get; init; } = true;
    public bool FusedPackedResidualNorm { get; init; } = false;
    // Loss-head dLogits feeds normal and transposed BF16 panels; build both
    // from one FP32 pass while preserving each panel's established layout.
    public bool FusedDualGradientPack { get; init; } = true;
    // Preserve post-bias BF16 rounding in the final GEMM epilogue, removing
    // the separate full-logit round pass from the measured packed loss head.
    public bool FusedLossHeadLogitRound { get; init; } = true;
    // Keep an existing BF16 leaf gradient packed while computing the next
    // microbatch's FP32 delta, then fuse the final add and BF16 publication.
    public bool Mix8_16FusedGradientAccumulation { get; init; } = false;
    // Reduce a secondary BF16 gradient directly into the primary BF16 buffer.
    public bool Mix8_16DirectGradientReduction { get; init; } = false;
    // Overlap the next secondary gradient read with the current primary upload.
    // Experimental until a full-step A/B demonstrates a benefit.
    public bool Mix8_16PipelinedGradientReduction { get; init; } = false;
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
        InferenceKvCache = false, InferenceGemv = false, InferencePackedEmbedding = false,
        InferenceSmallRowNorm = false, InferenceFusedGemv = false,
        ResidentTensors = false, TiledMatrices = false, StreamingAttention = false, ResidentMuonIterations = false, ChunkedLossHead = false, PackedUploads = false, FusedNormalization = false, BufferPoolBytes = 0, BatchDispatch = false, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false,
        InlineMatrixGradient = false, FusedAttentionDkv = false, FusedAttentionDq = false, FusedAttentionPv = false,
        BlockResidualNorm = false, FusedNormGradient = false, DirectXmxMatrices = false, PackedMatrixStorage = false, DirectAttentionQk = false,
        SubgroupAttentionReduction = false, CachedAttentionProbabilities = false, CacheSizedAttentionBackward = false,
        SavedAttentionProbabilitiesDirect2048 = false,
        PackedReluBackward = false, FusedPackedReluBackward = false, FusedReluPackBias = false,
        CoalescedBfp8Publication = false, StreamedWeightGradientWorkspaceMiB = 64,
        TunedFp32Attention = false, LruBufferPool = false, ReuseRetiredBuffers = false,
        TransformerCheckpointing = false, TransformerFfnCheckpointing = false, AutomaticTransformerMemoryPlan = false,
        PipelineEventCollection = false, PowerOfTwoPackScales = false, BulkAttentionQkPanels = false,
        FusedBfp8Linear = false, Mix8_16Int8Linear = false, ExpandedXmxTiles = false,
        BlockIoAttention = false, MatrixPanelCacheMiB = 0,
        StreamedXmxMatrices = false, OrderedTiledNorm = false, PackedNormInput = false,
        Mix8_16PackedAttentionBackward = false, Mix8_16Bf16QkvActivations = false,
        Mix8_16Bf16Activations = false,
        Mix8_16LinearDualGradientPack = false, Mix8_16FusedNormResidualBackward = false,
        Mix8_16ReluDualGradientPack = false,
        Mix8_16DirectBf16NormOutput = false, Mix8_16CachedLossLogits = false,
        Mix8_16ParallelNormReduction = false,
        Mix8_16BiasOnlyGradientReduction = false,
        Mix8_16AttentionRowDelta = false, Mix8_16FusedAttentionDpDs = false,
        Mix8_16NativeExpAttention = false, Mix8_16DkvPackedSlm = false,
        Mix8_16DpDsK32 = false, Mix8_16PvM128 = false, Mix8_16DqM128 = false,
        Mix8_16PvPackedSlm = false, Mix8_16DqPackedSlm = false,
        Mix8_16DqPackedSlmBoth = false,
        Mix8_16FusedNormParameterGradients = false,
        Mix8_16FusedAttentionDpDsXmx = false, Mix8_16XmxAttentionProducts = false,
        Mix8_16XmxAttentionSlm = false,
        Mix8_16FusedGradientAccumulation = false, Mix8_16DirectGradientReduction = false };
}
