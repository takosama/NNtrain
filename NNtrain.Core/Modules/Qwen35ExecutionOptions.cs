namespace NNtrain;

/// <summary>Quantized projection implementations for Qwen3.5 comparisons.</summary>
public enum Qwen35QuantizedKernel { Reference, Cooperative, Subgroup, Auto }

public sealed record Qwen35ExecutionOptions
{
    public Qwen35QuantizedKernel QuantizedKernel { get; init; } = Qwen35QuantizedKernel.Auto;
    public bool LoraTraining { get; init; }
    public bool ParallelModelLoad { get; init; } = true;
    public bool ComputeModelFingerprintOnLoad { get; init; }
    public bool InferencePairedProjection { get; init; }
    // Bit 0: IQ2_S, bit 1: Q4_K, bit 2: IQ3_S. Q4 pairing was slower on Arc B580.
    public int InferencePairedProjectionTypes { get; init; } = 5;
    public bool InferencePairedLoraProjection { get; init; }
    public bool CacheKernelArguments { get; init; } = true;
    public bool CacheProgramBinary { get; init; } = true;
    public bool InferenceFastRmsNorm { get; init; } = true;
    public bool InferenceCooperativeLora { get; init; } = true;
    public int ProjectionWorkgroupSize { get; init; } = 32;
    public bool ParallelArgmax { get; init; } = true;
    public bool ParallelDeltaNorm { get; init; } = true;
    public bool InferenceFusedLora { get; init; } = true;
    // Experimental layerwise prompt prefill. Zero keeps the serial path.
    public int InferencePrefillChunkTokens { get; init; }
    // FP32 SG16 prefill projections share quantized decoding across rows.
    // One preserves the existing GEMV projection implementation.
    public int InferenceProjectionRows { get; init; } = 1;
    // Arc XMX prefill keeps FP32 accumulators and splits both matrix operands
    // into high/low FP16 components. Decode remains on device and bounded.
    public bool InferenceXmxPrefill { get; init; }
    public bool InferenceXmxPackedPrefill { get; init; }
    // Retain integer IQ grids exactly and apply their coefficients in FP32.
    // At 384+ rows use a bounded temporary panel; shorter chunks decode on chip.
    public bool InferenceXmxFactoredPrefill { get; init; }
    // Decode original IQ2 GGUF blocks once per B-only SLM tile. Original
    // single-token projection/storage remains available without recoding.
    public bool InferenceXmxGgufBslmPrefill { get; init; }
    // Exactly recode IQ2 block projections into small, resident XMX panels.
    // Embeddings, output heads and every training path keep GGUF storage.
    public bool InferenceResidentIq2Panels { get; init; }
    public int InferenceBufferPoolMiB { get; init; } = 64;
    public int InferenceDeferredReleaseMiB { get; init; }
    // Mixed prompt attention can aggregate normalization and KV work across
    // rows. Keep selectable for device and model-specific performance checks.
    public bool InferenceBatchMixedAttention { get; init; } = true;
    // Batched prompt convolution and token-ordered width-128 DeltaNet state
    // avoid per-token dispatches while preserving the existing FP32 order.
    public bool InferenceBatchRecurrent { get; init; } = true;
    // SG16 leaders retain the serial RMS sum while avoiding two row barriers.
    // Every recurrent output and final state matches the sequential path bitwise.
    public bool InferenceSubgroupRecurrentRms { get; init; } = true;
    public int LoraReductionSize { get; init; } = 1024;
    public bool UnrollQ4 { get; init; }
    public bool NativeHalfScale { get; init; } = true;
    // Full diagnostics opt back into per-kernel events; completion fences are
    // retained when timing is disabled. Step loss/time telemetry is independent.
    public bool CollectKernelTimings { get; init; }
    public bool TrainingBatchGradientNorm { get; init; } = true;
    public int TrainingNormSplits { get; init; } = 8;
    public bool TrainingCooperativeLora { get; init; } = true;
    public bool TrainingCooperativeDelta { get; init; } = true;
    public bool TrainingResponseOnlyHead { get; init; } = true;
    public bool TrainingGpuCheckpoints { get; init; } = true;
    // For the host-checkpoint fallback: retain only its upstream gradient
    // on the owning GPU, staging through the host only at a device boundary.
    public bool TrainingHostCheckpointGpuGradients { get; init; } = true;
    // Experimental host-checkpoint buffer handoff. The combined switch enables
    // both directions; the individual switches permit independent A/B checks.
    public bool TrainingHostCheckpointBufferHandoff { get; init; }
    public bool TrainingHostCheckpointForwardBufferHandoff { get; init; }
    public bool TrainingHostCheckpointBackwardBufferHandoff { get; init; }
    // Experimental: copy each forward output into the reusable layer input
    // buffer after writing its host checkpoint, then release the output tape.
    public bool TrainingHostCheckpointForwardCopyHandoff { get; init; }
    // Queue a host checkpoint readback without waiting while the next layer
    // consumes the retained GPU output. Exact FP32 values and order are kept.
    public bool TrainingHostCheckpointAsyncRead { get; init; }
    // Keep a bounded number of host-fallback layer inputs on their owning
    // Arc instead. Zero preserves the existing all-host fallback.
    public int TrainingHybridCheckpointMiBPerDevice { get; init; }
    // Store only causal attention scores/probabilities without rounding them.
    // Set false to retain dense storage; longer optimizer trajectories remain unvalidated.
    public bool TrainingPackedAttentionScores { get; init; } = true;
    // Recompute causal scores in bounded query tiles during backward instead
    // of retaining a quadratic score matrix. Zero keeps packed/dense attention.
    public int TrainingStreamedAttentionTileRows { get; init; }
    // Experimental one-workgroup-per-row packed attention. Fuses score and
    // normalization in both directions while retaining FP32 probabilities.
    public bool TrainingFusedAttentionRows { get; init; }
    // Additional experimental fusion of context aggregation and gate into the
    // row-forward kernel. Requires TrainingFusedAttentionRows.
    public bool TrainingFusedAttentionOutput { get; init; }
    // Experimental exact-FP32 IQ2_S base-output cache for long-sequence recomputation.
    // Zero leaves the existing recomputation path unchanged.
    public int TrainingIQ2ProjectionCacheMiB { get; init; }
    // Opt in to selecting projections by estimated saved IQ2_S work per host byte.
    // The default retains the original forward-order cache selection.
    public bool TrainingIQ2ProjectionCachePrioritize { get; init; }
    // Experimental exact-FP32 base-output cache on each Arc device. Zero disables it.
    // The requested cap applies separately to each selected device.
    public int TrainingIQ2GpuProjectionCacheMiB { get; init; }
    // Experimental IQ2_S training forward: round activations and decoded
    // weights to BF16 and use Arc XMX. Off by default: this changes numerics.
    public bool TrainingIQ2Bf16XmxForward { get; init; }
    // Experimental FP16 alternative with finer precision and a narrower
    // numeric range. Keep disabled until model-level loss/gradient validation.
    public bool TrainingIQ2Fp16XmxForward { get; init; }
    public int TrainingTransposeRows { get; init; } = 16;
    // Eight packed components share decode work across eight sequence rows.
    // Set a tile to zero to compare with the scalar transpose implementation.
    public int TrainingTransposeOctetRows { get; init; } = 8;
    // Preserve the FP32 split order while reducing IQ2_S transpose scratch
    // through one rolling [rows, input] accumulator. Experimental and off.
    public bool TrainingIQ2RollingTranspose { get; init; }
    public int TrainingQ4TransposeOctetRows { get; init; } = 8;
    public int TrainingIQ3TransposeOctetRows { get; init; } = 8;
    // Eight packed Q5_K components share decode work across eight token rows.
    // Set zero to compare with the scalar transpose path.
    public int TrainingQ5TransposeOctetRows { get; init; } = 8;
    public int TrainingForwardRows { get; init; } = 4;
    // The Q5_K vocabulary head reuses each GGUF decode across eight rows.
    public int TrainingQ5ForwardRows { get; init; } = 8;
    public int TrainingBufferPoolMiB { get; init; } = 2048;
    public bool DetailedProfiling { get; init; }
    public bool FusedDelta { get; init; } = true;
    private int? _queuedKernelLimit;
    // The physical device budget still bounds retained buffers and pending work.
    public int QueuedKernelLimit
    {
        get => _queuedKernelLimit ?? 4096;
        init => _queuedKernelLimit = value;
    }
}
