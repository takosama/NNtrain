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
