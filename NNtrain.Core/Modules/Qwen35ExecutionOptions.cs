namespace NNtrain;

/// <summary>Quantized projection implementations for Qwen3.5 comparisons.</summary>
public enum Qwen35QuantizedKernel { Reference, Cooperative, Subgroup, Auto }

public sealed record Qwen35ExecutionOptions
{
    public Qwen35QuantizedKernel QuantizedKernel { get; init; } = Qwen35QuantizedKernel.Auto;
    public bool LoraTraining { get; init; }
    public bool InferenceCooperativeLora { get; init; } = true;
    public int ProjectionWorkgroupSize { get; init; } = 32;
    public bool ParallelArgmax { get; init; } = true;
    public bool ParallelDeltaNorm { get; init; } = true;
    public bool InferenceFusedLora { get; init; } = true;
    public int LoraReductionSize { get; init; } = 1024;
    public bool UnrollQ4 { get; init; }
    public bool NativeHalfScale { get; init; } = true;
    // Full diagnostics opt back into per-kernel events; completion fences are
    // retained when timing is disabled. Training always retains its telemetry.
    public bool CollectKernelTimings { get; init; }
    public bool TrainingBatchGradientNorm { get; init; } = true;
    public int TrainingNormSplits { get; init; } = 8;
    public bool TrainingCooperativeLora { get; init; } = true;
    public bool TrainingCooperativeDelta { get; init; } = true;
    public bool TrainingResponseOnlyHead { get; init; } = true;
    public int TrainingTransposeRows { get; init; } = 16;
    public int TrainingForwardRows { get; init; } = 4;
    public int TrainingBufferPoolMiB { get; init; } = 512;
    public bool DetailedProfiling { get; init; }
    public bool FusedDelta { get; init; } = true;
    private int? _queuedKernelLimit;
    // An explicit override applies to either workload; unchanged training
    // callers keep their original 512-command window.
    public int QueuedKernelLimit
    {
        get => _queuedKernelLimit ?? (LoraTraining ? 512 : 4096);
        init => _queuedKernelLimit = value;
    }
}
