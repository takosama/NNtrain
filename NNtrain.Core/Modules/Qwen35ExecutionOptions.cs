namespace NNtrain;

/// <summary>Quantized projection implementations for Qwen3.5 comparisons.</summary>
public enum Qwen35QuantizedKernel { Reference, Cooperative, Subgroup, Auto }

public sealed record Qwen35ExecutionOptions
{
    public Qwen35QuantizedKernel QuantizedKernel { get; init; } = Qwen35QuantizedKernel.Auto;
    public bool LoraTraining { get; init; }
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
    public int QueuedKernelLimit { get; init; } = 512;
}
