namespace NNtrain;

/// <summary>Quantized projection implementations for Qwen3.5 inference comparisons.</summary>
public enum Qwen35QuantizedKernel { Reference, Cooperative, Subgroup, Auto }

public sealed record Qwen35ExecutionOptions
{
    public Qwen35QuantizedKernel QuantizedKernel { get; init; } = Qwen35QuantizedKernel.Auto;
    public bool LoraTraining { get; init; }
    public bool FusedDelta { get; init; } = true;
    public int QueuedKernelLimit { get; init; } = 512;
}
