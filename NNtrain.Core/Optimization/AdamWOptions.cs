namespace NNtrain;

public sealed record AdamWOptions
{
    public float LearningRate { get; init; } = 1e-3f;

    public float Beta1 { get; init; } = 0.9f;

    public float Beta2 { get; init; } = 0.999f;

    public float Epsilon { get; init; } = 1e-8f;

    public float WeightDecay { get; init; } = 5e-2f;

    public bool Decay1D { get; init; }

    /// <summary>
    /// Stores the signed first moment in bfloat16 between steps. This reduces
    /// optimizer memory traffic at the cost of bfloat16 moment precision;
    /// checkpoints remain serialized as float32 arrays.
    /// </summary>
    public bool UseBFloat16FirstMoment { get; init; }

    /// <summary>
    /// Stores the non-negative second moment in bfloat16 between steps.
    /// This is an opt-in memory-bandwidth optimization; checkpoints remain
    /// serialized as float32 arrays.
    /// </summary>
    public bool UseBFloat16SecondMoment { get; init; }
}
