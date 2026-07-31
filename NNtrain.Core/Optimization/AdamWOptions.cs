namespace NNtrain;

public sealed record AdamWOptions
{
    public float LearningRate { get; init; } = 1e-3f;

    public float Beta1 { get; init; } = 0.9f;

    public float Beta2 { get; init; } = 0.999f;

    public float Epsilon { get; init; } = 1e-8f;

    public float WeightDecay { get; init; } = 1e-2f;

    public bool Decay1D { get; init; }
}
