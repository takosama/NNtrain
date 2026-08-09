namespace NNtrain;

public sealed record GainShareAdamWOptions
{
    public float LearningRate { get; init; } = 3e-4f;

    public float Beta1 { get; init; } = 0.9f;

    public float Beta2 { get; init; } = 0.999f;

    public float Epsilon { get; init; } = 1e-8f;

    public float Rho { get; init; } = 0.95f;

    public float Gamma { get; init; } = 1f;

    public float MinScale { get; init; } = 0.5f;

    public float MaxScale { get; init; } = 2f;

    public float WeightDecay { get; init; } = 5e-4f;

    public bool Decay1D { get; init; }
}
