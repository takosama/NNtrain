namespace NNtrain;

public sealed record LionOptions
{
    public float LearningRate { get; init; } = 3e-4f;

    public float Beta1 { get; init; } = 0.9f;

    public float Beta2 { get; init; } = 0.99f;

    public float WeightDecay { get; init; } = 1e-2f;

    public bool Decay1D { get; init; }
}
