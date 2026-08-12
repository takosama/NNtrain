namespace NNtrain;

public sealed record NekoMuonOptions
{
    public float LearningRate { get; init; } = 3e-4f;

    public float BetaFast { get; init; } = 0.9f;

    public float BetaSlow { get; init; } = 0.99f;

    public float Rho { get; init; } = 0.9f;

    public float Epsilon { get; init; } = 1e-7f;

    public int MaxNewtonSchulzSteps { get; init; } = 5;

    public int NewtonSchulzInterval { get; init; } = 5;

    public float WeightDecay { get; init; } = 1e-2f;

    public bool Decay1D { get; init; }
}
