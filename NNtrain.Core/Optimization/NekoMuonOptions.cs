namespace NNtrain;

public enum NekoMuonNewtonSchulzDepthMode
{
    Adaptive = 0,
    Minimum = 1,
    Fixed = 2,
}

public sealed record NekoMuonOptions
{
    public float LearningRate { get; init; } = 3e-4f;

    public float BetaFast { get; init; } = 0.9f;

    public float BetaSlow { get; init; } = 0.99f;

    /// <summary>
    /// Uses the ordinary Muon Nesterov direction
    /// beta * m_t + (1 - beta) * g_t as the Newton-Schulz input. The default
    /// is false so existing NekoMuon checkpoints and callers retain their
    /// original fast-moment semantics.
    /// </summary>
    public bool Nesterov { get; init; }

    public float Rho { get; init; } = 0.9f;

    public float Epsilon { get; init; } = 1e-7f;

    public int MaxNewtonSchulzSteps { get; init; } = 5;

    public int NewtonSchulzInterval { get; init; } = 5;

    public NekoMuonNewtonSchulzDepthMode NewtonSchulzDepthMode { get; init; } =
        NekoMuonNewtonSchulzDepthMode.Adaptive;

    public float NewtonSchulzDepth { get; init; }

    public float WeightDecay { get; init; } = 1e-2f;

    public bool Decay1D { get; init; }
}
