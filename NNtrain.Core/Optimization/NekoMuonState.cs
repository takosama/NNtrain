namespace NNtrain;

public sealed record NekoMuonState(
    int FormatVersion,
    int Step,
    NekoMuonOptions Options,
    NekoMuonParameterState[] ParameterStates)
{
    public const int CurrentFormatVersion = 2;

    /// <summary>
    /// Product of every fast-moment decay used through <see cref="Step"/>.
    /// Tracking the product, rather than assuming BetaFast^Step, keeps bias
    /// correction exact when a resumed run explicitly changes BetaFast.
    /// Null is reserved for legacy format-version 1 checkpoints.
    /// </summary>
    public double? FastDecayProduct { get; init; } =
        Math.Pow(Options.BetaFast, Step);

    /// <summary>
    /// Product of every slow-moment decay used through <see cref="Step"/>.
    /// Null is reserved for legacy format-version 1 checkpoints.
    /// </summary>
    public double? SlowDecayProduct { get; init; } =
        Math.Pow(Options.BetaSlow, Step);
}

public sealed record NekoMuonParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] FastMoment,
    float[] SlowMoment,
    float Confidence);
