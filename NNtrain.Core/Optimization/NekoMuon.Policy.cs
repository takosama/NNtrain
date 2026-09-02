namespace NNtrain;

public sealed partial class NekoMuon
{
    /// <summary>
    /// Changes the fast-moment decay while preserving accumulated optimizer
    /// state. This is used for an explicit runtime policy override after a
    /// checkpoint restore.
    /// </summary>
    public void SetBetaFast(float betaFast)
    {
        NekoMuonOptions options = _state.Options with
        {
            BetaFast = betaFast,
        };
        ValidateOptions(options, nameof(betaFast));
        _state = _state with { Options = options };
    }
}
