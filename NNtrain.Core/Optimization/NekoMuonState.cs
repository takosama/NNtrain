namespace NNtrain;

public sealed record NekoMuonState(
    int FormatVersion,
    int Step,
    NekoMuonOptions Options,
    NekoMuonParameterState[] ParameterStates)
{
    public const int CurrentFormatVersion = 1;
}

public sealed record NekoMuonParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] FastMoment,
    float[] SlowMoment,
    float Confidence);
