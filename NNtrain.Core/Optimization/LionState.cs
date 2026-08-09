namespace NNtrain;

public sealed record LionState(
    int FormatVersion,
    int Step,
    LionOptions Options,
    LionParameterState[] ParameterStates)
{
    public const int CurrentFormatVersion = 1;
}

public sealed record LionParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] Momentum);
