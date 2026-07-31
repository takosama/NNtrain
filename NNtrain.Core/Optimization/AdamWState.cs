namespace NNtrain;

public sealed record AdamWState(
    int FormatVersion,
    int Step,
    AdamWOptions Options,
    AdamWParameterState[] ParameterStates)
{
    public const int CurrentFormatVersion = 1;
}

public sealed record AdamWParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] FirstMoment,
    float[] SecondMoment);
