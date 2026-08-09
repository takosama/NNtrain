namespace NNtrain;

public sealed record GainShareAdamWState(
    int FormatVersion,
    int Step,
    GainShareAdamWOptions Options,
    GainShareAdamWParameterState[] ParameterStates,
    GainShareAdamWGroupState[] GroupStates)
{
    public const int CurrentFormatVersion = 1;
}

public sealed record GainShareAdamWParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] FirstMoment,
    float[] SecondMoment);

public sealed record GainShareAdamWGroupState(
    int Index,
    int[] ParameterIndices,
    double? AlignmentEma);
