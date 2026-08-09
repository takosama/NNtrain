namespace NNtrain;

public sealed record ModuleState(
    int FormatVersion,
    ModuleParameterState[] Parameters)
{
    public const int CurrentFormatVersion = 1;
}

public sealed record ModuleParameterState(
    int Index,
    string Name,
    int[] Shape,
    float[] Values);
