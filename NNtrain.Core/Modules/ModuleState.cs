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
    float[] Values,
    TensorDType DType = TensorDType.Float32,
    [property: System.Text.Json.Serialization.JsonIgnore(
        Condition = System.Text.Json.Serialization.JsonIgnoreCondition
            .WhenWritingNull)]
    TensorStorageMetadata? StorageMetadata = null);
