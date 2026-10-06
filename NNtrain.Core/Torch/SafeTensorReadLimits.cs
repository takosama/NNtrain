namespace NNtrain;

/// <summary>Header and logical float32 expansion limits, shared by eager and streamed readers.</summary>
internal sealed record SafeTensorReadLimits
{
    public int MaximumHeaderBytes { get; init; } = 64 * 1024 * 1024;
    public int MaximumTensors { get; init; } = 100_000;
    public int MaximumRank { get; init; } = 64;
    public long MaximumDecodedBytes { get; init; } = 16L * 1024 * 1024 * 1024;

    internal void Validate()
    {
        if (MaximumHeaderBytes < 0 || MaximumTensors < 0 || MaximumRank < 0 || MaximumDecodedBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(SafeTensorReadLimits));
    }
}
