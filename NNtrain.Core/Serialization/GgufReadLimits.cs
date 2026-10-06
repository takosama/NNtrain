namespace NNtrain;

/// <summary>Per-reader limits for metadata decoding, not tensor payload size.</summary>
public sealed record GgufReadLimits
{
    public int MaximumArrayElements { get; init; } = 4 * 1024 * 1024;
    public int MaximumStringBytes { get; init; } = 16 * 1024 * 1024;
    public int MaximumMetadataEntries { get; init; } = 100_000;
    public int MaximumTensors { get; init; } = 1_000_000;
    public int MaximumNestingDepth { get; init; } = 32;
    // Conservative allocation accounting includes strings, temporary UTF8 bytes,
    // object-array slots, boxed values and table entries; not a GC heap measurement.
    public long MaximumDecodedBytes { get; init; } = 512L * 1024 * 1024;

    internal void Validate()
    {
        if (MaximumArrayElements < 0 || MaximumStringBytes < 0
            || MaximumMetadataEntries < 0 || MaximumTensors < 0
            || MaximumNestingDepth is < 0 or > 128 || MaximumDecodedBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(GgufReadLimits));
    }
}
