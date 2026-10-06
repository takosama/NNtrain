using System.Buffers.Binary;
using System.Text.Json;

namespace NNtrain.Audio;

/// <summary>Host FP16 storage for ASR weights; never materializes the FP32 checkpoint.</summary>
public sealed class AsrHalfCheckpoint
{
    public sealed record Weight(int[] Shape, Half[] Values);
    public IReadOnlyDictionary<string, Weight> Weights { get; }
    public long WeightBytes { get; }
    public const int StagingBytes = 64 * 1024;

    private AsrHalfCheckpoint(Dictionary<string, Weight> weights, long bytes)
        => (Weights, WeightBytes) = (weights, bytes);

    public static AsrHalfCheckpoint Load(string path, long maximumWeightBytes,
        CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumWeightBytes);
        using var file = File.OpenRead(path);
        Span<byte> prefix = stackalloc byte[8];
        file.ReadExactly(prefix);
        ulong headerSize = BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (headerSize is 0 or > 16 * 1024 * 1024 || headerSize > (ulong)Math.Max(0, file.Length - 8))
            throw new InvalidDataException("Invalid safetensors header size.");
        byte[] header = new byte[(int)headerSize];
        file.ReadExactly(header);
        using var json = JsonDocument.Parse(header);
        long dataStart = 8 + (long)headerSize, totalBytes = 0;
        var descriptors = new List<(string Name, int[] Shape, int Count, int Width, long Start, long End)>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var property in json.RootElement.EnumerateObject())
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (property.Name == "__metadata__") continue;
            if (!names.Add(property.Name)) throw new InvalidDataException("Duplicate weight name.");
            var tensor = property.Value;
            int width = tensor.GetProperty("dtype").GetString() switch
            {
                "F32" => 4, "F16" => 2,
                _ => throw new InvalidDataException("ASR weights must be F32 or F16.")
            };
            int[] shape = tensor.GetProperty("shape").EnumerateArray().Select(x => x.GetInt32()).ToArray();
            long count = 1;
            foreach (int dimension in shape)
            {
                if (dimension <= 0) throw new InvalidDataException("Invalid weight dimension.");
                count = checked(count * dimension);
            }
            if (count > int.MaxValue) throw new InvalidDataException("Weight tensor exceeds array limit.");
            var offsets = tensor.GetProperty("data_offsets");
            if (offsets.GetArrayLength() != 2) throw new InvalidDataException("Invalid weight offsets.");
            long start = offsets[0].GetInt64(), end = offsets[1].GetInt64();
            if (start < 0 || end < start || end > file.Length - dataStart || end - start != checked(count * width))
                throw new InvalidDataException("Weight size or offset mismatch.");
            totalBytes = checked(totalBytes + count * 2);
            if (totalBytes > maximumWeightBytes) throw new InvalidDataException("FP16 weights exceed configured host memory budget.");
            descriptors.Add((property.Name, shape, (int)count, width, start, end));
        }
        long expected = 0;
        foreach (var descriptor in descriptors.OrderBy(x => x.Start))
        {
            if (descriptor.Start != expected) throw new InvalidDataException("Overlapping or noncontiguous weight data.");
            expected = descriptor.End;
        }
        if (expected != file.Length - dataStart) throw new InvalidDataException("Unclaimed checkpoint data.");
        var weights = new Dictionary<string, Weight>(StringComparer.Ordinal);
        byte[] staging = new byte[StagingBytes];
        foreach (var descriptor in descriptors)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var values = new Half[descriptor.Count];
            file.Position = dataStart + descriptor.Start;
            for (int offset = 0; offset < values.Length;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int count = Math.Min(values.Length - offset, staging.Length / descriptor.Width);
                file.ReadExactly(staging.AsSpan(0, count * descriptor.Width));
                for (int i = 0; i < count; i++)
                {
                    Half value = descriptor.Width == 2
                        ? BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(staging.AsSpan(i * 2, 2)))
                        : (Half)BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(staging.AsSpan(i * 4, 4)));
                    if (!Half.IsFinite(value)) throw new InvalidDataException($"Nonfinite or FP16-overflowing weight: {descriptor.Name}.");
                    values[offset + i] = value;
                }
                offset += count;
            }
            weights.Add(descriptor.Name, new Weight(descriptor.Shape, values));
        }
        return new AsrHalfCheckpoint(weights, totalBytes);
    }
}
