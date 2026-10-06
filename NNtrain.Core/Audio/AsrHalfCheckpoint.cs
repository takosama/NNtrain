using System.Buffers.Binary;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
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
        => Load(path, maximumWeightBytes, cancellationToken, null);

    internal static AsrHalfCheckpoint Load(string path, long maximumWeightBytes,
        CancellationToken cancellationToken, Action<string, double>? timingObserver)
    {
        long started = Stopwatch.GetTimestamp();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumWeightBytes);
        cancellationToken.ThrowIfCancellationRequested();
        // Deny concurrent writes; every load validates the current source, without a cache.
        using var file = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1, FileOptions.SequentialScan);
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
        timingObserver?.Invoke("checkpoint.headerValidation", Stopwatch.GetElapsedTime(started).TotalMilliseconds);
        double allocateMilliseconds = 0, readMilliseconds = 0, convertMilliseconds = 0;
        var weights = new Dictionary<string, Weight>(StringComparer.Ordinal);
        byte[] staging = new byte[StagingBytes];
        foreach (var descriptor in descriptors)
        {
            cancellationToken.ThrowIfCancellationRequested();
            long phase = Stopwatch.GetTimestamp();
            var values = new Half[descriptor.Count];
            allocateMilliseconds += Stopwatch.GetElapsedTime(phase).TotalMilliseconds;
            file.Position = dataStart + descriptor.Start;
            for (int offset = 0; offset < values.Length;)
            {
                cancellationToken.ThrowIfCancellationRequested();
                int count = Math.Min(values.Length - offset, staging.Length / descriptor.Width);
                Span<Half> destination = values.AsSpan(offset, count);
                phase = Stopwatch.GetTimestamp();
                if (descriptor.Width == 2 && BitConverter.IsLittleEndian)
                    file.ReadExactly(MemoryMarshal.AsBytes(destination));
                else
                    file.ReadExactly(staging.AsSpan(0, count * descriptor.Width));
                readMilliseconds += Stopwatch.GetElapsedTime(phase).TotalMilliseconds;
                phase = Stopwatch.GetTimestamp();
                if (descriptor.Width == 4 && BitConverter.IsLittleEndian)
                    TensorStorageCodec.EncodeFloat16(MemoryMarshal.Cast<byte, float>(staging.AsSpan(0, count * 4)), destination);
                else if (!BitConverter.IsLittleEndian)
                {
                    for (int i = 0; i < count; i++)
                        destination[i] = descriptor.Width == 2
                            ? BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(staging.AsSpan(i * 2, 2)))
                            : (Half)BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadInt32LittleEndian(staging.AsSpan(i * 4, 4)));
                }
                ValidateFinite(destination, descriptor.Name);
                cancellationToken.ThrowIfCancellationRequested();
                convertMilliseconds += Stopwatch.GetElapsedTime(phase).TotalMilliseconds;
                offset += count;
            }
            weights.Add(descriptor.Name, new Weight(descriptor.Shape, values));
        }
        timingObserver?.Invoke("checkpoint.allocate", allocateMilliseconds);
        timingObserver?.Invoke("checkpoint.read", readMilliseconds);
        timingObserver?.Invoke("checkpoint.convertValidate", convertMilliseconds);
        cancellationToken.ThrowIfCancellationRequested();
        return new AsrHalfCheckpoint(weights, totalBytes);
    }

    private static void ValidateFinite(ReadOnlySpan<Half> values, string name)
    {
        ReadOnlySpan<ushort> bits = MemoryMarshal.Cast<Half, ushort>(values);
        var exponent = new Vector<ushort>(0x7c00);
        int i = 0;
        if (Vector.IsHardwareAccelerated)
            for (; i <= bits.Length - Vector<ushort>.Count; i += Vector<ushort>.Count)
                if (Vector.EqualsAny(new Vector<ushort>(bits.Slice(i, Vector<ushort>.Count)) & exponent, exponent))
                    throw new InvalidDataException($"Nonfinite or FP16-overflowing weight: {name}.");
        for (; i < bits.Length; i++)
            if ((bits[i] & 0x7c00) == 0x7c00)
                throw new InvalidDataException($"Nonfinite or FP16-overflowing weight: {name}.");
    }
}
