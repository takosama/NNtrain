using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace NNtrain;

public sealed partial class Qwen35QuantizedModel
{
    private sealed record LoraEntry(string Name, int Input, int Output);
    private sealed record LoraHeader(int Version, string ModelSha256, string? TrainingIdentity,
        Qwen35LoraOptions Options, int Step, LoraEntry[] Entries);
    private static readonly byte[] LoraMagic = "NNQ35LR1"u8.ToArray();

    /// <summary>Atomic NNtrain adapter checkpoint containing only LoRA and Adam states.</summary>
    public void SaveLora(string path, string? trainingIdentity = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_options.LoraTraining || _loraOptions is null || _faulted || _loraFaulted)
            throw new InvalidOperationException("Only a valid training model can save a LoRA checkpoint.");
        path = Path.GetFullPath(path);
        if (path.Equals(_modelPath, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Adapter checkpoint must not overwrite the base GGUF.");
        var entries = _lora.OrderBy(pair => pair.Key, StringComparer.Ordinal)
            .Select(pair => new LoraEntry(pair.Key, pair.Value.Input, pair.Value.Output)).ToArray();
        var header = new LoraHeader(1, ModelFingerprint(), trainingIdentity, _loraOptions, LoraStep, entries);
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        string temporary = path + "." + Guid.NewGuid().ToString("N") + ".tmp";
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write,
                FileShare.None, bufferSize: 64 * 1024, options: FileOptions.SequentialScan))
            {
                WriteLoraCheckpoint(stream, json,
                    entries.SelectMany(entry => _lora[entry.Name].ReadState()));
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporary, path, overwrite: true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }

    /// <summary>Validates checksum/base/config/shape before attaching any loaded state.</summary>
    public void LoadLora(string path, string? expectedTrainingIdentity = null)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_lora.Count != 0) throw new InvalidOperationException("A LoRA adapter is already attached.");
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        if (stream.Length < 44) throw new InvalidDataException("Truncated LoRA checkpoint.");
        long payloadEnd = stream.Length - 32;
        byte[] digest = HashPrefix(stream, payloadEnd), savedDigest = new byte[32];
        stream.ReadExactly(savedDigest);
        if (!CryptographicOperations.FixedTimeEquals(digest, savedDigest))
            throw new InvalidDataException("LoRA checkpoint checksum mismatch.");
        stream.Position = 0;
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (!reader.ReadBytes(8).SequenceEqual(LoraMagic)) throw new InvalidDataException("Not an NNtrain Qwen3.5 LoRA checkpoint.");
        int length = reader.ReadInt32();
        if (length < 1 || length > 1024 * 1024 || length > payloadEnd - stream.Position)
            throw new InvalidDataException("Invalid LoRA header length.");
        LoraHeader header = JsonSerializer.Deserialize<LoraHeader>(reader.ReadBytes(length))
            ?? throw new InvalidDataException("Missing LoRA header.");
        if (header.Version != 1 || header.Step < 0 || header.Options is null || header.Entries is null
            || header.Entries.Length == 0 || header.ModelSha256 != ModelFingerprint())
            throw new InvalidDataException("LoRA checkpoint base/version mismatch.");
        if (expectedTrainingIdentity is not null && header.TrainingIdentity != expectedTrainingIdentity)
            throw new InvalidDataException("LoRA training data/config identity mismatch.");
        header.Options.Validate();
        if (header.Options.Layers?.Any(layer => layer >= Descriptor.LayerCount) == true)
            throw new InvalidDataException("Invalid LoRA layers.");
        string[] expected = _matrices.Keys.Where(name =>
        {
            if (!name.StartsWith("blk.", StringComparison.Ordinal)) return false;
            string[] parts = name.Split('.');
            return (header.Options.Layers is null || header.Options.Layers.Contains(int.Parse(parts[1])))
                && header.Options.Targets.Contains(parts[2]);
        }).Concat(header.Options.IncludeOutput ? ["output.weight"] : Array.Empty<string>())
            .Order(StringComparer.Ordinal).ToArray();
        if (!header.Entries.Select(entry => entry.Name).SequenceEqual(expected))
            throw new InvalidDataException("LoRA target directory mismatch.");
        var states = new Dictionary<string, float[][]>(StringComparer.Ordinal);
        foreach (LoraEntry entry in header.Entries)
        {
            Matrix matrix = entry.Name == "output.weight" ? OutputMatrix : _matrices[entry.Name];
            if (entry.Input != matrix._inputWidth || entry.Output != matrix.OutputWidth)
                throw new InvalidDataException("LoRA matrix shape mismatch.");
            var arrays = new float[6][];
            for (int i = 0; i < arrays.Length; i++)
            {
                int count = checked((i % 2 == 0 ? entry.Input : entry.Output) * header.Options.Rank);
                if (4L * count > payloadEnd - stream.Position) throw new InvalidDataException("Truncated LoRA tensor.");
                float[] values = new float[count];
                if (BitConverter.IsLittleEndian) stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
                else for (int j = 0; j < count; j++) values[j] = reader.ReadSingle();
                if (values.Any(value => !float.IsFinite(value) || (i >= 4 && value < 0)))
                    throw new InvalidDataException("Invalid LoRA parameter/optimizer state.");
                arrays[i] = values;
            }
            states.Add(entry.Name, arrays);
        }
        if (stream.Position != payloadEnd) throw new InvalidDataException("Unexpected trailing LoRA tensor data.");
        try
        {
            AttachLora(header.Options);
            foreach (var pair in states) _lora[pair.Key].RestoreState(pair.Value, _options.LoraTraining);
            LoraStep = header.Step; Reset();
        }
        catch
        {
            foreach (var adapter in _lora.Values) adapter.Dispose();
            _lora.Clear(); _loraOptions = null; LoraStep = 0;
            throw;
        }
    }

    private string ModelFingerprint()
    {
        if (_modelFingerprint is null)
        {
            FileStream stream = _modelSource ?? throw new InvalidOperationException("Base GGUF source is closed.");
            stream.Position = 0;
            _modelFingerprint = Convert.ToHexString(SHA256.HashData(stream));
        }
        return _modelFingerprint;
    }
    // Hash the same bytes as they are written; no second read of a large adapter
    // checkpoint is needed. The v1 little-endian payload and trailing digest stay
    // byte-for-byte compatible with BinaryWriter plus HashPrefix.
    internal static void WriteLoraCheckpoint(Stream stream, byte[] json, IEnumerable<float[]> states)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        WriteHashed(stream, hash, LoraMagic);
        Span<byte> headerLength = stackalloc byte[sizeof(int)];
        BinaryPrimitives.WriteInt32LittleEndian(headerLength, json.Length);
        WriteHashed(stream, hash, headerLength);
        WriteHashed(stream, hash, json);
        foreach (float[] values in states)
        {
            foreach (float value in values)
                if (!float.IsFinite(value)) throw new ArithmeticException("Cannot save non-finite LoRA state.");
            WriteFloats(stream, hash, values);
        }
        stream.Write(hash.GetHashAndReset());
    }

    private static void WriteFloats(Stream stream, IncrementalHash hash, float[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            WriteHashed(stream, hash, MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }
        Span<byte> buffer = stackalloc byte[4096];
        for (int offset = 0; offset < values.Length;)
        {
            int count = Math.Min(buffer.Length / sizeof(float), values.Length - offset);
            for (int i = 0; i < count; i++)
                BinaryPrimitives.WriteSingleLittleEndian(buffer.Slice(i * sizeof(float), sizeof(float)), values[offset + i]);
            WriteHashed(stream, hash, buffer[..(count * sizeof(float))]);
            offset += count;
        }
    }

    private static void WriteHashed(Stream stream, IncrementalHash hash, ReadOnlySpan<byte> bytes)
    {
        stream.Write(bytes);
        hash.AppendData(bytes);
    }
    private static byte[] HashPrefix(Stream stream, long length)
    {
        stream.Position = 0;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var buffer = new byte[1024 * 1024]; long remaining = length;
        while (remaining > 0)
        {
            int read = stream.Read(buffer, 0, (int)Math.Min(buffer.Length, remaining));
            if (read == 0) throw new EndOfStreamException();
            hash.AppendData(buffer, 0, read); remaining -= read;
        }
        return hash.GetHashAndReset();
    }
}
