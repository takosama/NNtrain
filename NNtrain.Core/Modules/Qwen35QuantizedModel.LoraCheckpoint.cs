using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
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
        if (Path.GetExtension(path).Equals(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            if (expectedTrainingIdentity is not null)
                throw new NotSupportedException("GGUF LoRA cannot resume training; use the NNtrain adapter.bin checkpoint.");
            LoadLoraGguf(path);
            return;
        }
        using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 64 * 1024, options: FileOptions.SequentialScan);
        LoraHeader? header = null;
        var loaded = ReadLoraCheckpoint(stream, _options.LoraTraining, json =>
        {
            header = JsonSerializer.Deserialize<LoraHeader>(json)
                ?? throw new InvalidDataException("Missing LoRA header.");
            if (header.Version != 1 || header.Step < 0 || header.Options is null || header.Entries is null
                || header.Entries.Length == 0)
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
            var shapes = new (int Input, int Output, int Rank)[header.Entries.Length];
            for (int i = 0; i < header.Entries.Length; i++)
            {
                LoraEntry entry = header.Entries[i];
                Matrix matrix = entry.Name == "output.weight" ? OutputMatrix : _matrices[entry.Name];
                if (entry.Input != matrix._inputWidth || entry.Output != matrix.OutputWidth)
                    throw new InvalidDataException("LoRA matrix shape mismatch.");
                shapes[i] = (entry.Input, entry.Output, header.Options.Rank);
            }
            return shapes;
        });
        // The entire adapter has passed tensor and checksum validation. The
        // protected base snapshot is still hashed in full before any attachment.
        LoraHeader validatedHeader = header ?? throw new InvalidDataException("Missing LoRA header.");
        if (validatedHeader.ModelSha256 != ModelFingerprint())
            throw new InvalidDataException("LoRA checkpoint base/version mismatch.");
        try
        {
            AttachLora(validatedHeader.Options);
            for (int i = 0; i < validatedHeader.Entries.Length; i++)
                _lora[validatedHeader.Entries[i].Name].RestoreState(loaded.States[i], _options.LoraTraining);
            LoraStep = validatedHeader.Step; Reset();
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
            _modelFingerprint = ComputeModelFingerprint(stream);
        }
        return _modelFingerprint;
    }

    // Read the entire protected model snapshot with bounded storage. Large reads
    // avoid a native SHA update and file read for every 4 KiB of a multi-GB model.
    internal static string ComputeModelFingerprint(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        stream.Position = 0;
        const int bufferBytes = 4 * 1024 * 1024;
        byte[] buffer = ArrayPool<byte>.Shared.Rent(bufferBytes);
        try
        {
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            int count;
            while ((count = stream.Read(buffer.AsSpan(0, bufferBytes))) != 0)
                hash.AppendData(buffer.AsSpan(0, count));
            return Convert.ToHexString(hash.GetHashAndReset());
        }
        finally { ArrayPool<byte>.Shared.Return(buffer); }
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
    /// <summary>Consumes and verifies the checkpoint once; inference retains only A/B.</summary>
    internal static (byte[] Header, float[][][] States) ReadLoraCheckpoint(Stream stream, bool training,
        Func<byte[], (int Input, int Output, int Rank)[]> validateHeader)
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentNullException.ThrowIfNull(validateHeader);
        if (stream.Position != 0) throw new ArgumentException("Read a LoRA checkpoint from its beginning.", nameof(stream));
        if (stream.Length < 44) throw new InvalidDataException("Truncated LoRA checkpoint.");
        long payloadEnd = stream.Length - 32, consumed = 0;
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> prefix = stackalloc byte[12];
        ReadPayload(prefix);
        if (!prefix[..8].SequenceEqual(LoraMagic))
            throw new InvalidDataException("Not an NNtrain Qwen3.5 LoRA checkpoint.");
        int headerLength = BinaryPrimitives.ReadInt32LittleEndian(prefix[8..]);
        if (headerLength < 1 || headerLength > 1024 * 1024 || headerLength > payloadEnd - consumed)
            throw new InvalidDataException("Invalid LoRA header length.");
        byte[] json = new byte[headerLength];
        ReadPayload(json);
        var shapes = validateHeader(json);
        if (shapes is null || shapes.Length == 0) throw new InvalidDataException("Missing LoRA tensor directory.");
        var states = new float[shapes.Length][][];
        float[]? scratch = training ? null : ArrayPool<float>.Shared.Rent(16 * 1024);
        try
        {
            for (int entry = 0; entry < shapes.Length; entry++)
            {
                var shape = shapes[entry];
                if (shape.Input < 1 || shape.Output < 1 || shape.Rank < 1)
                    throw new InvalidDataException("Invalid LoRA tensor dimensions.");
                var arrays = new float[6][];
                for (int field = 0; field < arrays.Length; field++)
                {
                    int count = checked((field % 2 == 0 ? shape.Input : shape.Output) * shape.Rank);
                    if (4L * count > payloadEnd - consumed) throw new InvalidDataException("Truncated LoRA tensor.");
                    if (training || field < 2)
                    {
                        arrays[field] = new float[count];
                        ReadValues(arrays[field], field >= 4);
                    }
                    else
                    {
                        // Validate every optimizer value, even though an
                        // inference-only model cannot train or save a checkpoint.
                        arrays[field] = [];
                        for (int offset = 0; offset < count;)
                        {
                            int chunk = Math.Min(scratch!.Length, count - offset);
                            ReadValues(scratch.AsSpan(0, chunk), field >= 4);
                            offset += chunk;
                        }
                    }
                }
                states[entry] = arrays;
            }
            if (consumed != payloadEnd) throw new InvalidDataException("Unexpected trailing LoRA tensor data.");
            Span<byte> savedDigest = stackalloc byte[32];
            stream.ReadExactly(savedDigest);
            if (!CryptographicOperations.FixedTimeEquals(hash.GetHashAndReset(), savedDigest))
                throw new InvalidDataException("LoRA checkpoint checksum mismatch.");
            return (json, states);
        }
        finally { if (scratch is not null) ArrayPool<float>.Shared.Return(scratch); }

        void ReadPayload(Span<byte> destination)
        {
            if (destination.Length > payloadEnd - consumed) throw new InvalidDataException("Truncated LoRA checkpoint.");
            stream.ReadExactly(destination);
            hash.AppendData(destination);
            consumed += destination.Length;
        }
        void ReadValues(Span<float> values, bool nonnegative)
        {
            Span<byte> bytes = MemoryMarshal.AsBytes(values);
            ReadPayload(bytes);
            if (!BitConverter.IsLittleEndian)
                for (int i = 0; i < values.Length; i++)
                    values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.Slice(i * sizeof(float), sizeof(float)));
            foreach (float value in values)
                if (!float.IsFinite(value) || (nonnegative && value < 0))
                    throw new InvalidDataException("Invalid LoRA parameter/optimizer state.");
        }
    }
}
