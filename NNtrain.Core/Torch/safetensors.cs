#pragma warning disable CS8981

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace NNtrain;

/// <summary>SafeTensors-compatible model state serialization.</summary>
public static class safetensors
{
    public static class torch
    {
        public static void save_file(ModuleState state, string path)
            => SafeTensorFile.Save(state, path);

        public static ModuleState load_file(string path)
            => SafeTensorFile.Load(path);

        /// <summary>
        /// Loads a SafeTensors file whose tensor keys do not follow the
        /// NNtrain <c>"&lt;index&gt;:&lt;name&gt;"</c> convention, such as a
        /// file written by PyTorch or downloaded from a model hub. Each model
        /// parameter is matched to the file entry with the same shape; header
        /// key order is deliberately ignored because the reference writer
        /// sorts keys alphabetically. Ambiguous shapes are rejected rather
        /// than guessed - pass an explicit key list in that case.
        /// </summary>
        public static ModuleState load_file(string path, Module model)
            => SafeTensorFile.LoadForModel(path, model, keys: null);

        /// <summary>
        /// Loads a SafeTensors file using an explicit file key for each model
        /// parameter slot, in <see cref="Module.Parameters"/> order.
        /// </summary>
        public static ModuleState load_file(
            string path,
            Module model,
            IReadOnlyList<string> keys)
            => SafeTensorFile.LoadForModel(path, model, keys);

        /// <summary>Lists the tensor keys stored in a SafeTensors file.</summary>
        public static IReadOnlyList<string> load_keys(string path)
            => SafeTensorFile.ReadKeys(path);
    }
}

internal static class SafeTensorFile
{
    private const int LengthPrefixSize = sizeof(long);
    private const int Float32Size = sizeof(float);
    private const int Float16Size = sizeof(ushort);
    private const int MaximumHeaderBytes = 64 * 1024 * 1024;
    private const string Float32FormatMetadata =
        "nntrain.module_state.f32.v1";
    private const string MixedFormatMetadata =
        "nntrain.module_state.mixed.v1";

    internal static void Save(ModuleState state, string path)
    {
        ArgumentNullException.ThrowIfNull(state);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ValidateState(state);

        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string temporaryPath = fullPath + ".tmp";

        try
        {
            byte[] header = CreateHeader(state);
            using (var stream = new FileStream(
                temporaryPath,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 1024 * 1024,
                FileOptions.SequentialScan))
            {
                Span<byte> prefix = stackalloc byte[LengthPrefixSize];
                BinaryPrimitives.WriteUInt64LittleEndian(
                    prefix,
                    checked((ulong)header.Length));
                stream.Write(prefix);
                stream.Write(header);
                foreach (ModuleParameterState parameter in
                    state.Parameters.OrderBy(parameter => parameter.Index))
                {
                    WriteValues(stream, parameter.Values, parameter.DType);
                }
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        catch
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
            throw;
        }
    }

    /// <summary>One tensor as it is physically stored in the file.</summary>
    private sealed record SafeTensorEntry(
        string Key,
        int[] Shape,
        TensorDType DType,
        float[] Values);

    internal static IReadOnlyList<string> ReadKeys(string path)
        => ReadEntries(path).Select(entry => entry.Key).ToArray();

    internal static ModuleState Load(string path)
    {
        List<SafeTensorEntry> entries = ReadEntries(path);
        var parameters = new List<ModuleParameterState>(entries.Count);
        var seenIndexes = new HashSet<int>();
        foreach (SafeTensorEntry entry in entries)
        {
            if (!TrySplitIndexedKey(entry.Key, out int index, out string name)
                || !seenIndexes.Add(index))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter key '{entry.Key}' is invalid. " +
                    "Keys must use the NNtrain '<index>:<name>' convention. " +
                    "Files written by other frameworks can be loaded with " +
                    "safetensors.torch.load_file(path, model).");
            }

            parameters.Add(
                new ModuleParameterState(
                    index,
                    name,
                    entry.Shape,
                    entry.Values,
                    entry.DType));
        }

        return Order(parameters);
    }

    /// <summary>
    /// Loads a file whose keys are chosen by another framework. Parameters are
    /// resolved either from an explicit key list or by unique shape match, so
    /// the result never depends on the header's key ordering.
    /// </summary>
    internal static ModuleState LoadForModel(
        string path,
        Module model,
        IReadOnlyList<string>? keys)
    {
        ArgumentNullException.ThrowIfNull(model);
        List<SafeTensorEntry> entries = ReadEntries(path);
        if (keys is null
            && entries.All(entry => TrySplitIndexedKey(entry.Key, out _, out _)))
        {
            return Load(path);
        }

        Parameter[] parameters = model.Parameters().ToArray();
        if (keys is not null && keys.Count != parameters.Length)
        {
            throw new ArgumentException(
                $"The key list has {keys.Count} entries but the model has " +
                $"{parameters.Length} parameters.",
                nameof(keys));
        }

        if (keys is null && entries.Count != parameters.Length)
        {
            throw new InvalidDataException(
                $"SafeTensors file '{path}' holds {entries.Count} tensors but " +
                $"the model has {parameters.Length} parameters. Keys in the " +
                $"file: {string.Join(", ", entries.Select(e => e.Key))}.");
        }

        var states = new ModuleParameterState[parameters.Length];
        var used = new bool[entries.Count];
        for (int index = 0; index < parameters.Length; index++)
        {
            Parameter parameter = parameters[index];
            int[] shape = parameter.T.Shape.ToArray();
            int selected;
            if (keys is not null)
            {
                selected = entries.FindIndex(
                    entry => string.Equals(
                        entry.Key,
                        keys[index],
                        StringComparison.Ordinal));
                if (selected < 0)
                {
                    throw new InvalidDataException(
                        $"SafeTensors file '{path}' has no tensor named " +
                        $"'{keys[index]}'. Keys in the file: " +
                        $"{string.Join(", ", entries.Select(e => e.Key))}.");
                }
            }
            else
            {
                selected = -1;
                for (int candidate = 0; candidate < entries.Count; candidate++)
                {
                    if (used[candidate]
                        || !entries[candidate].Shape.SequenceEqual(shape))
                    {
                        continue;
                    }

                    if (selected >= 0)
                    {
                        throw new InvalidDataException(
                            $"SafeTensors file '{path}' has more than one " +
                            $"unassigned tensor of shape " +
                            $"[{string.Join('x', shape)}] for parameter slot " +
                            $"{index} ('{parameter.Name}'), so the mapping is " +
                            "ambiguous. Pass an explicit key list to " +
                            "safetensors.torch.load_file(path, model, keys). " +
                            $"Keys in the file: " +
                            $"{string.Join(", ", entries.Select(e => e.Key))}.");
                    }

                    selected = candidate;
                }

                if (selected < 0)
                {
                    throw new InvalidDataException(
                        $"SafeTensors file '{path}' has no unassigned tensor " +
                        $"of shape [{string.Join('x', shape)}] for parameter " +
                        $"slot {index} ('{parameter.Name}'). Keys in the " +
                        $"file: {string.Join(", ", entries.Select(e => e.Key))}.");
                }
            }

            SafeTensorEntry entry = entries[selected];
            if (!entry.Shape.SequenceEqual(shape))
            {
                throw new InvalidDataException(
                    $"SafeTensors tensor '{entry.Key}' has shape " +
                    $"[{string.Join('x', entry.Shape)}] but parameter slot " +
                    $"{index} ('{parameter.Name}') expects " +
                    $"[{string.Join('x', shape)}].");
            }

            used[selected] = true;
            states[index] = new ModuleParameterState(
                index,
                parameter.Name,
                entry.Shape,
                entry.Values,
                entry.DType);
        }

        return new ModuleState(ModuleState.CurrentFormatVersion, states);
    }

    private static bool TrySplitIndexedKey(
        string key,
        out int index,
        out string name)
    {
        index = 0;
        name = string.Empty;
        int separator = key.IndexOf(':');
        if (separator <= 0
            || !int.TryParse(key.AsSpan(0, separator), out index)
            || index < 0)
        {
            return false;
        }

        name = key[(separator + 1)..];
        return true;
    }

    private static ModuleState Order(List<ModuleParameterState> parameters)
    {
        ModuleParameterState[] ordered = parameters
            .OrderBy(parameter => parameter.Index)
            .ToArray();
        for (int index = 0; index < ordered.Length; index++)
        {
            if (ordered[index].Index != index)
            {
                throw new InvalidDataException(
                    "SafeTensors parameter indexes must be contiguous.");
            }
        }

        return new ModuleState(ModuleState.CurrentFormatVersion, ordered);
    }

    private static List<SafeTensorEntry> ReadEntries(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string fullPath = Path.GetFullPath(path);
        using var stream = new FileStream(
            fullPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 1024 * 1024,
            FileOptions.RandomAccess);
        Span<byte> prefix = stackalloc byte[LengthPrefixSize];
        ReadExactly(stream, prefix);
        ulong encodedHeaderLength =
            BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (encodedHeaderLength == 0
            || encodedHeaderLength > MaximumHeaderBytes
            || encodedHeaderLength > (ulong)(stream.Length - LengthPrefixSize))
        {
            throw new InvalidDataException(
                "SafeTensors header length is invalid.");
        }

        int headerLength = checked((int)encodedHeaderLength);
        var header = new byte[headerLength];
        ReadExactly(stream, header);
        long dataStart = checked(LengthPrefixSize + (long)headerLength);
        long dataLength = stream.Length - dataStart;
        using JsonDocument document = ParseHeader(header);
        var entries = new List<SafeTensorEntry>();
        var seenKeys = new HashSet<string>(StringComparer.Ordinal);
        foreach (JsonProperty property in
            document.RootElement.EnumerateObject())
        {
            if (property.NameEquals("__metadata__"))
                continue;
            if (!seenKeys.Add(property.Name))
            {
                throw new InvalidDataException(
                    $"SafeTensors key '{property.Name}' appears twice.");
            }

            JsonElement descriptor = property.Value;
            string? dtype = descriptor.GetProperty("dtype").GetString();
            TensorDType tensorDType = dtype switch
            {
                "F32" => TensorDType.Float32,
                "F16" => TensorDType.Float16,
                "BF16" => TensorDType.BFloat16,
                _ => throw new InvalidDataException(
                    $"SafeTensors parameter '{property.Name}' uses " +
                    $"unsupported dtype '{dtype}'. Only F32, F16, and BF16 are " +
                    "supported."),
            };
            int elementSize = GetElementSize(tensorDType);
            int[] shape = descriptor.GetProperty("shape")
                .EnumerateArray()
                .Select(ReadDimension)
                .ToArray();
            JsonElement.ArrayEnumerator offsets = descriptor
                .GetProperty("data_offsets")
                .EnumerateArray();
            if (!offsets.MoveNext())
                throw InvalidOffsets(property.Name);
            long start = offsets.Current.GetInt64();
            if (!offsets.MoveNext())
                throw InvalidOffsets(property.Name);
            long end = offsets.Current.GetInt64();
            if (offsets.MoveNext()
                || start < 0
                || end < start
                || end > dataLength)
            {
                throw InvalidOffsets(property.Name);
            }

            int elementCount = GetElementCount(shape);
            if (end - start != checked((long)elementCount * elementSize))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{property.Name}' shape and " +
                    "byte range do not match.");
            }
            var values = new float[elementCount];
            stream.Position = checked(dataStart + start);
            ReadValues(stream, values, tensorDType);
            entries.Add(
                new SafeTensorEntry(
                    property.Name,
                    shape,
                    tensorDType,
                    values));
        }

        return entries;
    }

    private static byte[] CreateHeader(ModuleState state)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            writer.WriteString(
                "format",
                state.Parameters.Any(
                    parameter => parameter.DType != TensorDType.Float32)
                    ? MixedFormatMetadata
                    : Float32FormatMetadata);
            writer.WriteString(
                "module_state_format_version",
                state.FormatVersion.ToString(
                    System.Globalization.CultureInfo.InvariantCulture));
            writer.WriteEndObject();

            long offset = 0;
            foreach (ModuleParameterState parameter in
                state.Parameters.OrderBy(parameter => parameter.Index))
            {
                writer.WritePropertyName(
                    $"{parameter.Index:D8}:{parameter.Name}");
                writer.WriteStartObject();
                writer.WriteString("dtype", parameter.DType switch
                {
                    TensorDType.Float16 => "F16",
                    TensorDType.BFloat16 => "BF16",
                    _ => "F32",
                });
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (int dimension in parameter.Shape)
                    writer.WriteNumberValue(dimension);
                writer.WriteEndArray();
                long end = checked(
                    offset
                    + (long)parameter.Values.Length
                        * GetElementSize(parameter.DType));
                writer.WritePropertyName("data_offsets");
                writer.WriteStartArray();
                writer.WriteNumberValue(offset);
                writer.WriteNumberValue(end);
                writer.WriteEndArray();
                writer.WriteEndObject();
                offset = end;
            }
            writer.WriteEndObject();
        }

        byte[] json = buffer.ToArray();
        int paddedLength = checked((json.Length + 7) & ~7);
        var header = new byte[paddedLength];
        json.CopyTo(header, 0);
        header.AsSpan(json.Length).Fill((byte)' ');
        return header;
    }

    private static JsonDocument ParseHeader(byte[] header)
    {
        try
        {
            return JsonDocument.Parse(header);
        }
        catch (JsonException exception)
        {
            throw new InvalidDataException(
                "SafeTensors header is not valid JSON.",
                exception);
        }
    }

    private static void ValidateState(ModuleState state)
    {
        if (state.FormatVersion != ModuleState.CurrentFormatVersion
            || state.Parameters is null)
        {
            throw new ArgumentException(
                "Module state format is unsupported.",
                nameof(state));
        }
        ModuleParameterState[] ordered = state.Parameters
            .OrderBy(parameter => parameter.Index)
            .ToArray();
        for (int index = 0; index < ordered.Length; index++)
        {
            ModuleParameterState parameter = ordered[index]
                ?? throw new ArgumentException(
                    "Module state contains a null parameter.",
                    nameof(state));
            if (parameter.Index != index
                || parameter.Shape is null
                || parameter.Values is null
                || parameter.Values.Length
                    != GetElementCount(parameter.Shape)
                || parameter.DType is not TensorDType.Float32
                    and not TensorDType.Float16
                    and not TensorDType.BFloat16)
            {
                throw new ArgumentException(
                    $"Module state parameter {index} is invalid.",
                    nameof(state));
            }

            if (parameter.StorageMetadata is { IsRaw: false })
            {
                throw new ArgumentException(
                    $"Module state parameter {index} contains storage " +
                    "metadata that SafeTensors cannot serialize yet. " +
                    "Only raw Float32, Float16, and BFloat16 payloads are supported.",
                    nameof(state));
            }
        }
    }

    private static int ReadDimension(JsonElement element)
    {
        int dimension = element.GetInt32();
        if (dimension <= 0)
            throw new InvalidDataException("SafeTensors shape is invalid.");
        return dimension;
    }

    private static int GetElementCount(int[] shape)
    {
        ArgumentNullException.ThrowIfNull(shape);
        int count = 1;
        foreach (int dimension in shape)
        {
            if (dimension <= 0)
                throw new InvalidDataException("Tensor shape is invalid.");
            count = checked(count * dimension);
        }
        return count;
    }

    private static InvalidDataException InvalidOffsets(string name)
        => new($"SafeTensors parameter '{name}' has invalid data offsets.");

    private static int GetElementSize(TensorDType dtype)
        => dtype switch
        {
            TensorDType.Float32 => Float32Size,
            TensorDType.Float16 => Float16Size,
            TensorDType.BFloat16 => Float16Size,
            _ => throw new ArgumentOutOfRangeException(
                nameof(dtype),
                dtype,
                "SafeTensors supports only Float32, Float16, and BFloat16 states."),
        };

    private static void WriteValues(
        Stream stream,
        float[] values,
        TensorDType dtype)
    {
        if (dtype == TensorDType.Float16)
        {
            WriteFloat16Values(stream, values);
            return;
        }
        if (dtype == TensorDType.BFloat16)
        {
            WriteBFloat16Values(stream, values);
            return;
        }

        WriteFloat32Values(stream, values);
    }

    private static void ReadValues(
        Stream stream,
        float[] values,
        TensorDType dtype)
    {
        if (dtype == TensorDType.Float16)
        {
            ReadFloat16Values(stream, values);
            return;
        }
        if (dtype == TensorDType.BFloat16)
        {
            ReadBFloat16Values(stream, values);
            return;
        }

        ReadFloat32Values(stream, values);
    }

    private static void WriteFloat32Values(Stream stream, float[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }
        Span<byte> bytes = stackalloc byte[Float32Size];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
            stream.Write(bytes);
        }
    }

    private static void ReadFloat32Values(Stream stream, float[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            ReadExactly(stream, MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }
        Span<byte> bytes = stackalloc byte[Float32Size];
        for (int index = 0; index < values.Length; index++)
        {
            ReadExactly(stream, bytes);
            values[index] = BinaryPrimitives.ReadSingleLittleEndian(bytes);
        }
    }

    private static void WriteFloat16Values(Stream stream, float[] values)
    {
        const int ValuesPerChunk = 4096;
        Span<byte> bytes = stackalloc byte[ValuesPerChunk * Float16Size];
        int offset = 0;
        while (offset < values.Length)
        {
            int count = Math.Min(ValuesPerChunk, values.Length - offset);
            for (int index = 0; index < count; index++)
            {
                ushort bits = BitConverter.HalfToUInt16Bits(
                    (Half)values[offset + index]);
                BinaryPrimitives.WriteUInt16LittleEndian(
                    bytes.Slice(index * Float16Size, Float16Size),
                    bits);
            }
            stream.Write(bytes[..(count * Float16Size)]);
            offset += count;
        }
    }

    private static void ReadFloat16Values(Stream stream, float[] values)
    {
        const int ValuesPerChunk = 4096;
        Span<byte> bytes = stackalloc byte[ValuesPerChunk * Float16Size];
        int offset = 0;
        while (offset < values.Length)
        {
            int count = Math.Min(ValuesPerChunk, values.Length - offset);
            Span<byte> active = bytes[..(count * Float16Size)];
            ReadExactly(stream, active);
            for (int index = 0; index < count; index++)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(
                    active.Slice(index * Float16Size, Float16Size));
                values[offset + index] =
                    (float)BitConverter.UInt16BitsToHalf(bits);
            }
            offset += count;
        }
    }

    private static void WriteBFloat16Values(Stream stream, float[] values)
    {
        const int ValuesPerChunk = 4096;
        Span<byte> bytes = stackalloc byte[ValuesPerChunk * Float16Size];
        int offset = 0;
        while (offset < values.Length)
        {
            int count = Math.Min(ValuesPerChunk, values.Length - offset);
            for (int index = 0; index < count; index++)
            {
                ushort bits = TensorStorageCodec.EncodeBFloat16(
                    values[offset + index]);
                BinaryPrimitives.WriteUInt16LittleEndian(
                    bytes.Slice(index * Float16Size, Float16Size),
                    bits);
            }
            stream.Write(bytes[..(count * Float16Size)]);
            offset += count;
        }
    }

    private static void ReadBFloat16Values(Stream stream, float[] values)
    {
        const int ValuesPerChunk = 4096;
        Span<byte> bytes = stackalloc byte[ValuesPerChunk * Float16Size];
        int offset = 0;
        while (offset < values.Length)
        {
            int count = Math.Min(ValuesPerChunk, values.Length - offset);
            Span<byte> active = bytes[..(count * Float16Size)];
            ReadExactly(stream, active);
            for (int index = 0; index < count; index++)
            {
                ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(
                    active.Slice(index * Float16Size, Float16Size));
                values[offset + index] =
                    TensorStorageCodec.DecodeBFloat16(bits);
            }
            offset += count;
        }
    }

    private static void ReadExactly(Stream stream, Span<byte> destination)
    {
        while (!destination.IsEmpty)
        {
            int read = stream.Read(destination);
            if (read == 0)
                throw new EndOfStreamException("SafeTensors file is truncated.");
            destination = destination[read..];
        }
    }
}
