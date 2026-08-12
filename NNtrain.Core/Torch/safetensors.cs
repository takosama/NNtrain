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
    }
}

internal static class SafeTensorFile
{
    private const int LengthPrefixSize = sizeof(long);
    private const int FloatSize = sizeof(float);
    private const int MaximumHeaderBytes = 64 * 1024 * 1024;
    private const string FormatMetadata = "nntrain.module_state.f32.v1";

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
                    WriteFloats(stream, parameter.Values);
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

    internal static ModuleState Load(string path)
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
        var parameters = new List<ModuleParameterState>();
        var seenIndexes = new HashSet<int>();
        foreach (JsonProperty property in
            document.RootElement.EnumerateObject())
        {
            if (property.NameEquals("__metadata__"))
                continue;
            int separator = property.Name.IndexOf(':');
            if (separator <= 0
                || !int.TryParse(
                    property.Name.AsSpan(0, separator),
                    out int index)
                || index < 0
                || !seenIndexes.Add(index))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter key '{property.Name}' is invalid.");
            }

            JsonElement descriptor = property.Value;
            string? dtype = descriptor.GetProperty("dtype").GetString();
            if (!string.Equals(dtype, "F32", StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{property.Name}' uses " +
                    $"unsupported dtype '{dtype}'. Only F32 is supported.");
            }
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
            if (end - start != checked((long)elementCount * FloatSize))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{property.Name}' shape and " +
                    "byte range do not match.");
            }
            var values = new float[elementCount];
            stream.Position = checked(dataStart + start);
            ReadFloats(stream, values);
            parameters.Add(
                new ModuleParameterState(
                    index,
                    property.Name[(separator + 1)..],
                    shape,
                    values));
        }

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

    private static byte[] CreateHeader(ModuleState state)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            writer.WriteString("format", FormatMetadata);
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
                writer.WriteString("dtype", "F32");
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (int dimension in parameter.Shape)
                    writer.WriteNumberValue(dimension);
                writer.WriteEndArray();
                long end = checked(
                    offset + (long)parameter.Values.Length * FloatSize);
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
                    != GetElementCount(parameter.Shape))
            {
                throw new ArgumentException(
                    $"Module state parameter {index} is invalid.",
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

    private static void WriteFloats(Stream stream, float[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }
        Span<byte> bytes = stackalloc byte[FloatSize];
        foreach (float value in values)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
            stream.Write(bytes);
        }
    }

    private static void ReadFloats(Stream stream, float[] values)
    {
        if (BitConverter.IsLittleEndian)
        {
            ReadExactly(stream, MemoryMarshal.AsBytes(values.AsSpan()));
            return;
        }
        Span<byte> bytes = stackalloc byte[FloatSize];
        for (int index = 0; index < values.Length; index++)
        {
            ReadExactly(stream, bytes);
            values[index] = BinaryPrimitives.ReadSingleLittleEndian(bytes);
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
