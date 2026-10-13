using System.Text.Json;

namespace NNtrain;

internal static partial class SafeTensorFile
{
    private static byte[] CreateHeader(
        int formatVersion,
        bool hasMixedDTypes,
        Action<Utf8JsonWriter> writeParameters)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            writer.WriteString(
                "format",
                hasMixedDTypes ? MixedFormatMetadata : Float32FormatMetadata);
            writer.WriteString(
                "module_state_format_version",
                formatVersion.ToString(
                    System.Globalization.CultureInfo.InvariantCulture));
            writer.WriteEndObject();

            writeParameters(writer);
            writer.WriteEndObject();
        }

        byte[] json = buffer.ToArray();
        int paddedLength = checked((json.Length + 7) & ~7);
        var header = new byte[paddedLength];
        json.CopyTo(header, 0);
        header.AsSpan(json.Length).Fill((byte)' ');
        return header;
    }

    private static long WriteParameterHeader(
        Utf8JsonWriter writer,
        int index,
        string name,
        IReadOnlyList<int> shape,
        int elementCount,
        SafeTensorDTypeCodec codec,
        long offset)
    {
        writer.WritePropertyName($"{index:D8}:{name}");
        writer.WriteStartObject();
        writer.WriteString("dtype", codec.DescriptorName);
        writer.WritePropertyName("shape");
        writer.WriteStartArray();
        foreach (int dimension in shape)
            writer.WriteNumberValue(dimension);
        writer.WriteEndArray();
        long end = checked(
            offset + (long)elementCount * codec.EncodedElementSize);
        writer.WritePropertyName("data_offsets");
        writer.WriteStartArray();
        writer.WriteNumberValue(offset);
        writer.WriteNumberValue(end);
        writer.WriteEndArray();
        writer.WriteEndObject();
        return end;
    }
}
