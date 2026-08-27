using System.Globalization;
using System.Text;
using System.Text.Json;

namespace NNtrain;

/// <summary>
/// Restores the embedded JSON ModuleState through a bounded character buffer.
/// Classification checkpoints keep exact FP32 master values in this field,
/// so mixed-precision resume cannot substitute the quantized SafeTensors
/// compatibility artifact.
/// </summary>
internal static class StreamingModuleStateJsonReader
{
    internal static void Restore(
        string path,
        string propertyName,
        Module model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentException.ThrowIfNullOrWhiteSpace(propertyName);
        ArgumentNullException.ThrowIfNull(model);

        using var stream = new FileStream(
            Path.GetFullPath(path),
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 64 * 1024,
            FileOptions.SequentialScan);
        using var text = new StreamReader(
            stream,
            new UTF8Encoding(
                encoderShouldEmitUTF8Identifier: false,
                throwOnInvalidBytes: true),
            detectEncodingFromByteOrderMarks: true,
            bufferSize: 64 * 1024,
            leaveOpen: false);
        using var staging = new CheckpointFloatStagingBuffer();
        var json = new JsonCursor(text);
        Parameter[] parameters = model.parameters().ToArray();

        json.BeginObject();
        bool first = true;
        bool restored = false;
        while (json.TryReadProperty(ref first, out string name))
        {
            if (!string.Equals(name, propertyName, StringComparison.Ordinal))
            {
                json.SkipValue();
                continue;
            }

            RestoreModuleState(json, parameters, staging);
            restored = true;
            break;
        }
        if (!restored)
        {
            throw new JsonException(
                $"Checkpoint property '{propertyName}' is missing.");
        }
    }

    private static void RestoreModuleState(
        JsonCursor json,
        IReadOnlyList<Parameter> parameters,
        CheckpointFloatStagingBuffer staging)
    {
        json.BeginObject();
        bool first = true;
        int formatVersion = 0;
        int restoredParameters = -1;
        while (json.TryReadProperty(ref first, out string name))
        {
            if (name == nameof(ModuleState.FormatVersion))
                formatVersion = json.ReadInt32();
            else if (name == nameof(ModuleState.Parameters))
                restoredParameters = RestoreParameters(
                    json,
                    parameters,
                    staging);
            else
                json.SkipValue();
        }

        if (formatVersion != ModuleState.CurrentFormatVersion
            || restoredParameters != parameters.Count)
        {
            throw new JsonException(
                "Checkpoint model state is incompatible.");
        }
    }

    private static int RestoreParameters(
        JsonCursor json,
        IReadOnlyList<Parameter> parameters,
        CheckpointFloatStagingBuffer staging)
    {
        json.BeginArray();
        bool first = true;
        int index = 0;
        while (json.TryReadArrayValue(ref first))
        {
            if (index >= parameters.Count)
            {
                throw new JsonException(
                    "Checkpoint has more parameters than the model.");
            }
            RestoreParameter(json, parameters[index], index, staging);
            index++;
        }
        return index;
    }

    private static void RestoreParameter(
        JsonCursor json,
        Parameter parameter,
        int expectedIndex,
        CheckpointFloatStagingBuffer staging)
    {
        json.BeginObject();
        bool first = true;
        int index = -1;
        string? name = null;
        int[]? shape = null;
        bool restored = false;
        while (json.TryReadProperty(ref first, out string propertyName))
        {
            if (propertyName == nameof(ModuleParameterState.Index))
            {
                index = json.ReadInt32();
            }
            else if (propertyName == nameof(ModuleParameterState.Name))
            {
                name = json.ReadString();
            }
            else if (propertyName == nameof(ModuleParameterState.Shape))
            {
                shape = ReadShape(json);
            }
            else if (propertyName == nameof(ModuleParameterState.Values))
            {
                if (index != expectedIndex
                    || !string.Equals(
                        name,
                        parameter.Name,
                        StringComparison.Ordinal)
                    || shape is null
                    || !shape.SequenceEqual(parameter.T.Shape))
                {
                    throw new JsonException(
                        $"Checkpoint parameter slot {expectedIndex} is " +
                        "incompatible with the model.");
                }
                RestoreValues(json, parameter, staging);
                restored = true;
            }
            else
            {
                json.SkipValue();
            }
        }

        if (!restored)
        {
            throw new JsonException(
                $"Checkpoint parameter slot {expectedIndex} has no values.");
        }
    }

    private static int[] ReadShape(JsonCursor json)
    {
        json.BeginArray();
        bool first = true;
        var dimensions = new List<int>();
        while (json.TryReadArrayValue(ref first))
        {
            int dimension = json.ReadInt32();
            if (dimension <= 0)
                throw new JsonException("Checkpoint tensor shape is invalid.");
            dimensions.Add(dimension);
        }
        return dimensions.ToArray();
    }

    private static void RestoreValues(
        JsonCursor json,
        Parameter parameter,
        CheckpointFloatStagingBuffer staging)
    {
        json.BeginArray();
        bool first = true;
        int capacity = Math.Min(
            CheckpointFloatStagingBuffer.MaximumElementCount,
            parameter.T.Numel);
        Span<float> chunk = staging.GetManagedSpan(capacity);
        int chunkCount = 0;
        int totalCount = 0;
        using Tensor.CheckpointRestoreWriter destination =
            parameter.T.BeginCheckpointRestore();
        while (json.TryReadArrayValue(ref first))
        {
            float value = json.ReadSingle();
            if (!float.IsFinite(value)
                || (parameter.T.DType == TensorDType.Float16
                    && !Half.IsFinite((Half)value)))
            {
                throw new JsonException(
                    "Checkpoint tensor contains a non-finite or " +
                    "out-of-range value.");
            }
            if (totalCount >= parameter.T.Numel)
                throw new JsonException("Checkpoint tensor has too many values.");

            chunk[chunkCount++] = value;
            totalCount++;
            if (chunkCount == chunk.Length)
            {
                destination.WriteNext(chunk);
                chunkCount = 0;
            }
        }
        if (chunkCount > 0)
            destination.WriteNext(chunk[..chunkCount]);
        if (totalCount != parameter.T.Numel)
        {
            throw new JsonException(
                "Checkpoint tensor value count does not match its shape.");
        }
        destination.Complete();
    }

    private sealed class JsonCursor(StreamReader reader)
    {
        private const int MaximumTokenLength = 1024 * 1024;

        internal void BeginObject() => Expect('{');
        internal void BeginArray() => Expect('[');

        internal bool TryReadProperty(ref bool first, out string name)
        {
            SkipWhitespace();
            if (TryConsume('}'))
            {
                name = string.Empty;
                return false;
            }
            if (!first)
                Expect(',');
            first = false;
            name = ReadString();
            Expect(':');
            return true;
        }

        internal bool TryReadArrayValue(ref bool first)
        {
            SkipWhitespace();
            if (TryConsume(']'))
                return false;
            if (!first)
                Expect(',');
            first = false;
            return true;
        }

        internal int ReadInt32()
            => int.Parse(
                ReadLiteral(),
                NumberStyles.Integer,
                CultureInfo.InvariantCulture);

        internal float ReadSingle()
        {
            SkipWhitespace();
            string token = Peek() == '"' ? ReadString() : ReadLiteral();
            if (!float.TryParse(
                token,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out float value))
            {
                throw new JsonException("Checkpoint value is not a float.");
            }
            return value;
        }

        internal string ReadString()
        {
            SkipWhitespace();
            ExpectRaw('"');
            var value = new StringBuilder();
            while (true)
            {
                int next = reader.Read();
                if (next < 0)
                    throw UnexpectedEnd();
                char character = (char)next;
                if (character == '"')
                    return value.ToString();
                if (character == '\\')
                {
                    int escaped = reader.Read();
                    if (escaped < 0)
                        throw UnexpectedEnd();
                    character = (char)escaped switch
                    {
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        'b' => '\b',
                        'f' => '\f',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        'u' => ReadUnicodeEscape(),
                        _ => throw new JsonException(
                            "Checkpoint JSON string escape is invalid."),
                    };
                }
                else if (character < ' ')
                {
                    throw new JsonException(
                        "Checkpoint JSON string contains a control character.");
                }
                value.Append(character);
                if (value.Length > MaximumTokenLength)
                    throw new JsonException("Checkpoint JSON token is too long.");
            }
        }

        internal void SkipValue()
        {
            SkipWhitespace();
            int next = Peek();
            if (next == '"')
            {
                _ = ReadString();
                return;
            }
            if (next == '{')
            {
                BeginObject();
                bool first = true;
                while (TryReadProperty(ref first, out _))
                    SkipValue();
                return;
            }
            if (next == '[')
            {
                BeginArray();
                bool first = true;
                while (TryReadArrayValue(ref first))
                    SkipValue();
                return;
            }
            _ = ReadLiteral();
        }

        private string ReadLiteral()
        {
            SkipWhitespace();
            var token = new StringBuilder();
            while (true)
            {
                int next = reader.Peek();
                if (next < 0
                    || char.IsWhiteSpace((char)next)
                    || next is ',' or ']' or '}')
                {
                    break;
                }
                token.Append((char)reader.Read());
                if (token.Length > 256)
                    throw new JsonException("Checkpoint literal is too long.");
            }
            if (token.Length == 0)
                throw new JsonException("Checkpoint JSON literal is missing.");
            return token.ToString();
        }

        private char ReadUnicodeEscape()
        {
            Span<char> digits = stackalloc char[4];
            for (int index = 0; index < digits.Length; index++)
            {
                int next = reader.Read();
                if (next < 0)
                    throw UnexpectedEnd();
                digits[index] = (char)next;
            }
            if (!ushort.TryParse(
                digits,
                NumberStyles.AllowHexSpecifier,
                CultureInfo.InvariantCulture,
                out ushort value))
            {
                throw new JsonException(
                    "Checkpoint unicode escape is invalid.");
            }
            return (char)value;
        }

        private void Expect(char expected)
        {
            SkipWhitespace();
            ExpectRaw(expected);
        }

        private void ExpectRaw(char expected)
        {
            int actual = reader.Read();
            if (actual != expected)
            {
                throw new JsonException(
                    $"Expected JSON character '{expected}'.");
            }
        }

        private bool TryConsume(char expected)
        {
            if (Peek() != expected)
                return false;
            _ = reader.Read();
            return true;
        }

        private int Peek()
        {
            int value = reader.Peek();
            if (value < 0)
                throw UnexpectedEnd();
            return value;
        }

        private void SkipWhitespace()
        {
            while (reader.Peek() is int next
                && next >= 0
                && char.IsWhiteSpace((char)next))
            {
                _ = reader.Read();
            }
        }

        private static JsonException UnexpectedEnd()
            => new("Checkpoint JSON ended unexpectedly.");
    }
}
