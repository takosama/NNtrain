using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;

namespace NNtrain;

/// <summary>
/// Streams typed optimizer state without building an intermediate
/// <see cref="JsonElement"/> DOM. This is used by large checkpoints whose raw
/// optimizer JSON can exceed the practical size of one contiguous buffer.
/// </summary>
public static class OptimizerStateStream
{
    private static readonly byte[] BinaryMagic =
        "NNOPT\0\r\n"u8.ToArray();
    private const int BinaryFormatVersion = 1;
    private const int MaximumMetadataBytes = 1024 * 1024;
    internal const int BFloat16ConversionChunkElements = 64 * 1024;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };

    public static IReadOnlyList<IOptimizer> GetLeafOptimizers(
        IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        var leaves = new List<IOptimizer>();
        AddLeaves(optimizer, leaves);
        return leaves;
    }

    public static string GetStateType(IOptimizer optimizer)
        => optimizer switch
        {
            NekoMuon => "NekoMuon",
            AdamW => "AdamW",
            Lion => "Lion",
            GainShareAdamW => "GainShareAdamW",
            CompositeOptimizer => "CompositeOptimizer",
            _ => throw new NotSupportedException(
                $"Optimizer '{optimizer.GetType().Name}' does not support " +
                "streaming checkpoint state."),
        };

    public static void LoadStateJson(IOptimizer optimizer, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(stream);
        switch (optimizer)
        {
            case NekoMuon nekoMuon:
                nekoMuon.RestoreStateOwned(
                    Deserialize<NekoMuonState>(stream));
                break;
            case AdamW adamW:
                adamW.RestoreStateOwned(
                    Deserialize<AdamWState>(stream));
                break;
            case Lion lion:
                lion.RestoreStateOwned(
                    Deserialize<LionState>(stream));
                break;
            case GainShareAdamW gainShare:
                gainShare.RestoreStateOwned(
                    Deserialize<GainShareAdamWState>(stream));
                break;
            case CompositeOptimizer:
                throw new ArgumentException(
                    "Load each leaf returned by GetLeafOptimizers when " +
                    "restoring a composite optimizer.",
                    nameof(optimizer));
            default:
                throw new NotSupportedException(
                    $"Optimizer '{optimizer.GetType().Name}' does not " +
                    "support streaming checkpoint state.");
        }
    }

    /// <summary>
    /// Restores an optimizer from the compact binary checkpoint format. Float
    /// arrays are read directly into their final owned buffers; no JSON DOM,
    /// base64 string, or aggregate payload buffer is materialized.
    /// </summary>
    public static void LoadStateBinary(IOptimizer optimizer, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanRead)
            throw new ArgumentException("The stream must be readable.", nameof(stream));

        Span<byte> magic = stackalloc byte[BinaryMagic.Length];
        stream.ReadExactly(magic);
        if (!magic.SequenceEqual(BinaryMagic))
            throw new InvalidDataException("Optimizer state binary header is invalid.");

        using var reader = new BinaryReader(
            stream,
            Encoding.UTF8,
            leaveOpen: true);
        int formatVersion = reader.ReadInt32();
        if (formatVersion != BinaryFormatVersion)
        {
            throw new InvalidDataException(
                $"Unsupported optimizer binary format version " +
                $"'{formatVersion}'.");
        }

        string serializedType = ReadString(reader, stream);
        string expectedType = GetStateType(optimizer);
        if (!string.Equals(serializedType, expectedType, StringComparison.Ordinal))
        {
            throw new InvalidDataException(
                $"Optimizer binary contains '{serializedType}', but " +
                $"'{expectedType}' was expected.");
        }

        switch (optimizer)
        {
            case NekoMuon nekoMuon:
                nekoMuon.RestoreStateOwned(ReadNekoMuonState(reader, stream));
                break;
            case AdamW adamW:
                adamW.RestoreStateOwned(ReadAdamWState(reader, stream));
                break;
            case Lion lion:
                lion.RestoreStateOwned(ReadLionState(reader, stream));
                break;
            case GainShareAdamW gainShare:
                gainShare.RestoreStateOwned(
                    ReadGainShareAdamWState(reader, stream));
                break;
            case CompositeOptimizer:
                throw new ArgumentException(
                    "Load each leaf returned by GetLeafOptimizers when " +
                    "restoring a composite optimizer.",
                    nameof(optimizer));
            default:
                throw new NotSupportedException(
                    $"Optimizer '{optimizer.GetType().Name}' does not " +
                    "support streaming checkpoint state.");
        }

        if (stream.CanSeek && stream.Position != stream.Length)
        {
            throw new InvalidDataException(
                "Optimizer binary contains trailing data.");
        }
    }

    public static void SaveStateJson(IOptimizer optimizer, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(stream);
        switch (optimizer)
        {
            case NekoMuon nekoMuon:
                JsonSerializer.Serialize(
                    stream,
                    nekoMuon.CaptureStateForStreaming(),
                    JsonOptions);
                break;
            case AdamW adamW:
                JsonSerializer.Serialize(
                    stream,
                    adamW.CaptureStateForStreaming(),
                    JsonOptions);
                break;
            case Lion lion:
                JsonSerializer.Serialize(
                    stream,
                    lion.CaptureStateForStreaming(),
                    JsonOptions);
                break;
            case GainShareAdamW gainShare:
                JsonSerializer.Serialize(
                    stream,
                    gainShare.CaptureStateForStreaming(),
                    JsonOptions);
                break;
            case CompositeOptimizer:
                throw new ArgumentException(
                    "Save each leaf returned by GetLeafOptimizers when " +
                    "serializing a composite optimizer.",
                    nameof(optimizer));
            default:
                throw new NotSupportedException(
                    $"Optimizer '{optimizer.GetType().Name}' does not " +
                    "support streaming checkpoint state.");
        }
    }

    /// <summary>
    /// Writes an optimizer in a compact little-endian binary format. Existing
    /// moment arrays are copied straight to the output stream as raw IEEE-754
    /// bytes, keeping peak memory independent of total optimizer-state size.
    /// </summary>
    public static void SaveStateBinary(IOptimizer optimizer, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(stream);
        if (!stream.CanWrite)
            throw new ArgumentException("The stream must be writable.", nameof(stream));

        stream.Write(BinaryMagic);
        using var writer = new BinaryWriter(
            stream,
            Encoding.UTF8,
            leaveOpen: true);
        writer.Write(BinaryFormatVersion);
        WriteString(writer, GetStateType(optimizer));

        switch (optimizer)
        {
            case NekoMuon nekoMuon:
                WriteNekoMuonState(
                    writer,
                    stream,
                    nekoMuon.CaptureStateForStreaming());
                break;
            case AdamW adamW:
                WriteAdamWState(writer, stream, adamW);
                break;
            case Lion lion:
                WriteLionState(
                    writer,
                    stream,
                    lion.CaptureStateForStreaming());
                break;
            case GainShareAdamW gainShare:
                WriteGainShareAdamWState(
                    writer,
                    stream,
                    gainShare.CaptureStateForStreaming());
                break;
            case CompositeOptimizer:
                throw new ArgumentException(
                    "Save each leaf returned by GetLeafOptimizers when " +
                    "serializing a composite optimizer.",
                    nameof(optimizer));
            default:
                throw new NotSupportedException(
                    $"Optimizer '{optimizer.GetType().Name}' does not " +
                    "support streaming checkpoint state.");
        }
        writer.Flush();
    }

    private static void WriteAdamWState(
        BinaryWriter writer,
        Stream stream,
        AdamW optimizer)
    {
        optimizer.SynchronizeStateForStreaming();
        WriteStateHeader(
            writer,
            AdamWState.CurrentFormatVersion,
            optimizer.StreamingStep,
            optimizer.StreamingOptions,
            optimizer.StreamingParameterCount);
        for (int index = 0; index < optimizer.StreamingParameterCount; index++)
        {
            AdamWStreamingParameterState parameter =
                optimizer.GetStreamingParameterState(index);
            WriteParameterMetadata(
                writer,
                parameter.Index,
                parameter.Name,
                parameter.Shape);
            WriteAdamWMoment(
                writer,
                stream,
                parameter.FirstMoment,
                parameter.FirstMomentBFloat16);
            WriteAdamWMoment(
                writer,
                stream,
                parameter.SecondMoment,
                parameter.SecondMomentBFloat16);
        }
    }

    private static void WriteAdamWMoment(
        BinaryWriter writer,
        Stream stream,
        float[] values,
        short[]? bfloat16Values)
    {
        if (bfloat16Values is null)
        {
            WriteFloatArray(writer, stream, values);
            return;
        }

        writer.Write(bfloat16Values.Length);
        writer.Flush();
        float[] conversion = ArrayPool<float>.Shared.Rent(
            Math.Min(
                BFloat16ConversionChunkElements,
                Math.Max(1, bfloat16Values.Length)));
        try
        {
            int offset = 0;
            while (offset < bfloat16Values.Length)
            {
                int count = Math.Min(
                    BFloat16ConversionChunkElements,
                    bfloat16Values.Length - offset);
                for (int index = 0; index < count; index++)
                {
                    uint bits = (uint)(ushort)bfloat16Values[offset + index]
                        << 16;
                    conversion[index] = BitConverter.UInt32BitsToSingle(bits);
                }
                stream.Write(MemoryMarshal.AsBytes(conversion.AsSpan(0, count)));
                offset += count;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(conversion);
        }
    }

    private static AdamWState ReadAdamWState(
        BinaryReader reader,
        Stream stream)
    {
        (int formatVersion, int step, AdamWOptions options, int count) =
            ReadStateHeader<AdamWOptions>(reader, stream);
        var states = new AdamWParameterState[count];
        for (int index = 0; index < states.Length; index++)
        {
            (int slot, string name, int[] shape) =
                ReadParameterMetadata(reader, stream);
            states[index] = new AdamWParameterState(
                slot,
                name,
                shape,
                ReadFloatArray(reader, stream),
                ReadFloatArray(reader, stream));
        }
        return new AdamWState(formatVersion, step, options, states);
    }

    private static void WriteNekoMuonState(
        BinaryWriter writer,
        Stream stream,
        NekoMuonState state)
    {
        WriteStateHeader(
            writer,
            state.FormatVersion,
            state.Step,
            state.Options,
            state.ParameterStates.Length);
        foreach (NekoMuonParameterState parameter in state.ParameterStates)
        {
            WriteParameterMetadata(
                writer,
                parameter.Index,
                parameter.Name,
                parameter.Shape);
            WriteFloatArray(writer, stream, parameter.FastMoment);
            WriteFloatArray(writer, stream, parameter.SlowMoment);
            writer.Write(parameter.Confidence);
        }
    }

    private static NekoMuonState ReadNekoMuonState(
        BinaryReader reader,
        Stream stream)
    {
        (int formatVersion, int step, NekoMuonOptions options, int count) =
            ReadStateHeader<NekoMuonOptions>(reader, stream);
        var states = new NekoMuonParameterState[count];
        for (int index = 0; index < states.Length; index++)
        {
            (int slot, string name, int[] shape) =
                ReadParameterMetadata(reader, stream);
            states[index] = new NekoMuonParameterState(
                slot,
                name,
                shape,
                ReadFloatArray(reader, stream),
                ReadFloatArray(reader, stream),
                reader.ReadSingle());
        }
        return new NekoMuonState(formatVersion, step, options, states);
    }

    private static void WriteLionState(
        BinaryWriter writer,
        Stream stream,
        LionState state)
    {
        WriteStateHeader(
            writer,
            state.FormatVersion,
            state.Step,
            state.Options,
            state.ParameterStates.Length);
        foreach (LionParameterState parameter in state.ParameterStates)
        {
            WriteParameterMetadata(
                writer,
                parameter.Index,
                parameter.Name,
                parameter.Shape);
            WriteFloatArray(writer, stream, parameter.Momentum);
        }
    }

    private static LionState ReadLionState(
        BinaryReader reader,
        Stream stream)
    {
        (int formatVersion, int step, LionOptions options, int count) =
            ReadStateHeader<LionOptions>(reader, stream);
        var states = new LionParameterState[count];
        for (int index = 0; index < states.Length; index++)
        {
            (int slot, string name, int[] shape) =
                ReadParameterMetadata(reader, stream);
            states[index] = new LionParameterState(
                slot,
                name,
                shape,
                ReadFloatArray(reader, stream));
        }
        return new LionState(formatVersion, step, options, states);
    }

    private static void WriteGainShareAdamWState(
        BinaryWriter writer,
        Stream stream,
        GainShareAdamWState state)
    {
        WriteStateHeader(
            writer,
            state.FormatVersion,
            state.Step,
            state.Options,
            state.ParameterStates.Length);
        foreach (GainShareAdamWParameterState parameter in state.ParameterStates)
        {
            WriteParameterMetadata(
                writer,
                parameter.Index,
                parameter.Name,
                parameter.Shape);
            WriteFloatArray(writer, stream, parameter.FirstMoment);
            WriteFloatArray(writer, stream, parameter.SecondMoment);
        }

        writer.Write(state.GroupStates.Length);
        foreach (GainShareAdamWGroupState group in state.GroupStates)
        {
            writer.Write(group.Index);
            WriteIntArray(writer, group.ParameterIndices);
            writer.Write(group.AlignmentEma.HasValue);
            if (group.AlignmentEma.HasValue)
                writer.Write(group.AlignmentEma.Value);
        }
    }

    private static GainShareAdamWState ReadGainShareAdamWState(
        BinaryReader reader,
        Stream stream)
    {
        (int formatVersion, int step, GainShareAdamWOptions options, int count) =
            ReadStateHeader<GainShareAdamWOptions>(reader, stream);
        var parameterStates = new GainShareAdamWParameterState[count];
        for (int index = 0; index < parameterStates.Length; index++)
        {
            (int slot, string name, int[] shape) =
                ReadParameterMetadata(reader, stream);
            parameterStates[index] = new GainShareAdamWParameterState(
                slot,
                name,
                shape,
                ReadFloatArray(reader, stream),
                ReadFloatArray(reader, stream));
        }

        int groupCount = ReadCount(reader, stream, sizeof(int) + 1);
        var groupStates = new GainShareAdamWGroupState[groupCount];
        for (int index = 0; index < groupStates.Length; index++)
        {
            int slot = reader.ReadInt32();
            int[] parameterIndices = ReadIntArray(reader, stream);
            bool hasAlignment = reader.ReadBoolean();
            groupStates[index] = new GainShareAdamWGroupState(
                slot,
                parameterIndices,
                hasAlignment ? reader.ReadDouble() : null);
        }
        return new GainShareAdamWState(
            formatVersion,
            step,
            options,
            parameterStates,
            groupStates);
    }

    private static void WriteStateHeader<TOptions>(
        BinaryWriter writer,
        int formatVersion,
        int step,
        TOptions options,
        int parameterCount)
    {
        writer.Write(formatVersion);
        writer.Write(step);
        byte[] metadata = JsonSerializer.SerializeToUtf8Bytes(
            options,
            JsonOptions);
        writer.Write(metadata.Length);
        writer.Write(metadata);
        writer.Write(parameterCount);
    }

    private static (
        int FormatVersion,
        int Step,
        TOptions Options,
        int ParameterCount) ReadStateHeader<TOptions>(
            BinaryReader reader,
            Stream stream)
    {
        int formatVersion = reader.ReadInt32();
        int step = reader.ReadInt32();
        int metadataLength = ReadBoundedLength(
            reader,
            stream,
            MaximumMetadataBytes);
        byte[] metadata = reader.ReadBytes(metadataLength);
        if (metadata.Length != metadataLength)
            throw new EndOfStreamException();
        TOptions options = JsonSerializer.Deserialize<TOptions>(
            metadata,
            JsonOptions)
            ?? throw new InvalidDataException(
                $"Optimizer options '{typeof(TOptions).Name}' were null.");
        int parameterCount = ReadCount(reader, stream, sizeof(int) * 4);
        return (formatVersion, step, options, parameterCount);
    }

    private static void WriteParameterMetadata(
        BinaryWriter writer,
        int index,
        string name,
        int[] shape)
    {
        writer.Write(index);
        WriteString(writer, name);
        WriteIntArray(writer, shape);
    }

    private static (int Index, string Name, int[] Shape)
        ReadParameterMetadata(BinaryReader reader, Stream stream)
        => (
            reader.ReadInt32(),
            ReadString(reader, stream),
            ReadIntArray(reader, stream));

    private static void WriteFloatArray(
        BinaryWriter writer,
        Stream stream,
        float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        writer.Write(values.Length);
        writer.Flush();
        stream.Write(MemoryMarshal.AsBytes(values.AsSpan()));
    }

    private static float[] ReadFloatArray(
        BinaryReader reader,
        Stream stream)
    {
        int count = ReadCount(reader, stream, sizeof(float));
        var values = new float[count];
        stream.ReadExactly(MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    private static void WriteIntArray(BinaryWriter writer, int[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        writer.Write(values.Length);
        foreach (int value in values)
            writer.Write(value);
    }

    private static int[] ReadIntArray(BinaryReader reader, Stream stream)
    {
        int count = ReadCount(reader, stream, sizeof(int));
        var values = new int[count];
        for (int index = 0; index < values.Length; index++)
            values[index] = reader.ReadInt32();
        return values;
    }

    private static void WriteString(BinaryWriter writer, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write(bytes.Length);
        writer.Write(bytes);
    }

    private static string ReadString(BinaryReader reader, Stream stream)
    {
        int length = ReadBoundedLength(
            reader,
            stream,
            MaximumMetadataBytes);
        byte[] bytes = reader.ReadBytes(length);
        if (bytes.Length != length)
            throw new EndOfStreamException();
        return Encoding.UTF8.GetString(bytes);
    }

    private static int ReadCount(
        BinaryReader reader,
        Stream stream,
        int minimumBytesPerItem)
    {
        int count = reader.ReadInt32();
        if (count < 0)
            throw new InvalidDataException("Optimizer binary count is negative.");
        if (stream.CanSeek
            && (long)count * minimumBytesPerItem
                > stream.Length - stream.Position)
        {
            throw new InvalidDataException(
                "Optimizer binary count exceeds the remaining payload.");
        }
        return count;
    }

    private static int ReadBoundedLength(
        BinaryReader reader,
        Stream stream,
        int maximum)
    {
        int length = reader.ReadInt32();
        if (length < 0 || length > maximum)
            throw new InvalidDataException("Optimizer binary length is invalid.");
        if (stream.CanSeek && length > stream.Length - stream.Position)
        {
            throw new InvalidDataException(
                "Optimizer binary length exceeds the remaining payload.");
        }
        return length;
    }

    private static T Deserialize<T>(Stream stream)
        => JsonSerializer.Deserialize<T>(stream, JsonOptions)
            ?? throw new InvalidDataException(
                $"Optimizer state '{typeof(T).Name}' was JSON null.");

    private static void AddLeaves(
        IOptimizer optimizer,
        List<IOptimizer> leaves)
    {
        if (optimizer is CompositeOptimizer composite)
        {
            foreach (IOptimizer child in composite.Optimizers)
                AddLeaves(child, leaves);
            return;
        }
        _ = GetStateType(optimizer);
        leaves.Add(optimizer);
    }
}
