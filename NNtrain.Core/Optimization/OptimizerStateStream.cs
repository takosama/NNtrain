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
    private const int LegacyBinaryFormatVersion = 1;
    private const int PackedBFloat16BinaryFormatVersion = 2;
    private const int MaximumMetadataBytes = 1024 * 1024;
    internal const int BFloat16ConversionChunkElements = 64 * 1024;

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
    };
    private static readonly OptimizerStateCodecRegistry CodecRegistry =
        CreateCodecRegistry();

    public static IReadOnlyList<IOptimizer> GetLeafOptimizers(
        IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        var leaves = new List<IOptimizer>();
        AddLeaves(optimizer, leaves);
        return leaves;
    }

    public static string GetStateType(IOptimizer optimizer)
        => optimizer is IOptimizerContainer
            ? "CompositeOptimizer"
            : ResolveCodec(optimizer).StateType;

    public static void LoadStateJson(IOptimizer optimizer, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(stream);
        if (optimizer is IOptimizerContainer)
        {
            throw new ArgumentException(
                "Load each leaf returned by GetLeafOptimizers when " +
                "restoring a composite optimizer.",
                nameof(optimizer));
        }
        ResolveCodec(optimizer).LoadJson(optimizer, stream);
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
        if (formatVersion is not (LegacyBinaryFormatVersion or PackedBFloat16BinaryFormatVersion))
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

        if (optimizer is IOptimizerContainer)
        {
            throw new ArgumentException(
                "Load each leaf returned by GetLeafOptimizers when " +
                "restoring a composite optimizer.",
                nameof(optimizer));
        }
        ResolveCodec(optimizer).LoadBinary(optimizer, reader, stream, formatVersion);

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
        if (optimizer is IOptimizerContainer)
        {
            throw new ArgumentException(
                "Save each leaf returned by GetLeafOptimizers when " +
                "serializing a composite optimizer.",
                nameof(optimizer));
        }
        ResolveCodec(optimizer).SaveJson(optimizer, stream);
    }

    /// <summary>
    /// Writes an optimizer in a compact little-endian binary format. Arc
    /// mix8_16 moments are written as physical BF16 bytes; other modes retain
    /// the version 1 FP32 representation. Writes are bounded by one chunk.
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
        bool packedBFloat16 = optimizer switch
        {
            AdamW adam => adam.UsesPackedBFloat16Checkpoint,
            NekoMuon muon => muon.UsesPackedBFloat16Checkpoint,
            _ => false,
        };
        writer.Write(packedBFloat16
            ? PackedBFloat16BinaryFormatVersion
            : LegacyBinaryFormatVersion);
        WriteString(writer, GetStateType(optimizer));

        if (optimizer is IOptimizerContainer)
        {
            throw new ArgumentException(
                "Save each leaf returned by GetLeafOptimizers when " +
                "serializing a composite optimizer.",
                nameof(optimizer));
        }
        ResolveCodec(optimizer).SaveBinary(optimizer, writer, stream);
        writer.Flush();
    }

    private static OptimizerStateCodecRegistry CreateCodecRegistry()
    {
        var registry = new OptimizerStateCodecRegistry();
        registry.Register(
            new OptimizerStateCodec<NekoMuon>(
                "NekoMuon",
                (optimizer, stream) => optimizer.RestoreStateOwned(
                    Deserialize<NekoMuonState>(stream)),
                (optimizer, reader, stream, binaryVersion) => optimizer.RestoreStateOwned(
                    ReadNekoMuonState(reader, stream, binaryVersion)),
                (optimizer, stream) => JsonSerializer.Serialize(
                    stream,
                    optimizer.CaptureStateForStreaming(),
                    JsonOptions),
                (optimizer, writer, stream) => WriteNekoMuonState(
                    writer, stream, optimizer)));
        registry.Register(
            new OptimizerStateCodec<AdamW>(
                "AdamW",
                (optimizer, stream) => optimizer.RestoreStateOwned(
                    Deserialize<AdamWState>(stream)),
                (optimizer, reader, stream, binaryVersion) => optimizer.RestoreStateOwned(
                    ReadAdamWState(reader, stream, binaryVersion)),
                (optimizer, stream) => JsonSerializer.Serialize(
                    stream,
                    optimizer.CaptureStateForStreaming(),
                    JsonOptions),
                (optimizer, writer, stream) => WriteAdamWState(
                    writer,
                    stream,
                    optimizer)));
        registry.Register(
            new OptimizerStateCodec<Lion>(
                "Lion",
                (optimizer, stream) => optimizer.RestoreStateOwned(
                    Deserialize<LionState>(stream)),
                (optimizer, reader, stream, binaryVersion) => optimizer.RestoreStateOwned(
                    ReadLionState(reader, stream)),
                (optimizer, stream) => JsonSerializer.Serialize(
                    stream,
                    optimizer.CaptureStateForStreaming(),
                    JsonOptions),
                (optimizer, writer, stream) => WriteLionState(
                    writer,
                    stream,
                    optimizer.CaptureStateForStreaming())));
        registry.Register(
            new OptimizerStateCodec<GainShareAdamW>(
                "GainShareAdamW",
                (optimizer, stream) => optimizer.RestoreStateOwned(
                    Deserialize<GainShareAdamWState>(stream)),
                (optimizer, reader, stream, binaryVersion) => optimizer.RestoreStateOwned(
                    ReadGainShareAdamWState(reader, stream)),
                (optimizer, stream) => JsonSerializer.Serialize(
                    stream,
                    optimizer.CaptureStateForStreaming(),
                    JsonOptions),
                (optimizer, writer, stream) => WriteGainShareAdamWState(
                    writer,
                    stream,
                    optimizer.CaptureStateForStreaming())));
        return registry;
    }

    private static IOptimizerStateCodec ResolveCodec(IOptimizer optimizer)
    {
        if (CodecRegistry.TryResolve(optimizer, out IOptimizerStateCodec? codec))
            return codec!;
        throw new NotSupportedException(
            $"Optimizer '{optimizer.GetType().Name}' does not support " +
            "streaming checkpoint state.");
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
            if (optimizer.UsesPackedBFloat16Checkpoint)
            {
                (ushort[] first, ushort[] second) = optimizer.GetStreamingPackedMoments(index);
                WritePackedBFloat16Array(writer, stream, first);
                WritePackedBFloat16Array(writer, stream, second);
            }
            else
            {
                WriteAdamWMoment(writer, stream, parameter.FirstMoment,
                    parameter.FirstMomentBFloat16);
                WriteAdamWMoment(writer, stream, parameter.SecondMoment,
                    parameter.SecondMomentBFloat16);
            }
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
        Stream stream,
        int binaryVersion)
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
                ReadOptimizerMoment(reader, stream, binaryVersion),
                ReadOptimizerMoment(reader, stream, binaryVersion));
        }
        return new AdamWState(formatVersion, step, options, states);
    }

    private static void WriteNekoMuonState(
        BinaryWriter writer,
        Stream stream,
        NekoMuon optimizer)
    {
        NekoMuonState state = optimizer.UsesPackedBFloat16Checkpoint
            ? optimizer.GetPackedStreamingState()
            : optimizer.CaptureStateForStreaming();
        WriteStateHeader(
            writer,
            state.FormatVersion,
            state.Step,
            state.Options,
            state.ParameterStates.Length);
        if (state.FormatVersion >= 2)
        {
            writer.Write(state.FastDecayProduct
                ?? throw new InvalidDataException(
                    "NekoMuon fast decay product is missing."));
            writer.Write(state.SlowDecayProduct
                ?? throw new InvalidDataException(
                    "NekoMuon slow decay product is missing."));
        }
        for (int index = 0; index < state.ParameterStates.Length; index++)
        {
            NekoMuonParameterState parameter = state.ParameterStates[index];
            WriteParameterMetadata(
                writer,
                parameter.Index,
                parameter.Name,
                parameter.Shape);
            if (optimizer.UsesPackedBFloat16Checkpoint)
            {
                (ushort[] fast, ushort[] slow) = optimizer.GetStreamingPackedMoments(index);
                WritePackedBFloat16Array(writer, stream, fast);
                WritePackedBFloat16Array(writer, stream, slow);
            }
            else
            {
                WriteFloatArray(writer, stream, parameter.FastMoment);
                WriteFloatArray(writer, stream, parameter.SlowMoment);
            }
            writer.Write(parameter.Confidence);
        }
    }

    private static NekoMuonState ReadNekoMuonState(
        BinaryReader reader,
        Stream stream,
        int binaryVersion)
    {
        (int formatVersion, int step, NekoMuonOptions options, int count) =
            ReadStateHeader<NekoMuonOptions>(reader, stream);
        double? fastDecayProduct = formatVersion >= 2
            ? reader.ReadDouble()
            : null;
        double? slowDecayProduct = formatVersion >= 2
            ? reader.ReadDouble()
            : null;
        var states = new NekoMuonParameterState[count];
        for (int index = 0; index < states.Length; index++)
        {
            (int slot, string name, int[] shape) =
                ReadParameterMetadata(reader, stream);
            states[index] = new NekoMuonParameterState(
                slot,
                name,
                shape,
                ReadOptimizerMoment(reader, stream, binaryVersion),
                ReadOptimizerMoment(reader, stream, binaryVersion),
                reader.ReadSingle());
        }
        return new NekoMuonState(formatVersion, step, options, states)
        {
            FastDecayProduct = fastDecayProduct,
            SlowDecayProduct = slowDecayProduct,
        };
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

    private static void WritePackedBFloat16Array(
        BinaryWriter writer,
        Stream stream,
        float[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        // Negative length marks a physical BF16 moment. Version 1 lengths
        // are always non-negative and continue to decode as raw FP32.
        writer.Write(~values.Length);
        writer.Flush();
        ushort[] chunk = ArrayPool<ushort>.Shared.Rent(
            Math.Min(BFloat16ConversionChunkElements, Math.Max(1, values.Length)));
        try
        {
            for (int offset = 0; offset < values.Length;)
            {
                int count = Math.Min(chunk.Length, values.Length - offset);
                for (int i = 0; i < count; i++)
                    chunk[i] = TensorStorageCodec.EncodeBFloat16(values[offset + i]);
                stream.Write(MemoryMarshal.AsBytes(chunk.AsSpan(0, count)));
                offset += count;
            }
        }
        finally { ArrayPool<ushort>.Shared.Return(chunk); }
    }

    private static void WritePackedBFloat16Array(
        BinaryWriter writer,
        Stream stream,
        ushort[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        writer.Write(~values.Length);
        writer.Flush();
        for (int offset = 0; offset < values.Length;)
        {
            int count = Math.Min(BFloat16ConversionChunkElements,
                values.Length - offset);
            stream.Write(MemoryMarshal.AsBytes(values.AsSpan(offset, count)));
            offset += count;
        }
    }

    private static float[] ReadOptimizerMoment(
        BinaryReader reader,
        Stream stream,
        int binaryVersion)
    {
        if (binaryVersion == LegacyBinaryFormatVersion)
            return ReadFloatArray(reader, stream);
        int marker = reader.ReadInt32();
        if (marker >= 0)
            throw new InvalidDataException("Version 2 optimizer moments must use physical BF16 storage.");
        int count = ~marker;
        if (stream.CanSeek && (long)count * sizeof(ushort) > stream.Length - stream.Position)
            throw new InvalidDataException("BF16 optimizer moment exceeds the remaining payload.");
        var values = new float[count];
        ushort[] chunk = ArrayPool<ushort>.Shared.Rent(
            Math.Min(BFloat16ConversionChunkElements, Math.Max(1, count)));
        try
        {
            for (int offset = 0; offset < count;)
            {
                int length = Math.Min(chunk.Length, count - offset);
                stream.ReadExactly(MemoryMarshal.AsBytes(chunk.AsSpan(0, length)));
                for (int i = 0; i < length; i++)
                    values[offset + i] = TensorStorageCodec.DecodeBFloat16(chunk[i]);
                offset += length;
            }
        }
        finally { ArrayPool<ushort>.Shared.Return(chunk); }
        return values;
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
        if (optimizer is IOptimizerContainer container)
        {
            foreach (IOptimizer child in container.Optimizers)
                AddLeaves(child, leaves);
            return;
        }
        _ = GetStateType(optimizer);
        leaves.Add(optimizer);
    }
}
