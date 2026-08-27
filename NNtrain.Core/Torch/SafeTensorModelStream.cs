using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace NNtrain;

/// <summary>
/// Reusable host staging for checkpoint parameter chunks. CUDA transfers use
/// cudaMallocHost memory; CPU-only saves use one pinned managed array. The
/// active allocation never exceeds <see cref="MaximumByteLength"/>.
/// </summary>
internal sealed unsafe class CheckpointFloatStagingBuffer : IDisposable
{
    internal const int MaximumByteLength = 16 * 1024 * 1024;
    internal const int MaximumElementCount =
        MaximumByteLength / sizeof(float);

    private float[]? _managed;
    private nint _cudaPinned;
    private int _disposed;

    internal nint Pointer
        => _cudaPinned != 0
            ? _cudaPinned
            : throw new InvalidOperationException(
                "CUDA-pinned checkpoint staging is not active.");

    internal Span<float> GetManagedSpan(int length)
    {
        ValidateLength(length);
        ThrowIfDisposed();
        if (_cudaPinned != 0)
        {
            _ = NativeCudaRuntime.HostFreeNative(_cudaPinned);
            _cudaPinned = 0;
        }
        _managed ??= GC.AllocateUninitializedArray<float>(
            MaximumElementCount,
            pinned: true);
        return _managed.AsSpan(0, length);
    }

    internal Span<float> GetCudaPinnedSpan(int length)
    {
        ValidateLength(length);
        ThrowIfDisposed();
        _managed = null;
        if (_cudaPinned == 0)
        {
            NativeCudaRuntime.Check(
                NativeCudaRuntime.HostAllocateNative(
                    MaximumByteLength,
                    out _cudaPinned),
                "cudaMallocHost(checkpoint staging)");
        }
        return new Span<float>((void*)_cudaPinned, length);
    }

    internal Span<ushort> GetEncodedPrefix(int length)
    {
        ValidateLength(length);
        return new Span<ushort>((void*)Pointer, length);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        nint pointer = _cudaPinned;
        _cudaPinned = 0;
        _managed = null;
        if (pointer != 0)
            _ = NativeCudaRuntime.HostFreeNative(pointer);
    }

    private static void ValidateLength(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (length > MaximumElementCount)
        {
            throw new ArgumentOutOfRangeException(
                nameof(length),
                length,
                $"Checkpoint staging is limited to {MaximumByteLength:N0} bytes.");
        }
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}

internal static partial class SafeTensorFile
{
    private sealed record SafeTensorDescriptor(
        string Key,
        int[] Shape,
        SafeTensorDTypeCodec Codec,
        long Start,
        long End)
    {
        internal TensorDType DType => Codec.DType;
    }

    /// <summary>
    /// Saves a model without first constructing a ModuleState or cloning all
    /// parameter values onto the host.
    /// </summary>
    internal static void SaveModel(
        Module model,
        string path,
        Action<int>? stagingChunkObserved = null,
        TensorDType? artifactDTypeOverride = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (artifactDTypeOverride is { } artifactDType)
            TensorDTypeContract.ValidateImplemented(
                artifactDType,
                nameof(artifactDTypeOverride));

        Parameter[] parameters = model.Parameters().ToArray();
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string temporaryPath = fullPath + ".tmp";

        try
        {
            byte[] header = CreateModelHeader(
                parameters,
                artifactDTypeOverride);
            using var staging = new CheckpointFloatStagingBuffer();
            using (var stream = new FileStream(
                temporaryPath,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 1024 * 1024,
                FileOptions.SequentialScan))
            {
                Span<byte> prefix = stackalloc byte[sizeof(long)];
                BinaryPrimitives.WriteUInt64LittleEndian(
                    prefix,
                    checked((ulong)header.Length));
                stream.Write(prefix);
                stream.Write(header);

                foreach (Parameter parameter in parameters)
                {
                    Tensor tensor = parameter.T;
                    int maximumChunkElements = tensor.DType == TensorDType.Bfp8
                        ? CheckpointFloatStagingBuffer.MaximumElementCount / 2
                        : CheckpointFloatStagingBuffer.MaximumElementCount;
                    for (int offset = 0; offset < tensor.Numel;)
                    {
                        int count = Math.Min(
                            maximumChunkElements,
                            tensor.Numel - offset);
                        ReadOnlySpan<float> values =
                            tensor.CopyCheckpointRangeTo(
                                offset,
                                count,
                                staging,
                                preferMaster: true);
                        stagingChunkObserved?.Invoke(
                            checked(count * sizeof(float)));
                        WriteStreamedValues(
                            stream,
                            values,
                            artifactDTypeOverride ?? tensor.DType);
                        offset += count;
                    }
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

    /// <summary>
    /// Restores a SafeTensors payload directly into an existing model. Header
    /// metadata is retained, but tensor payloads are decoded through one
    /// bounded staging block instead of a model-sized ModuleState graph.
    /// </summary>
    internal static void LoadModel(
        string path,
        Module model,
        IReadOnlyList<string>? keys = null,
        Action<int>? stagingChunkObserved = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(model);

        string fullPath = Path.GetFullPath(path);
        using var stream = new FileStream(
            fullPath,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 1024 * 1024,
            FileOptions.RandomAccess);
        (long dataStart, List<SafeTensorDescriptor> descriptors) =
            ReadDescriptors(stream);
        Parameter[] parameters = model.Parameters().ToArray();
        SafeTensorDescriptor[] mapped = MapDescriptors(
            fullPath,
            descriptors,
            parameters,
            keys);

        using var staging = new CheckpointFloatStagingBuffer();
        for (int parameterIndex = 0;
            parameterIndex < parameters.Length;
            parameterIndex++)
        {
            Parameter parameter = parameters[parameterIndex];
            SafeTensorDescriptor descriptor = mapped[parameterIndex];
            if (parameter.T.RequiresTwoPassBfp8CheckpointRestore)
            {
                RestorePureBfp8Parameter(
                    stream,
                    checked(dataStart + descriptor.Start),
                    descriptor,
                    parameter,
                    staging,
                    stagingChunkObserved);
                continue;
            }
            stream.Position = checked(dataStart + descriptor.Start);
            using Tensor.CheckpointRestoreWriter destination =
                parameter.T.BeginCheckpointRestore();
            int remaining = parameter.T.Numel;
            while (remaining > 0)
            {
                int count = Math.Min(
                    CheckpointFloatStagingBuffer.MaximumElementCount,
                    remaining);
                Span<float> values = staging.GetManagedSpan(count);
                ReadStreamedValues(
                    stream,
                    values,
                    descriptor.Codec);
                ValidateRestoredValues(
                    values,
                    descriptor.Key,
                    parameter.T.DType);
                stagingChunkObserved?.Invoke(
                    checked(count * sizeof(float)));
                destination.WriteNext(values);
                remaining -= count;
            }
            destination.Complete();
        }
    }

    private static void RestorePureBfp8Parameter(
        FileStream stream,
        long payloadStart,
        SafeTensorDescriptor descriptor,
        Parameter parameter,
        CheckpointFloatStagingBuffer staging,
        Action<int>? stagingChunkObserved)
    {
        using Tensor.Bfp8CheckpointRestoreWriter destination =
            parameter.T.BeginBfp8CheckpointRestore();
        stream.Position = payloadStart;
        int remaining = parameter.T.Numel;
        while (remaining > 0)
        {
            int count = Math.Min(
                CheckpointFloatStagingBuffer.MaximumElementCount,
                remaining);
            Span<float> values = staging.GetManagedSpan(count);
            ReadStreamedValues(stream, values, descriptor.Codec);
            ValidateRestoredValues(values, descriptor.Key, TensorDType.Bfp8);
            stagingChunkObserved?.Invoke(checked(count * sizeof(float)));
            destination.AccumulateScale(values);
            remaining -= count;
        }

        destination.PrepareEncoding();
        stream.Position = payloadStart;
        remaining = parameter.T.Numel;
        while (remaining > 0)
        {
            int count = Math.Min(
                CheckpointFloatStagingBuffer.MaximumElementCount,
                remaining);
            Span<float> values = staging.GetManagedSpan(count);
            ReadStreamedValues(stream, values, descriptor.Codec);
            stagingChunkObserved?.Invoke(checked(count * sizeof(float)));
            destination.WriteNext(values);
            remaining -= count;
        }
        destination.Complete();
    }

    private static (
        long DataStart,
        List<SafeTensorDescriptor> Descriptors)
        ReadDescriptors(FileStream stream)
    {
        Span<byte> prefix = stackalloc byte[LengthPrefixSize];
        ReadExactly(stream, prefix);
        ulong encodedHeaderLength =
            BinaryPrimitives.ReadUInt64LittleEndian(prefix);
        if (encodedHeaderLength == 0
            || encodedHeaderLength > MaximumHeaderBytes
            || encodedHeaderLength
                > (ulong)(stream.Length - LengthPrefixSize))
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
        var descriptors = new List<SafeTensorDescriptor>();
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
            SafeTensorDTypeCodec codec;
            try
            {
                codec = SafeTensorDTypeCodecs.Parse(dtype);
            }
            catch (NotSupportedException exception)
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{property.Name}' uses " +
                    $"unsupported dtype '{dtype}'. Only F32, F16, and BF16 " +
                    "are supported.",
                    exception);
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
            if (end - start
                != checked((long)elementCount * codec.EncodedElementSize))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{property.Name}' shape and " +
                    "byte range do not match.");
            }
            descriptors.Add(
                new SafeTensorDescriptor(
                    property.Name,
                    shape,
                    codec,
                    start,
                    end));
        }
        return (dataStart, descriptors);
    }

    private static SafeTensorDescriptor[] MapDescriptors(
        string path,
        IReadOnlyList<SafeTensorDescriptor> descriptors,
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<string>? keys)
    {
        if (keys is not null && keys.Count != parameters.Count)
        {
            throw new ArgumentException(
                $"The key list has {keys.Count} entries but the model has " +
                $"{parameters.Count} parameters.",
                nameof(keys));
        }
        if (keys is null && descriptors.Count != parameters.Count)
        {
            throw new InvalidDataException(
                $"SafeTensors file '{path}' holds {descriptors.Count} " +
                $"tensors but the model has {parameters.Count} parameters. " +
                $"Keys in the file: " +
                $"{string.Join(", ", descriptors.Select(item => item.Key))}.");
        }

        if (keys is null
            && descriptors.All(item =>
                TrySplitIndexedKey(item.Key, out _, out _)))
        {
            SafeTensorDescriptor[] ordered = descriptors
                .OrderBy(item =>
                {
                    _ = TrySplitIndexedKey(item.Key, out int index, out _);
                    return index;
                })
                .ToArray();
            if (ordered.Length != parameters.Count)
                throw new InvalidDataException("Parameter count mismatch.");
            for (int index = 0; index < ordered.Length; index++)
            {
                _ = TrySplitIndexedKey(
                    ordered[index].Key,
                    out int encodedIndex,
                    out string encodedName);
                if (encodedIndex != index
                    || !string.Equals(
                        encodedName,
                        parameters[index].Name,
                        StringComparison.Ordinal))
                {
                    throw new InvalidDataException(
                        $"SafeTensors parameter '{ordered[index].Key}' is " +
                        $"incompatible with model slot {index} " +
                        $"('{parameters[index].Name}').");
                }
                ValidateMappedShape(ordered[index], parameters[index], index);
            }
            return ordered;
        }

        var mapped = new SafeTensorDescriptor[parameters.Count];
        var used = new bool[descriptors.Count];
        for (int index = 0; index < parameters.Count; index++)
        {
            Parameter parameter = parameters[index];
            int selected;
            if (keys is not null)
            {
                selected = descriptors
                    .Select((item, itemIndex) => (item, itemIndex))
                    .Where(pair => string.Equals(
                        pair.item.Key,
                        keys[index],
                        StringComparison.Ordinal))
                    .Select(pair => pair.itemIndex)
                    .DefaultIfEmpty(-1)
                    .First();
                if (selected < 0)
                {
                    throw new InvalidDataException(
                        $"SafeTensors file '{path}' has no tensor named " +
                        $"'{keys[index]}'. Keys in the file: " +
                        $"{string.Join(", ", descriptors.Select(item => item.Key))}.");
                }
                if (used[selected])
                {
                    throw new InvalidDataException(
                        $"SafeTensors tensor '{keys[index]}' is mapped to " +
                        "more than one model parameter.");
                }
            }
            else
            {
                selected = -1;
                for (int candidate = 0;
                    candidate < descriptors.Count;
                    candidate++)
                {
                    if (used[candidate]
                        || !descriptors[candidate].Shape.SequenceEqual(
                            parameter.T.Shape))
                    {
                        continue;
                    }
                    if (selected >= 0)
                    {
                        throw new InvalidDataException(
                            $"SafeTensors file '{path}' has more than one " +
                            $"unassigned tensor of shape " +
                            $"[{string.Join('x', parameter.T.Shape)}] for " +
                            $"parameter slot {index} ('{parameter.Name}'), " +
                            "so the mapping is ambiguous. Pass an explicit " +
                            "key list.");
                    }
                    selected = candidate;
                }
                if (selected < 0)
                {
                    throw new InvalidDataException(
                        $"SafeTensors file '{path}' has no unassigned tensor " +
                        $"of shape [{string.Join('x', parameter.T.Shape)}] " +
                        $"for parameter slot {index} ('{parameter.Name}').");
                }
            }

            ValidateMappedShape(descriptors[selected], parameter, index);
            used[selected] = true;
            mapped[index] = descriptors[selected];
        }
        return mapped;
    }

    private static void ValidateMappedShape(
        SafeTensorDescriptor descriptor,
        Parameter parameter,
        int index)
    {
        if (descriptor.Shape.SequenceEqual(parameter.T.Shape))
            return;
        throw new InvalidDataException(
            $"SafeTensors tensor '{descriptor.Key}' has shape " +
            $"[{string.Join('x', descriptor.Shape)}] but parameter slot " +
            $"{index} ('{parameter.Name}') expects " +
            $"[{string.Join('x', parameter.T.Shape)}].");
    }

    private static void ReadStreamedValues(
        Stream stream,
        Span<float> values,
        SafeTensorDTypeCodec codec)
        => codec.Read(stream, values);

    private static void ValidateRestoredValues(
        ReadOnlySpan<float> values,
        string key,
        TensorDType destinationDType)
    {
        foreach (float value in values)
        {
            if (!float.IsFinite(value))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{key}' contains a non-finite " +
                    "value.");
            }
            if (destinationDType == TensorDType.Float16
                && !Half.IsFinite((Half)value))
            {
                throw new InvalidDataException(
                    $"SafeTensors parameter '{key}' contains a value " +
                    "outside the finite Float16 range.");
            }
        }
    }

    private static byte[] CreateModelHeader(
        IReadOnlyList<Parameter> parameters,
        TensorDType? artifactDTypeOverride)
    {
        using var buffer = new MemoryStream();
        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            writer.WriteString(
                "format",
                parameters.Any(parameter =>
                    (artifactDTypeOverride ?? parameter.T.DType)
                        != TensorDType.Float32)
                        ? "nntrain.module_state.mixed.v1"
                        : "nntrain.module_state.f32.v1");
            writer.WriteString(
                "module_state_format_version",
                ModuleState.CurrentFormatVersion.ToString(
                    System.Globalization.CultureInfo.InvariantCulture));
            writer.WriteEndObject();

            long offset = 0;
            for (int index = 0; index < parameters.Count; index++)
            {
                Parameter parameter = parameters[index];
                Tensor tensor = parameter.T;
                TensorDType artifactDType =
                    artifactDTypeOverride ?? tensor.DType;
                SafeTensorDTypeCodec codec =
                    SafeTensorDTypeCodecs.Get(artifactDType);
                writer.WritePropertyName($"{index:D8}:{parameter.Name}");
                writer.WriteStartObject();
                writer.WriteString("dtype", codec.DescriptorName);
                writer.WritePropertyName("shape");
                writer.WriteStartArray();
                foreach (int dimension in tensor.Shape)
                    writer.WriteNumberValue(dimension);
                writer.WriteEndArray();
                long end = checked(
                    offset
                    + (long)tensor.Numel * codec.EncodedElementSize);
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

    private static void WriteStreamedValues(
        Stream stream,
        ReadOnlySpan<float> values,
        TensorDType dtype)
        => SafeTensorDTypeCodecs.Get(dtype).Write(stream, values);
}
