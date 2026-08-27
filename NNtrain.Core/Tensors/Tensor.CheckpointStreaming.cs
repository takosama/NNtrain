namespace NNtrain;

public partial class Tensor
{
    /// <summary>
    /// Starts a sequential checkpoint restore without first synchronizing the
    /// value being replaced from CUDA. Existing device replicas are discarded
    /// immediately, which both avoids a useless D2H transfer and releases VRAM
    /// before file payloads are staged.
    /// </summary>
    internal CheckpointRestoreWriter BeginCheckpointRestore()
        => new(this);

    internal bool RequiresTwoPassBfp8CheckpointRestore
        => DType == TensorDType.Bfp8 && _masterData is null;

    internal Bfp8CheckpointRestoreWriter BeginBfp8CheckpointRestore()
        => new(this);

    /// <summary>
    /// Copies one checkpoint range without materializing the complete tensor
    /// on the host. CUDA-resident data is copied directly from its
    /// authoritative replica into page-locked staging memory.
    /// </summary>
    internal ReadOnlySpan<float> CopyCheckpointRangeTo(
        int sourceOffset,
        int length,
        CheckpointFloatStagingBuffer staging,
        bool preferMaster)
    {
        ArgumentNullException.ThrowIfNull(staging);
        ArgumentOutOfRangeException.ThrowIfNegative(sourceOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(length);
        if (sourceOffset > Numel - length)
            throw new ArgumentOutOfRangeException(nameof(length));

        lock (_deviceSync)
        {
            if (_hostDataCurrent)
            {
                Span<float> destination = staging.GetManagedSpan(length);
                if (preferMaster && _masterData is not null)
                {
                    _masterData.AsSpan(sourceOffset, length)
                        .CopyTo(destination);
                }
                else
                {
                    _data.CopyRangeTo(sourceOffset, destination);
                }
                return destination;
            }

            int deviceIndex = _cudaDeviceIndex;
            if (preferMaster
                && _cudaMasterBuffers.TryGetValue(
                    deviceIndex,
                    out DeviceBuffer? master)
                && master.Version == _dataVersion)
            {
                return CopyCudaFloatCheckpointRange(
                    master.Buffer,
                    sourceOffset,
                    length,
                    staging);
            }

            if (_cudaBuffers.TryGetValue(
                    deviceIndex,
                    out DeviceBuffer? values)
                && values.Version == _dataVersion)
            {
                return CopyCudaFloatCheckpointRange(
                    values.Buffer,
                    sourceOffset,
                    length,
                    staging);
            }

            if (_cudaBFloat16Buffers.TryGetValue(
                    deviceIndex,
                    out BFloat16DeviceBuffer? encoded)
                && encoded.Version == _dataVersion)
            {
                return CopyCudaBFloat16CheckpointRange(
                    encoded.Buffer,
                    sourceOffset,
                    length,
                    staging);
            }

            if (_cudaBfp8Buffers.TryGetValue(
                    deviceIndex,
                    out Bfp8DeviceBuffer? bfp8)
                && bfp8.Version == _dataVersion)
            {
                return CopyCudaBfp8CheckpointRange(
                    bfp8.View,
                    sourceOffset,
                    length,
                    staging);
            }

            // A secondary replica may be the only current copy after a
            // data-parallel operation. Prefer a direct chunked read from it
            // over the legacy whole-tensor host synchronization path.
            DeviceBuffer? anyMaster = preferMaster
                ? _cudaMasterBuffers.Values.FirstOrDefault(
                    candidate => candidate.Version == _dataVersion)
                : null;
            if (anyMaster is not null)
            {
                return CopyCudaFloatCheckpointRange(
                    anyMaster.Buffer,
                    sourceOffset,
                    length,
                    staging);
            }

            DeviceBuffer? anyFloat = _cudaBuffers.Values.FirstOrDefault(
                candidate => candidate.Version == _dataVersion);
            if (anyFloat is not null)
            {
                return CopyCudaFloatCheckpointRange(
                    anyFloat.Buffer,
                    sourceOffset,
                    length,
                    staging);
            }

            BFloat16DeviceBuffer? anyBFloat16 =
                _cudaBFloat16Buffers.Values.FirstOrDefault(
                    candidate => candidate.Version == _dataVersion);
            if (anyBFloat16 is not null)
            {
                return CopyCudaBFloat16CheckpointRange(
                    anyBFloat16.Buffer,
                    sourceOffset,
                    length,
                    staging);
            }


            Bfp8DeviceBuffer? anyBfp8 =
                _cudaBfp8Buffers.Values.FirstOrDefault(
                    candidate => candidate.Version == _dataVersion);
            if (anyBfp8 is not null)
            {
                return CopyCudaBfp8CheckpointRange(
                    anyBfp8.View,
                    sourceOffset,
                    length,
                    staging);
            }

            throw new InvalidOperationException(
                "A CUDA-resident tensor has no authoritative checkpoint replica.");
        }
    }

    private static ReadOnlySpan<float> CopyCudaFloatCheckpointRange(
        NativeCudaBuffer<float> source,
        int sourceOffset,
        int length,
        CheckpointFloatStagingBuffer staging)
    {
        Span<float> destination = staging.GetCudaPinnedSpan(length);
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostNative(
                source.Device.Index,
                staging.Pointer,
                source.NativePtr + checked(sourceOffset * sizeof(float)),
                checked((nuint)length * sizeof(float))),
            "cudaMemcpy(D2H checkpoint parameter chunk)");
        return destination;
    }

    private static ReadOnlySpan<float> CopyCudaBFloat16CheckpointRange(
        NativeCudaBuffer<ushort> source,
        int sourceOffset,
        int length,
        CheckpointFloatStagingBuffer staging)
    {
        Span<float> destination = staging.GetCudaPinnedSpan(length);
        Span<ushort> encoded = staging.GetEncodedPrefix(length);
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostNative(
                source.Device.Index,
                staging.Pointer,
                source.NativePtr + checked(sourceOffset * sizeof(ushort)),
                checked((nuint)length * sizeof(ushort))),
            "cudaMemcpy(D2H checkpoint BF16 parameter chunk)");

        // Source and destination intentionally share the same staging block.
        // Expanding from the end prevents a float write from overwriting an
        // encoded value that has not been consumed yet.
        for (int index = length - 1; index >= 0; index--)
        {
            destination[index] =
                TensorStorageCodec.DecodeBFloat16(encoded[index]);
        }
        return destination;
    }

    private static unsafe ReadOnlySpan<float> CopyCudaBfp8CheckpointRange(
        CudaBfp8BufferView source,
        int sourceOffset,
        int length,
        CheckpointFloatStagingBuffer staging)
    {
        if (length > CheckpointFloatStagingBuffer.MaximumElementCount / 2)
        {
            throw new ArgumentOutOfRangeException(
                nameof(length),
                "BFP8 checkpoint chunks reserve half of staging for scales.");
        }

        Span<float> destination = staging.GetCudaPinnedSpan(length);
        var payload = new Span<sbyte>((void*)staging.Pointer, length);
        int blockSize = source.Descriptor.GetEffectiveBlockSize(
            source.Payload.Length);
        int firstScale = sourceOffset / blockSize;
        int lastScale = checked(sourceOffset + length - 1) / blockSize;
        int scaleCount = checked(lastScale - firstScale + 1);
        nint scalePointer = staging.Pointer + checked(length * sizeof(float));
        var scales = new Span<float>((void*)scalePointer, scaleCount);

        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostNative(
                source.Payload.Device.Index,
                staging.Pointer,
                source.Payload.NativePtr + sourceOffset,
                checked((nuint)length)),
            "cudaMemcpy(D2H checkpoint BFP8 payload chunk)");
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostNative(
                source.Scales.Device.Index,
                scalePointer,
                source.Scales.NativePtr + checked(firstScale * sizeof(float)),
                checked((nuint)scaleCount * sizeof(float))),
            "cudaMemcpy(D2H checkpoint BFP8 scale chunk)");

        for (int index = length - 1; index >= 0; index--)
        {
            int scaleIndex = checked(sourceOffset + index) / blockSize
                - firstScale;
            destination[index] = payload[index] * scales[scaleIndex];
        }
        return destination;
    }

    internal sealed class CheckpointRestoreWriter : IDisposable
    {
        private Tensor? _owner;
        private int _written;

        internal CheckpointRestoreWriter(Tensor owner)
        {
            _owner = owner;
            // The previous value is about to be replaced. Do not make an
            // authoritative CUDA replica current on the host merely to throw
            // it away. Invalidate also releases the old VRAM allocation.
            owner.InvalidateCudaBuffers();
        }

        internal void WriteNext(ReadOnlySpan<float> values)
        {
            Tensor owner = _owner
                ?? throw new ObjectDisposedException(
                    nameof(CheckpointRestoreWriter));
            if (values.Length > owner.Numel - _written)
            {
                throw new InvalidDataException(
                    "Checkpoint tensor payload exceeds the parameter shape.");
            }

            lock (owner._deviceSync)
            {
                if (owner._masterData is not null)
                {
                    values.CopyTo(owner._masterData.AsSpan(_written));
                }
                else
                {
                    owner._data.CopyRangeFromFloat32(values, _written);
                }
            }
            _written += values.Length;
        }

        internal void Complete()
        {
            Tensor owner = _owner
                ?? throw new ObjectDisposedException(
                    nameof(CheckpointRestoreWriter));
            if (_written != owner.Numel)
            {
                throw new InvalidDataException(
                    $"Checkpoint tensor payload contains {_written} values " +
                    $"but the parameter requires {owner.Numel}.");
            }

            lock (owner._deviceSync)
            {
                if (owner._masterData is not null)
                    owner._data.CopyFrom(owner._masterData);
                owner.MarkDataMutated();
            }
            _owner = null;
        }

        public void Dispose()
        {
            Tensor? owner = _owner;
            _owner = null;
            if (owner is null)
                return;

            // A failed load leaves the model unusable, but its authority and
            // version must still describe the partially replaced host value.
            lock (owner._deviceSync)
            {
                if (owner._masterData is not null)
                    owner._data.CopyFrom(owner._masterData);
                owner.MarkDataMutated();
            }
        }
    }

    internal sealed class Bfp8CheckpointRestoreWriter : IDisposable
    {
        private Tensor? _owner;
        private readonly sbyte[] _payload;
        private readonly float[] _scales;
        private readonly int _blockSize;
        private int _scanned;
        private int _encoded;
        private bool _readyToEncode;

        internal Bfp8CheckpointRestoreWriter(Tensor owner)
        {
            if (!owner.RequiresTwoPassBfp8CheckpointRestore)
            {
                throw new InvalidOperationException(
                    "Two-pass BFP8 restore requires pure BFP8 storage.");
            }
            _owner = owner;
            owner.InvalidateCudaBuffers();
            lock (owner._deviceSync)
            {
                if (!owner._data.TryGetBfp8Buffers(
                        out _payload,
                        out _scales,
                        out Bfp8QuantizationDescriptor descriptor))
                {
                    throw new InvalidOperationException(
                        "BFP8 checkpoint destination has no encoded storage.");
                }
                _blockSize = descriptor.GetEffectiveBlockSize(owner.Numel);
                Array.Clear(_payload);
                Array.Clear(_scales);
            }
        }

        internal void AccumulateScale(ReadOnlySpan<float> values)
        {
            Tensor owner = GetOwner();
            if (_readyToEncode || values.Length > owner.Numel - _scanned)
                throw new InvalidOperationException("Invalid BFP8 scale pass.");
            for (int index = 0; index < values.Length; index++)
            {
                float value = values[index];
                if (!float.IsFinite(value))
                {
                    throw new InvalidDataException(
                        "BFP8 checkpoint values must be finite.");
                }
                int scaleIndex = checked(_scanned + index) / _blockSize;
                _scales[scaleIndex] = MathF.Max(
                    _scales[scaleIndex],
                    MathF.Abs(value));
            }
            _scanned += values.Length;
        }

        internal void PrepareEncoding()
        {
            Tensor owner = GetOwner();
            if (_scanned != owner.Numel || _readyToEncode)
                throw new InvalidOperationException("BFP8 scale pass is incomplete.");
            for (int index = 0; index < _scales.Length; index++)
            {
                _scales[index] = _scales[index] == 0f
                    ? 1f
                    : _scales[index] / 127f;
            }
            _readyToEncode = true;
        }

        internal void WriteNext(ReadOnlySpan<float> values)
        {
            Tensor owner = GetOwner();
            if (!_readyToEncode || values.Length > owner.Numel - _encoded)
                throw new InvalidOperationException("Invalid BFP8 encode pass.");
            for (int index = 0; index < values.Length; index++)
            {
                int destination = checked(_encoded + index);
                float scale = _scales[destination / _blockSize];
                float rounded = MathF.Round(
                    values[index] / scale,
                    MidpointRounding.ToEven);
                _payload[destination] = (sbyte)Math.Clamp(
                    (int)rounded,
                    -127,
                    127);
            }
            _encoded += values.Length;
        }

        internal void Complete()
        {
            Tensor owner = GetOwner();
            if (!_readyToEncode || _encoded != owner.Numel)
                throw new InvalidOperationException("BFP8 encode pass is incomplete.");
            lock (owner._deviceSync)
                owner.MarkDataMutated();
            _owner = null;
        }

        public void Dispose()
        {
            Tensor? owner = _owner;
            _owner = null;
            if (owner is null)
                return;
            Array.Clear(_payload);
            Array.Fill(_scales, 1f);
            lock (owner._deviceSync)
                owner.MarkDataMutated();
        }

        private Tensor GetOwner()
            => _owner
                ?? throw new ObjectDisposedException(
                    nameof(Bfp8CheckpointRestoreWriter));
    }
}
