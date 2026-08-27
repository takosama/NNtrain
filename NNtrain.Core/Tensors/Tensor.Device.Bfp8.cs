namespace NNtrain;

internal readonly record struct CudaBfp8BufferView(
    NativeCudaBuffer<sbyte> Payload,
    NativeCudaBuffer<float> Scales,
    Bfp8QuantizationDescriptor Descriptor);

public partial class Tensor
{
    private readonly Dictionary<int, Bfp8DeviceBuffer> _cudaBfp8Buffers = [];

    internal CudaBfp8BufferView EnsureCudaBfp8Buffer(int deviceIndex = -1)
    {
        if (DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "A physical CUDA BFP8 buffer requires BFP8 dtype.");
        }

        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        lock (_deviceSync)
        {
            // A CUDA-produced tensor is already authoritative. Return it
            // without materializing the lazy host placeholder or performing a
            // payload/scale round trip.
            if (_cudaBfp8Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8DeviceBuffer? current)
                && current.Version == _dataVersion)
            {
                return current.View;
            }

            if (!_hostDataCurrent
                && (!_cudaBfp8Buffers.TryGetValue(
                        resolvedDeviceIndex,
                        out Bfp8DeviceBuffer? requested)
                    || requested.Version != _dataVersion))
            {
                SynchronizeHostFromCudaLocked(_cudaDeviceIndex);
            }

            if (!_data.TryGetBfp8Buffers(
                    out sbyte[] hostPayload,
                    out float[] hostScales,
                    out Bfp8QuantizationDescriptor descriptor))
            {
                throw new InvalidOperationException(
                    "BFP8 tensor storage is missing its encoded payload.");
            }

            if (!_cudaBfp8Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8DeviceBuffer? buffer)
                || buffer.Payload.Length != Numel
                || buffer.Scales.Length != hostScales.Length
                || buffer.Descriptor != descriptor)
            {
                buffer?.Dispose();
                NativeCudaBuffer<sbyte>? payload = null;
                NativeCudaBuffer<float>? scales = null;
                try
                {
                    payload = accelerator.Allocate1D(hostPayload);
                    scales = accelerator.Allocate1D(hostScales);
                    buffer = new Bfp8DeviceBuffer(
                        payload,
                        scales,
                        descriptor,
                        _dataVersion);
                    _cudaBfp8Buffers[resolvedDeviceIndex] = buffer;
                    return buffer.View;
                }
                catch
                {
                    payload?.Dispose();
                    scales?.Dispose();
                    throw;
                }
            }

            if (buffer.Version != _dataVersion)
            {
                buffer.Payload.CopyFromCPU(hostPayload);
                buffer.Scales.CopyFromCPU(hostScales);
                buffer.Version = _dataVersion;
            }
            return buffer.View;
        }
    }

    /// <summary>
    /// Returns a versioned, device-resident BF16 decode of this BFP8 tensor.
    /// The cache belongs to the encoded replica and is invalidated whenever
    /// that replica's generation changes.
    /// </summary>
    internal NativeCudaBuffer<ushort> EnsureCudaBfp8BFloat16Buffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        _ = EnsureCudaBfp8Buffer(resolvedDeviceIndex);
        lock (_deviceSync)
        {
            if (!_cudaBfp8Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8DeviceBuffer? buffer)
                || buffer.Version != _dataVersion)
            {
                throw new InvalidOperationException(
                    "The authoritative CUDA BFP8 replica disappeared while " +
                    "creating its BF16 decode cache.");
            }

            return buffer.GetOrCreateBFloat16(resolvedDeviceIndex);
        }
    }

    internal NativeCudaBuffer<sbyte> EnsureCudaBfp8ColumnMajorPayload(
        int rows,
        int columns,
        int deviceIndex = -1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rows);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(columns);
        if (checked(rows * columns) != Numel)
        {
            throw new ArgumentException(
                "The BFP8 Int8 layout transform must cover the full tensor.");
        }

        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        _ = EnsureCudaBfp8Buffer(resolvedDeviceIndex);
        lock (_deviceSync)
        {
            if (!_cudaBfp8Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8DeviceBuffer? buffer)
                || buffer.Version != _dataVersion)
            {
                throw new InvalidOperationException(
                    "The authoritative CUDA BFP8 replica disappeared while " +
                    "creating its Int8 layout cache.");
            }
            return buffer.GetOrCreateColumnMajor(
                resolvedDeviceIndex,
                rows,
                columns);
        }
    }

    internal void AdoptCudaBfp8Buffers(
        NativeCudaBuffer<sbyte> payload,
        NativeCudaBuffer<float> scales,
        Bfp8QuantizationDescriptor descriptor,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(payload);
        ArgumentNullException.ThrowIfNull(scales);
        ArgumentNullException.ThrowIfNull(descriptor);
        if (DType != TensorDType.Bfp8)
            throw new InvalidOperationException("Tensor dtype must be BFP8.");
        if (payload.Device.Index != deviceIndex
            || scales.Device.Index != deviceIndex
            || payload.Length != Numel
            || scales.Length != descriptor.GetScaleCount(Numel)
            || descriptor != Bfp8Quantization)
        {
            throw new ArgumentException(
                "CUDA BFP8 buffers must match the tensor, descriptor, and device.");
        }

        lock (_deviceSync)
        {
            if (_cudaBfp8Buffers.Remove(
                    deviceIndex,
                    out Bfp8DeviceBuffer? previous))
            {
                previous.Dispose();
            }
            _cudaBfp8Buffers[deviceIndex] = new Bfp8DeviceBuffer(
                payload,
                scales,
                descriptor,
                _dataVersion);
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    private void DisposeCudaBfp8BuffersLocked()
    {
        foreach (Bfp8DeviceBuffer buffer in _cudaBfp8Buffers.Values)
            buffer.Dispose();
        _cudaBfp8Buffers.Clear();
    }

    private sealed class Bfp8DeviceBuffer : IDisposable
    {
        private int _disposed;
        private long _version;
        private NativeCudaBuffer<ushort>? _bfloat16Cache;
        private NativeCudaBuffer<sbyte>? _columnMajorCache;
        private int _columnMajorRows;
        private int _columnMajorColumns;

        internal Bfp8DeviceBuffer(
            NativeCudaBuffer<sbyte> payload,
            NativeCudaBuffer<float> scales,
            Bfp8QuantizationDescriptor descriptor,
            long version)
        {
            Payload = payload;
            Scales = scales;
            Descriptor = descriptor;
            _version = version;
        }

        internal NativeCudaBuffer<sbyte> Payload { get; }
        internal NativeCudaBuffer<float> Scales { get; }
        internal Bfp8QuantizationDescriptor Descriptor { get; }
        internal long Version
        {
            get => _version;
            set
            {
                if (_version == value)
                    return;
                _version = value;
                DisposeBFloat16Cache();
                DisposeColumnMajorCache();
            }
        }
        internal CudaBfp8BufferView View => new(Payload, Scales, Descriptor);

        internal NativeCudaBuffer<ushort> GetOrCreateBFloat16(int deviceIndex)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            if (_bfloat16Cache is not null)
                return _bfloat16Cache;

            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<ushort>? decoded = null;
            try
            {
                decoded = accelerator.Allocate1D<ushort>(Payload.Length);
                CudaBfp8GemmTelemetry.RecordBFloat16DecodeCacheMiss();
                CudaBfp8Native.DequantizeBFloat16(
                    deviceIndex,
                    Payload,
                    Scales,
                    decoded,
                    Descriptor,
                    accelerator.DefaultStream);
                _bfloat16Cache = decoded;
                return decoded;
            }
            catch
            {
                decoded?.Dispose();
                throw;
            }
        }

        internal NativeCudaBuffer<sbyte> GetOrCreateColumnMajor(
            int deviceIndex,
            int rows,
            int columns)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            if (_columnMajorCache is not null
                && _columnMajorRows == rows
                && _columnMajorColumns == columns)
            {
                return _columnMajorCache;
            }

            DisposeColumnMajorCache();
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<sbyte>? transformed = null;
            try
            {
                transformed = accelerator.Allocate1D<sbyte>(Payload.Length);
                CudaBfp8GemmTelemetry.RecordInt8LayoutTransformCacheMiss();
                CudaBfp8Native.TransposeInt8RowToColumn(
                    deviceIndex,
                    Payload,
                    transformed,
                    rows,
                    columns,
                    accelerator.DefaultStream);
                _columnMajorRows = rows;
                _columnMajorColumns = columns;
                _columnMajorCache = transformed;
                return transformed;
            }
            catch
            {
                transformed?.Dispose();
                throw;
            }
        }

        private void DisposeBFloat16Cache()
        {
            NativeCudaBuffer<ushort>? cache = _bfloat16Cache;
            _bfloat16Cache = null;
            cache?.Dispose();
        }

        private void DisposeColumnMajorCache()
        {
            NativeCudaBuffer<sbyte>? cache = _columnMajorCache;
            _columnMajorCache = null;
            _columnMajorRows = 0;
            _columnMajorColumns = 0;
            cache?.Dispose();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            try
            {
                try
                {
                    try
                    {
                        DisposeBFloat16Cache();
                    }
                    finally
                    {
                        DisposeColumnMajorCache();
                    }
                }
                finally
                {
                    Payload.Dispose();
                }
            }
            finally
            {
                Scales.Dispose();
            }
        }
    }
}
