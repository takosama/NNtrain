using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

public partial class Tensor
{
    private readonly object _deviceSync = new();
    private TensorDevice _device;
    private int _cudaDeviceIndex;
    private readonly Dictionary<int, DeviceBuffer> _cudaBuffers = [];
    private readonly Dictionary<int, BFloat16DeviceBuffer> _cudaBFloat16Buffers = [];
    private readonly Dictionary<int, DeviceBuffer> _cudaMasterBuffers = [];
    private readonly Dictionary<int, GradientDeviceBuffer> _cudaGradientBuffers = [];
    private readonly Dictionary<int, DeviceBuffer> _cudaStagingBuffers = [];
    private long _cudaBufferDataVersion = -1;
    private bool _hostDataCurrent = true;
    private long _gradientVersion;
    private bool _hostGradientCurrent = true;

    public TensorDevice Device => _device;

    public TorchDevice device
        => new(
            _device,
            _device == TensorDevice.Cuda ? _cudaDeviceIndex : 0);

    public Tensor To(TensorDevice device)
        => to(new TorchDevice(
            device,
            device == TensorDevice.Cuda ? CudaDeviceIndex : 0));

    public Tensor to(TorchDevice device)
    {
        if (device.IsCuda)
        {
            if (DType == TensorDType.BFloat16)
                EnsureCudaBFloat16Buffer(device.Index);
            else
                EnsureCudaFloat32Buffer(device.Index);
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = device.Index;
            return this;
        }

        if (_device == TensorDevice.Cuda)
            SynchronizeHostFromCuda();
        _device = TensorDevice.Cpu;
        return this;
    }

    public Tensor to(TensorDevice device) => To(device);

    private int ResolveCudaDeviceIndex(int requestedDeviceIndex)
    {
        if (requestedDeviceIndex >= 0)
            return requestedDeviceIndex;
        return ExecutionDevice == TensorDevice.Cuda
            ? CudaDeviceIndex
            : _cudaDeviceIndex;
    }

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaFloat32Buffer(
        int deviceIndex = -1)
    {
        if (DType == TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "BFloat16 tensors must use their physical 16-bit CUDA buffer; " +
                "implicit expansion to a float32 device buffer is forbidden.");
        }
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostDataCurrent
            && !_cudaBuffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostFromCuda();
        }
        lock (_deviceSync)
        {
            if (!_cudaBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D(GetPhysicalFloat32ComputeCache()),
                    resolvedDeviceIndex);
                _cudaBuffers[resolvedDeviceIndex] = buffer;
                _cudaBufferDataVersion = _dataVersion;
                return buffer.Buffer;
            }

            if (_cudaBufferDataVersion != _dataVersion)
            {
                buffer.Buffer.CopyFromCPU(GetPhysicalFloat32ComputeCache());
                _cudaBufferDataVersion = _dataVersion;
            }
            return buffer.Buffer;
        }
    }

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaMasterFloat32Buffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        if (DType == TensorDType.Float32)
            return EnsureCudaFloat32Buffer(resolvedDeviceIndex);
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostDataCurrent
            && !_cudaMasterBuffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostFromCuda();
        }
        lock (_deviceSync)
        {
            if (!_cudaMasterBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D(DataBuffer),
                    resolvedDeviceIndex);
                _cudaMasterBuffers[resolvedDeviceIndex] = buffer;
            }
            return buffer.Buffer;
        }
    }

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaGradientBuffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostGradientCurrent
            && !_cudaGradientBuffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostGradientFromCuda();
        }
        lock (_deviceSync)
        {
            if (!_cudaGradientBuffers.TryGetValue(
                resolvedDeviceIndex,
                out GradientDeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                MemoryBuffer1D<float, Stride1D.Dense> gradientBuffer;
                if (_grad.Length == 0)
                {
                    gradientBuffer = accelerator.Allocate1D<float>(Numel);
                    gradientBuffer.MemSetToZero();
                }
                else
                {
                    gradientBuffer = accelerator.Allocate1D(_grad);
                }
                buffer = new GradientDeviceBuffer(
                    gradientBuffer,
                    _gradientVersion,
                    resolvedDeviceIndex);
                _cudaGradientBuffers[resolvedDeviceIndex] = buffer;
                return buffer.Buffer;
            }

            if (buffer.Version != _gradientVersion)
            {
                // A different CUDA adapter may have produced another local
                // gradient for the same data-parallel step. Keep this
                // adapter's local buffer until the explicit all-reduce.
                if (!_hostGradientCurrent)
                    return buffer.Buffer;
                buffer.Buffer.CopyFromCPU(_grad);
                buffer.Version = _gradientVersion;
            }
            return buffer.Buffer;
        }
    }

    internal MemoryBuffer1D<float, Stride1D.Dense> EnsureCudaStagingBuffer(
        int deviceIndex)
    {
        CudaAccelerator accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        lock (_deviceSync)
        {
            if (!_cudaStagingBuffers.TryGetValue(
                deviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D<float>(Numel),
                    deviceIndex);
                _cudaStagingBuffers[deviceIndex] = buffer;
            }
            return buffer.Buffer;
        }
    }

    internal void MarkCudaGradientMutated(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            if (!_cudaGradientBuffers.TryGetValue(
                resolvedDeviceIndex,
                out GradientDeviceBuffer? buffer))
            {
                throw new InvalidOperationException(
                    "Cannot mark a CUDA gradient modified before allocating it.");
            }
            unchecked
            {
                _gradientVersion++;
            }
            buffer.Version = _gradientVersion;
            _hostGradientCurrent = false;
        }
    }

    internal void PrepareCudaGradientBuffers(IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        foreach (int deviceIndex in deviceIndices)
            EnsureCudaGradientBuffer(deviceIndex);
    }

    internal void MarkCudaGradientsSynchronized(IReadOnlyList<int> deviceIndices)
    {
        lock (_deviceSync)
        {
            unchecked
            {
                _gradientVersion++;
            }
            foreach (int deviceIndex in deviceIndices)
            {
                if (_cudaGradientBuffers.TryGetValue(
                    deviceIndex,
                    out GradientDeviceBuffer? buffer))
                {
                    buffer.Version = _gradientVersion;
                }
            }
            _hostGradientCurrent = false;
        }
    }

    internal void SetCudaGradient(float[] values, int deviceIndex = -1)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length != Numel)
            throw new ArgumentException("Gradient length must match the tensor.", nameof(values));
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        MemoryBuffer1D<float, Stride1D.Dense> buffer =
            EnsureCudaGradientBuffer(resolvedDeviceIndex);
        buffer.CopyFromCPU(values);
        MarkCudaGradientMutated(resolvedDeviceIndex);
    }

    internal void AdoptCudaFloat32Buffer(
        MemoryBuffer1D<float, Stride1D.Dense> buffer,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (buffer.Length != Numel)
            throw new ArgumentException("CUDA buffer length must match the tensor.", nameof(buffer));
        lock (_deviceSync)
        {
            if (_cudaBuffers.Remove(deviceIndex, out DeviceBuffer? previous))
                previous.Dispose();
            _cudaBuffers[deviceIndex] = new DeviceBuffer(buffer, deviceIndex);
            _cudaBufferDataVersion = _dataVersion;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    internal static MemoryBuffer1D<float, Stride1D.Dense> RentCudaFloatBuffer(
        int deviceIndex,
        int length)
        => CudaFloatBufferPool.Rent(deviceIndex, length);

    internal static void ReturnCudaFloatBuffer(
        CudaAccelerator accelerator,
        MemoryBuffer1D<float, Stride1D.Dense> buffer)
        => CudaFloatBufferPool.Return(accelerator, buffer);

    internal static MemoryBuffer1D<int, Stride1D.Dense> RentCudaIntBuffer(
        int deviceIndex,
        int[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        MemoryBuffer1D<int, Stride1D.Dense> buffer =
            CudaIntBufferPool.Rent(deviceIndex, values.Length);
        buffer.CopyFromCPU(values);
        return buffer;
    }

    internal static void ReturnCudaIntBuffer(
        CudaAccelerator accelerator,
        MemoryBuffer1D<int, Stride1D.Dense> buffer)
        => CudaIntBufferPool.Return(accelerator, buffer);

    internal static void ClearCudaFloatBufferPool(int deviceIndex)
    {
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaFloatBufferPool.Clear(accelerator);
        CudaBFloat16BufferPool.Clear(accelerator);
        CudaIntBufferPool.Clear(accelerator);
    }

    private static Tensor FromCudaResult(
        MemoryBuffer1D<float, Stride1D.Dense> buffer,
        int deviceIndex,
        int[] shape,
        Tensor[] parents,
        TensorDType? dtype = null)
    {
        TensorDType resultDType = dtype ?? TensorDTypeContract.Promote(parents);
        var result = new Tensor(
            TensorStorage.CreateDevicePlaceholder(
                checked((int)buffer.Length),
                resultDType),
            shape,
            parents);
        result.AdoptCudaFloat32Buffer(buffer, deviceIndex);
        if (!AutogradContext.IsRecordingEnabled)
            CudaInferenceScope.Track(result, deviceIndex);
        return result;
    }

    private static Tensor FromCudaResult(
        MemoryBuffer1D<ushort, Stride1D.Dense> buffer,
        int deviceIndex,
        int[] shape,
        Tensor[] parents,
        TensorDType? dtype = null)
    {
        TensorDType resultDType = dtype ?? TensorDTypeContract.Promote(parents);
        if (resultDType != TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "A physical CUDA bfloat16 result requires BFloat16 dtype.");
        }
        var result = new Tensor(
            TensorStorage.CreateDevicePlaceholder(
                checked((int)buffer.Length),
                resultDType),
            shape,
            parents);
        result.AdoptCudaBFloat16Buffer(buffer, deviceIndex);
        if (!AutogradContext.IsRecordingEnabled)
            CudaInferenceScope.Track(result, deviceIndex);
        return result;
    }

    internal void SynchronizeHostFromCuda(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            bool hasFloat = _cudaBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer);
            bool hasBFloat16 = _cudaBFloat16Buffers.TryGetValue(
                resolvedDeviceIndex,
                out BFloat16DeviceBuffer? bfloat16Buffer);
            if (!hasFloat && !hasBFloat16)
            {
                return;
            }

            if (_hostDataCurrent)
                return;

            float[] data = DataBuffer;
            if (_cudaMasterBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? masterBuffer))
            {
                masterBuffer.Buffer.CopyToCPU(data);
            }
            else if (buffer is not null)
            {
                buffer.Buffer.CopyToCPU(data);
            }
            else
            {
                var encoded = new ushort[Numel];
                bfloat16Buffer!.Buffer.CopyToCPU(encoded);
                TensorStorageCodec.DecodeBFloat16(encoded, data);
            }
            if (DType != TensorDType.Float32)
                _data.CopyFrom(data);
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = true;
            _cudaBufferDataVersion = _dataVersion;
        }
    }

    internal void MarkCudaDataMutated(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            if (!_cudaBuffers.ContainsKey(resolvedDeviceIndex)
                && !_cudaBFloat16Buffers.ContainsKey(resolvedDeviceIndex))
            {
                throw new InvalidOperationException(
                    "Cannot mark CUDA data modified before allocating its buffer.");
            }
            unchecked
            {
                _dataVersion++;
            }
            _cudaBufferDataVersion = _dataVersion;
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = resolvedDeviceIndex;
        }
    }

    private void EnsureHostDataCurrent()
    {
        if (!_hostDataCurrent)
            SynchronizeHostFromCuda();
    }

    private void EnsureHostGradientCurrent()
    {
        if (!_hostGradientCurrent)
            SynchronizeHostGradientFromCuda();
    }

    internal void EnsureHostGradientStorage()
    {
        EnsureHostGradientCurrent();
        if (_grad.Length == 0)
            _grad = new float[Numel];
    }

    private void SynchronizeHostGradientFromCuda(int deviceIndex = -1)
    {
        lock (_deviceSync)
            SynchronizeHostGradientFromCudaLocked(deviceIndex);
    }

    private void SynchronizeHostGradientFromCudaLocked(int deviceIndex = -1)
    {
        if (_hostGradientCurrent)
            return;
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        if (!_cudaGradientBuffers.TryGetValue(
            resolvedDeviceIndex,
            out GradientDeviceBuffer? buffer))
        {
            buffer = _cudaGradientBuffers.Values.FirstOrDefault(
                candidate => candidate.Version == _gradientVersion);
        }
        if (buffer is null)
            return;
        if (_grad.Length == 0)
            _grad = new float[Numel];
        buffer.Buffer.CopyToCPU(_grad);
        _hostGradientCurrent = true;
    }

    private void MarkHostGradientMutable()
    {
        unchecked
        {
            _gradientVersion++;
        }
        _hostGradientCurrent = true;
    }

    private void ClearCudaGradients()
    {
        lock (_deviceSync)
        {
            unchecked
            {
                _gradientVersion++;
            }
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
            {
                buffer.Buffer.MemSetToZero();
                buffer.Version = _gradientVersion;
            }
            _hostGradientCurrent = true;
        }
    }

    internal void InvalidateCudaBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.Dispose();
            foreach (BFloat16DeviceBuffer buffer in _cudaBFloat16Buffers.Values)
                buffer.Dispose();
            foreach (DeviceBuffer buffer in _cudaMasterBuffers.Values)
                buffer.Dispose();
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
                buffer.Dispose();
            foreach (DeviceBuffer buffer in _cudaStagingBuffers.Values)
                buffer.Dispose();
            _cudaBuffers.Clear();
            _cudaBFloat16Buffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaStagingBuffers.Clear();
            _cudaBufferDataVersion = -1;
            _hostDataCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    internal void ReleaseCudaGraphBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.ReturnToPool();
            foreach (BFloat16DeviceBuffer buffer in _cudaBFloat16Buffers.Values)
                buffer.ReturnToPool();
            foreach (DeviceBuffer buffer in _cudaMasterBuffers.Values)
                buffer.ReturnToPool();
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
                buffer.ReturnToPool();
            foreach (DeviceBuffer buffer in _cudaStagingBuffers.Values)
                buffer.ReturnToPool();
            _cudaBuffers.Clear();
            _cudaBFloat16Buffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaStagingBuffers.Clear();
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    internal void ReleaseCudaInferenceBuffers()
    {
        lock (_deviceSync)
        {
            foreach (DeviceBuffer buffer in _cudaBuffers.Values)
                buffer.ReturnToPool();
            foreach (BFloat16DeviceBuffer buffer in _cudaBFloat16Buffers.Values)
                buffer.ReturnToPool();
            foreach (DeviceBuffer buffer in _cudaMasterBuffers.Values)
                buffer.ReturnToPool();
            foreach (GradientDeviceBuffer buffer in _cudaGradientBuffers.Values)
                buffer.Dispose();
            foreach (DeviceBuffer buffer in _cudaStagingBuffers.Values)
                buffer.ReturnToPool();
            _cudaBuffers.Clear();
            _cudaBFloat16Buffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaStagingBuffers.Clear();
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    private sealed class DeviceBuffer : IDisposable
    {
        private int _disposed;
        private readonly CudaAccelerator _accelerator;

        internal DeviceBuffer(
            MemoryBuffer1D<float, Stride1D.Dense> buffer,
            int deviceIndex)
        {
            Buffer = buffer;
            _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        }

        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Buffer.Dispose();
        }

        internal void ReturnToPool()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            CudaFloatBufferPool.Return(_accelerator, Buffer);
        }
    }

    private sealed class GradientDeviceBuffer(
        MemoryBuffer1D<float, Stride1D.Dense> buffer,
        long version,
        int deviceIndex) : IDisposable
    {
        private int _disposed;
        private readonly CudaAccelerator _accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        internal MemoryBuffer1D<float, Stride1D.Dense> Buffer { get; } = buffer;
        internal long Version { get; set; } = version;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            Buffer.Dispose();
        }

        internal void ReturnToPool()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            CudaFloatBufferPool.Return(_accelerator, Buffer);
        }
    }

    /// <summary>
    /// Reuses transient CUDA storage between autograd graphs.  The pool is
    /// deliberately bounded: cached buffers are an optimization, not a claim
    /// on all available VRAM.
    /// </summary>
    private static class CudaFloatBufferPool
    {
        // Training graphs routinely need several GiB of transient activation
        // storage. Keep a bounded high-water set resident so the next
        // fixed-shape batch can reuse it, while reserving most VRAM for live
        // activations, gradients, parameters, and optimizer state.
        // A miss evicts stale shape buckets before allocating a new shape.
        private const long MinimumCachedBytesPerDevice = 512L * 1024 * 1024;
        private const int CacheMemoryPercent = 25;
        private const long MinimumReservedFreeBytes = 1536L * 1024 * 1024;
        private const int ReservedFreeMemoryPercent = 25;
        private const int MaximumBuffersPerSize = 64;
        private static readonly object Sync = new();
        private static readonly Dictionary<
            (CudaAccelerator Accelerator, int Length),
            Stack<MemoryBuffer1D<float, Stride1D.Dense>>> Buffers = [];
        private static readonly HashSet<
            MemoryBuffer1D<float, Stride1D.Dense>> PooledBuffers = [];
        private static readonly Dictionary<CudaAccelerator, long>
            CachedBytes = [];

        internal static MemoryBuffer1D<float, Stride1D.Dense> Rent(
            int deviceIndex,
            int length)
        {
            CudaAccelerator accelerator =
                NNtrain.ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var dispose = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            lock (Sync)
            {
                var key = (accelerator, length);
                if (Buffers.TryGetValue(key, out var bucket)
                    && bucket.Count > 0)
                {
                    CachedBytes[accelerator] = CachedBytes.GetValueOrDefault(
                        accelerator) - checked((long)length * sizeof(float));
                    MemoryBuffer1D<float, Stride1D.Dense> buffer = bucket.Pop();
                    PooledBuffers.Remove(buffer);
                    return buffer;
                }

                long requestedBytes = checked((long)length * sizeof(float));
                long cachedBytes = CachedBytes.GetValueOrDefault(accelerator);
                long capacityTargetBytes = Math.Max(
                    0L,
                    GetCacheBudget(accelerator) - requestedBytes);
                long freeBytes = accelerator.GetFreeMemory();
                long bytesNeededForReserve = Math.Max(
                    0L,
                    checked(GetReservedFreeBytes(accelerator)
                        + requestedBytes - freeBytes));
                long reserveTargetBytes = Math.Max(
                    0L,
                    cachedBytes - bytesNeededForReserve);
                long targetCachedBytes = Math.Min(
                    capacityTargetBytes,
                    reserveTargetBytes);
                TrimToBytes(accelerator, targetCachedBytes, dispose);
            }

            foreach (MemoryBuffer1D<float, Stride1D.Dense> buffer in dispose)
                buffer.Dispose();

            try
            {
                return accelerator.Allocate1D<float>(length);
            }
            catch (CudaException exception) when (IsOutOfMemory(exception))
            {
                // A shape change can leave an otherwise valid but unusable
                // set of cached blocks. Flush only transient storage on this
                // adapter, then retry once. A genuine live-set OOM still
                // propagates from the second allocation.
                accelerator.Synchronize();
                Clear(accelerator);
                return accelerator.Allocate1D<float>(length);
            }
        }

        internal static void Return(
            CudaAccelerator accelerator,
            MemoryBuffer1D<float, Stride1D.Dense> buffer)
        {
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(float));
            lock (Sync)
            {
                // A context and its adopted result tensor can share storage.
                // Never place the same native allocation in a bucket twice.
                if (!PooledBuffers.Add(buffer))
                    return;
                var key = (accelerator, length);
                if (!Buffers.TryGetValue(key, out var bucket))
                {
                    bucket = [];
                    Buffers[key] = bucket;
                }

                if (bucket.Count < MaximumBuffersPerSize
                    && CachedBytes.GetValueOrDefault(accelerator) + bytes
                        <= GetCacheBudget(accelerator))
                {
                    bucket.Push(buffer);
                    CachedBytes[accelerator] =
                        CachedBytes.GetValueOrDefault(accelerator) + bytes;
                    return;
                }
                PooledBuffers.Remove(buffer);
            }

            buffer.Dispose();
        }

        private static long GetCacheBudget(CudaAccelerator accelerator)
            => Math.Max(
                MinimumCachedBytesPerDevice,
                checked(accelerator.MemorySize * CacheMemoryPercent / 100));

        private static long GetReservedFreeBytes(CudaAccelerator accelerator)
            => Math.Min(
                accelerator.MemorySize / 3,
                Math.Max(
                    MinimumReservedFreeBytes,
                    checked(accelerator.MemorySize
                        * ReservedFreeMemoryPercent / 100)));

        private static bool IsOutOfMemory(CudaException exception)
            => string.Equals(
                    exception.Error,
                    nameof(CudaError.CUDA_ERROR_OUT_OF_MEMORY),
                    StringComparison.Ordinal)
                || exception.Message.Contains(
                    "out of memory",
                    StringComparison.OrdinalIgnoreCase);

        private static void TrimToBytes(
            CudaAccelerator accelerator,
            long targetBytes,
            List<MemoryBuffer1D<float, Stride1D.Dense>> dispose)
        {
            long cached = CachedBytes.GetValueOrDefault(accelerator);
            if (cached <= targetBytes)
                return;
            var keys = Buffers.Keys
                .Where(key => ReferenceEquals(key.Accelerator, accelerator))
                .ToArray();
            foreach (var key in keys)
            {
                Stack<MemoryBuffer1D<float, Stride1D.Dense>> bucket =
                    Buffers[key];
                while (bucket.Count > 0 && cached > targetBytes)
                {
                    MemoryBuffer1D<float, Stride1D.Dense> buffer = bucket.Pop();
                    PooledBuffers.Remove(buffer);
                    cached -= checked((long)key.Length * sizeof(float));
                    dispose.Add(buffer);
                }
                if (bucket.Count == 0)
                    Buffers.Remove(key);
                if (cached <= targetBytes)
                    break;
            }
            CachedBytes[accelerator] = cached;
        }

        internal static void Clear(CudaAccelerator accelerator)
        {
            var dispose = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
            lock (Sync)
            {
                var keys = Buffers.Keys
                    .Where(key => ReferenceEquals(key.Accelerator, accelerator))
                    .ToArray();
                foreach (var key in keys)
                {
                    Stack<MemoryBuffer1D<float, Stride1D.Dense>> bucket =
                        Buffers[key];
                    while (bucket.Count > 0)
                    {
                        var buffer = bucket.Pop();
                        PooledBuffers.Remove(buffer);
                        CachedBytes[accelerator] =
                            CachedBytes.GetValueOrDefault(accelerator)
                            - checked((long)key.Length * sizeof(float));
                        dispose.Add(buffer);
                    }
                    Buffers.Remove(key);
                }
                CachedBytes.Remove(accelerator);
            }
            foreach (var buffer in dispose)
                buffer.Dispose();
        }
    }

    private static class CudaIntBufferPool
    {
        private const int MaximumBuffersPerSize = 64;
        private static readonly object Sync = new();
        private static readonly Dictionary<
            (CudaAccelerator Accelerator, int Length),
            Stack<MemoryBuffer1D<int, Stride1D.Dense>>> Buffers = [];
        private static readonly HashSet<
            MemoryBuffer1D<int, Stride1D.Dense>> PooledBuffers = [];

        internal static MemoryBuffer1D<int, Stride1D.Dense> Rent(
            int deviceIndex,
            int length)
        {
            CudaAccelerator accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            lock (Sync)
            {
                var key = (accelerator, length);
                if (Buffers.TryGetValue(key, out var bucket)
                    && bucket.Count > 0)
                {
                    MemoryBuffer1D<int, Stride1D.Dense> buffer = bucket.Pop();
                    PooledBuffers.Remove(buffer);
                    return buffer;
                }
            }
            return accelerator.Allocate1D<int>(length);
        }

        internal static void Return(
            CudaAccelerator accelerator,
            MemoryBuffer1D<int, Stride1D.Dense> buffer)
        {
            lock (Sync)
            {
                if (!PooledBuffers.Add(buffer))
                    return;
                var key = (accelerator, checked((int)buffer.Length));
                if (!Buffers.TryGetValue(key, out var bucket))
                {
                    bucket = [];
                    Buffers[key] = bucket;
                }
                if (bucket.Count < MaximumBuffersPerSize)
                {
                    bucket.Push(buffer);
                    return;
                }
                PooledBuffers.Remove(buffer);
            }
            buffer.Dispose();
        }

        internal static void Clear(CudaAccelerator accelerator)
        {
            var dispose = new List<MemoryBuffer1D<int, Stride1D.Dense>>();
            lock (Sync)
            {
                var keys = Buffers.Keys
                    .Where(key => ReferenceEquals(key.Accelerator, accelerator))
                    .ToArray();
                foreach (var key in keys)
                {
                    foreach (MemoryBuffer1D<int, Stride1D.Dense> buffer
                        in Buffers[key])
                    {
                        PooledBuffers.Remove(buffer);
                        dispose.Add(buffer);
                    }
                    Buffers.Remove(key);
                }
            }
            foreach (MemoryBuffer1D<int, Stride1D.Dense> buffer in dispose)
                buffer.Dispose();
        }
    }

}
