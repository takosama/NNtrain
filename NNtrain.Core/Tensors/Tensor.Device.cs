
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
    private readonly Dictionary<int, BFloat16GradientDeviceBuffer>
        _cudaBFloat16GradientBuffers = [];
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

    internal NativeCudaBuffer<float> EnsureCudaFloat32Buffer(
        int deviceIndex = -1)
    {
        if (DType == TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "BFloat16 tensors must use their physical 16-bit CUDA buffer; " +
                "implicit expansion to a float32 device buffer is forbidden.");
        }
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaDevice accelerator =
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

    internal NativeCudaBuffer<float> EnsureCudaMasterFloat32Buffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        if (DType == TensorDType.Float32)
            return EnsureCudaFloat32Buffer(resolvedDeviceIndex);
        NativeCudaDevice accelerator =
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

    internal NativeCudaBuffer<float> EnsureCudaGradientBuffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        if (!_hostGradientCurrent
            && !_cudaGradientBuffers.ContainsKey(resolvedDeviceIndex)
            && !_cudaBFloat16GradientBuffers.ContainsKey(resolvedDeviceIndex))
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
                NativeCudaBuffer<float> gradientBuffer;
                if (_cudaBFloat16GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out BFloat16GradientDeviceBuffer? encoded)
                    && encoded.Version == _gradientVersion)
                {
                    gradientBuffer = CudaFloatBufferPool.Rent(
                        resolvedDeviceIndex, Numel);
                    CudaTensorNative.DecodeBFloat16(
                        resolvedDeviceIndex,
                        encoded.Buffer.NativePtr,
                        gradientBuffer.NativePtr,
                        Numel);
                }
                else if (_grad.Length == 0)
                {
                    gradientBuffer = CudaFloatBufferPool.Rent(
                        resolvedDeviceIndex, Numel);
                    gradientBuffer.MemSetToZero();
                }
                else
                {
                    gradientBuffer = CudaFloatBufferPool.Rent(
                        resolvedDeviceIndex, Numel);
                    gradientBuffer.CopyFromCPU(_grad);
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

    internal void BindCudaGradientArena(
        int deviceIndex,
        NativeCudaBuffer<float> slice)
    {
        ArgumentNullException.ThrowIfNull(slice);
        if (slice.Device.Index != deviceIndex || slice.Length != Numel
            || slice.Arena is null)
        {
            throw new ArgumentException(
                "Gradient arena slice must match the tensor and CUDA device.",
                nameof(slice));
        }
        lock (_deviceSync)
        {
            if (_cudaGradientBuffers.TryGetValue(
                deviceIndex, out GradientDeviceBuffer? current))
            {
                if (ReferenceEquals(current.Buffer.Arena, slice.Arena)
                    && current.Buffer.NativePtr == slice.NativePtr)
                {
                    slice.Dispose();
                    return;
                }
                current.Dispose();
            }
            _cudaGradientBuffers[deviceIndex] = new GradientDeviceBuffer(
                slice, _gradientVersion, deviceIndex);
            _hostGradientCurrent = true;
        }
    }

    internal void UnbindCudaGradientArena(
        int deviceIndex,
        NativeCudaArena<float> arena)
    {
        lock (_deviceSync)
        {
            if (!_cudaGradientBuffers.TryGetValue(
                    deviceIndex, out GradientDeviceBuffer? current)
                || !ReferenceEquals(current.Buffer.Arena, arena))
            {
                return;
            }
            _cudaGradientBuffers.Remove(deviceIndex);
            current.Dispose();
            _hostGradientCurrent = true;
        }
    }

    internal NativeCudaArena<float>? GetCudaGradientArena(
        int deviceIndex)
    {
        lock (_deviceSync)
        {
            return _cudaGradientBuffers.TryGetValue(
                deviceIndex, out GradientDeviceBuffer? buffer)
                ? buffer.Buffer.Arena
                : null;
        }
    }

    internal NativeCudaBuffer<float> EnsureCudaStagingBuffer(
        int deviceIndex)
    {
        NativeCudaDevice accelerator =
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
            if (_cudaBFloat16GradientBuffers.Remove(
                resolvedDeviceIndex,
                out BFloat16GradientDeviceBuffer? encoded))
            {
                encoded.ReturnToPool();
            }
            unchecked
            {
                _gradientVersion++;
            }
            buffer.Buffer.MarkGradientStorageDirty();
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
        NativeCudaBuffer<float> buffer =
            EnsureCudaGradientBuffer(resolvedDeviceIndex);
        buffer.CopyFromCPU(values);
        MarkCudaGradientMutated(resolvedDeviceIndex);
    }

    internal void AdoptCudaFloat32Buffer(
        NativeCudaBuffer<float> buffer,
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

    internal static NativeCudaBuffer<float> RentCudaFloatBuffer(
        int deviceIndex,
        int length)
        => CudaFloatBufferPool.Rent(deviceIndex, length);

    internal static void ReturnCudaFloatBuffer(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<float> buffer)
        => CudaFloatBufferPool.Return(accelerator, buffer);

    internal static NativeCudaBuffer<int> RentCudaIntBuffer(
        int deviceIndex,
        int[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        NativeCudaBuffer<int> buffer =
            CudaIntBufferPool.Rent(deviceIndex, values.Length);
        CudaIntBufferPool.Upload(deviceIndex, buffer, values);
        return buffer;
    }

    internal static void ReturnCudaIntBuffer(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<int> buffer)
        => CudaIntBufferPool.Return(accelerator, buffer);

    internal static void ClearCudaFloatBufferPool(int deviceIndex)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        CudaFloatBufferPool.Clear(accelerator);
        CudaBFloat16BufferPool.Clear(accelerator);
        CudaIntBufferPool.Clear(accelerator);
    }

    private static Tensor FromCudaResult(
        NativeCudaBuffer<float> buffer,
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
        NativeCudaBuffer<ushort> buffer,
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
        if (_grad.Length == 0)
            _grad = new float[Numel];
        if (buffer is not null)
        {
            buffer.Buffer.CopyToCPU(_grad);
        }
        else
        {
            if (!_cudaBFloat16GradientBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out BFloat16GradientDeviceBuffer? encoded))
            {
                encoded = _cudaBFloat16GradientBuffers.Values.FirstOrDefault(
                    candidate => candidate.Version == _gradientVersion);
            }
            if (encoded is null)
                return;
            var encodedHost = new ushort[Numel];
            encoded.Buffer.CopyToCPU(encodedHost);
            TensorStorageCodec.DecodeBFloat16(encodedHost, _grad);
        }
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
                buffer.Buffer.ClearGradientStorage();
                buffer.Version = _gradientVersion;
            }
            foreach (BFloat16GradientDeviceBuffer buffer
                in _cudaBFloat16GradientBuffers.Values)
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
            foreach (BFloat16GradientDeviceBuffer buffer
                in _cudaBFloat16GradientBuffers.Values)
            {
                buffer.Dispose();
            }
            foreach (DeviceBuffer buffer in _cudaStagingBuffers.Values)
                buffer.Dispose();
            _cudaBuffers.Clear();
            _cudaBFloat16Buffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaBFloat16GradientBuffers.Clear();
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
            foreach (BFloat16GradientDeviceBuffer buffer
                in _cudaBFloat16GradientBuffers.Values)
            {
                buffer.ReturnToPool();
            }
            foreach (DeviceBuffer buffer in _cudaStagingBuffers.Values)
                buffer.ReturnToPool();
            _cudaBuffers.Clear();
            _cudaBFloat16Buffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaBFloat16GradientBuffers.Clear();
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
            foreach (BFloat16GradientDeviceBuffer buffer
                in _cudaBFloat16GradientBuffers.Values)
            {
                buffer.ReturnToPool();
            }
            foreach (DeviceBuffer buffer in _cudaStagingBuffers.Values)
                buffer.ReturnToPool();
            _cudaBuffers.Clear();
            _cudaBFloat16Buffers.Clear();
            _cudaMasterBuffers.Clear();
            _cudaGradientBuffers.Clear();
            _cudaBFloat16GradientBuffers.Clear();
            _cudaStagingBuffers.Clear();
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _device = TensorDevice.Cpu;
        }
    }

    private sealed class DeviceBuffer : IDisposable
    {
        private int _disposed;
        private readonly NativeCudaDevice _accelerator;

        internal DeviceBuffer(
            NativeCudaBuffer<float> buffer,
            int deviceIndex)
        {
            Buffer = buffer;
            _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        }

        internal NativeCudaBuffer<float> Buffer { get; }

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
        NativeCudaBuffer<float> buffer,
        long version,
        int deviceIndex) : IDisposable
    {
        private int _disposed;
        private readonly NativeCudaDevice _accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        internal NativeCudaBuffer<float> Buffer { get; } = buffer;
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
        private const int MaximumBuffersPerSize = 64;
        private static readonly System.Collections.Concurrent
            .ConcurrentDictionary<NativeCudaDevice, PoolState> Pools = new();

        private sealed class PoolState
        {
            internal object Sync { get; } = new();
            internal Dictionary<int, Stack<NativeCudaBuffer<float>>> Buffers
                { get; } = [];
            internal HashSet<NativeCudaBuffer<float>> PooledBuffers
                { get; } = [];
        }

        internal static NativeCudaBuffer<float> Rent(
            int deviceIndex,
            int length)
        {
            NativeCudaDevice accelerator =
                NNtrain.ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                if (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<float> buffer = bucket.Pop();
                    state.PooledBuffers.Remove(buffer);
                    CudaTransientBufferBudget.Release(
                        accelerator,
                        checked((long)length * sizeof(float)));
                    return buffer;
                }
            }

            try
            {
                return accelerator.Allocate1D<float>(length);
            }
            catch (NativeCudaException exception) when (IsOutOfMemory(exception))
            {
                // A shape change can leave an otherwise valid but unusable
                // set of cached blocks. Flush only transient storage on this
                // adapter, then retry once. A genuine live-set OOM still
                // propagates from the second allocation.
                accelerator.Synchronize();
                Clear(accelerator);
                CudaBFloat16BufferPool.Clear(accelerator);
                CudaIntBufferPool.Clear(accelerator);
                return accelerator.Allocate1D<float>(length);
            }
        }

        internal static void Return(
            NativeCudaDevice accelerator,
            NativeCudaBuffer<float> buffer)
        {
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(float));
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                // A context and its adopted result tensor can share storage.
                // Never place the same native allocation in a bucket twice.
                if (!state.PooledBuffers.Add(buffer))
                    return;
                if (!state.Buffers.TryGetValue(length, out var bucket))
                {
                    bucket = [];
                    state.Buffers[length] = bucket;
                }

                if (bucket.Count < MaximumBuffersPerSize
                    && CudaTransientBufferBudget.TryReserve(
                        accelerator, bytes))
                {
                    bucket.Push(buffer);
                    return;
                }
                state.PooledBuffers.Remove(buffer);
            }

            buffer.Dispose();
        }

        private static bool IsOutOfMemory(NativeCudaException exception)
            => exception.Status == 2
                || exception.Message.Contains(
                    "out of memory",
                    StringComparison.OrdinalIgnoreCase);

        internal static void Clear(NativeCudaDevice accelerator)
        {
            var dispose = new List<NativeCudaBuffer<float>>();
            if (!Pools.TryGetValue(accelerator, out PoolState? state))
                return;
            long releasedBytes = 0;
            lock (state.Sync)
            {
                foreach ((int length, Stack<NativeCudaBuffer<float>> bucket)
                    in state.Buffers)
                {
                    while (bucket.Count > 0)
                    {
                        var buffer = bucket.Pop();
                        state.PooledBuffers.Remove(buffer);
                        releasedBytes += checked((long)length * sizeof(float));
                        dispose.Add(buffer);
                    }
                }
                state.Buffers.Clear();
            }
            CudaTransientBufferBudget.Release(accelerator, releasedBytes);
            foreach (var buffer in dispose)
                buffer.Dispose();
        }
    }

    /// <summary>
    /// One high-water budget shared by all transient element types. Fixed
    /// training shapes stay resident after warmup; a real OOM flushes the
    /// cache and retries. Accounting and locks are per device, so two CUDA
    /// dispatch threads never serialize on allocator bookkeeping.
    /// </summary>
    private static class CudaTransientBufferBudget
    {
        // Leave room for resident weights, master weights, gradient arenas,
        // optimizer moments/workspaces, and the Windows display allocation.
        // This budget counts only idle transient buffers, so 45% still keeps
        // the full activation working set reusable without crowding out the
        // next step's persistent allocations.
        private const int CacheMemoryPercent = 45;
        private static readonly System.Collections.Concurrent
            .ConcurrentDictionary<NativeCudaDevice, BudgetState> States = new();

        private sealed class BudgetState(long maximumBytes)
        {
            internal object Sync { get; } = new();
            internal long MaximumBytes { get; } = maximumBytes;
            internal long CachedBytes;
        }

        internal static bool TryReserve(
            NativeCudaDevice accelerator,
            long bytes)
        {
            BudgetState state = States.GetOrAdd(
                accelerator,
                static device => new BudgetState(checked(
                    device.MemorySize * CacheMemoryPercent / 100)));
            lock (state.Sync)
            {
                if (bytes > state.MaximumBytes - state.CachedBytes)
                    return false;
                state.CachedBytes += bytes;
                return true;
            }
        }

        internal static void Release(
            NativeCudaDevice accelerator,
            long bytes)
        {
            if (bytes <= 0
                || !States.TryGetValue(accelerator, out BudgetState? state))
            {
                return;
            }
            lock (state.Sync)
                state.CachedBytes = Math.Max(0, state.CachedBytes - bytes);
        }
    }

    private static class CudaIntBufferPool
    {
        private const int MaximumBuffersPerSize = 64;
        private static readonly System.Collections.Concurrent
            .ConcurrentDictionary<NativeCudaDevice, PoolState> Pools = new();

        private sealed class PoolState
        {
            internal object Sync { get; } = new();
            internal Dictionary<int, Stack<NativeCudaBuffer<int>>> Buffers
                { get; } = [];
            internal HashSet<NativeCudaBuffer<int>> PooledBuffers
                { get; } = [];
            internal Dictionary<NativeCudaBuffer<int>,
                NativeCudaPinnedUpload<int>> Staging { get; } = [];
        }

        internal static NativeCudaBuffer<int> Rent(
            int deviceIndex,
            int length)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                if (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<int> buffer = bucket.Pop();
                    state.PooledBuffers.Remove(buffer);
                    CudaTransientBufferBudget.Release(
                        accelerator,
                        checked((long)length * sizeof(int)));
                    return buffer;
                }
            }
            return accelerator.Allocate1D<int>(length);
        }

        internal static void Upload(
            int deviceIndex,
            NativeCudaBuffer<int> buffer,
            ReadOnlySpan<int> values)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            NativeCudaPinnedUpload<int> staging;
            lock (state.Sync)
            {
                if (!state.Staging.TryGetValue(buffer, out staging!))
                {
                    staging = new NativeCudaPinnedUpload<int>(
                        deviceIndex, values.Length);
                    state.Staging.Add(buffer, staging);
                }
            }
            staging.Upload(values, buffer, accelerator.DefaultStream);
        }

        internal static void Return(
            NativeCudaDevice accelerator,
            NativeCudaBuffer<int> buffer)
        {
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(int));
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            NativeCudaPinnedUpload<int>? releaseStaging = null;
            lock (state.Sync)
            {
                if (!state.PooledBuffers.Add(buffer))
                    return;
                if (!state.Buffers.TryGetValue(length, out var bucket))
                {
                    bucket = [];
                    state.Buffers[length] = bucket;
                }
                if (bucket.Count < MaximumBuffersPerSize
                    && CudaTransientBufferBudget.TryReserve(
                        accelerator, bytes))
                {
                    bucket.Push(buffer);
                    return;
                }
                state.PooledBuffers.Remove(buffer);
                if (state.Staging.Remove(buffer, out var staging))
                    releaseStaging = staging;
            }
            releaseStaging?.Dispose();
            buffer.Dispose();
        }

        internal static void Clear(NativeCudaDevice accelerator)
        {
            var dispose = new List<NativeCudaBuffer<int>>();
            var stagingToDispose = new List<NativeCudaPinnedUpload<int>>();
            if (!Pools.TryGetValue(accelerator, out PoolState? state))
                return;
            long releasedBytes = 0;
            lock (state.Sync)
            {
                foreach ((int length, Stack<NativeCudaBuffer<int>> bucket)
                    in state.Buffers)
                {
                    while (bucket.Count > 0)
                    {
                        NativeCudaBuffer<int> buffer = bucket.Pop();
                        state.PooledBuffers.Remove(buffer);
                        if (state.Staging.Remove(buffer, out var staging))
                            stagingToDispose.Add(staging);
                        releasedBytes += checked((long)length * sizeof(int));
                        dispose.Add(buffer);
                    }
                }
                state.Buffers.Clear();
            }
            CudaTransientBufferBudget.Release(accelerator, releasedBytes);
            foreach (NativeCudaPinnedUpload<int> staging in stagingToDispose)
                staging.Dispose();
            foreach (NativeCudaBuffer<int> buffer in dispose)
                buffer.Dispose();
        }
    }

}
