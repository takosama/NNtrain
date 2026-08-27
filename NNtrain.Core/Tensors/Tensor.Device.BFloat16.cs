
namespace NNtrain;

public partial class Tensor
{
    internal NativeCudaBuffer<ushort> EnsureCudaBFloat16Buffer(
        int deviceIndex = -1)
    {
        if (DType != TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "A physical CUDA bfloat16 buffer requires BFloat16 dtype.");
        }
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        lock (_deviceSync)
        {
            if (!_hostDataCurrent
                && (!_cudaBFloat16Buffers.TryGetValue(
                        resolvedDeviceIndex,
                        out BFloat16DeviceBuffer? requestedBuffer)
                    || requestedBuffer.Version != _dataVersion))
            {
                SynchronizeHostFromCudaLocked(_cudaDeviceIndex);
            }
            if (!_cudaBFloat16Buffers.TryGetValue(
                resolvedDeviceIndex,
                out BFloat16DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel)
            {
                buffer?.Dispose();
                var encoded = new ushort[Numel];
                TensorStorageCodec.EncodeBFloat16(DataBuffer, encoded);
                buffer = new BFloat16DeviceBuffer(
                    accelerator.Allocate1D(encoded),
                    _dataVersion,
                    resolvedDeviceIndex);
                _cudaBFloat16Buffers[resolvedDeviceIndex] = buffer;
                return buffer.Buffer;
            }

            if (buffer.Version != _dataVersion)
            {
                var encoded = new ushort[Numel];
                TensorStorageCodec.EncodeBFloat16(DataBuffer, encoded);
                buffer.Buffer.CopyFromCPU(encoded);
                buffer.Version = _dataVersion;
            }
            return buffer.Buffer;
        }
    }

    internal void AdoptCudaBFloat16Buffer(
        NativeCudaBuffer<ushort> buffer,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (DType != TensorDType.BFloat16)
            throw new InvalidOperationException("Tensor dtype must be BFloat16.");
        if (buffer.Length != Numel)
            throw new ArgumentException("CUDA buffer length must match the tensor.", nameof(buffer));
        lock (_deviceSync)
        {
            if (_cudaBFloat16Buffers.Remove(
                deviceIndex,
                out BFloat16DeviceBuffer? previous))
            {
                previous.Dispose();
            }
            _cudaBFloat16Buffers[deviceIndex] =
                new BFloat16DeviceBuffer(buffer, _dataVersion, deviceIndex);
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    internal void AdoptCudaBFloat16GradientBuffer(
        NativeCudaBuffer<ushort> buffer,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        if (buffer.Device.Index != deviceIndex || buffer.Length != Numel)
        {
            throw new ArgumentException(
                "CUDA BF16 gradient must match the tensor and device.",
                nameof(buffer));
        }
        lock (_deviceSync)
        {
            if (_cudaGradientBuffers.Remove(
                deviceIndex, out GradientDeviceBuffer? previousFloat))
            {
                previousFloat.Dispose();
            }
            if (_cudaBFloat16GradientBuffers.Remove(
                deviceIndex,
                out BFloat16GradientDeviceBuffer? previousBFloat16))
            {
                previousBFloat16.ReturnToPool();
            }
            unchecked
            {
                _gradientVersion++;
            }
            _cudaBFloat16GradientBuffers[deviceIndex] =
                new BFloat16GradientDeviceBuffer(
                    buffer, _gradientVersion, deviceIndex);
            _hostGradientCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    internal bool TryGetCudaBFloat16GradientBuffer(
        int deviceIndex,
        out NativeCudaBuffer<ushort>? buffer)
    {
        lock (_deviceSync)
        {
            if (_cudaBFloat16GradientBuffers.TryGetValue(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? encoded)
                && encoded.Version == _gradientVersion)
            {
                buffer = encoded.Buffer;
                return true;
            }
            buffer = null;
            return false;
        }
    }

    internal static NativeCudaBuffer<ushort>
        RentCudaBFloat16Buffer(int deviceIndex, int length)
        => CudaBFloat16BufferPool.Rent(deviceIndex, length);

    internal static void ReturnCudaBFloat16Buffer(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> buffer)
        => CudaBFloat16BufferPool.Return(accelerator, buffer);

    private sealed class BFloat16DeviceBuffer : IDisposable
    {
        private int _disposed;
        private readonly NativeCudaDevice _accelerator;

        internal BFloat16DeviceBuffer(
            NativeCudaBuffer<ushort> buffer,
            long version,
            int deviceIndex)
        {
            Buffer = buffer;
            Version = version;
            _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        }

        internal NativeCudaBuffer<ushort> Buffer { get; }
        internal long Version { get; set; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                Buffer.Dispose();
        }

        internal void ReturnToPool()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                CudaBFloat16BufferPool.Return(_accelerator, Buffer);
        }
    }

    private sealed class BFloat16GradientDeviceBuffer(
        NativeCudaBuffer<ushort> buffer,
        long version,
        int deviceIndex) : IDisposable
    {
        private int _disposed;
        private readonly NativeCudaDevice _accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        internal NativeCudaBuffer<ushort> Buffer { get; } = buffer;
        internal long Version { get; set; } = version;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                Buffer.Dispose();
        }

        internal void ReturnToPool()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0)
                CudaBFloat16BufferPool.Return(_accelerator, Buffer);
        }
    }

    private static class CudaBFloat16BufferPool
    {
        private const int MaximumBuffersPerSize = 128;
        private static readonly System.Collections.Concurrent
            .ConcurrentDictionary<NativeCudaDevice, PoolState> Pools = new();

        private sealed class PoolState
        {
            internal object Sync { get; } = new();
            internal Dictionary<int, Stack<NativeCudaBuffer<ushort>>> Buffers
                { get; } = [];
            internal HashSet<NativeCudaBuffer<ushort>> PooledBuffers
                { get; } = [];
        }

        internal static NativeCudaBuffer<ushort> Rent(
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
                    NativeCudaBuffer<ushort> buffer = bucket.Pop();
                    state.PooledBuffers.Remove(buffer);
                    CudaTransientBufferBudget.Release(
                        accelerator,
                        checked((long)length * sizeof(ushort)));
                    return buffer;
                }
            }

            try
            {
                return accelerator.Allocate1D<ushort>(length);
            }
            catch (NativeCudaException exception) when (IsOutOfMemory(exception))
            {
                accelerator.Synchronize();
                CudaFloatBufferPool.Clear(accelerator);
                CudaIntBufferPool.Clear(accelerator);
                Clear(accelerator);
                return accelerator.Allocate1D<ushort>(length);
            }
        }

        internal static void Return(
            NativeCudaDevice accelerator,
            NativeCudaBuffer<ushort> buffer)
        {
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(ushort));
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
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
            var dispose = new List<NativeCudaBuffer<ushort>>();
            if (!Pools.TryGetValue(accelerator, out PoolState? state))
                return;
            long releasedBytes = 0;
            lock (state.Sync)
            {
                foreach ((int length, Stack<NativeCudaBuffer<ushort>> bucket)
                    in state.Buffers)
                {
                    while (bucket.Count > 0)
                    {
                        var buffer = bucket.Pop();
                        state.PooledBuffers.Remove(buffer);
                        releasedBytes += checked(
                            (long)length * sizeof(ushort));
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
}
