using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace NNtrain;

public partial class Tensor
{
    internal MemoryBuffer1D<ushort, Stride1D.Dense> EnsureCudaBFloat16Buffer(
        int deviceIndex = -1)
    {
        if (DType != TensorDType.BFloat16)
        {
            throw new InvalidOperationException(
                "A physical CUDA bfloat16 buffer requires BFloat16 dtype.");
        }
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        CudaAccelerator accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        if (!_hostDataCurrent
            && !_cudaBFloat16Buffers.ContainsKey(resolvedDeviceIndex))
        {
            SynchronizeHostFromCuda();
        }
        lock (_deviceSync)
        {
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
                    resolvedDeviceIndex);
                _cudaBFloat16Buffers[resolvedDeviceIndex] = buffer;
                _cudaBufferDataVersion = _dataVersion;
                return buffer.Buffer;
            }

            if (_cudaBufferDataVersion != _dataVersion)
            {
                var encoded = new ushort[Numel];
                TensorStorageCodec.EncodeBFloat16(DataBuffer, encoded);
                buffer.Buffer.CopyFromCPU(encoded);
                _cudaBufferDataVersion = _dataVersion;
            }
            return buffer.Buffer;
        }
    }

    internal void AdoptCudaBFloat16Buffer(
        MemoryBuffer1D<ushort, Stride1D.Dense> buffer,
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
                new BFloat16DeviceBuffer(buffer, deviceIndex);
            _cudaBufferDataVersion = _dataVersion;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    internal static MemoryBuffer1D<ushort, Stride1D.Dense>
        RentCudaBFloat16Buffer(int deviceIndex, int length)
        => CudaBFloat16BufferPool.Rent(deviceIndex, length);

    internal static void ReturnCudaBFloat16Buffer(
        CudaAccelerator accelerator,
        MemoryBuffer1D<ushort, Stride1D.Dense> buffer)
        => CudaBFloat16BufferPool.Return(accelerator, buffer);

    private sealed class BFloat16DeviceBuffer : IDisposable
    {
        private int _disposed;
        private readonly CudaAccelerator _accelerator;

        internal BFloat16DeviceBuffer(
            MemoryBuffer1D<ushort, Stride1D.Dense> buffer,
            int deviceIndex)
        {
            Buffer = buffer;
            _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        }

        internal MemoryBuffer1D<ushort, Stride1D.Dense> Buffer { get; }

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
        private const long MinimumCachedBytesPerDevice = 256L * 1024 * 1024;
        private const int CacheMemoryPercent = 12;
        private const long MinimumReservedFreeBytes = 1536L * 1024 * 1024;
        private const int ReservedFreeMemoryPercent = 25;
        private const int MaximumBuffersPerSize = 64;
        private static readonly object Sync = new();
        private static readonly Dictionary<
            (CudaAccelerator Accelerator, int Length),
            Stack<MemoryBuffer1D<ushort, Stride1D.Dense>>> Buffers = [];
        private static readonly HashSet<MemoryBuffer1D<ushort, Stride1D.Dense>>
            PooledBuffers = [];
        private static readonly Dictionary<CudaAccelerator, long> CachedBytes = [];

        internal static MemoryBuffer1D<ushort, Stride1D.Dense> Rent(
            int deviceIndex,
            int length)
        {
            CudaAccelerator accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            var dispose = new List<MemoryBuffer1D<ushort, Stride1D.Dense>>();
            lock (Sync)
            {
                var key = (accelerator, length);
                if (Buffers.TryGetValue(key, out var bucket)
                    && bucket.Count > 0)
                {
                    MemoryBuffer1D<ushort, Stride1D.Dense> buffer = bucket.Pop();
                    PooledBuffers.Remove(buffer);
                    CachedBytes[accelerator] = CachedBytes.GetValueOrDefault(accelerator)
                        - checked((long)length * sizeof(ushort));
                    return buffer;
                }

                long requestedBytes = checked((long)length * sizeof(ushort));
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
                TrimToBytes(
                    accelerator,
                    Math.Min(capacityTargetBytes, reserveTargetBytes),
                    dispose);
            }

            foreach (MemoryBuffer1D<ushort, Stride1D.Dense> buffer in dispose)
                buffer.Dispose();

            try
            {
                return accelerator.Allocate1D<ushort>(length);
            }
            catch (CudaException exception) when (IsOutOfMemory(exception))
            {
                accelerator.Synchronize();
                CudaFloatBufferPool.Clear(accelerator);
                CudaIntBufferPool.Clear(accelerator);
                Clear(accelerator);
                return accelerator.Allocate1D<ushort>(length);
            }
        }

        internal static void Return(
            CudaAccelerator accelerator,
            MemoryBuffer1D<ushort, Stride1D.Dense> buffer)
        {
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(ushort));
            lock (Sync)
            {
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
            List<MemoryBuffer1D<ushort, Stride1D.Dense>> dispose)
        {
            long cached = CachedBytes.GetValueOrDefault(accelerator);
            if (cached <= targetBytes)
                return;
            var keys = Buffers.Keys
                .Where(key => ReferenceEquals(key.Accelerator, accelerator))
                .ToArray();
            foreach (var key in keys)
            {
                Stack<MemoryBuffer1D<ushort, Stride1D.Dense>> bucket =
                    Buffers[key];
                while (bucket.Count > 0 && cached > targetBytes)
                {
                    MemoryBuffer1D<ushort, Stride1D.Dense> buffer = bucket.Pop();
                    PooledBuffers.Remove(buffer);
                    cached -= checked((long)key.Length * sizeof(ushort));
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
            var dispose = new List<MemoryBuffer1D<ushort, Stride1D.Dense>>();
            lock (Sync)
            {
                var keys = Buffers.Keys
                    .Where(key => ReferenceEquals(key.Accelerator, accelerator))
                    .ToArray();
                foreach (var key in keys)
                {
                    while (Buffers[key].Count > 0)
                    {
                        var buffer = Buffers[key].Pop();
                        PooledBuffers.Remove(buffer);
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
}
