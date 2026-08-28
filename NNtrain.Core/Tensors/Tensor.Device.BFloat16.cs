
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
                    || requestedBuffer.Version != _dataVersion
                    || !IsReplicaUsableInCurrentSession(
                        requestedBuffer.Buffer)))
            {
                SynchronizeHostFromCudaLocked(_cudaDeviceIndex);
            }
            if (!_cudaBFloat16Buffers.TryGetValue(
                resolvedDeviceIndex,
                out BFloat16DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
            {
                buffer?.Dispose();
                var encoded = new ushort[Numel];
                TensorStorageCodec.EncodeBFloat16(DataBuffer, encoded);
                buffer = new BFloat16DeviceBuffer(
                    accelerator.Allocate1D(encoded),
                    _dataVersion,
                    resolvedDeviceIndex);
                _cudaBFloat16Buffers[resolvedDeviceIndex] = buffer;
                RegisterSessionReplicaLocked(buffer.Buffer);
                return buffer.Buffer;
            }

            if (buffer.Version != _dataVersion)
            {
                var encoded = new ushort[Numel];
                TensorStorageCodec.EncodeBFloat16(DataBuffer, encoded);
                buffer.Buffer.CopyFromCPU(encoded);
                buffer.Version = _dataVersion;
            }
            RegisterSessionReplicaLocked(buffer.Buffer);
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
            RegisterSessionReplicaLocked(buffer);
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
            RegisterSessionReplicaLocked(buffer);
            _gradientAuthority = GradientStorageAuthority.CudaBFloat16;
            _gradientAuthorityDeviceIndex = deviceIndex;
            _hostGradientCurrent = false;
            MarkCudaGradientLocalLocked(deviceIndex);
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    /// <summary>
    /// Binds a reducer-owned BF16 arena slice as this tensor's gradient
    /// storage. Backward kernels borrow the slice directly, so a completed
    /// bucket is already packed and requires no per-leaf copy or conversion.
    /// </summary>
    internal void BindCudaBFloat16GradientArena(
        int deviceIndex,
        NativeCudaBuffer<ushort> slice)
    {
        ArgumentNullException.ThrowIfNull(slice);
        if (DType != TensorDType.BFloat16
            || slice.Device.Index != deviceIndex
            || slice.Length != Numel
            || slice.Arena is null)
        {
            throw new ArgumentException(
                "BF16 gradient arena slice must match the tensor and CUDA " +
                "device.",
                nameof(slice));
        }
        lock (_deviceSync)
        {
            if (_cudaBFloat16GradientBuffers.TryGetValue(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? current)
                && ReferenceEquals(current.Buffer.Arena, slice.Arena)
                && current.Buffer.NativePtr == slice.NativePtr)
            {
                slice.Dispose();
                RegisterSessionReplicaLocked(current.Buffer);
                return;
            }
            if (_cudaGradientBuffers.Remove(
                    deviceIndex,
                    out GradientDeviceBuffer? previousFloat))
            {
                previousFloat.Dispose();
            }
            if (_cudaBFloat16GradientBuffers.Remove(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? previousBFloat16))
            {
                previousBFloat16.ReturnToPool();
            }
            _cudaBFloat16GradientBuffers[deviceIndex] =
                new BFloat16GradientDeviceBuffer(
                    slice,
                    _gradientVersion,
                    deviceIndex,
                    ownsBuffer: true,
                    isArenaSlice: true);
            RegisterSessionReplicaLocked(slice);
        }
    }

    internal NativeCudaArena<ushort>? GetCudaBFloat16GradientArena(
        int deviceIndex)
    {
        lock (_deviceSync)
        {
            return _cudaBFloat16GradientBuffers.TryGetValue(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? buffer)
                ? buffer.Buffer.Arena
                : null;
        }
    }

    internal bool HasAuthoritativeCudaBFloat16Gradient
    {
        get
        {
            lock (_deviceSync)
            {
                return _gradientAuthority
                        == GradientStorageAuthority.CudaBFloat16
                    && _cudaBFloat16GradientBuffers.Values.Any(
                        buffer => buffer.Version == _gradientVersion
                            && IsReplicaUsableInCurrentSession(
                                buffer.Buffer));
            }
        }
    }

    /// <summary>
    /// Publishes an in-place mutation of an already-owned BF16 gradient
    /// replica. This is the BF16 counterpart of
    /// <see cref="MarkCudaGradientMutated(int)"/> and is required after a
    /// zero_grad/reuse cycle, where the physical buffer survives but its
    /// previous coherence generation has been consumed.
    /// </summary>
    internal void MarkCudaBFloat16GradientMutated(int deviceIndex)
    {
        lock (_deviceSync)
        {
            ThrowIfReducerOwnedGradientZeroPendingLocked(deviceIndex);
            if (!_cudaBFloat16GradientBuffers.TryGetValue(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? buffer)
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
            {
                throw new InvalidOperationException(
                    "Cannot mark a CUDA BF16 gradient modified before " +
                    "allocating it.");
            }
            if (_cudaGradientBuffers.Remove(
                    deviceIndex,
                    out GradientDeviceBuffer? decoded))
            {
                decoded.Dispose();
            }
            // Acquire only returns an existing BF16 target when this buffer is
            // already bound to the current logical gradient generation. The
            // write publishes bytes for that generation; it must not create a
            // second generation per data-parallel lane. Incrementing here
            // made device 0 immediately stale device 1's reducer-owned arena
            // slice, so device 1 replaced the slice with a pooled buffer and
            // the reducer lost its binding. zero_grad (or a fresh adoption)
            // is the operation that advances _gradientVersion.
            buffer.Buffer.MarkGradientStorageDirty();
            buffer.Version = _gradientVersion;
            _gradientAuthority = GradientStorageAuthority.CudaBFloat16;
            _gradientAuthorityDeviceIndex = deviceIndex;
            _hostGradientCurrent = false;
            MarkCudaGradientLocalLocked(deviceIndex);
        }
    }

    internal void UnbindCudaBFloat16GradientArena(
        int deviceIndex,
        NativeCudaArena<ushort> arena)
    {
        ArgumentNullException.ThrowIfNull(arena);
        lock (_deviceSync)
        {
            if (!_cudaBFloat16GradientBuffers.TryGetValue(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? current)
                || !ReferenceEquals(current.Buffer.Arena, arena))
            {
                return;
            }
            SynchronizeHostGradientFromCudaLocked(deviceIndex);
            _cudaBFloat16GradientBuffers.Remove(deviceIndex);
            current.Dispose();
        }
    }

    internal void MarkCudaBFloat16GradientsSynchronized(
        IReadOnlyList<int> deviceIndices)
        => MarkCudaBFloat16GradientsSynchronized(
            deviceIndices,
            PreserveOrCreateGradientReductionStamp(deviceIndices));

    internal void MarkCudaBFloat16GradientsSynchronized(
        IReadOnlyList<int> deviceIndices,
        CudaGradientReductionStamp reductionStamp)
    {
        ValidateCudaGradientDeviceSet(deviceIndices);
        if (!reductionStamp.IsValid)
            throw new ArgumentException("Reduction stamp must be valid.");
        lock (_deviceSync)
        {
            unchecked
            {
                _gradientVersion++;
            }
            foreach (int deviceIndex in deviceIndices)
            {
                if (!_cudaBFloat16GradientBuffers.TryGetValue(
                        deviceIndex,
                        out BFloat16GradientDeviceBuffer? buffer))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no BF16 gradient " +
                        "replica to publish.");
                }
                buffer.Version = _gradientVersion;
            }
            _gradientAuthority = GradientStorageAuthority.CudaBFloat16;
            _gradientAuthorityDeviceIndex = deviceIndices.Count == 0
                ? -1
                : deviceIndices[0];
            _hostGradientCurrent = false;
            CommitCudaGradientReductionLocked(
                deviceIndices,
                reductionStamp);
        }
    }

    /// <summary>
    /// Publishes a BF16 gradient that was written in-place over this tensor's
    /// one-shot data buffer. The data replica remains the sole owner; the
    /// gradient entry is a non-owning alias used only by the next autograd
    /// node before graph release.
    /// </summary>
    internal void MarkCudaBFloat16DataAsGradientInPlace(
        NativeCudaBuffer<ushort> buffer,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(buffer);
        lock (_deviceSync)
        {
            if (!_allowInPlaceBFloat16Gradient
                || !_cudaBFloat16Buffers.TryGetValue(
                    deviceIndex,
                    out BFloat16DeviceBuffer? data)
                || !ReferenceEquals(data.Buffer, buffer)
                || data.Version != _dataVersion)
            {
                throw new InvalidOperationException(
                    "The BF16 data replica is not eligible for an in-place " +
                    "gradient publication.");
            }
            if (_cudaGradientBuffers.Remove(
                deviceIndex,
                out GradientDeviceBuffer? previousFloat))
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
                    buffer,
                    _gradientVersion,
                    deviceIndex,
                    ownsBuffer: false);
            RegisterSessionReplicaLocked(buffer);
            _gradientAuthority = GradientStorageAuthority.CudaBFloat16;
            _gradientAuthorityDeviceIndex = deviceIndex;
            _hostGradientCurrent = false;
            MarkCudaGradientLocalLocked(deviceIndex);
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
                && encoded.Version == _gradientVersion
                && IsReplicaUsableInCurrentSession(encoded.Buffer))
            {
                RegisterSessionReplicaLocked(encoded.Buffer);
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

    internal sealed class BFloat16DeviceBuffer : IDisposable
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

    internal sealed class BFloat16GradientDeviceBuffer : IDisposable
    {
        private int _disposed;
        private readonly NativeCudaDevice _accelerator;
        private readonly bool _ownsBuffer;
        private readonly bool _isArenaSlice;

        internal BFloat16GradientDeviceBuffer(
            NativeCudaBuffer<ushort> buffer,
            long version,
            int deviceIndex,
            bool ownsBuffer = true,
            bool isArenaSlice = false)
        {
            Buffer = buffer;
            Version = version;
            _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            _ownsBuffer = ownsBuffer;
            _isArenaSlice = isArenaSlice;
        }

        internal NativeCudaBuffer<ushort> Buffer { get; }
        internal long Version { get; set; }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && _ownsBuffer)
                Buffer.Dispose();
        }

        internal void ReturnToPool()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0 || !_ownsBuffer)
                return;
            if (_isArenaSlice)
                Buffer.Dispose();
            else
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
            if (TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out _))
            {
                return accelerator.Allocate1D<ushort>(
                    length,
                    NNtrain.Cuda.Memory.CudaMemoryKind.Transient);
            }
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                if (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<ushort> buffer = bucket.Pop();
                    if (bucket.Count == 0)
                        state.Buffers.Remove(length);
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
            if (buffer.IsLaneManagedReusable)
            {
                buffer.Dispose();
                return;
            }
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(ushort));
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                if (!state.PooledBuffers.Add(buffer))
                    return;
                state.Buffers.TryGetValue(length, out var bucket);
                if ((bucket is null
                        || bucket.Count < MaximumBuffersPerSize)
                    && CudaTransientBufferBudget.TryReserve(
                        accelerator, bytes))
                {
                    if (bucket is null)
                    {
                        bucket = [];
                        state.Buffers.Add(length, bucket);
                    }
                    bucket.Push(buffer);
                    return;
                }
                state.PooledBuffers.Remove(buffer);
                if (bucket is { Count: 0 })
                    state.Buffers.Remove(length);
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
            DisposeTransientBuffersAll(
                dispose,
                "CUDA bfloat16 transient buffer cleanup failed.");
        }
    }
}
