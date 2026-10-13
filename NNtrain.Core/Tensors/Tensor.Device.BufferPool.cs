using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
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
        if (CudaGraphBatchInputs.TryBorrow(
                deviceIndex,
                values,
                out NativeCudaBuffer<int> fixedBuffer))
        {
            return fixedBuffer;
        }
        NativeCudaBuffer<int> buffer =
            CudaIntBufferPool.Rent(deviceIndex, values.Length);
        using (DeviceTransferGuard.AllowBatchHostToDevice())
            CudaIntBufferPool.Upload(deviceIndex, buffer, values);
        return buffer;
    }

    internal static NativeCudaBuffer<int> RentCudaIntBuffer(
        int deviceIndex,
        int length)
        => CudaIntBufferPool.Rent(deviceIndex, length);

    internal static void ReturnCudaIntBuffer(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<int> buffer)
    {
        ArgumentNullException.ThrowIfNull(accelerator);
        ArgumentNullException.ThrowIfNull(buffer);
        if (CudaGraphBatchInputs.TryReturn(accelerator.Index, buffer))
            return;
        CudaIntBufferPool.Return(accelerator, buffer);
    }

    internal static BoundedUploadSlotCacheTelemetry
        GetCudaIntUploadSlotTelemetry(int deviceIndex)
        => CudaIntBufferPool.GetLaneTelemetry(deviceIndex);

    internal static void ClearCudaFloatBufferPool(int deviceIndex)
    {
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        List<Exception>? failures = null;
        TryTransientPoolCleanup(
            () => CudaFloatBufferPool.Clear(accelerator),
            ref failures);
        TryTransientPoolCleanup(
            () => CudaBFloat16BufferPool.Clear(accelerator),
            ref failures);
        TryTransientPoolCleanup(
            () => CudaIntBufferPool.Clear(accelerator),
            ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                $"CUDA transient pool cleanup failed on device " +
                $"{deviceIndex}.",
                failures);
        }
    }

    private static void TryTransientPoolCleanup(
        Action cleanup,
        ref List<Exception>? failures)
    {
        try
        {
            cleanup();
        }
        catch (AggregateException aggregate)
        {
            (failures ??= []).AddRange(
                aggregate.Flatten().InnerExceptions);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    private static void DisposeTransientBuffersAll<T>(
        IEnumerable<NativeCudaBuffer<T>> buffers,
        string failureMessage)
        where T : unmanaged
    {
        List<Exception>? failures = null;
        foreach (NativeCudaBuffer<T> buffer in buffers)
        {
            try
            {
                buffer.Dispose();
            }
            catch (AggregateException aggregate)
            {
                (failures ??= []).AddRange(
                    aggregate.Flatten().InnerExceptions);
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (failures is not null)
            throw new AggregateException(failureMessage, failures);
    }

    /// <summary>
    /// Reuses transient CUDA storage between autograd graphs.  The pool is
    /// deliberately bounded: cached buffers are an optimization, not a claim
    /// on all available VRAM.
    /// </summary>
    private static class CudaFloatBufferPool
    {
        // A 16-layer transformer can have more than 64 simultaneously-live
        // buffers with the same flattened shape.  Keep the observed fixed
        // shape high-water mark resident; the shared byte budget below still
        // provides the actual memory bound.
        private const int MaximumBuffersPerSize = 128;
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
            if (TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out _))
            {
                return accelerator.Allocate1D<float>(
                    length,
                    NNtrain.Cuda.Memory.CudaMemoryKind.Transient);
            }
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                while (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<float> buffer = bucket.Pop();
                    if (bucket.Count == 0)
                        state.Buffers.Remove(length);
                    state.PooledBuffers.Remove(buffer);
                    CudaTransientBufferBudget.Release(
                        accelerator,
                        checked((long)length * sizeof(float)));
                    if (buffer.IsAlive && buffer.SessionGeneration == 0 && buffer.Arena is null)
                        return buffer;
                    buffer.Dispose();
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
            // Persistent and arena-backed session buffers are session-owned
            // too: caching their wrappers past teardown creates dangling rents.
            if (!buffer.IsAlive || buffer.SessionGeneration != 0 || buffer.Arena is not null)
            {
                buffer.Dispose();
                return;
            }
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
            DisposeTransientBuffersAll(
                dispose,
                "CUDA float transient buffer cleanup failed.");
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
        // Start conservatively, then learn the fixed-shape high-water mark.
        // Returned buffers are already allocated, so retaining them does not
        // lower the free-memory value observed at that point.  Expansion is
        // allowed only while a real emergency reserve remains, and a genuine
        // allocation OOM still flushes every transient pool before retrying.
        private const int InitialCacheMemoryPercent = 45;
        private const int MaximumCacheMemoryPercent = 65;
        private const long MinimumFreeReserveBytes = 512L * 1024 * 1024;
        private static readonly System.Collections.Concurrent
            .ConcurrentDictionary<NativeCudaDevice, BudgetState> States = new();

        private sealed class BudgetState(
            long initialMaximumBytes,
            long hardMaximumBytes)
        {
            internal object Sync { get; } = new();
            internal long MaximumBytes = initialMaximumBytes;
            internal long HardMaximumBytes { get; } = hardMaximumBytes;
            internal long CachedBytes;
        }

        internal static bool TryReserve(
            NativeCudaDevice accelerator,
            long bytes)
        {
            BudgetState state = States.GetOrAdd(
                accelerator,
                static device => new BudgetState(
                    checked(device.MemorySize
                        * InitialCacheMemoryPercent / 100),
                    checked(device.MemorySize
                        * MaximumCacheMemoryPercent / 100)));
            lock (state.Sync)
            {
                if (bytes > state.MaximumBytes - state.CachedBytes)
                {
                    long requested = checked(state.CachedBytes + bytes);
                    if (requested > state.HardMaximumBytes)
                        return false;

                    long freeBytes;
                    try
                    {
                        freeBytes = accelerator.GetFreeMemory();
                    }
                    catch (NativeCudaException)
                    {
                        return false;
                    }
                    if (freeBytes < MinimumFreeReserveBytes)
                        return false;

                    // Grow only as far as the high-water mark actually seen.
                    // Subsequent steady steps therefore avoid cudaMemGetInfo.
                    state.MaximumBytes = requested;
                }
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
        private const int MaximumBuffersPerSize = 128;
        private const int LaneUploadSlotsPerLength = 2;
        private const int LaneUploadMaximumLengths = 3;
        private static readonly System.Collections.Concurrent
            .ConcurrentDictionary<NativeCudaDevice, PoolState> Pools = new();
        private static readonly object LaneUploadPoolsSync = new();
        private static readonly System.Runtime.CompilerServices
            .ConditionalWeakTable<IStreamExecutionLane, LaneUploadSlots>
            LaneUploadPools = new();

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
            if (TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out _))
            {
                return accelerator.Allocate1D<int>(
                    length,
                    NNtrain.Cuda.Memory.CudaMemoryKind.Transient);
            }
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            lock (state.Sync)
            {
                while (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<int> buffer = bucket.Pop();
                    if (bucket.Count == 0)
                        state.Buffers.Remove(length);
                    state.PooledBuffers.Remove(buffer);
                    CudaTransientBufferBudget.Release(
                        accelerator,
                        checked((long)length * sizeof(int)));
                    if (buffer.IsAlive && buffer.SessionGeneration == 0 && buffer.Arena is null)
                        return buffer;
                    if (state.Staging.Remove(buffer, out var staging)) staging.Dispose();
                    buffer.Dispose();
                }
            }
            return accelerator.Allocate1D<int>(length);
        }

        internal static void Upload(
            int deviceIndex,
            NativeCudaBuffer<int> buffer,
            int[] values)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            if (TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out IStreamExecutionLane lane))
            {
                lane.ActivateComputeStream();
                GetOrCreateLaneUploadSlots(lane).Upload(
                    values,
                    buffer,
                    lane.ComputeStreamHandle);
                return;
            }
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

        internal static BoundedUploadSlotCacheTelemetry GetLaneTelemetry(
            int deviceIndex)
        {
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    deviceIndex,
                    out IStreamExecutionLane lane))
            {
                return default;
            }
            lock (LaneUploadPoolsSync)
            {
                return LaneUploadPools.TryGetValue(
                    lane,
                    out LaneUploadSlots? slots)
                        ? slots.Telemetry
                        : default;
            }
        }

        internal static void Return(
            NativeCudaDevice accelerator,
            NativeCudaBuffer<int> buffer)
        {
            if (!buffer.IsAlive || buffer.SessionGeneration != 0 || buffer.Arena is not null)
            {
                buffer.Dispose();
                return;
            }
            int length = checked((int)buffer.Length);
            long bytes = checked((long)length * sizeof(int));
            PoolState state = Pools.GetOrAdd(
                accelerator, static _ => new PoolState());
            NativeCudaPinnedUpload<int>? releaseStaging = null;
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
                if (state.Staging.Remove(buffer, out var staging))
                    releaseStaging = staging;
            }
            List<Exception>? failures = null;
            if (releaseStaging is not null)
                TryDispose(releaseStaging, ref failures);
            TryDispose(buffer, ref failures);
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA int buffer cleanup failed.",
                    failures);
            }
        }

        internal static void Clear(NativeCudaDevice accelerator)
        {
            var dispose = new List<NativeCudaBuffer<int>>();
            var stagingToDispose = new List<NativeCudaPinnedUpload<int>>();
            List<Exception>? failures = null;
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
                TryDispose(staging, ref failures);
            foreach (NativeCudaBuffer<int> buffer in dispose)
                TryDispose(buffer, ref failures);
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA int buffer pool cleanup failed.",
                    failures);
            }
        }

        private static LaneUploadSlots GetOrCreateLaneUploadSlots(
            IStreamExecutionLane lane)
        {
            lock (LaneUploadPoolsSync)
            {
                if (LaneUploadPools.TryGetValue(
                        lane,
                        out LaneUploadSlots? existing))
                {
                    return existing;
                }

                var created = new LaneUploadSlots(lane.DeviceIndex);
                LaneUploadSlots owned = ExecutionLaneResources.Attach(
                    lane,
                    created);
                LaneUploadPools.Add(lane, owned);
                return owned;
            }
        }

        private sealed class LaneUploadSlots : IDisposable
        {
            private readonly BoundedUploadSlotCache<
                NativeCudaPinnedUpload<int>> _slots;

            internal LaneUploadSlots(int deviceIndex)
            {
                _slots = new BoundedUploadSlotCache<
                    NativeCudaPinnedUpload<int>>(
                    length => new NativeCudaPinnedUpload<int>(
                        deviceIndex,
                        length),
                    LaneUploadSlotsPerLength,
                    LaneUploadMaximumLengths);
            }

            internal BoundedUploadSlotCacheTelemetry Telemetry
                => _slots.Telemetry;

            internal void Upload(
                int[] source,
                NativeCudaBuffer<int> destination,
                nint stream)
                => _slots.Use(
                    source.Length,
                    slot => slot.Upload(source, destination, stream));

            public void Dispose() => _slots.Dispose();
        }

        private static void TryDispose(
            IDisposable resource,
            ref List<Exception>? failures)
        {
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
    }

}
