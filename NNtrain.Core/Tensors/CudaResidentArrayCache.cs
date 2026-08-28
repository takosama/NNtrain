using System.Runtime.CompilerServices;
using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Keeps immutable Float32 compute views on every CUDA adapter that consumes
/// them. Entries follow the lifetime of their host array and are invalidated
/// explicitly when a tensor is updated.
/// </summary>
internal static class CudaResidentArrayCache
{
    internal const int MaximumLaneEntries = 64;
    private static readonly ConditionalWeakTable<float[], Entry> Entries = new();
    private static readonly ConditionalWeakTable<
        IStreamExecutionLane,
        Lazy<LaneArrayCache>> LaneCaches = new();
    private static int _activeLaneCacheCount;

    internal static int ActiveLaneCacheCount =>
        Volatile.Read(ref _activeLaneCacheCount);

    internal static CudaResidentArrayLease GetOrUpload(
        NativeCudaDevice accelerator,
        float[] values)
    {
        if (TensorExecutionContext.TryGetCudaStreamLane(
                accelerator.Index,
                out IStreamExecutionLane lane)
            && lane.ComputeStreamHandle == accelerator.DefaultStream)
        {
            return LaneCaches.GetValue(
                lane,
                static owner => new Lazy<LaneArrayCache>(
                    () => ExecutionLaneResources.Attach(
                        owner,
                        new LaneArrayCache(
                            owner.DeviceIndex,
                            owner.ComputeStreamHandle)),
                    LazyThreadSafetyMode.ExecutionAndPublication))
                .Value.GetOrUpload(accelerator, values);
        }
        Entry entry = Entries.GetValue(values, static _ => new Entry());
        return new CudaResidentArrayLease(
            values,
            entry.GetOrUpload(accelerator, values),
            owner: null);
    }

    internal static int GetActiveLaneEntryCount(int deviceIndex)
    {
        if (!TensorExecutionContext.TryGetCudaStreamLane(
                deviceIndex,
                out IStreamExecutionLane lane)
            || !LaneCaches.TryGetValue(
                lane,
                out Lazy<LaneArrayCache>? cache)
            || !cache.IsValueCreated)
        {
            return 0;
        }
        return cache.Value.Count;
    }

    internal static void Invalidate(float[]? values)
    {
        if (values is null)
            return;
        List<Exception>? failures = null;
        if (Entries.TryGetValue(values, out Entry? entry))
        {
            try
            {
                entry.Dispose();
            }
            catch (Exception exception)
            {
                AddFailure(ref failures, exception);
            }
            finally
            {
                Entries.Remove(values);
            }
        }
        foreach (KeyValuePair<
            IStreamExecutionLane,
            Lazy<LaneArrayCache>> pair in LaneCaches)
        {
            if (pair.Value.IsValueCreated)
            {
                try
                {
                    pair.Value.Value.Invalidate(values);
                }
                catch (Exception exception)
                {
                    AddFailure(ref failures, exception);
                }
            }
        }
        if (failures is not null)
        {
            throw new AggregateException(
                "One or more CUDA resident array replicas failed to invalidate.",
                failures);
        }
    }

    private static void AddFailure(
        ref List<Exception>? failures,
        Exception exception)
    {
        failures ??= [];
        if (exception is AggregateException aggregate)
            failures.AddRange(aggregate.Flatten().InnerExceptions);
        else
            failures.Add(exception);
    }

    private sealed class LaneArrayCache : IDisposable
    {
        private readonly object _sync = new();
        private readonly BoundedDisposableLeaseCache<
            float[],
            ResidentBufferOwner> _buffers = new(
                MaximumLaneEntries,
                ReferenceEqualityComparer.Instance);
        private readonly int _deviceIndex;
        private readonly nint _computeStream;
        private bool _disposed;

        internal LaneArrayCache(int deviceIndex, nint computeStream)
        {
            _deviceIndex = deviceIndex;
            _computeStream = computeStream;
            Interlocked.Increment(ref _activeLaneCacheCount);
        }

        internal int Count => _buffers.Count;

        internal CudaResidentArrayLease GetOrUpload(
            NativeCudaDevice accelerator,
            float[] values)
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                BoundedDisposableLeaseCache<
                    float[],
                    ResidentBufferOwner>.Lease lease =
                    _buffers.Acquire(
                        values,
                        source => new ResidentBufferOwner(
                            _deviceIndex,
                            _computeStream,
                            accelerator.Allocate1D(source)))
                    ?? throw new InvalidOperationException(
                        "A CUDA resident buffer could not be created.");
                try
                {
                    return new CudaResidentArrayLease(
                        values,
                        lease.Value.Buffer,
                        lease);
                }
                catch
                {
                    lease.Dispose();
                    throw;
                }
            }
        }

        internal void Invalidate(float[] values)
        {
            lock (_sync)
            {
                if (_disposed)
                    return;
                _buffers.Remove(values);
            }
        }

        public void Dispose()
        {
            lock (_sync)
            {
                if (_disposed)
                    return;
                _disposed = true;
            }
            try
            {
                _buffers.Dispose();
            }
            catch (Exception exception)
            {
                throw new AggregateException(
                    $"CUDA resident array cache cleanup failed on device " +
                    $"{_deviceIndex}.",
                    exception);
            }
            finally
            {
                Interlocked.Decrement(ref _activeLaneCacheCount);
            }
        }

        private sealed class ResidentBufferOwner(
            int deviceIndex,
            nint computeStream,
            NativeCudaBuffer<float> buffer) : IDisposable
        {
            private int _disposed;

            internal NativeCudaBuffer<float> Buffer { get; } = buffer;

            public void Dispose()
            {
                if (Interlocked.Exchange(ref _disposed, 1) != 0)
                    return;
                NativeCudaRuntime.DisposeAfterStreamFence(
                    deviceIndex,
                    computeStream,
                    Buffer.Dispose);
            }
        }
    }

    private sealed class Entry : IDisposable
    {
        private readonly object _sync = new();
        private readonly Dictionary<NativeCudaDevice,
            NativeCudaBuffer<float>> _buffers =
            new(ReferenceEqualityComparer.Instance);
        private bool _disposed;

        internal NativeCudaBuffer<float> GetOrUpload(
            NativeCudaDevice accelerator,
            float[] values)
        {
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (!_buffers.TryGetValue(accelerator, out var buffer))
                {
                    buffer = accelerator.Allocate1D(values);
                    _buffers.Add(accelerator, buffer);
                }
                return buffer;
            }
        }

        public void Dispose()
        {
            NativeCudaBuffer<float>[] buffers;
            lock (_sync)
            {
                if (_disposed)
                    return;
                _disposed = true;
                buffers = _buffers.Values.ToArray();
                _buffers.Clear();
            }
            List<Exception>? failures = null;
            try
            {
                foreach (NativeCudaBuffer<float> buffer in buffers)
                {
                    try
                    {
                        buffer.Dispose();
                    }
                    catch (Exception exception)
                    {
                        AddFailure(ref failures, exception);
                    }
                }
            }
            finally
            {
                GC.SuppressFinalize(this);
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA resident array fallback cleanup failed.",
                    failures);
            }
        }

        ~Entry()
        {
            try
            {
                Dispose();
            }
            catch
            {
                // Finalizers must remain non-throwing. Explicit invalidation
                // and lane disposal surface aggregated cleanup failures.
            }
        }
    }
}

/// <summary>
/// Keeps a host array and its CUDA replica alive for one native launch. Lane
/// cache eviction can retire the replica concurrently, but actual release is
/// deferred until this lease returns and the compute-stream fence completes.
/// </summary>
internal sealed class CudaResidentArrayLease : IDisposable
{
    private object? _hostOwner;
    private IDisposable? _owner;

    internal CudaResidentArrayLease(
        float[] hostOwner,
        NativeCudaBuffer<float> buffer,
        IDisposable? owner)
    {
        _hostOwner = hostOwner ?? throw new ArgumentNullException(
            nameof(hostOwner));
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _owner = owner;
    }

    internal NativeCudaBuffer<float> Buffer { get; }

    public void Dispose()
    {
        IDisposable? owner = Interlocked.Exchange(ref _owner, null);
        try
        {
            owner?.Dispose();
        }
        finally
        {
            Interlocked.Exchange(ref _hostOwner, null);
        }
    }
}
