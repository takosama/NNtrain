using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
    internal enum ReplicaReleaseMode
    {
        Dispose,
        ReturnGraphToPool,
        ReturnInferenceToPool,
    }

    /// <summary>
    /// Owns every CUDA replica and the coherence state that gives those
    /// buffers meaning.  All creation, publication, and release paths lock
    /// <see cref="Sync"/>, so two GPU workers cannot publish competing owners
    /// for the same tensor generation.
    /// </summary>
    internal sealed class DeviceReplicaSet
    {
        internal Dictionary<int, DeviceBuffer> Float32Data { get; } = [];
        internal Dictionary<int, BFloat16DeviceBuffer> BFloat16Data { get; } = [];
        internal Dictionary<int, Bfp8DeviceBuffer> Bfp8Data { get; } = [];
        internal Dictionary<int, DeviceBuffer> Float32Masters { get; } = [];
        internal Dictionary<int, GradientDeviceBuffer> Float32Gradients { get; } = [];
        internal Dictionary<int, BFloat16GradientDeviceBuffer>
            BFloat16Gradients { get; } = [];
        internal Dictionary<int, Bfp8GradientDeviceBuffer> Bfp8Gradients { get; } = [];
        internal Dictionary<int, DeviceBuffer> Staging { get; } = [];
        internal Dictionary<long, IDisposable> SessionRegistrations { get; } = [];

        internal object Sync { get; } = new();
        internal TensorDevice Device { get; set; }
        internal int CudaDeviceIndex { get; set; }
        internal bool HostDataCurrent { get; set; } = true;
        internal long GradientVersion { get; set; }
        internal bool HostGradientCurrent { get; set; } = true;
        internal GradientStorageAuthority GradientAuthority { get; set; }
            = GradientStorageAuthority.Host;
        internal int GradientAuthorityDeviceIndex { get; set; } = -1;
        internal CudaGradientCoherenceKind GradientCoherenceKind { get; set; }
            = CudaGradientCoherenceKind.Host;
        internal int GradientLocalDeviceIndex { get; set; } = -1;
        internal int[] GradientReducedDevices { get; set; } = [];
        internal CudaGradientReductionStamp GradientReductionStamp { get; set; }
        internal int[] PendingGradientReductionDevices { get; set; } = [];
        internal CudaGradientReductionStamp PendingGradientReductionStamp { get; set; }
        internal long RegisteredGradientReducerGeneration { get; set; }
        internal long GradientZeroOwnerGeneration { get; set; }
        internal int[] GradientZeroOwnerDevices { get; set; } = [];
        internal ulong ReducerOwnedGradientZeroPendingMask { get; set; }
        internal long OptimizerConsumedGradientVersion { get; set; } = -1;
        internal CudaGradientReductionStamp OptimizerConsumedReductionStamp { get; set; }

        internal int DataReplicaCount
        {
            get
            {
                lock (Sync)
                {
                    return Float32Data.Count
                        + BFloat16Data.Count
                        + Bfp8Data.Count;
                }
            }
        }

        internal int GradientReplicaCount
        {
            get
            {
                lock (Sync)
                {
                    return Float32Gradients.Count
                        + BFloat16Gradients.Count
                        + Bfp8Gradients.Count;
                }
            }
        }

        internal List<Exception>? ReleaseResourcesLocked(
            ReplicaReleaseMode mode)
        {
            if (!Monitor.IsEntered(Sync))
            {
                throw new SynchronizationLockException(
                    "CUDA replica release requires the replica-set lock.");
            }

            List<Exception>? failures = null;
            try
            {
                foreach (DeviceBuffer buffer in Float32Data.Values)
                {
                    Attempt(
                        mode == ReplicaReleaseMode.Dispose
                            ? buffer.Dispose
                            : buffer.ReturnToPool,
                        ref failures);
                }
                foreach (BFloat16DeviceBuffer buffer in BFloat16Data.Values)
                {
                    Attempt(
                        mode == ReplicaReleaseMode.Dispose
                            ? buffer.Dispose
                            : buffer.ReturnToPool,
                        ref failures);
                }
                foreach (Bfp8DeviceBuffer buffer in Bfp8Data.Values)
                    Attempt(buffer.Dispose, ref failures);
                foreach (DeviceBuffer buffer in Float32Masters.Values)
                {
                    Attempt(
                        mode == ReplicaReleaseMode.Dispose
                            ? buffer.Dispose
                            : buffer.ReturnToPool,
                        ref failures);
                }
                foreach (GradientDeviceBuffer buffer in Float32Gradients.Values)
                {
                    Attempt(
                        mode == ReplicaReleaseMode.ReturnGraphToPool
                            ? buffer.ReturnToPool
                            : buffer.Dispose,
                        ref failures);
                }
                foreach (BFloat16GradientDeviceBuffer buffer
                    in BFloat16Gradients.Values)
                {
                    Attempt(
                        mode == ReplicaReleaseMode.Dispose
                            ? buffer.Dispose
                            : buffer.ReturnToPool,
                        ref failures);
                }
                foreach (Bfp8GradientDeviceBuffer buffer in Bfp8Gradients.Values)
                    Attempt(buffer.Dispose, ref failures);
                foreach (DeviceBuffer buffer in Staging.Values)
                {
                    Attempt(
                        mode == ReplicaReleaseMode.Dispose
                            ? buffer.Dispose
                            : buffer.ReturnToPool,
                        ref failures);
                }
                foreach (IDisposable registration
                    in SessionRegistrations.Values)
                {
                    Attempt(registration.Dispose, ref failures);
                }
            }
            finally
            {
                // Ownership is relinquished even when one native resource
                // reports a cleanup error. No later release may double-free a
                // resource, and one failure never strands subsequent buffers.
                Float32Data.Clear();
                BFloat16Data.Clear();
                Bfp8Data.Clear();
                Float32Masters.Clear();
                Float32Gradients.Clear();
                BFloat16Gradients.Clear();
                Bfp8Gradients.Clear();
                Staging.Clear();
                SessionRegistrations.Clear();
            }

            return failures;
        }

        internal List<Exception>? ReleaseSessionGenerationLocked(
            long sessionGeneration)
        {
            if (!Monitor.IsEntered(Sync))
            {
                throw new SynchronizationLockException(
                    "CUDA replica retirement requires the replica-set lock.");
            }
            if (sessionGeneration <= 0)
                throw new ArgumentOutOfRangeException(nameof(sessionGeneration));

            List<Exception>? failures = null;
            ReleaseMatching(
                Float32Data,
                buffer => buffer.Buffer.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                BFloat16Data,
                buffer => buffer.Buffer.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                Bfp8Data,
                buffer => buffer.Payload.SessionGeneration == sessionGeneration
                    || buffer.Scales.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                Float32Masters,
                buffer => buffer.Buffer.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                Float32Gradients,
                buffer => buffer.Buffer.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                BFloat16Gradients,
                buffer => buffer.Buffer.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                Bfp8Gradients,
                buffer => buffer.Payload.SessionGeneration == sessionGeneration
                    || buffer.Scales.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);
            ReleaseMatching(
                Staging,
                buffer => buffer.Buffer.SessionGeneration == sessionGeneration,
                static buffer => buffer.Dispose(),
                ref failures);

            if (SessionRegistrations.Remove(
                    sessionGeneration,
                    out IDisposable? registration))
            {
                Attempt(registration.Dispose, ref failures);
            }
            return failures;
        }

        private static void ReleaseMatching<TBuffer>(
            Dictionary<int, TBuffer> buffers,
            Func<TBuffer, bool> belongsToSession,
            Action<TBuffer> release,
            ref List<Exception>? failures)
            where TBuffer : class
        {
            foreach (int deviceIndex in buffers
                .Where(pair => belongsToSession(pair.Value))
                .Select(static pair => pair.Key)
                .ToArray())
            {
                TBuffer buffer = buffers[deviceIndex];
                buffers.Remove(deviceIndex);
                Attempt(() => release(buffer), ref failures);
            }
        }

        private static void Attempt(
            Action release,
            ref List<Exception>? failures)
        {
            try
            {
                release();
            }
            catch (Exception exception)
            {
                failures ??= [];
                if (exception is AggregateException aggregate)
                    failures.AddRange(aggregate.Flatten().InnerExceptions);
                else
                    failures.Add(exception);
            }
        }
    }

    private static bool IsReplicaUsableInCurrentSession<T>(
        NativeCudaBuffer<T> buffer)
        where T : unmanaged
    {
        if (!buffer.IsAlive)
            return false;
        if (buffer.SessionGeneration == 0)
            return true;
        return ExecutionSession.Current?.Generation
            == buffer.SessionGeneration;
    }

    private void RegisterSessionReplicaLocked<T>(NativeCudaBuffer<T> buffer)
        where T : unmanaged
    {
        if (!Monitor.IsEntered(_deviceSync))
        {
            throw new SynchronizationLockException(
                "CUDA session registration requires the replica-set lock.");
        }

        long generation = buffer.SessionGeneration;
        if (generation == 0
            || _value.Replicas.SessionRegistrations.ContainsKey(generation))
        {
            return;
        }
        if (!buffer.TryGetOwnerSession(out ExecutionSession? session)
            || session is null
            || session.IsDisposed)
        {
            return;
        }

        IDisposable registration;
        try
        {
            registration = session.RegisterBeforeDispose(
                this,
                owner => ((Tensor)owner)
                    .RetireSessionCudaReplicas(generation));
        }
        catch (ObjectDisposedException)
        {
            // The lane manager will close the allocation. Ensure paths also
            // reject a closed lease, so a concurrently-ending session cannot
            // publish this replica as usable in a later generation.
            return;
        }

        if (!_value.Replicas.SessionRegistrations.TryAdd(
                generation,
                registration))
        {
            registration.Dispose();
        }
    }

    private void RetireSessionCudaReplicas(long sessionGeneration)
    {
        List<Exception>? failures = null;
        lock (_deviceSync)
        {
            int dataDevice = FindAuthoritativeDataDeviceForSessionLocked(
                sessionGeneration);
            if (!_hostDataCurrent && dataDevice >= 0)
            {
                try
                {
                    SynchronizeHostFromCudaLocked(dataDevice);
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }

            int gradientDevice =
                FindAuthoritativeGradientDeviceForSessionLocked(
                    sessionGeneration);
            if (!_hostGradientCurrent && gradientDevice >= 0)
            {
                try
                {
                    SynchronizeHostGradientFromCudaLocked(gradientDevice);
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }

            List<Exception>? releaseFailures = _value.Replicas
                .ReleaseSessionGenerationLocked(sessionGeneration);
            if (releaseFailures is not null)
                (failures ??= []).AddRange(releaseFailures);

            int remainingDataDevice = FindAuthoritativeDataDeviceLocked();
            if (remainingDataDevice < 0)
            {
                _device = TensorDevice.Cpu;
                _cudaDeviceIndex = 0;
                if (!_hostDataCurrent)
                {
                    (failures ??= []).Add(new InvalidOperationException(
                        "An authoritative CUDA tensor replica could not be " +
                        "preserved before its execution session ended."));
                }
            }
            else if (_cudaDeviceIndex == dataDevice)
            {
                _cudaDeviceIndex = remainingDataDevice;
            }

            if (FindAuthoritativeGradientDeviceLocked() < 0)
            {
                if (_hostGradientCurrent)
                {
                    _gradientAuthority = GradientStorageAuthority.Host;
                    _gradientAuthorityDeviceIndex = -1;
                    ResetCudaGradientCoherenceLocked();
                }
                else
                {
                    (failures ??= []).Add(new InvalidOperationException(
                        "An authoritative CUDA gradient replica could not be " +
                        "preserved before its execution session ended."));
                }
            }
        }

        if (failures is not null)
        {
            throw new AggregateException(
                $"CUDA replicas for execution generation " +
                $"{sessionGeneration} failed to retire cleanly.",
                failures);
        }
    }

    private int FindAuthoritativeDataDeviceForSessionLocked(
        long sessionGeneration)
    {
        foreach ((int deviceIndex, long generation)
            in EnumerateAuthoritativeDataDevicesLocked())
        {
            if (generation == sessionGeneration)
                return deviceIndex;
        }
        return -1;
    }

    private int FindAuthoritativeDataDeviceLocked()
    {
        foreach ((int deviceIndex, _) in EnumerateAuthoritativeDataDevicesLocked())
            return deviceIndex;
        return -1;
    }

    private IEnumerable<(int DeviceIndex, long SessionGeneration)>
        EnumerateAuthoritativeDataDevicesLocked()
    {
        foreach ((int deviceIndex, DeviceBuffer buffer)
            in _cudaMasterBuffers)
        {
            if (buffer.Version == _dataVersion && buffer.Buffer.IsAlive)
                yield return (deviceIndex, buffer.Buffer.SessionGeneration);
        }
        foreach ((int deviceIndex, DeviceBuffer buffer) in _cudaBuffers)
        {
            if (buffer.Version == _dataVersion && buffer.Buffer.IsAlive)
                yield return (deviceIndex, buffer.Buffer.SessionGeneration);
        }
        foreach ((int deviceIndex, BFloat16DeviceBuffer buffer)
            in _cudaBFloat16Buffers)
        {
            if (buffer.Version == _dataVersion && buffer.Buffer.IsAlive)
                yield return (deviceIndex, buffer.Buffer.SessionGeneration);
        }
        foreach ((int deviceIndex, Bfp8DeviceBuffer buffer)
            in _cudaBfp8Buffers)
        {
            if (buffer.Version == _dataVersion
                && buffer.Payload.IsAlive
                && buffer.Scales.IsAlive)
            {
                yield return (
                    deviceIndex,
                    buffer.Payload.SessionGeneration);
            }
        }
    }

    private int FindAuthoritativeGradientDeviceForSessionLocked(
        long sessionGeneration)
    {
        foreach ((int deviceIndex, long generation)
            in EnumerateAuthoritativeGradientDevicesLocked())
        {
            if (generation == sessionGeneration)
                return deviceIndex;
        }
        return -1;
    }

    private int FindAuthoritativeGradientDeviceLocked()
    {
        foreach ((int deviceIndex, _)
            in EnumerateAuthoritativeGradientDevicesLocked())
        {
            return deviceIndex;
        }
        return -1;
    }

    private IEnumerable<(int DeviceIndex, long SessionGeneration)>
        EnumerateAuthoritativeGradientDevicesLocked()
    {
        switch (_gradientAuthority)
        {
            case GradientStorageAuthority.CudaFloat32:
                foreach ((int deviceIndex, GradientDeviceBuffer buffer)
                    in _cudaGradientBuffers)
                {
                    if (buffer.Version == _gradientVersion
                        && buffer.Buffer.IsAlive)
                    {
                        yield return (
                            deviceIndex,
                            buffer.Buffer.SessionGeneration);
                    }
                }
                break;
            case GradientStorageAuthority.CudaBFloat16:
                foreach ((int deviceIndex, BFloat16GradientDeviceBuffer buffer)
                    in _cudaBFloat16GradientBuffers)
                {
                    if (buffer.Version == _gradientVersion
                        && buffer.Buffer.IsAlive)
                    {
                        yield return (
                            deviceIndex,
                            buffer.Buffer.SessionGeneration);
                    }
                }
                break;
            case GradientStorageAuthority.CudaBfp8:
                foreach ((int deviceIndex, Bfp8GradientDeviceBuffer buffer)
                    in _cudaBfp8GradientBuffers)
                {
                    if (buffer.Version == _gradientVersion
                        && buffer.Payload.IsAlive
                        && buffer.Scales.IsAlive)
                    {
                        yield return (
                            deviceIndex,
                            buffer.Payload.SessionGeneration);
                    }
                }
                break;
        }
    }

    internal void InvalidateCudaBuffers()
    {
        lock (_deviceSync)
        {
            List<Exception>? failures = _value.Replicas
                .ReleaseResourcesLocked(ReplicaReleaseMode.Dispose);
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _gradientAuthority = GradientStorageAuthority.Host;
            _gradientAuthorityDeviceIndex = -1;
            ResetCudaGradientCoherenceLocked();
            _device = TensorDevice.Cpu;
            if (failures is not null)
            {
                throw new AggregateException(
                    "One or more CUDA replicas failed to invalidate.",
                    failures);
            }
        }
    }

    internal void ReleaseCudaGraphBuffers()
    {
        lock (_deviceSync)
        {
            List<Exception>? failures = _value.Replicas
                .ReleaseResourcesLocked(
                    ReplicaReleaseMode.ReturnGraphToPool);
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _gradientAuthority = GradientStorageAuthority.Host;
            _gradientAuthorityDeviceIndex = -1;
            ResetCudaGradientCoherenceLocked();
            _device = TensorDevice.Cpu;
            if (failures is not null)
            {
                throw new AggregateException(
                    "One or more CUDA graph replicas failed to release.",
                    failures);
            }
        }
    }

    internal void ReleaseCudaInferenceBuffers()
    {
        lock (_deviceSync)
        {
            List<Exception>? failures = _value.Replicas
                .ReleaseResourcesLocked(
                    ReplicaReleaseMode.ReturnInferenceToPool);
            _hostDataCurrent = true;
            _hostGradientCurrent = true;
            _gradientAuthority = GradientStorageAuthority.Host;
            _gradientAuthorityDeviceIndex = -1;
            ResetCudaGradientCoherenceLocked();
            _device = TensorDevice.Cpu;
            if (failures is not null)
            {
                throw new AggregateException(
                    "One or more CUDA inference replicas failed to release.",
                    failures);
            }
        }
    }

    private void ResetCudaGradientCoherenceLocked()
    {
        _gradientCoherenceKind = CudaGradientCoherenceKind.Host;
        _gradientLocalDeviceIndex = -1;
        _gradientReducedDevices = [];
        _gradientReductionStamp = default;
        _pendingGradientReductionDevices = [];
        _pendingGradientReductionStamp = default;
        _registeredGradientReducerGeneration = 0;
        ClearGradientZeroOwnerLocked();
        _optimizerConsumedGradientVersion = -1;
        _optimizerConsumedReductionStamp = default;
    }


}
