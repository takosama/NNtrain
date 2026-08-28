using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
    internal enum GradientStorageAuthority
    {
        Host,
        CudaFloat32,
        CudaBFloat16,
        CudaBfp8,
    }

    private object _deviceSync => _value.Replicas.Sync;
    private TensorDevice _device
    {
        get => _value.Replicas.Device;
        set => _value.Replicas.Device = value;
    }
    private int _cudaDeviceIndex
    {
        get => _value.Replicas.CudaDeviceIndex;
        set => _value.Replicas.CudaDeviceIndex = value;
    }
    private Dictionary<int, DeviceBuffer> _cudaBuffers
        => _value.Replicas.Float32Data;
    private Dictionary<int, BFloat16DeviceBuffer> _cudaBFloat16Buffers
        => _value.Replicas.BFloat16Data;
    private Dictionary<int, Bfp8DeviceBuffer> _cudaBfp8Buffers
        => _value.Replicas.Bfp8Data;
    private Dictionary<int, DeviceBuffer> _cudaMasterBuffers
        => _value.Replicas.Float32Masters;
    private Dictionary<int, GradientDeviceBuffer> _cudaGradientBuffers
        => _value.Replicas.Float32Gradients;
    private Dictionary<int, BFloat16GradientDeviceBuffer>
        _cudaBFloat16GradientBuffers
        => _value.Replicas.BFloat16Gradients;
    private Dictionary<int, Bfp8GradientDeviceBuffer>
        _cudaBfp8GradientBuffers
        => _value.Replicas.Bfp8Gradients;
    private Dictionary<int, DeviceBuffer> _cudaStagingBuffers
        => _value.Replicas.Staging;
    private bool _hostDataCurrent
    {
        get => _value.Replicas.HostDataCurrent;
        set => _value.Replicas.HostDataCurrent = value;
    }
    private long _gradientVersion
    {
        get => _value.Replicas.GradientVersion;
        set => _value.Replicas.GradientVersion = value;
    }
    private bool _hostGradientCurrent
    {
        get => _value.Replicas.HostGradientCurrent;
        set => _value.Replicas.HostGradientCurrent = value;
    }
    private GradientStorageAuthority _gradientAuthority
    {
        get => _value.Replicas.GradientAuthority;
        set => _value.Replicas.GradientAuthority = value;
    }
    private int _gradientAuthorityDeviceIndex
    {
        get => _value.Replicas.GradientAuthorityDeviceIndex;
        set => _value.Replicas.GradientAuthorityDeviceIndex = value;
    }
    private CudaGradientCoherenceKind _gradientCoherenceKind
    {
        get => _value.Replicas.GradientCoherenceKind;
        set => _value.Replicas.GradientCoherenceKind = value;
    }
    private int _gradientLocalDeviceIndex
    {
        get => _value.Replicas.GradientLocalDeviceIndex;
        set => _value.Replicas.GradientLocalDeviceIndex = value;
    }
    private int[] _gradientReducedDevices
    {
        get => _value.Replicas.GradientReducedDevices;
        set => _value.Replicas.GradientReducedDevices = value;
    }
    private CudaGradientReductionStamp _gradientReductionStamp
    {
        get => _value.Replicas.GradientReductionStamp;
        set => _value.Replicas.GradientReductionStamp = value;
    }
    private int[] _pendingGradientReductionDevices
    {
        get => _value.Replicas.PendingGradientReductionDevices;
        set => _value.Replicas.PendingGradientReductionDevices = value;
    }
    private CudaGradientReductionStamp _pendingGradientReductionStamp
    {
        get => _value.Replicas.PendingGradientReductionStamp;
        set => _value.Replicas.PendingGradientReductionStamp = value;
    }
    private long _registeredGradientReducerGeneration
    {
        get => _value.Replicas.RegisteredGradientReducerGeneration;
        set => _value.Replicas.RegisteredGradientReducerGeneration = value;
    }
    private long _gradientZeroOwnerGeneration
    {
        get => _value.Replicas.GradientZeroOwnerGeneration;
        set => _value.Replicas.GradientZeroOwnerGeneration = value;
    }
    private int[] _gradientZeroOwnerDevices
    {
        get => _value.Replicas.GradientZeroOwnerDevices;
        set => _value.Replicas.GradientZeroOwnerDevices = value;
    }
    private ulong _reducerOwnedGradientZeroPendingMask
    {
        get => _value.Replicas.ReducerOwnedGradientZeroPendingMask;
        set => _value.Replicas.ReducerOwnedGradientZeroPendingMask = value;
    }
    private long _optimizerConsumedGradientVersion
    {
        get => _value.Replicas.OptimizerConsumedGradientVersion;
        set => _value.Replicas.OptimizerConsumedGradientVersion = value;
    }
    private CudaGradientReductionStamp _optimizerConsumedReductionStamp
    {
        get => _value.Replicas.OptimizerConsumedReductionStamp;
        set => _value.Replicas.OptimizerConsumedReductionStamp = value;
    }

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

    public TensorDevice Device => _device;

    public TorchDevice device
        => new(
            _device,
            _device == TensorDevice.Cuda ? _cudaDeviceIndex : 0);

    internal int[] GetResidentCudaDeviceIndices()
    {
        lock (_deviceSync)
        {
            return _cudaBuffers.Keys
                .Concat(_cudaBFloat16Buffers.Keys)
                .Concat(_cudaBfp8Buffers.Keys)
                .Concat(_cudaMasterBuffers.Keys)
                .Distinct()
                .Order()
                .ToArray();
        }
    }

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
            else if (DType == TensorDType.Bfp8)
                EnsureCudaBfp8Buffer(device.Index);
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
        if (DType is TensorDType.BFloat16 or TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                $"{DType} tensors must use their physical CUDA buffer; " +
                "implicit expansion to a float32 device buffer is forbidden.");
        }
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaDevice accelerator =
            NNtrain.ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        lock (_deviceSync)
        {
            if (!_hostDataCurrent
                && (!_cudaBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out DeviceBuffer? requestedBuffer)
                    || requestedBuffer.Version != _dataVersion
                    || !IsReplicaUsableInCurrentSession(
                        requestedBuffer.Buffer)))
            {
                SynchronizeHostFromCudaLocked(_cudaDeviceIndex);
            }
            if (!_cudaBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D(GetPhysicalFloat32ComputeCache()),
                    _dataVersion,
                    resolvedDeviceIndex);
                _cudaBuffers[resolvedDeviceIndex] = buffer;
                RegisterSessionReplicaLocked(buffer.Buffer);
                return buffer.Buffer;
            }

            if (buffer.Version != _dataVersion)
            {
                buffer.Buffer.CopyFromCPU(GetPhysicalFloat32ComputeCache());
                buffer.Version = _dataVersion;
            }
            RegisterSessionReplicaLocked(buffer.Buffer);
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
        lock (_deviceSync)
        {
            if (!_hostDataCurrent
                && (!_cudaMasterBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out DeviceBuffer? requestedMaster)
                    || requestedMaster.Version != _dataVersion
                    || !IsReplicaUsableInCurrentSession(
                        requestedMaster.Buffer)))
            {
                SynchronizeHostFromCudaLocked(_cudaDeviceIndex);
            }
            if (!_cudaMasterBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D(DataBuffer),
                    _dataVersion,
                    resolvedDeviceIndex);
                _cudaMasterBuffers[resolvedDeviceIndex] = buffer;
            }
            else if (buffer.Version != _dataVersion)
            {
                buffer.Buffer.CopyFromCPU(DataBuffer);
                buffer.Version = _dataVersion;
            }
            RegisterSessionReplicaLocked(buffer.Buffer);
            return buffer.Buffer;
        }
    }

    internal bool HasCudaMasterFloat32Buffer(int deviceIndex)
    {
        lock (_deviceSync)
        {
            return _cudaMasterBuffers.TryGetValue(
                    deviceIndex,
                    out DeviceBuffer? buffer)
                && buffer.Version == _dataVersion
                && IsReplicaUsableInCurrentSession(buffer.Buffer);
        }
    }

    internal NativeCudaBuffer<float> EnsureCudaGradientBuffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            ThrowIfReducerOwnedGradientZeroPendingLocked(
                resolvedDeviceIndex);
            bool hasUsableFloatGradient = _cudaGradientBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out GradientDeviceBuffer? requestedFloatGradient)
                && IsReplicaUsableInCurrentSession(
                    requestedFloatGradient.Buffer);
            bool hasUsableBFloat16Gradient =
                _cudaBFloat16GradientBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out BFloat16GradientDeviceBuffer? requestedBFloat16Gradient)
                && IsReplicaUsableInCurrentSession(
                    requestedBFloat16Gradient.Buffer);
            bool hasUsableBfp8Gradient = _cudaBfp8GradientBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8GradientDeviceBuffer? requestedBfp8Gradient)
                && IsReplicaUsableInCurrentSession(requestedBfp8Gradient.Payload)
                && IsReplicaUsableInCurrentSession(requestedBfp8Gradient.Scales);
            if (!_hostGradientCurrent
                && !hasUsableFloatGradient
                && !hasUsableBFloat16Gradient
                && !hasUsableBfp8Gradient)
            {
                if (_gradientAuthority == GradientStorageAuthority.CudaBfp8)
                {
                    throw new InvalidOperationException(
                        $"BFP8 gradient generation {_gradientVersion} is " +
                        $"not resident on CUDA device {resolvedDeviceIndex}; " +
                        "implicit host synchronization is forbidden.");
                }
                SynchronizeHostGradientFromCudaLocked();
            }
            if (!_cudaGradientBuffers.TryGetValue(
                resolvedDeviceIndex,
                out GradientDeviceBuffer? buffer)
                || buffer.Buffer.Length != Numel
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
            {
                buffer?.Dispose();
                NativeCudaBuffer<float> gradientBuffer;
                if (_cudaBfp8GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out Bfp8GradientDeviceBuffer? bfp8Encoded)
                    && bfp8Encoded.Version == _gradientVersion
                    && IsReplicaUsableInCurrentSession(bfp8Encoded.Payload)
                    && IsReplicaUsableInCurrentSession(bfp8Encoded.Scales))
                {
                    gradientBuffer = CudaFloatBufferPool.Rent(
                        resolvedDeviceIndex, Numel);
                    CudaBfp8Native.DequantizeFloat32(
                        resolvedDeviceIndex,
                        bfp8Encoded.Payload,
                        bfp8Encoded.Scales,
                        gradientBuffer,
                        Bfp8QuantizationDescriptor.TensorWide);
                }
                else if (_cudaBFloat16GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out BFloat16GradientDeviceBuffer? encoded)
                    && encoded.Version == _gradientVersion
                    && IsReplicaUsableInCurrentSession(encoded.Buffer))
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
                RegisterSessionReplicaLocked(buffer.Buffer);
                return buffer.Buffer;
            }

            if (buffer.Version != _gradientVersion)
            {
                // A different CUDA adapter may have produced another local
                // gradient for the same data-parallel step. Keep this
                // adapter's local buffer until the explicit all-reduce.
                if (!_hostGradientCurrent
                    && _gradientAuthority
                        == GradientStorageAuthority.CudaFloat32)
                {
                    RegisterSessionReplicaLocked(buffer.Buffer);
                    return buffer.Buffer;
                }
                if (_cudaBfp8GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out Bfp8GradientDeviceBuffer? bfp8Encoded)
                    && bfp8Encoded.Version == _gradientVersion
                    && IsReplicaUsableInCurrentSession(bfp8Encoded.Payload)
                    && IsReplicaUsableInCurrentSession(bfp8Encoded.Scales))
                {
                    CudaBfp8Native.DequantizeFloat32(
                        resolvedDeviceIndex,
                        bfp8Encoded.Payload,
                        bfp8Encoded.Scales,
                        buffer.Buffer,
                        Bfp8QuantizationDescriptor.TensorWide);
                    buffer.Version = _gradientVersion;
                    RegisterSessionReplicaLocked(buffer.Buffer);
                    return buffer.Buffer;
                }
                if (_cudaBFloat16GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out BFloat16GradientDeviceBuffer? bfloat16Encoded)
                    && bfloat16Encoded.Version == _gradientVersion
                    && IsReplicaUsableInCurrentSession(
                        bfloat16Encoded.Buffer))
                {
                    CudaTensorNative.DecodeBFloat16(
                        resolvedDeviceIndex,
                        bfloat16Encoded.Buffer.NativePtr,
                        buffer.Buffer.NativePtr,
                        Numel);
                    buffer.Version = _gradientVersion;
                    RegisterSessionReplicaLocked(buffer.Buffer);
                    return buffer.Buffer;
                }
                buffer.Buffer.CopyFromCPU(_grad);
                buffer.Version = _gradientVersion;
            }
            RegisterSessionReplicaLocked(buffer.Buffer);
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
                    RegisterSessionReplicaLocked(current.Buffer);
                    return;
                }
                current.Dispose();
            }
            _cudaGradientBuffers[deviceIndex] = new GradientDeviceBuffer(
                slice, _gradientVersion, deviceIndex);
            RegisterSessionReplicaLocked(slice);
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
            // An arena is an allocation detail of a data-parallel plan, not
            // the owner of the tensor's logical gradient.  Preserve the
            // authoritative value before the plan releases its arena; merely
            // marking the host mirror current here would silently turn a
            // completed resident gradient into zeros after engine disposal.
            SynchronizeHostGradientFromCudaLocked(deviceIndex);
            _cudaGradientBuffers.Remove(deviceIndex);
            current.Dispose();
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
                || buffer.Buffer.Length != Numel
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
            {
                buffer?.Dispose();
                buffer = new DeviceBuffer(
                    accelerator.Allocate1D<float>(Numel),
                    version: -1,
                    deviceIndex: deviceIndex);
                _cudaStagingBuffers[deviceIndex] = buffer;
            }
            RegisterSessionReplicaLocked(buffer.Buffer);
            return buffer.Buffer;
        }
    }

    internal void MarkCudaGradientMutated(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            ThrowIfReducerOwnedGradientZeroPendingLocked(
                resolvedDeviceIndex);
            if (!_cudaGradientBuffers.TryGetValue(
                resolvedDeviceIndex,
                out GradientDeviceBuffer? buffer)
                || !IsReplicaUsableInCurrentSession(buffer.Buffer))
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
            _gradientAuthority = GradientStorageAuthority.CudaFloat32;
            _gradientAuthorityDeviceIndex = resolvedDeviceIndex;
            _hostGradientCurrent = false;
            MarkCudaGradientLocalLocked(resolvedDeviceIndex);
        }
    }

    internal long GradientVersion
    {
        get
        {
            lock (_deviceSync)
                return _gradientVersion;
        }
    }

    internal void PrepareCudaGradientBuffers(IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        foreach (int deviceIndex in deviceIndices)
            EnsureCudaGradientBuffer(deviceIndex);
    }

    internal void MarkCudaGradientsSynchronized(
        IReadOnlyList<int> deviceIndices)
        => MarkCudaGradientsSynchronized(
            deviceIndices,
            PreserveOrCreateGradientReductionStamp(deviceIndices));

    internal void MarkCudaGradientsSynchronized(
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
                if (!_cudaGradientBuffers.TryGetValue(
                        deviceIndex,
                        out GradientDeviceBuffer? buffer))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no Float32 " +
                        "gradient replica to publish.");
                }
                buffer.Version = _gradientVersion;
            }
            _gradientAuthority = GradientStorageAuthority.CudaFloat32;
            _gradientAuthorityDeviceIndex = deviceIndices.Count == 0
                ? -1
                : deviceIndices[0];
            _hostGradientCurrent = false;
            CommitCudaGradientReductionLocked(deviceIndices, reductionStamp);
        }
    }

    internal void RegisterCudaGradientReducer(
        long reducerGeneration,
        IReadOnlyList<int> deviceIndices,
        bool ownsGradientZeroing = false)
    {
        if (reducerGeneration <= 0)
            throw new ArgumentOutOfRangeException(nameof(reducerGeneration));
        ValidateCudaGradientDeviceSet(deviceIndices);
        lock (_deviceSync)
        {
            bool completedByAnotherReducer = _gradientCoherenceKind
                    == CudaGradientCoherenceKind.Reduced
                && _gradientReductionStamp.ReducerGeneration
                    != reducerGeneration;
            bool pendingFromAnotherReducer =
                _pendingGradientReductionStamp.IsValid
                && _pendingGradientReductionStamp.ReducerGeneration
                    != reducerGeneration;
            if (completedByAnotherReducer || pendingFromAnotherReducer)
            {
                InvalidateCudaGradientReductionLocked();
            }
            _registeredGradientReducerGeneration = reducerGeneration;
            if (ownsGradientZeroing)
            {
                if (deviceIndices.Count > 64)
                {
                    throw new ArgumentException(
                        "Reducer-owned gradient zeroing supports at most " +
                        "64 CUDA devices.",
                        nameof(deviceIndices));
                }
                _gradientZeroOwnerGeneration = reducerGeneration;
                _gradientZeroOwnerDevices = deviceIndices.ToArray();
                _reducerOwnedGradientZeroPendingMask = 0;
            }
            else if (_gradientZeroOwnerGeneration != 0
                && _gradientZeroOwnerGeneration != reducerGeneration)
            {
                ClearGradientZeroOwnerLocked();
            }
        }
    }

    internal void UnregisterCudaGradientReducer(long reducerGeneration)
    {
        if (reducerGeneration <= 0)
            throw new ArgumentOutOfRangeException(nameof(reducerGeneration));
        lock (_deviceSync)
        {
            if (_registeredGradientReducerGeneration == reducerGeneration)
                _registeredGradientReducerGeneration = 0;
            if (_gradientZeroOwnerGeneration == reducerGeneration)
                ClearGradientZeroOwnerLocked();
        }
    }

    internal NativeCudaBuffer<float>
        PrepareReducerOwnedCudaGradientBuffer(
            long reducerGeneration,
            int deviceIndex)
    {
        lock (_deviceSync)
        {
            ValidateGradientZeroOwnerLocked(
                reducerGeneration,
                deviceIndex);
            if (_cudaGradientBuffers.TryGetValue(
                    deviceIndex,
                    out GradientDeviceBuffer? existing)
                && existing.Buffer.Length == Numel
                && IsReplicaUsableInCurrentSession(existing.Buffer))
            {
                RegisterSessionReplicaLocked(existing.Buffer);
                return existing.Buffer;
            }

            existing?.Dispose();
            NativeCudaBuffer<float> gradientBuffer =
                CudaFloatBufferPool.Rent(deviceIndex, Numel);
            _cudaGradientBuffers[deviceIndex] = new GradientDeviceBuffer(
                gradientBuffer,
                version: -1,
                deviceIndex);
            RegisterSessionReplicaLocked(gradientBuffer);
            return gradientBuffer;
        }
    }

    internal void CompleteReducerOwnedCudaGradientZero(
        long reducerGeneration,
        int deviceIndex)
    {
        lock (_deviceSync)
        {
            int deviceSlot = ValidateGradientZeroOwnerLocked(
                reducerGeneration,
                deviceIndex);
            if (!_cudaGradientBuffers.TryGetValue(
                    deviceIndex,
                    out GradientDeviceBuffer? buffer))
            {
                if (!_cudaBFloat16GradientBuffers.TryGetValue(
                        deviceIndex,
                        out BFloat16GradientDeviceBuffer? encoded))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no reducer-owned " +
                        "gradient buffer to clear.");
                }
                encoded.Version = _gradientVersion;
            }
            else
            {
                buffer.Version = _gradientVersion;
            }
            _reducerOwnedGradientZeroPendingMask &= ~(1UL << deviceSlot);
        }
    }

    internal void BeginCudaGradientReduction(
        CudaGradientReductionStamp reductionStamp,
        IReadOnlyList<int> deviceIndices)
    {
        if (!reductionStamp.IsValid)
            throw new ArgumentException("Reduction stamp must be valid.");
        ValidateCudaGradientDeviceSet(deviceIndices);
        lock (_deviceSync)
        {
            if (_registeredGradientReducerGeneration
                != reductionStamp.ReducerGeneration)
            {
                throw new InvalidOperationException(
                    "The CUDA gradient reducer generation is stale.");
            }
            if (_pendingGradientReductionStamp.IsValid)
            {
                throw new InvalidOperationException(
                    "A CUDA gradient reduction is already pending.");
            }
            _pendingGradientReductionStamp = reductionStamp;
            _pendingGradientReductionDevices = deviceIndices.ToArray();
            InvalidateCudaGradientReductionLocked(clearPending: false);
        }
    }

    internal void AbortCudaGradientReduction(
        CudaGradientReductionStamp reductionStamp)
    {
        lock (_deviceSync)
        {
            bool pending = _pendingGradientReductionStamp
                == reductionStamp;
            bool partiallyPublished = _gradientCoherenceKind
                    == CudaGradientCoherenceKind.Reduced
                && _gradientReductionStamp == reductionStamp;
            if (!pending && !partiallyPublished)
                return;
            InvalidateCudaGradientReductionLocked();
        }
    }

    internal CudaGradientCoherenceSnapshot
        GetCudaGradientCoherenceSnapshot()
    {
        lock (_deviceSync)
        {
            return new CudaGradientCoherenceSnapshot(
                _gradientCoherenceKind,
                _gradientLocalDeviceIndex,
                (int[])_gradientReducedDevices.Clone(),
                _gradientReductionStamp,
                _pendingGradientReductionStamp,
                _gradientVersion,
                _optimizerConsumedGradientVersion,
                _optimizerConsumedReductionStamp);
        }
    }

    internal void ConsumeCudaGradientForOptimizer(
        long expectedGradientVersion,
        CudaGradientReductionStamp expectedReductionStamp)
    {
        lock (_deviceSync)
        {
            if (_gradientVersion != expectedGradientVersion
                || _optimizerConsumedGradientVersion
                    == expectedGradientVersion)
            {
                throw new InvalidOperationException(
                    "The CUDA gradient changed while the optimizer was " +
                    "claiming it.");
            }
            if (expectedReductionStamp.IsValid
                && (_gradientCoherenceKind
                        != CudaGradientCoherenceKind.Reduced
                    || _gradientReductionStamp
                        != expectedReductionStamp))
            {
                throw new InvalidOperationException(
                    "The CUDA reduction stamp changed while the optimizer " +
                    "was claiming it.");
            }
            _optimizerConsumedGradientVersion = expectedGradientVersion;
            if (expectedReductionStamp.IsValid)
            {
                _optimizerConsumedReductionStamp =
                    expectedReductionStamp;
            }
        }
    }

    private CudaGradientReductionStamp
        PreserveOrCreateGradientReductionStamp(
            IReadOnlyList<int> deviceIndices)
    {
        ValidateCudaGradientDeviceSet(deviceIndices);
        lock (_deviceSync)
        {
            if (_gradientCoherenceKind
                    == CudaGradientCoherenceKind.Reduced
                && _gradientReductionStamp.IsValid
                && _gradientReducedDevices.SequenceEqual(deviceIndices))
            {
                return _gradientReductionStamp;
            }
        }
        return CudaGradientReductionStampSource.CreateStandalone();
    }

    private static void ValidateCudaGradientDeviceSet(
        IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        if (deviceIndices.Count == 0
            || deviceIndices.Any(device => device < 0)
            || deviceIndices.Distinct().Count() != deviceIndices.Count)
        {
            throw new ArgumentException(
                "CUDA gradient devices must be unique and non-negative.",
                nameof(deviceIndices));
        }
    }

    private int ValidateGradientZeroOwnerLocked(
        long reducerGeneration,
        int deviceIndex)
    {
        if (_gradientZeroOwnerGeneration != reducerGeneration)
        {
            throw new InvalidOperationException(
                "The CUDA gradient-zero owner generation is stale.");
        }
        int deviceSlot = Array.IndexOf(
            _gradientZeroOwnerDevices,
            deviceIndex);
        if (deviceSlot < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        return deviceSlot;
    }

    private void ThrowIfReducerOwnedGradientZeroPendingLocked(
        int deviceIndex)
    {
        if (_gradientZeroOwnerGeneration == 0
            || _reducerOwnedGradientZeroPendingMask == 0)
        {
            return;
        }
        int deviceSlot = Array.IndexOf(
            _gradientZeroOwnerDevices,
            deviceIndex);
        if (deviceSlot >= 0
            && (_reducerOwnedGradientZeroPendingMask
                & (1UL << deviceSlot)) != 0)
        {
            throw new InvalidOperationException(
                $"CUDA device {deviceIndex} gradient storage is logically " +
                "zero but awaits its reducer worker's physical clear.");
        }
    }

    private void ClearGradientZeroOwnerLocked()
    {
        _gradientZeroOwnerGeneration = 0;
        _gradientZeroOwnerDevices = [];
        _reducerOwnedGradientZeroPendingMask = 0;
    }

    private void MarkCudaGradientLocalLocked(int deviceIndex)
    {
        _gradientCoherenceKind = CudaGradientCoherenceKind.Local;
        _gradientLocalDeviceIndex = deviceIndex;
        _gradientReducedDevices = [];
        _gradientReductionStamp = default;
        if (_pendingGradientReductionStamp.IsValid
            && !_pendingGradientReductionDevices.Contains(deviceIndex))
        {
            _pendingGradientReductionStamp = default;
            _pendingGradientReductionDevices = [];
        }
    }

    private void CommitCudaGradientReductionLocked(
        IReadOnlyList<int> deviceIndices,
        CudaGradientReductionStamp reductionStamp)
    {
        bool preservingCompletedReduction = _gradientCoherenceKind
                == CudaGradientCoherenceKind.Reduced
            && _gradientReductionStamp == reductionStamp
            && _gradientReducedDevices.SequenceEqual(deviceIndices);
        if (!preservingCompletedReduction
            && _pendingGradientReductionStamp.IsValid
            && (_pendingGradientReductionStamp != reductionStamp
                || !_pendingGradientReductionDevices.SequenceEqual(
                    deviceIndices)))
        {
            throw new InvalidOperationException(
                "CUDA gradient reduction completion does not match its " +
                "pending step/device set.");
        }
        if (!preservingCompletedReduction
            && _registeredGradientReducerGeneration != 0
            && reductionStamp.ReducerGeneration
                != _registeredGradientReducerGeneration)
        {
            throw new InvalidOperationException(
                "A stale CUDA gradient reducer attempted to publish after " +
                "its plan was replaced.");
        }
        if (!preservingCompletedReduction
            && _registeredGradientReducerGeneration != 0
            && _pendingGradientReductionStamp != reductionStamp)
        {
            throw new InvalidOperationException(
                "CUDA gradient reducer attempted to publish a step that was " +
                "not begun.");
        }
        _gradientCoherenceKind = CudaGradientCoherenceKind.Reduced;
        _gradientLocalDeviceIndex = -1;
        _gradientReducedDevices = deviceIndices.ToArray();
        _gradientReductionStamp = reductionStamp;
        _pendingGradientReductionStamp = default;
        _pendingGradientReductionDevices = [];
    }

    private void InvalidateCudaGradientReductionLocked(
        bool clearPending = true)
    {
        _gradientCoherenceKind = CudaGradientCoherenceKind.Local;
        _gradientLocalDeviceIndex = -1;
        _gradientReducedDevices = [];
        _gradientReductionStamp = default;
        if (clearPending)
        {
            _pendingGradientReductionStamp = default;
            _pendingGradientReductionDevices = [];
        }
    }

    internal void SetCudaGradient(float[] values, int deviceIndex = -1)
    {
        ArgumentNullException.ThrowIfNull(values);
        if (values.Length != Numel)
            throw new ArgumentException("Gradient length must match the tensor.", nameof(values));
        if (DType == TensorDType.Bfp8
            && values.Any(value => !float.IsFinite(value)))
        {
            throw new ArgumentException(
                "BFP8 gradient publication requires finite values.",
                nameof(values));
        }
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaBuffer<float> buffer =
            EnsureCudaGradientBuffer(resolvedDeviceIndex);
        buffer.CopyFromCPU(values);
        MarkCudaGradientMutated(resolvedDeviceIndex);
    }

    /// <summary>
    /// Uploads one dynamic batch tensor under the transfer guard's explicit
    /// batch authorization. Parameters and persistent state must never use
    /// this path.
    /// </summary>
    internal void PrepareCudaBatchInput(int deviceIndex = -1)
    {
        using IDisposable authorization =
            DeviceTransferGuard.AllowBatchHostToDevice();
        _ = EnsureCudaFloat32Buffer(deviceIndex);
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
            _cudaBuffers[deviceIndex] = new DeviceBuffer(
                buffer, _dataVersion, deviceIndex);
            RegisterSessionReplicaLocked(buffer);
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
            parents,
            cudaResult: true);
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
            parents,
            cudaResult: true);
        result.AdoptCudaBFloat16Buffer(buffer, deviceIndex);
        if (!AutogradContext.IsRecordingEnabled)
            CudaInferenceScope.Track(result, deviceIndex);
        return result;
    }

    internal void SynchronizeHostFromCuda(int deviceIndex = -1)
    {
        lock (_deviceSync)
            SynchronizeHostFromCudaLocked(deviceIndex);
    }

    private void SynchronizeHostFromCudaLocked(int deviceIndex = -1)
    {
        if (_hostDataCurrent)
            return;

        // Host synchronization follows the tensor's authoritative replica,
        // not the ambient execution device.  This matters when a tensor
        // produced on GPU 0 is first materialized on GPU 1.
        int resolvedDeviceIndex = deviceIndex >= 0
            ? deviceIndex
            : _cudaDeviceIndex;
        bool hasFloat = _cudaBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? buffer)
            && buffer.Version == _dataVersion;
        bool hasBFloat16 = _cudaBFloat16Buffers.TryGetValue(
                resolvedDeviceIndex,
                out BFloat16DeviceBuffer? bfloat16Buffer)
            && bfloat16Buffer.Version == _dataVersion;
        bool hasBfp8 = _cudaBfp8Buffers.TryGetValue(
                resolvedDeviceIndex,
                out Bfp8DeviceBuffer? bfp8Buffer)
            && bfp8Buffer.Version == _dataVersion;
        bool hasMaster = _cudaMasterBuffers.TryGetValue(
                resolvedDeviceIndex,
                out DeviceBuffer? masterBuffer)
            && masterBuffer.Version == _dataVersion;
        if (!hasFloat && !hasBFloat16 && !hasBfp8 && !hasMaster)
            return;

        if (hasBfp8 && !hasMaster)
        {
            var payload = new sbyte[Numel];
            var scales = new float[bfp8Buffer!.Scales.Length];
            bfp8Buffer.Payload.CopyToCPU(payload);
            bfp8Buffer.Scales.CopyToCPU(scales);
            _data.CopyFromBfp8Encoded(payload, scales);
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = true;
            return;
        }

        float[] data = DataBuffer;
        if (hasMaster)
        {
            masterBuffer!.Buffer.CopyToCPU(data);
        }
        else if (hasFloat)
        {
            buffer!.Buffer.CopyToCPU(data);
        }
        else if (hasBFloat16)
        {
            var encoded = new ushort[Numel];
            bfloat16Buffer!.Buffer.CopyToCPU(encoded);
            TensorStorageCodec.DecodeBFloat16(encoded, data);
        }
        if (DType != TensorDType.Float32)
            _data.CopyFrom(data);
        _physicalFloat32CacheDataVersion = -1;
        _hostDataCurrent = true;
    }

    internal void MarkCudaDataMutated(int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            bool hasUsableFloat = _cudaBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out DeviceBuffer? floatReplica)
                && IsReplicaUsableInCurrentSession(floatReplica.Buffer);
            bool hasUsableBFloat16 = _cudaBFloat16Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out BFloat16DeviceBuffer? bfloat16Replica)
                && IsReplicaUsableInCurrentSession(bfloat16Replica.Buffer);
            bool hasUsableBfp8 = _cudaBfp8Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8DeviceBuffer? bfp8Replica)
                && IsReplicaUsableInCurrentSession(bfp8Replica.Payload)
                && IsReplicaUsableInCurrentSession(bfp8Replica.Scales);
            if (!hasUsableFloat && !hasUsableBFloat16 && !hasUsableBfp8)
            {
                throw new InvalidOperationException(
                    "Cannot mark CUDA data modified before allocating its buffer.");
            }
            unchecked
            {
                _dataVersion++;
            }
            if (_cudaBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out DeviceBuffer? floatBuffer))
            {
                floatBuffer.Version = _dataVersion;
            }
            if (_cudaBFloat16Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out BFloat16DeviceBuffer? bfloat16Buffer))
            {
                bfloat16Buffer.Version = _dataVersion;
            }
            if (_cudaBfp8Buffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8DeviceBuffer? bfp8Buffer))
            {
                bfp8Buffer.Version = _dataVersion;
            }
            if (_cudaMasterBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out DeviceBuffer? masterBuffer))
            {
                masterBuffer.Version = _dataVersion;
            }
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = resolvedDeviceIndex;
        }
    }

    private void EnsureHostDataCurrent()
    {
        lock (_deviceSync)
            SynchronizeHostFromCudaLocked();
    }

    private void EnsureHostGradientCurrent()
    {
        lock (_deviceSync)
            SynchronizeHostGradientFromCudaLocked();
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
        if (_grad.Length == 0)
            _grad = new float[Numel];
        int resolvedDeviceIndex = deviceIndex >= 0
            ? deviceIndex
            : _gradientAuthorityDeviceIndex >= 0
                ? _gradientAuthorityDeviceIndex
                : ResolveCudaDeviceIndex(deviceIndex);
        switch (_gradientAuthority)
        {
            case GradientStorageAuthority.CudaFloat32:
            {
                if (!_cudaGradientBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out GradientDeviceBuffer? buffer)
                    || buffer.Version != _gradientVersion)
                {
                    buffer = _cudaGradientBuffers.Values.FirstOrDefault(
                        candidate => candidate.Version == _gradientVersion);
                }
                if (buffer is null)
                {
                    throw new InvalidOperationException(
                        "The authoritative CUDA Float32 gradient replica " +
                        "is missing.");
                }
                buffer.Buffer.CopyToCPU(_grad);
                break;
            }
            case GradientStorageAuthority.CudaBFloat16:
            {
                if (!_cudaBFloat16GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out BFloat16GradientDeviceBuffer? encoded)
                    || encoded.Version != _gradientVersion)
                {
                    encoded = _cudaBFloat16GradientBuffers.Values
                        .FirstOrDefault(candidate =>
                            candidate.Version == _gradientVersion);
                }
                if (encoded is null)
                {
                    throw new InvalidOperationException(
                        "The authoritative CUDA BFloat16 gradient replica " +
                        "is missing.");
                }
                var encodedHost = new ushort[Numel];
                encoded.Buffer.CopyToCPU(encodedHost);
                TensorStorageCodec.DecodeBFloat16(encodedHost, _grad);
                break;
            }
            case GradientStorageAuthority.CudaBfp8:
            {
                if (!_cudaBfp8GradientBuffers.TryGetValue(
                        resolvedDeviceIndex,
                        out Bfp8GradientDeviceBuffer? encoded)
                    || encoded.Version != _gradientVersion)
                {
                    encoded = _cudaBfp8GradientBuffers.Values
                        .FirstOrDefault(candidate =>
                            candidate.Version == _gradientVersion);
                }
                if (encoded is null)
                {
                    throw new InvalidOperationException(
                        "The authoritative CUDA BFP8 gradient replica is " +
                        "missing.");
                }
                var payload = new sbyte[Numel];
                var scales = new float[1];
                encoded.Payload.CopyToCPU(payload);
                encoded.Scales.CopyToCPU(scales);
                Bfp8QuantizationCodec.Default.Decode(
                    payload,
                    scales,
                    Bfp8QuantizationDescriptor.TensorWide,
                    _grad);
                break;
            }
            case GradientStorageAuthority.Host:
                break;
            default:
                throw new InvalidOperationException(
                    "Unknown gradient storage authority.");
        }
        _hostGradientCurrent = true;
    }

    private void MarkHostGradientMutable()
    {
        unchecked
        {
            _gradientVersion++;
        }
        _gradientAuthority = GradientStorageAuthority.Host;
        _gradientAuthorityDeviceIndex = -1;
        _hostGradientCurrent = true;
        _gradientCoherenceKind = CudaGradientCoherenceKind.Host;
        _gradientLocalDeviceIndex = -1;
        _gradientReducedDevices = [];
        _gradientReductionStamp = default;
        _pendingGradientReductionStamp = default;
        _pendingGradientReductionDevices = [];
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
            foreach (Bfp8GradientDeviceBuffer buffer
                in _cudaBfp8GradientBuffers.Values)
            {
                buffer.Payload.MemSetToZero();
                buffer.Version = _gradientVersion;
            }
            _gradientAuthority = GradientStorageAuthority.Host;
            _gradientAuthorityDeviceIndex = -1;
            _hostGradientCurrent = true;
            _gradientCoherenceKind = CudaGradientCoherenceKind.Host;
            _gradientLocalDeviceIndex = -1;
            _gradientReducedDevices = [];
            _gradientReductionStamp = default;
            _pendingGradientReductionStamp = default;
            _pendingGradientReductionDevices = [];
        }
    }

    private bool TryClearResidentCudaGradients()
    {
        lock (_deviceSync)
        {
            if (_cudaGradientBuffers.Count == 0
                && _cudaBFloat16GradientBuffers.Count == 0
                && _cudaBfp8GradientBuffers.Count == 0)
            {
                return false;
            }
            if (_gradientZeroOwnerGeneration != 0)
            {
                unchecked
                {
                    _gradientVersion++;
                }
                // An empty host buffer is the canonical logical zero and
                // avoids writing either the full host mirror or hundreds of
                // CUDA arena slices on the coordinator stream. Each reducer
                // worker publishes its physical clear before backward starts.
                _grad = [];
                _hostGradientCurrent = true;
                _gradientAuthority = GradientStorageAuthority.Host;
                _gradientAuthorityDeviceIndex = -1;
                _gradientCoherenceKind = CudaGradientCoherenceKind.Host;
                _gradientLocalDeviceIndex = -1;
                _gradientReducedDevices = [];
                _gradientReductionStamp = default;
                _pendingGradientReductionStamp = default;
                _pendingGradientReductionDevices = [];
                _reducerOwnedGradientZeroPendingMask =
                    _gradientZeroOwnerDevices.Length == 64
                        ? ulong.MaxValue
                        : (1UL << _gradientZeroOwnerDevices.Length) - 1;
                return true;
            }
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
            foreach (Bfp8GradientDeviceBuffer buffer
                in _cudaBfp8GradientBuffers.Values)
            {
                buffer.Payload.MemSetToZero();
                buffer.Version = _gradientVersion;
            }
            if (DType == TensorDType.Bfp8
                && _cudaBfp8GradientBuffers.Count != 0)
            {
                _gradientAuthority = GradientStorageAuthority.CudaBfp8;
                _gradientAuthorityDeviceIndex =
                    _cudaBfp8GradientBuffers.Keys.First();
            }
            else if (DType == TensorDType.BFloat16
                && _cudaBFloat16GradientBuffers.Count != 0)
            {
                _gradientAuthority = GradientStorageAuthority.CudaBFloat16;
                _gradientAuthorityDeviceIndex =
                    _cudaBFloat16GradientBuffers.Keys.First();
            }
            else if (_cudaGradientBuffers.Count != 0)
            {
                _gradientAuthority = GradientStorageAuthority.CudaFloat32;
                _gradientAuthorityDeviceIndex =
                    _cudaGradientBuffers.Keys.First();
            }
            else
            {
                _gradientAuthority = GradientStorageAuthority.CudaBFloat16;
                _gradientAuthorityDeviceIndex =
                    _cudaBFloat16GradientBuffers.Keys.First();
            }
            _hostGradientCurrent = false;
            InvalidateCudaGradientReductionLocked();
            return true;
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

    internal sealed class DeviceBuffer : IDisposable
    {
        private int _disposed;
        private readonly NativeCudaDevice _accelerator;

        internal DeviceBuffer(
            NativeCudaBuffer<float> buffer,
            long version,
            int deviceIndex)
        {
            Buffer = buffer;
            Version = version;
            _accelerator = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        }

        internal NativeCudaBuffer<float> Buffer { get; }
        internal long Version { get; set; }

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

    internal sealed class GradientDeviceBuffer(
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
                if (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<float> buffer = bucket.Pop();
                    if (bucket.Count == 0)
                        state.Buffers.Remove(length);
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
            if (buffer.IsLaneManagedReusable)
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
                if (state.Buffers.TryGetValue(length, out var bucket)
                    && bucket.Count > 0)
                {
                    NativeCudaBuffer<int> buffer = bucket.Pop();
                    if (bucket.Count == 0)
                        state.Buffers.Remove(length);
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
            if (buffer.IsLaneManagedReusable)
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
