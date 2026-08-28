using NNtrain.Runtime.Execution;

namespace NNtrain;

internal readonly record struct CudaBfp8BufferView(
    NativeCudaBuffer<sbyte> Payload,
    NativeCudaBuffer<float> Scales,
    Bfp8QuantizationDescriptor Descriptor);

/// <summary>
/// Keeps a decoded BF16 operand alive while a CUDA operation is enqueued.
/// Leaf tensors (normally parameters) borrow their versioned replica cache;
/// transient autograd values return their decode to the lane allocator.
/// </summary>
internal sealed class CudaBfp8BFloat16Lease : IDisposable
{
    private NativeCudaBuffer<ushort>? _temporary;
    private NativeCudaDevice? _accelerator;

    internal CudaBfp8BFloat16Lease(
        NativeCudaBuffer<ushort> buffer,
        NativeCudaDevice? accelerator = null)
    {
        Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
        _temporary = accelerator is null ? null : buffer;
        _accelerator = accelerator;
    }

    internal NativeCudaBuffer<ushort> Buffer { get; }

    public void Dispose()
    {
        NativeCudaBuffer<ushort>? temporary = Interlocked.Exchange(
            ref _temporary,
            null);
        NativeCudaDevice? accelerator = Interlocked.Exchange(
            ref _accelerator,
            null);
        if (temporary is not null && accelerator is not null)
            Tensor.ReturnCudaBFloat16Buffer(accelerator, temporary);
    }
}

public partial class Tensor
{
    internal CudaBfp8BufferView EnsureCudaBfp8Buffer(int deviceIndex = -1)
    {
        if (DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "A physical CUDA BFP8 buffer requires BFP8 dtype.");
        }

        ValidateBfp8PrecisionContract();

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
                if (IsReplicaUsableInCurrentSession(current.Payload)
                    && IsReplicaUsableInCurrentSession(current.Scales))
                {
                    RegisterSessionReplicaLocked(current.Payload);
                    return current.View;
                }
            }

            if (!_hostDataCurrent
                && (!_cudaBfp8Buffers.TryGetValue(
                        resolvedDeviceIndex,
                        out Bfp8DeviceBuffer? requested)
                    || requested.Version != _dataVersion
                    || !IsReplicaUsableInCurrentSession(requested.Payload)
                    || !IsReplicaUsableInCurrentSession(requested.Scales)))
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
                || buffer.Descriptor != descriptor
                || !IsReplicaUsableInCurrentSession(buffer.Payload)
                || !IsReplicaUsableInCurrentSession(buffer.Scales))
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
                    RegisterSessionReplicaLocked(payload);
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
            RegisterSessionReplicaLocked(buffer.Payload);
            return buffer.View;
        }
    }

    /// <summary>
    /// Enforces the model-level BFP8 scale contract before a tensor can enter
    /// a CUDA kernel. Pure bfp8 uses one tensor-wide scale; mix8_32 requires
    /// independent block scales. With no active model policy the low-level
    /// Tensor API retains its descriptor-driven compatibility behavior.
    /// </summary>
    internal void ValidateBfp8PrecisionContract()
    {
        if (DType != TensorDType.Bfp8)
            return;

        Bfp8QuantizationDescriptor descriptor = Bfp8Quantization
            ?? throw new InvalidOperationException(
                "BFP8 tensor storage has no quantization descriptor.");
        PrecisionPolicy? policy = TensorExecutionContext.ActivePrecisionPolicy;
        if (policy?.Mode == PrecisionMode.Bfp8
            && descriptor != Bfp8QuantizationDescriptor.TensorWide)
        {
            throw new InvalidOperationException(
                "The bfp8 precision policy requires tensor-wide BFP8 scales; " +
                "block-scaled storage belongs to mix8_32.");
        }
        if (policy?.Mode == PrecisionMode.Mix8_32
            && descriptor.Granularity != Bfp8ScaleGranularity.Block)
        {
            throw new InvalidOperationException(
                "The mix8_32 precision policy requires block-scaled BFP8 " +
                "storage; a tensor-wide scale is not permitted.");
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

    /// <summary>
    /// Acquires a BF16 operand without permanently widening transient BFP8
    /// activations. Leaf tensors retain their versioned decode because their
    /// values are reused by forward and backward; graph intermediates use a
    /// lane-managed transient lease that is returned after kernel enqueue.
    /// </summary>
    internal CudaBfp8BFloat16Lease AcquireCudaBfp8BFloat16Buffer(
        int deviceIndex = -1)
    {
        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        if (Node.IsLeaf)
        {
            return new CudaBfp8BFloat16Lease(
                EnsureCudaBfp8BFloat16Buffer(resolvedDeviceIndex));
        }

        CudaBfp8BufferView source = EnsureCudaBfp8Buffer(resolvedDeviceIndex);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        NativeCudaBuffer<ushort>? decoded = null;
        try
        {
            decoded = RentCudaBFloat16Buffer(resolvedDeviceIndex, Numel);
            CudaBfp8GemmTelemetry.RecordBFloat16DecodeCacheMiss();
            CudaBfp8Native.DequantizeBFloat16(
                resolvedDeviceIndex,
                source.Payload,
                source.Scales,
                decoded,
                source.Descriptor,
                accelerator.DefaultStream);
            return new CudaBfp8BFloat16Lease(decoded, accelerator);
        }
        catch
        {
            if (decoded is not null)
                ReturnCudaBFloat16Buffer(accelerator, decoded);
            throw;
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
            RegisterSessionReplicaLocked(payload);
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndex;
        }
    }

    /// <summary>
    /// Quantizes the completed Float32 backward accumulator and publishes a
    /// tensor-wide signed-Int8 gradient replica.  The Float32 allocation may
    /// remain as a decode cache/arena lease, but it is no longer authoritative.
    /// </summary>
    internal CudaBfp8BufferView PublishCudaBfp8Gradient(
        int deviceIndex = -1,
        nint stream = 0)
    {
        if (DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "Only BFP8 tensors can publish BFP8 gradients.");
        }

        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(resolvedDeviceIndex);
        nint resolvedStream = stream == 0
            ? accelerator.DefaultStream
            : stream;
        using NativeCudaBuffer<int> finiteStatus =
            accelerator.Allocate1D<int>(
                1,
                Cuda.Memory.CudaMemoryKind.Workspace);
        finiteStatus.MemSetToZero();
        CudaBfp8BufferView view = QuantizeCudaBfp8Gradient(
            resolvedDeviceIndex,
            finiteStatus,
            resolvedStream);
        CudaGradientBuckets.Synchronize(
            accelerator,
            resolvedDeviceIndex,
            resolvedStream);
        var finite = new int[1];
        finiteStatus.CopyToCPU(finite);
        if (finite[0] != 0)
        {
            throw new InvalidOperationException(
                $"Non-finite CUDA gradient detected before BFP8 " +
                $"publication for tensor '{Name}' on device " +
                $"{resolvedDeviceIndex}.");
        }
        CommitCudaBfp8Gradient(resolvedDeviceIndex);
        return view;
    }

    internal CudaBfp8BufferView QuantizeCudaBfp8Gradient(
        int deviceIndex,
        NativeCudaBuffer<int> finiteStatus,
        nint stream,
        NativeCudaBuffer<double>? squaredSum = null)
    {
        ArgumentNullException.ThrowIfNull(finiteStatus);
        if (DType != TensorDType.Bfp8
            || Bfp8Quantization != Bfp8QuantizationDescriptor.TensorWide)
        {
            throw new InvalidOperationException(
                "Pure BFP8 gradient publication requires tensor-wide BFP8.");
        }
        NativeCudaBuffer<float> source = EnsureCudaGradientBuffer(deviceIndex);
        lock (_deviceSync)
        {
            Bfp8GradientDeviceBuffer encoded =
                GetOrCreateCudaBfp8GradientBufferLocked(deviceIndex);
            CudaBfp8GradientNative.Quantize(
                deviceIndex,
                source,
                encoded.View,
                finiteStatus,
                stream,
                squaredSum);
            return encoded.View;
        }
    }

    internal void CommitCudaBfp8Gradient(int deviceIndex)
    {
        lock (_deviceSync)
        {
            if (!_cudaBfp8GradientBuffers.TryGetValue(
                    deviceIndex,
                    out Bfp8GradientDeviceBuffer? encoded))
            {
                throw new InvalidOperationException(
                    $"CUDA device {deviceIndex} has no quantized BFP8 " +
                    "gradient to publish.");
            }
            encoded.Version = _gradientVersion;
            _gradientAuthority = GradientStorageAuthority.CudaBfp8;
            _gradientAuthorityDeviceIndex = deviceIndex;
            _hostGradientCurrent = false;
            MarkCudaGradientLocalLocked(deviceIndex);
        }
    }

    internal CudaBfp8BufferView EnsureCudaBfp8GradientBuffer(
        int deviceIndex = -1)
    {
        if (DType != TensorDType.Bfp8)
        {
            throw new InvalidOperationException(
                "A physical CUDA BFP8 gradient requires BFP8 dtype.");
        }

        int resolvedDeviceIndex = ResolveCudaDeviceIndex(deviceIndex);
        lock (_deviceSync)
        {
            if (_cudaBfp8GradientBuffers.TryGetValue(
                    resolvedDeviceIndex,
                    out Bfp8GradientDeviceBuffer? current)
                && current.Version == _gradientVersion
                && _gradientAuthority == GradientStorageAuthority.CudaBfp8
                && IsReplicaUsableInCurrentSession(current.Payload)
                && IsReplicaUsableInCurrentSession(current.Scales))
            {
                RegisterSessionReplicaLocked(current.Payload);
                return current.View;
            }

            if (_gradientAuthority == GradientStorageAuthority.CudaBfp8)
            {
                throw new InvalidOperationException(
                    $"The authoritative BFP8 gradient generation is not " +
                    $"resident on CUDA device {resolvedDeviceIndex}. An " +
                    "explicit device reduction/broadcast is required; host " +
                    "fallback is forbidden.");
            }
        }

        return PublishCudaBfp8Gradient(resolvedDeviceIndex);
    }

    internal bool TryGetCudaBfp8GradientBuffer(
        int deviceIndex,
        out CudaBfp8BufferView view)
    {
        lock (_deviceSync)
        {
            if (_cudaBfp8GradientBuffers.TryGetValue(
                    deviceIndex,
                    out Bfp8GradientDeviceBuffer? encoded)
                && encoded.Version == _gradientVersion
                && IsReplicaUsableInCurrentSession(encoded.Payload)
                && IsReplicaUsableInCurrentSession(encoded.Scales))
            {
                RegisterSessionReplicaLocked(encoded.Payload);
                view = encoded.View;
                return true;
            }
        }
        view = default;
        return false;
    }

    internal bool HasAuthoritativeCudaBfp8Gradient
    {
        get
        {
            lock (_deviceSync)
            {
                return _gradientAuthority
                    == GradientStorageAuthority.CudaBfp8
                    && _cudaBfp8GradientBuffers.Values.Any(
                        buffer => buffer.Version == _gradientVersion
                            && IsReplicaUsableInCurrentSession(
                                buffer.Payload)
                            && IsReplicaUsableInCurrentSession(
                                buffer.Scales));
            }
        }
    }

    internal CudaBfp8BufferView PrepareCudaBfp8GradientReplica(
        int deviceIndex)
    {
        if (DType != TensorDType.Bfp8)
            throw new InvalidOperationException("Tensor dtype must be BFP8.");
        lock (_deviceSync)
            return GetOrCreateCudaBfp8GradientBufferLocked(deviceIndex).View;
    }

    internal void MarkCudaBfp8GradientsSynchronized(
        IReadOnlyList<int> deviceIndices)
        => MarkCudaBfp8GradientsSynchronized(
            deviceIndices,
            PreserveOrCreateGradientReductionStamp(deviceIndices));

    internal void MarkCudaBfp8GradientsSynchronized(
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
                if (!_cudaBfp8GradientBuffers.TryGetValue(
                        deviceIndex,
                        out Bfp8GradientDeviceBuffer? encoded))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no BFP8 gradient " +
                        "replica to publish.");
                }
                encoded.Version = _gradientVersion;
            }
            _gradientAuthority = GradientStorageAuthority.CudaBfp8;
            _gradientAuthorityDeviceIndex = deviceIndices[0];
            _hostGradientCurrent = false;
            CommitCudaGradientReductionLocked(deviceIndices, reductionStamp);
        }
    }

    internal void MarkCudaBfp8DataReplicasSynchronized(
        IReadOnlyList<int> deviceIndices)
    {
        ArgumentNullException.ThrowIfNull(deviceIndices);
        if (DType != TensorDType.Bfp8 || deviceIndices.Count == 0)
        {
            throw new InvalidOperationException(
                "BFP8 data publication requires at least one BFP8 replica.");
        }
        lock (_deviceSync)
        {
            foreach (int deviceIndex in deviceIndices)
            {
                if (!_cudaBfp8Buffers.TryGetValue(
                        deviceIndex,
                        out Bfp8DeviceBuffer? replica)
                    || !IsReplicaUsableInCurrentSession(replica.Payload)
                    || !IsReplicaUsableInCurrentSession(replica.Scales))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has no usable BFP8 data " +
                        "replica in the current execution session.");
                }
                if (_cudaMasterBuffers.TryGetValue(
                        deviceIndex,
                        out DeviceBuffer? master)
                    && !IsReplicaUsableInCurrentSession(master.Buffer))
                {
                    throw new InvalidOperationException(
                        $"CUDA device {deviceIndex} has a stale BFP8 master " +
                        "replica.");
                }
            }
            unchecked
            {
                _dataVersion++;
            }
            foreach (int deviceIndex in deviceIndices)
            {
                _cudaBfp8Buffers[deviceIndex].Version = _dataVersion;
                if (_cudaMasterBuffers.TryGetValue(
                        deviceIndex,
                        out DeviceBuffer? master))
                {
                    // mix8_32 keeps its FP32 master authoritative across
                    // steps. Advancing the block-scaled replica generation
                    // must not make that resident master appear stale and
                    // trigger a parameter-sized host round trip next step.
                    master.Version = _dataVersion;
                }
            }
            _physicalFloat32CacheDataVersion = -1;
            _hostDataCurrent = false;
            _device = TensorDevice.Cuda;
            _cudaDeviceIndex = deviceIndices[0];
        }
    }

    private Bfp8GradientDeviceBuffer
        GetOrCreateCudaBfp8GradientBufferLocked(int deviceIndex)
    {
        if (_cudaBfp8GradientBuffers.TryGetValue(
                deviceIndex,
                out Bfp8GradientDeviceBuffer? existing)
            && existing.Payload.Length == Numel
            && IsReplicaUsableInCurrentSession(existing.Payload)
            && IsReplicaUsableInCurrentSession(existing.Scales))
        {
            RegisterSessionReplicaLocked(existing.Payload);
            return existing;
        }

        existing?.Dispose();
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        NativeCudaBuffer<sbyte>? payload = null;
        NativeCudaBuffer<float>? scales = null;
        try
        {
            payload = accelerator.Allocate1D<sbyte>(Numel);
            scales = accelerator.Allocate1D<float>(1);
            var created = new Bfp8GradientDeviceBuffer(
                payload,
                scales,
                _gradientVersion);
            _cudaBfp8GradientBuffers[deviceIndex] = created;
            RegisterSessionReplicaLocked(payload);
            return created;
        }
        catch
        {
            payload?.Dispose();
            scales?.Dispose();
            throw;
        }
    }

    internal sealed class Bfp8GradientDeviceBuffer : IDisposable
    {
        private int _disposed;

        internal Bfp8GradientDeviceBuffer(
            NativeCudaBuffer<sbyte> payload,
            NativeCudaBuffer<float> scales,
            long version)
        {
            Payload = payload;
            Scales = scales;
            Version = version;
        }

        internal NativeCudaBuffer<sbyte> Payload { get; }
        internal NativeCudaBuffer<float> Scales { get; }
        internal long Version { get; set; }
        internal CudaBfp8BufferView View => new(
            Payload,
            Scales,
            Bfp8QuantizationDescriptor.TensorWide);

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            List<Exception>? failures = null;
            try
            {
                Payload.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            try
            {
                Scales.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA BFP8 gradient payload/scale cleanup failed.",
                    failures);
            }
        }
    }

    internal sealed class Bfp8DeviceBuffer : IDisposable
    {
        private int _disposed;
        private long _version;
        private NativeCudaBuffer<ushort>? _bfloat16Cache;
        private long _bfloat16CacheVersion;
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
                // The decoded operand is a persistent leaf cache. An
                // optimizer publication makes its contents stale, but its
                // allocation is still the right size and belongs to the same
                // device/descriptor replica. Keep the pointer and refresh it
                // in place on the next use instead of introducing a
                // cudaFree/cudaMalloc pair into every training step.
                DisposeColumnMajorCache();
            }
        }
        internal CudaBfp8BufferView View => new(Payload, Scales, Descriptor);

        internal NativeCudaBuffer<ushort> GetOrCreateBFloat16(int deviceIndex)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            // A compiled CUDA training graph retains the cache pointer, but
            // managed version checks do not execute during replay. Record one
            // dequantization node per leaf replica into the graph even when
            // this cache is current at capture time. Without it, optimizer
            // updates leave the graph reading an old BF16 snapshot until an
            // unrelated inference call happens to refresh the same cache.
            bool recordGraphRefresh =
                CudaGraphBfp8ParameterRefreshScope.Register(
                    deviceIndex,
                    this);
            if (_bfloat16Cache is not null
                && _bfloat16CacheVersion == _version
                && !recordGraphRefresh)
            {
                return _bfloat16Cache;
            }

            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            NativeCudaBuffer<ushort>? decoded = _bfloat16Cache;
            bool created = decoded is null;
            try
            {
                decoded ??= accelerator.Allocate1D<ushort>(Payload.Length);
                CudaBfp8GemmTelemetry.RecordBFloat16DecodeCacheMiss();
                CudaBfp8Native.DequantizeBFloat16(
                    deviceIndex,
                    Payload,
                    Scales,
                    decoded,
                    Descriptor,
                    accelerator.DefaultStream);
                _bfloat16Cache = decoded;
                _bfloat16CacheVersion = _version;
                return decoded;
            }
            catch
            {
                // A failed refresh must never publish a current generation.
                // Preserve an existing allocation for a later retry, but
                // release a newly-created one because it has not become part
                // of the replica yet.
                if (created)
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
            List<Exception>? failures = null;
            try
            {
                DisposeBFloat16Cache();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            try
            {
                DisposeColumnMajorCache();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            try
            {
                Payload.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            try
            {
                Scales.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA BFP8 data payload/scale/cache cleanup failed.",
                    failures);
            }
        }
    }
}
