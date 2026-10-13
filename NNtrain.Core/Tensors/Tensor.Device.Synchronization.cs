using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
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
        SynchronizeArcHostData();
        ThrowIfForwardValuesRetired();
        lock (_deviceSync)
            SynchronizeHostFromCudaLocked();
    }

    private void EnsureHostGradientCurrent()
    {
        SynchronizeArcHostGradient();
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
        InvalidateArcGradient();
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


}
