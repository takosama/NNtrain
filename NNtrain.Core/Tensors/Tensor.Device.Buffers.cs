using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
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


}
