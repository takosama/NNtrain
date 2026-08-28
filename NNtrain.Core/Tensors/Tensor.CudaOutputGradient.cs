namespace NNtrain;

public partial class Tensor
{
    /// <summary>
    /// Publishes an autograd-root seed on the owning CUDA device. Scalar
    /// roots stay argument-only; an explicit vector seed is dynamic input and
    /// is uploaded once into a reusable staging buffer, then accumulated on
    /// the compute stream. The training transfer guard deliberately rejects
    /// this vector path, because production training must root backward at a
    /// device-resident scalar loss.
    /// </summary>
    internal bool TryAccumulateCudaOutputGradient(float[]? seed)
    {
        if (ExecutionDevice != TensorDevice.Cuda)
            return false;
        if (Numel == 1)
            return TryAccumulateCudaOutputGradientScalar(
                seed is null ? 1f : seed[0]);
        if (seed is null)
            return false;

        int deviceIndex = ResolveCudaDeviceIndex(-1);
        NativeCudaBuffer<float> staging =
            EnsureCudaStagingBuffer(deviceIndex);
        staging.CopyFromCPU(seed);
        if (DType == TensorDType.BFloat16
            && TensorExecutionContext.UsesBFloat16GradientStorage)
        {
            // Pure BF16 means the root gradient is BF16-authoritative too.
            // Encode the explicit seed directly into the resident target;
            // publishing an FP32 gradient here would silently turn the first
            // leaf of an otherwise pure-BF16 graph into mixed precision.
            using CudaPureBFloat16GradientTarget target =
                CudaPureBFloat16GradientTarget.Acquire(this, deviceIndex);
            target.AccumulateFloat32(staging, Numel);
            target.Commit();
            return true;
        }
        NativeCudaBuffer<float> gradient =
            EnsureCudaGradientBuffer(deviceIndex);
        CudaTensorNative.Accumulate(
            deviceIndex,
            staging.NativePtr,
            gradient.NativePtr,
            Numel);
        MarkCudaGradientMutated(deviceIndex);
        return true;
    }

    /// <summary>
    /// Accumulates a scalar autograd-root seed without publishing a host
    /// gradient or issuing an H2D copy. Returns false when this tensor is not
    /// a CUDA-resident scalar so the ordinary CPU/non-scalar path can run.
    /// </summary>
    internal bool TryAccumulateCudaOutputGradientScalar(float seed)
    {
        if (Numel != 1 || ExecutionDevice != TensorDevice.Cuda)
            return false;

        int deviceIndex = ResolveCudaDeviceIndex(-1);
        lock (_deviceSync)
        {
            GradientDeviceBuffer gradient;
            bool accumulate;
            float launchValue = seed;

            if (_cudaGradientBuffers.TryGetValue(
                    deviceIndex,
                    out GradientDeviceBuffer? current)
                && current.Buffer.Length == 1
                && current.Version == _gradientVersion
                && IsReplicaUsableInCurrentSession(current.Buffer))
            {
                gradient = current;
                accumulate = true;
            }
            else if (_cudaBfp8GradientBuffers.TryGetValue(
                    deviceIndex,
                    out Bfp8GradientDeviceBuffer? bfp8)
                && bfp8.Version == _gradientVersion
                && IsReplicaUsableInCurrentSession(bfp8.Payload)
                && IsReplicaUsableInCurrentSession(bfp8.Scales))
            {
                gradient = GetOrCreateScalarFloatGradientBufferLocked(
                    deviceIndex,
                    current);
                CudaBfp8Native.DequantizeFloat32(
                    deviceIndex,
                    bfp8.Payload,
                    bfp8.Scales,
                    gradient.Buffer,
                    Bfp8QuantizationDescriptor.TensorWide);
                gradient.Version = _gradientVersion;
                accumulate = true;
            }
            else if (_cudaBFloat16GradientBuffers.TryGetValue(
                    deviceIndex,
                    out BFloat16GradientDeviceBuffer? bfloat16)
                && bfloat16.Version == _gradientVersion
                && IsReplicaUsableInCurrentSession(bfloat16.Buffer))
            {
                gradient = GetOrCreateScalarFloatGradientBufferLocked(
                    deviceIndex,
                    current);
                CudaTensorNative.DecodeBFloat16(
                    deviceIndex,
                    bfloat16.Buffer.NativePtr,
                    gradient.Buffer.NativePtr,
                    1);
                gradient.Version = _gradientVersion;
                accumulate = true;
            }
            else if (_hostGradientCurrent)
            {
                // A prior CPU traversal may have left a scalar host gradient.
                // Fold it into the kernel argument and assign on-device; this
                // preserves accumulation without copying a one-element array.
                launchValue += _grad.Length == 0 ? 0f : _grad[0];
                gradient = GetOrCreateScalarFloatGradientBufferLocked(
                    deviceIndex,
                    current);
                gradient.Version = _gradientVersion;
                accumulate = false;
            }
            else
            {
                throw new InvalidOperationException(
                    $"The authoritative scalar gradient is not resident on " +
                    $"CUDA device {deviceIndex}; an explicit replica " +
                    "synchronization is required before backward.");
            }

            CudaTensorNative.AccumulateScalar(
                deviceIndex,
                gradient.Buffer.NativePtr,
                launchValue,
                accumulate);
            MarkCudaGradientMutated(deviceIndex);
            if (DType == TensorDType.BFloat16
                && TensorExecutionContext.UsesBFloat16GradientStorage)
            {
                // The scalar kernel takes its value as a launch argument and
                // therefore needs no H2D seed upload. Convert and publish on
                // the same stream before returning so the storage authority
                // still obeys the pure-BF16 policy.
                using CudaPureBFloat16GradientTarget target =
                    CudaPureBFloat16GradientTarget.Acquire(
                        this,
                        deviceIndex);
                target.Commit();
            }
            return true;
        }
    }

    private GradientDeviceBuffer GetOrCreateScalarFloatGradientBufferLocked(
        int deviceIndex,
        GradientDeviceBuffer? current)
    {
        if (current is not null
            && current.Buffer.Length == 1
            && IsReplicaUsableInCurrentSession(current.Buffer))
        {
            RegisterSessionReplicaLocked(current.Buffer);
            return current;
        }

        current?.Dispose();
        var replacement = new GradientDeviceBuffer(
            CudaFloatBufferPool.Rent(deviceIndex, 1),
            _gradientVersion,
            deviceIndex);
        _cudaGradientBuffers[deviceIndex] = replacement;
        RegisterSessionReplicaLocked(replacement.Buffer);
        return replacement;
    }
}
