namespace NNtrain;

/// <summary>
/// Owns the authoritative CUDA BF16 gradient target used by pure-bfloat16
/// execution.  Existing BF16 gradients are accumulated in place.  A legacy
/// FP32 CUDA gradient is converted on the same device exactly once, so a DAG
/// can cross an older backward node without synchronizing through the host.
/// </summary>
internal sealed class CudaPureBFloat16GradientTarget : IDisposable
{
    private readonly Tensor _tensor;
    private readonly int _deviceIndex;
    private readonly NativeCudaDevice _accelerator;
    private NativeCudaBuffer<ushort>? _ownedBuffer;
    private bool _committed;

    private CudaPureBFloat16GradientTarget(
        Tensor tensor,
        int deviceIndex,
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> buffer,
        NativeCudaBuffer<ushort>? ownedBuffer,
        bool hasValue)
    {
        _tensor = tensor;
        _deviceIndex = deviceIndex;
        _accelerator = accelerator;
        Buffer = buffer;
        _ownedBuffer = ownedBuffer;
        HasValue = hasValue;
    }

    internal NativeCudaBuffer<ushort> Buffer { get; }

    /// <summary>
    /// True when <see cref="Buffer"/> already contains a logical gradient.
    /// Direct GEMM callers use this as their beta=1/0 decision.  Once a full
    /// contribution has been written, subsequent branches must accumulate.
    /// </summary>
    internal bool HasValue { get; private set; }

    internal static CudaPureBFloat16GradientTarget Acquire(
        Tensor tensor,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        if (tensor.DType != TensorDType.BFloat16
            || !TensorExecutionContext.UsesBFloat16GradientStorage)
        {
            throw new InvalidOperationException(
                "A pure-BF16 CUDA gradient target requires the bfloat16 " +
                "precision policy and BFloat16 tensor storage.");
        }

        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (tensor.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out NativeCudaBuffer<ushort>? existing))
        {
            return new CudaPureBFloat16GradientTarget(
                tensor,
                deviceIndex,
                accelerator,
                existing!,
                ownedBuffer: null,
                hasValue: true);
        }

        NativeCudaBuffer<ushort> rented =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, tensor.Numel);
        try
        {
            bool hasValue = tensor.HasGradientBuffer;
            if (hasValue)
            {
                // EnsureCudaGradientBuffer resolves the current CUDA replica
                // on this device.  Encoding is an ordered device kernel; no
                // host mirror or scalar synchronization is involved.
                CudaTensorNative.EncodeBFloat16(
                    deviceIndex,
                    tensor.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                    rented.NativePtr,
                    tensor.Numel);
            }

            return new CudaPureBFloat16GradientTarget(
                tensor,
                deviceIndex,
                accelerator,
                rented,
                rented,
                hasValue);
        }
        catch
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, rented);
            throw;
        }
    }

    internal void MarkFullContributionWritten() => HasValue = true;

    internal void EnsureZeroInitialized()
    {
        if (HasValue)
            return;
        Buffer.MemSetToZero();
        HasValue = true;
    }

    internal void AccumulateFloat32(
        NativeCudaBuffer<float> contribution,
        int length)
    {
        ArgumentNullException.ThrowIfNull(contribution);
        if (length != _tensor.Numel || contribution.Length < length)
        {
            throw new ArgumentException(
                "CUDA gradient contribution must cover the tensor.",
                nameof(contribution));
        }

        if (!HasValue)
        {
            CudaTensorNative.EncodeBFloat16(
                _deviceIndex,
                contribution.NativePtr,
                Buffer.NativePtr,
                length);
            HasValue = true;
            return;
        }

        NativeCudaBuffer<ushort> encoded =
            Tensor.RentCudaBFloat16Buffer(_deviceIndex, length);
        try
        {
            CudaTensorNative.EncodeBFloat16(
                _deviceIndex,
                contribution.NativePtr,
                encoded.NativePtr,
                length);
            CudaPublicOpsNative.ShapeAccumulateBFloat16Gradient(
                _deviceIndex,
                encoded.NativePtr,
                Buffer.NativePtr,
                length,
                sourceOffset: 0,
                destinationOffset: 0,
                _accelerator.DefaultStream);
        }
        finally
        {
            Tensor.ReturnCudaBFloat16Buffer(_accelerator, encoded);
        }
    }

    internal void Commit()
    {
        if (_committed)
            return;
        if (!HasValue)
        {
            throw new InvalidOperationException(
                "Cannot publish an unwritten CUDA BF16 gradient target.");
        }
        Buffer.MarkGradientStorageDirty();
        if (_ownedBuffer is not null)
        {
            _tensor.AdoptCudaBFloat16GradientBuffer(
                _ownedBuffer,
                _deviceIndex);
            _ownedBuffer = null;
        }
        else
        {
            // Reused BF16 storage still represents a new logical gradient
            // generation. Merely dirtying the bytes leaves zero_grad's
            // coherence state (or an optimizer-consumed version) in place.
            _tensor.MarkCudaBFloat16GradientMutated(_deviceIndex);
        }
        _committed = true;
    }

    public void Dispose()
    {
        if (_ownedBuffer is not null)
        {
            Tensor.ReturnCudaBFloat16Buffer(_accelerator, _ownedBuffer);
            _ownedBuffer = null;
        }
    }
}

/// <summary>
/// Borrows an authoritative BF16 upstream gradient or creates an ordered,
/// device-resident BF16 view of the current FP32 upstream gradient.
/// </summary>
internal sealed class CudaBFloat16GradientSource : IDisposable
{
    private readonly NativeCudaDevice _accelerator;
    private NativeCudaBuffer<ushort>? _ownedBuffer;

    private CudaBFloat16GradientSource(
        NativeCudaDevice accelerator,
        NativeCudaBuffer<ushort> buffer,
        NativeCudaBuffer<ushort>? ownedBuffer)
    {
        _accelerator = accelerator;
        Buffer = buffer;
        _ownedBuffer = ownedBuffer;
    }

    internal NativeCudaBuffer<ushort> Buffer { get; }

    internal static CudaBFloat16GradientSource Acquire(
        Tensor output,
        int deviceIndex)
    {
        ArgumentNullException.ThrowIfNull(output);
        NativeCudaDevice accelerator =
            ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
        if (output.TryGetCudaBFloat16GradientBuffer(
                deviceIndex,
                out NativeCudaBuffer<ushort>? existing))
        {
            return new CudaBFloat16GradientSource(
                accelerator,
                existing!,
                ownedBuffer: null);
        }

        NativeCudaBuffer<ushort> rented =
            Tensor.RentCudaBFloat16Buffer(deviceIndex, output.Numel);
        try
        {
            CudaTensorNative.EncodeBFloat16(
                deviceIndex,
                output.EnsureCudaGradientBuffer(deviceIndex).NativePtr,
                rented.NativePtr,
                output.Numel);
            return new CudaBFloat16GradientSource(
                accelerator,
                rented,
                rented);
        }
        catch
        {
            Tensor.ReturnCudaBFloat16Buffer(accelerator, rented);
            throw;
        }
    }

    public void Dispose()
    {
        if (_ownedBuffer is not null)
        {
            Tensor.ReturnCudaBFloat16Buffer(_accelerator, _ownedBuffer);
            _ownedBuffer = null;
        }
    }
}

/// <summary>
/// Deduplicates gradient targets by tensor identity.  This is required for
/// legal shared-parent graphs such as x+x and for fused operators whose
/// residual, branch, or affine parameters may alias in small test shapes.
/// </summary>
internal sealed class CudaPureBFloat16GradientTargetSet : IDisposable
{
    private readonly int _deviceIndex;
    private readonly Dictionary<Tensor, CudaPureBFloat16GradientTarget>
        _targets = new(ReferenceEqualityComparer.Instance);

    internal CudaPureBFloat16GradientTargetSet(int deviceIndex)
        => _deviceIndex = deviceIndex;

    internal CudaPureBFloat16GradientTarget Get(Tensor tensor)
    {
        if (_targets.TryGetValue(
                tensor,
                out CudaPureBFloat16GradientTarget? target))
        {
            return target;
        }
        target = CudaPureBFloat16GradientTarget.Acquire(
            tensor,
            _deviceIndex);
        _targets.Add(tensor, target);
        return target;
    }

    internal void CommitAll()
    {
        foreach (CudaPureBFloat16GradientTarget target in _targets.Values)
            target.Commit();
    }

    public void Dispose()
    {
        foreach (CudaPureBFloat16GradientTarget target in _targets.Values)
            target.Dispose();
        _targets.Clear();
    }
}
