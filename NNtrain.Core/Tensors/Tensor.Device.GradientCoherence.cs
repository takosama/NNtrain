using NNtrain.Runtime.Execution;

namespace NNtrain;

public partial class Tensor
{
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


}
