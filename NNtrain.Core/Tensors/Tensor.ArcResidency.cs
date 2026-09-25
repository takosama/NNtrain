using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain;

partial class Tensor
{
    private ArcReplica? _arcReplica;
    private bool _arcValuesReleased;
    internal static bool ArcResident => ExecutionDevice == TensorDevice.Arc && ArcLane.Options.ResidentTensors;
    internal static bool ArcPackedBackwardGradients => ArcResident
        && TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16;
    internal (long FloatBytes, long BFloat16Bytes) ArcGradientResidentBytes
        => (_arcReplica?.Gradient?.ByteLength ?? 0,
            _arcReplica?.BFloat16Gradient?.ByteLength ?? 0);

    // Packed payload, optimizer master and gradient have independent host authority.
    // Only explicit inspection/checkpoint/session closure materializes host arrays.
    private sealed class ArcReplica(ArcExecutionLane lane) : IDisposable
    {
        internal readonly ArcExecutionLane Lane = lane;
        internal ArcBuffer? Value, Scales, Gradient, Master;
        internal ArcBuffer? BFloat16Gradient, BFloat16Master;
        internal ArcMatrixPanelCache? MatrixPanels;
        internal bool DataDirty, GradientDirty, MasterDirty;
        internal bool GradientIsDelta;
        internal IDisposable? Registration;
        public void Dispose()
        {
            MatrixPanels?.Dispose(); MatrixPanels = null;
            Value?.Dispose(); Scales?.Dispose(); Gradient?.Dispose(); Master?.Dispose();
            BFloat16Gradient?.Dispose(); BFloat16Master?.Dispose();
            Value = Scales = Gradient = Master = BFloat16Gradient = BFloat16Master = null;
            Registration?.Dispose(); Registration = null;
            GC.SuppressFinalize(this);
        }
        ~ArcReplica() { try { Dispose(); } catch { /* lane owns the final native cleanup */ } }
    }

    private ArcReplica ArcOwner()
    {
        if (_arcReplica is { } old && !ReferenceEquals(old.Lane, ArcLane)) ReleaseArcReplica(preserve: true);
        if (_arcReplica is null)
        {
            _arcReplica = new(ArcLane);
            _arcReplica.Registration = ExecutionSession.Current!.RegisterBeforeDispose(this,
                static owner => ((Tensor)owner).ReleaseArcReplica(preserve: true));
        }
        return _arcReplica;
    }

    private void EnsureArcPacked()
    {
        if (_arcValuesReleased) throw new InvalidOperationException("Arc forward values were released by BackwardAndRelease.");
        ArcReplica state = ArcOwner();
        if (state.Value is not null) return;
        EnsureHostDataCurrent();
        if (_data.TryGetBfp8Buffers(out sbyte[] bytes, out float[] scales, out _))
        {
            state.Value = state.Lane.UploadRaw(bytes);
            state.Scales = state.Lane.Upload(scales);
        }
        else if (_data.TryGetBFloat16Buffer(out ushort[] bf16)) state.Value = state.Lane.UploadRaw(bf16);
        else state.Value = state.Lane.Upload(_data.ToFloat32Array());
    }

    private ArcBuffer ArcResidentValues(bool matrixOperand = false)
    {
        EnsureArcPacked();
        return DecodeArcReplica(_arcReplica!, matrixOperand);
    }

    private ArcBuffer DecodeArcReplica(ArcReplica state, bool matrixOperand = false)
    {
        if (DType == TensorDType.Float32) return state.Value!.Borrow();
        var output = state.Lane.Allocate(Numel);
        try
        {
            if (DType == TensorDType.Bfp8)
                state.Lane.Run("decode_bfp8", Numel, 0, state.Value!, state.Scales!, output, Numel,
                    Bfp8Quantization!.GetEffectiveBlockSize(Numel), matrixOperand ? 1 : 0);
            else state.Lane.Run("decode_bf16", Numel, 0, state.Value!, output, Numel);
            return output;
        }
        catch { output.Dispose(); throw; }
    }

    private void PublishArcValues(ArcBuffer source)
    {
        ArcReplica state = ArcOwner();
        state.MatrixPanels?.Clear();
        state.Value ??= state.Lane.AllocateBytes(checked(Numel * (DType == TensorDType.Float32 ? 4 : DType == TensorDType.BFloat16 ? 2 : 1)));
        if (DType == TensorDType.Bfp8)
        {
            int groups = Bfp8Quantization!.GetScaleCount(Numel);
            state.Scales ??= state.Lane.Allocate(groups);
            int block = Bfp8Quantization.GetEffectiveBlockSize(Numel);
            bool coalesced = state.Lane.Options.CoalescedBfp8Publication && state.Lane.Options.XmxMatrices
                && state.Lane.Device.SupportsXmx && state.Lane.Device.MinimumSubgroupSize == 16 && block is 32 or 128;
            string kernel = !coalesced ? "resident_bfp8" : block == 32 ? "resident_bfp8_quad4_32" : "resident_bfp8_sg16_128";
            state.Lane.Run(kernel, groups * (coalesced ? block == 32 ? 4L : 16L : 1L), coalesced ? 256 : 0,
                source, state.Value, state.Scales, state.Lane.NumericStatus, Numel, block);
        }
        else if (DType == TensorDType.BFloat16) state.Lane.Run("resident_bf16", Numel, 0, source, state.Value, Numel);
        else state.Lane.Run("copy_scale", Numel, 0, source, state.Value, Numel, 1f, 0);
        state.DataDirty = true;
        _device = TensorDevice.Arc;
        _arcDeviceIndex = state.Lane.DeviceIndex;
    }

    private static Tensor ArcDeviceResult(ArcBuffer values, int[] shape, Tensor[] parents, TensorDType? dtype = null)
    {
        TensorDType format = dtype ?? (ArcMayPublishBFloat16Activation(parents)
            ? TensorDType.BFloat16 : TensorDTypeContract.Promote(parents));
        if (format is not (TensorDType.Float32 or TensorDType.BFloat16 or TensorDType.Bfp8))
            throw new NotSupportedException($"Arc does not implement storage format {format}.");
        int length = checked(shape.Aggregate(1, (a, b) => checked(a * b)));
        TensorStorage placeholder = format == TensorDType.Bfp8
            ? TensorStorage.CreateDeviceBfp8Placeholder(length, parents.Any(p => p.DType == TensorDType.Bfp8)
                ? SelectBfp8ResultDescriptor(parents) : Bfp8QuantizationDescriptor.TensorWide)
            : TensorStorage.CreateDevicePlaceholder(length, format);
        Tensor result = FromStorageResult(placeholder, shape, parents);
        try { result.PublishArcValues(values); ArcInferenceFrame.Current?.Add(result); ArcCheckpointFrame.Current?.Add(result); return result; }
        catch { result.ReleaseArcReplica(preserve: false); throw; }
    }

    // Takes ownership of an already encoded BF16 result. The producing kernel
    // has performed the usual publication rounding, so no FP32 copy or second
    // resident_bf16 launch is needed.
    private static Tensor ArcDeviceBFloat16Result(ArcBuffer values, int[] shape, Tensor[] parents)
    {
        int length = checked(shape.Aggregate(1, (a, b) => checked(a * b)));
        if (values.ByteLength != checked((long)length * sizeof(ushort)))
            throw new ArgumentException("Encoded BF16 result has the wrong byte length.", nameof(values));
        Tensor result = FromStorageResult(TensorStorage.CreateDevicePlaceholder(length, TensorDType.BFloat16),
            shape, parents);
        try
        {
            ArcReplica state = result.ArcOwner();
            state.Value = values;
            state.DataDirty = true;
            result._device = TensorDevice.Arc;
            result._arcDeviceIndex = state.Lane.DeviceIndex;
            ArcInferenceFrame.Current?.Add(result);
            ArcCheckpointFrame.Current?.Add(result);
            return result;
        }
        catch { result.ReleaseArcReplica(preserve: false); throw; }
    }

    private static bool ArcMayPublishBFloat16Activation(Tensor[] parents)
    {
        PrecisionPolicy? policy = TensorExecutionContext.ActivePrecisionPolicy;
        return ArcLane.Options.Mix8_16Bf16Activations
            && policy is { Mode: PrecisionMode.Mix8_16,
                NonWeightSelection: NonWeightSelectionPolicy.FastestAvailable }
            && (policy.AllowedActivationStorageFormats & NumericFormatSet.BFloat16) != 0
            && parents.Length > 0
            && parents.All(parent => parent.DType is TensorDType.Bfp8 or TensorDType.BFloat16);
    }

    internal ArcBuffer ArcGradient()
    {
        ArcReplica state = ArcOwner();
        if (state.GradientIsDelta && !AutogradEngine.IsArcBackwardTraversal)
            _ = ArcBFloat16Gradient();
        if (state.Gradient is null)
        {
            state.Gradient = state.Lane.Allocate(Numel);
            if (state.BFloat16Gradient is not null)
            {
                bool freshDelta = ArcPackedBackwardGradients && Node.IsLeaf
                    && AutogradEngine.IsArcBackwardTraversal
                    && state.Lane.Options.Mix8_16FusedGradientAccumulation;
                if (freshDelta)
                {
                    state.Lane.Run("resident_zero", Numel, 0, state.Gradient, Numel);
                    state.GradientIsDelta = true;
                }
                else
                {
                    state.Lane.Run("unpack_bf16_state", Numel, 0,
                        state.BFloat16Gradient, state.Gradient, Numel);
                    state.BFloat16Gradient.Dispose();
                    state.BFloat16Gradient = null;
                }
            }
            else if (_grad.Length == Numel)
                state.Lane.Write(state.Gradient, _grad);
            else
                state.Lane.Run("resident_zero", Numel, 0, state.Gradient, Numel);
        }
        // Conservatively dirty on access: kernels can accumulate into it.
        state.GradientDirty = true;
        return state.Gradient;
    }

    /// <summary>Returns the physical BF16 gradient retained after backward/clip.</summary>
    internal ArcBuffer ArcBFloat16Gradient()
    {
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode != PrecisionMode.Mix8_16)
            throw new InvalidOperationException("A BF16 Arc gradient requires mix8_16 precision.");
        ArcReplica state = ArcOwner();
        if (state.BFloat16Gradient is not null && state.GradientIsDelta)
        {
            state.Lane.Run("add_float_to_bf16_state", Numel, 0,
                state.BFloat16Gradient, state.Gradient!, Numel);
            state.Gradient?.Dispose();
            state.Gradient = null;
            state.GradientIsDelta = false;
        }
        else if (state.BFloat16Gradient is null)
        {
            ArcBuffer floatGradient = ArcGradient();
            ArcBuffer packed = state.Lane.AllocateBytes(checked(Numel * sizeof(ushort)));
            try
            {
                state.Lane.Run("pack_bf16_state", Numel, 0, floatGradient, packed, Numel);
                state.BFloat16Gradient = packed;
                state.Gradient?.Dispose();
                state.Gradient = null;
                state.GradientIsDelta = false;
            }
            catch { packed.Dispose(); throw; }
        }
        state.GradientDirty = true;
        return state.BFloat16Gradient;
    }

    /// <summary>
    /// Once the final graph consumer of a leaf has run, retain its gradient
    /// in BF16 until the next microbatch or optimizer consumer needs it.
    /// </summary>
    internal void PackArcLeafGradientAfterBackward()
    {
        if (!ArcPackedBackwardGradients
            || (_arcReplica?.Gradient is null && _grad.Length != Numel))
            return;
        _ = ArcBFloat16Gradient();
    }

    /// <summary>
    /// Rounds the FP32 backward accumulator at the numerical BF16 boundary.
    /// Packing it into physical BF16 storage is deferred until the final
    /// gradient consumer requests ArcBFloat16Gradient.
    /// </summary>
    internal void RoundArcGradientToBFloat16()
    {
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode != PrecisionMode.Mix8_16)
            return;
        ArcBuffer gradient = ArcGradient();
        ArcLane.Run("round_bf16_values", Numel, 0, gradient, Numel);
    }

    /// <summary>
    /// Adds a host-staged FP32 gradient from another Arc lane to this lane's
    /// resident gradient. The optimizer continues to own the primary copy.
    /// </summary>
    internal void AccumulateArcGradient(float[] other)
    {
        if (!ArcResident || other is null || other.Length != Numel)
            throw new ArgumentException("Arc gradient reduction requires a matching resident gradient.", nameof(other));
        ArcBuffer destination = ArcGradient();
        ArcExecutionLane lane = ArcLane;
        using ArcBuffer source = lane.Upload(other);
        lane.Run("add", Numel, 0, destination, source, destination, Numel);
    }

    internal ushort[] CaptureArcBFloat16Gradient()
    {
        ArcReplica state = ArcOwner();
        ushort[] result = new ushort[Numel];
        state.Lane.ReadRaw(ArcBFloat16Gradient(), result);
        return result;
    }

    internal void AccumulateArcBFloat16Gradient(ushort[] other)
    {
        if (!ArcResident || other is null || other.Length != Numel)
            throw new ArgumentException("Arc BF16 gradient reduction requires matching resident gradients.", nameof(other));
        if (ArcLane.Options.Mix8_16DirectGradientReduction)
        {
            using ArcBuffer packedSource = ArcLane.UploadRaw(other);
            ArcLane.Run("add_bf16_to_bf16_state", Numel, 0,
                ArcBFloat16Gradient(), packedSource, Numel);
            return;
        }
        ArcBuffer destination = ArcGradient();
        using ArcBuffer source = ArcLane.UploadRaw(other);
        ArcLane.Run("add_bf16_state_to_float", Numel, 0, destination, source, Numel);
        _ = ArcBFloat16Gradient();
    }

    internal ArcBuffer ArcMaster()
    {
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16)
            throw new InvalidOperationException("mix8_16 uses ArcBFloat16Master for optimizer state.");
        ArcReplica state = ArcOwner();
        if (state.Master is null)
            state.Master = state.Lane.Upload(DataBuffer);
        return state.Master;
    }

    /// <summary>Returns the physical two-byte master weight for mix8_16.</summary>
    internal ArcBuffer ArcBFloat16Master()
    {
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode != PrecisionMode.Mix8_16)
            throw new InvalidOperationException("A BF16 Arc master requires mix8_16 precision.");
        ArcReplica state = ArcOwner();
        if (state.BFloat16Master is null)
            state.BFloat16Master = state.Lane.UploadRaw(GetOrCreateHostBFloat16Master());
        return state.BFloat16Master;
    }

    internal void CompleteArcUpdate()
    {
        ArcReplica state = ArcOwner();
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode == PrecisionMode.Mix8_16)
        {
            if (DType != TensorDType.Bfp8 || state.BFloat16Master is null)
                throw new InvalidOperationException("mix8_16 requires a BFP8 parameter and a resident BF16 master.");
            state.MatrixPanels?.Clear();
            state.Value ??= state.Lane.AllocateBytes(Numel);
            int groups = Bfp8Quantization!.GetScaleCount(Numel);
            state.Scales ??= state.Lane.Allocate(groups);
            int block = Bfp8Quantization.GetEffectiveBlockSize(Numel);
            bool coalesced = block == 32 && state.Lane.Options.CoalescedBfp8Publication
                && state.Lane.Options.XmxMatrices && state.Lane.Device.SupportsXmx
                && state.Lane.Device.MinimumSubgroupSize == 16;
            state.Lane.Run(coalesced ? "resident_bfp8_from_bf16_quad4_32" : "resident_bfp8_from_bf16",
                groups * (coalesced ? 4L : 1L), coalesced ? 256 : 0,
                state.BFloat16Master, state.Value, state.Scales, state.Lane.NumericStatus,
                Numel, block);
            state.DataDirty = true;
            _device = TensorDevice.Arc;
            _arcDeviceIndex = state.Lane.DeviceIndex;
        }
        else PublishArcValues(state.Master!);
        state.MasterDirty = true;
        unchecked { _dataVersion++; }
        _physicalFloat32CacheDataVersion = -1;
    }

    private void SynchronizeArcHostData()
    {
        if (_arcValuesReleased) throw new InvalidOperationException("Arc forward values were released by BackwardAndRelease.");
        if (_arcReplica is not { } state) return;
        if (state.DataDirty)
        {
            if (DType == TensorDType.Bfp8)
            {
                state.Lane.CheckNumericStatus();
                var bytes = new sbyte[Numel]; var scales = new float[Bfp8Quantization!.GetScaleCount(Numel)];
                state.Lane.ReadRaw(state.Value!, bytes); state.Lane.Read(state.Scales!, scales);
                _data.CopyFromBfp8Encoded(bytes, scales);
            }
            else if (DType == TensorDType.BFloat16)
            {
                var bytes = new ushort[Numel]; state.Lane.ReadRaw(state.Value!, bytes);
                _data.TryGetBFloat16Buffer(out ushort[] host);
                bytes.CopyTo(host, 0);
            }
            else
            {
                float[] host = _data.GetMutableFloat32Buffer();
                state.Lane.Read(state.Value!, host);
            }
            state.DataDirty = false;
        }
        if (state.MasterDirty)
        {
            if (state.BFloat16Master is not null)
            {
                ushort[] packed = new ushort[Numel];
                state.Lane.ReadRaw(state.BFloat16Master, packed);
                AdoptHostBFloat16Master(packed);
            }
            else
            {
                float[] host = DType == TensorDType.Float32
                    ? _data.GetMutableFloat32Buffer()
                    : _masterData ??= new float[Numel];
                state.Lane.Read(state.Master!, host);
            }
            state.MasterDirty = false;
        }
    }

    private void SynchronizeArcHostGradient()
    {
        if (_arcReplica is not { GradientDirty: true } state) return;
        if (state.GradientIsDelta)
        {
            state.Lane.Run("add_float_to_bf16_state", Numel, 0,
                state.BFloat16Gradient!, state.Gradient!, Numel);
            state.Gradient!.Dispose();
            state.Gradient = null;
            state.GradientIsDelta = false;
        }
        if (_grad.Length != Numel) _grad = new float[Numel];
        if (state.Gradient is not null) state.Lane.Read(state.Gradient, _grad);
        else if (state.BFloat16Gradient is not null)
        {
            ushort[] packed = new ushort[Numel];
            state.Lane.ReadRaw(state.BFloat16Gradient, packed);
            for (int i = 0; i < packed.Length; i++)
                _grad[i] = TensorStorageCodec.DecodeBFloat16(packed[i]);
        }
        state.GradientDirty = false;
    }

    private void InvalidateArcGradient()
    {
        if (_arcReplica is not { } state) return;
        state.Gradient?.Dispose(); state.BFloat16Gradient?.Dispose();
        state.Gradient = state.BFloat16Gradient = null; state.GradientDirty = false;
        state.GradientIsDelta = false;
    }

    private bool ClearArcGradient()
    {
        if (_arcReplica is not { } state || (state.Gradient is null && state.BFloat16Gradient is null)) return false;
        state.Gradient?.Dispose(); state.BFloat16Gradient?.Dispose();
        state.Gradient = state.BFloat16Gradient = null;
        state.GradientDirty = false;
        state.GradientIsDelta = false;
        _grad = [];
        return true;
    }

    private void InvalidateArcValues()
    {
        if (_arcReplica is not { } state) return;
        state.MatrixPanels?.Clear();
        state.Value?.Dispose(); state.Scales?.Dispose(); state.Master?.Dispose(); state.BFloat16Master?.Dispose();
        state.Value = state.Scales = state.Master = state.BFloat16Master = null;
        state.DataDirty = state.MasterDirty = false;
    }

    private void ReleaseArcReplica(bool preserve)
    {
        ArcReplica? state = _arcReplica;
        if (state is null) return;
        try { if (preserve) { SynchronizeArcHostData(); SynchronizeArcHostGradient(); } }
        finally { state.Dispose(); _arcReplica = null; }
    }

    internal void ReleaseArcGraph()
    {
        // Preserve an already-read scalar loss; do not transfer discarded activations.
        if (_arcReplica is { DataDirty: true }) _arcValuesReleased = true;
        ReleaseArcReplica(preserve: false);
        _grad = [];
    }

    internal static IDisposable? BeginArcInferenceFrame() => ArcResident ? new ArcInferenceFrame() : null;
    private sealed class ArcInferenceFrame : IDisposable
    {
        [ThreadStatic] internal static ArcInferenceFrame? Current;
        private readonly ArcInferenceFrame? _previous = Current;
        private readonly List<Tensor> _values = [];
        internal ArcInferenceFrame() => Current = this;
        internal void Add(Tensor value) { if (value.Node.IsDetached) _values.Add(value); }
        public void Dispose() { foreach (Tensor value in _values) value.ReleaseArcGraph(); _values.Clear(); Current = _previous; }
    }
}
