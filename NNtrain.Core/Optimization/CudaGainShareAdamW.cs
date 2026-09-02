using NNtrain.Cuda.Interop;
using NNtrain.Runtime.Execution;
using System.Runtime.InteropServices;

namespace NNtrain;

/// <summary>
/// Owns the fully resident CUDA implementation of GainShareAdamW.  The
/// managed optimizer remains the CPU reference and checkpoint facade; this
/// object owns precision-specific device state and deterministic lifetime.
/// </summary>
internal sealed class CudaGainShareAdamW : IDisposable
{
    private readonly IReadOnlyList<Parameter> _parameters;
    private readonly int[] _parameterGroupIndices;
    private readonly int _groupCount;
    private DeviceState[] _deviceStates = [];
    private PrecisionMode? _mode;
    private int _disposed;

    internal CudaGainShareAdamW(
        IReadOnlyList<Parameter> parameters,
        int[] parameterGroupIndices,
        int groupCount)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        ArgumentNullException.ThrowIfNull(parameterGroupIndices);
        if (parameters.Count == 0
            || parameterGroupIndices.Length != parameters.Count)
        {
            throw new ArgumentException(
                "GainShare CUDA parameter metadata is inconsistent.");
        }
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(groupCount);
        _parameters = parameters;
        _parameterGroupIndices = parameterGroupIndices;
        _groupCount = groupCount;
    }

    internal void Prepare(
        GainShareAdamWState hostState,
        IReadOnlyList<int> devices)
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0, this);
        ArgumentNullException.ThrowIfNull(hostState);
        int[] normalizedDevices = NormalizeDevices(devices);
        PrecisionMode mode = ResolveAndValidateMode();
        if (_mode == mode
            && _deviceStates.Length == normalizedDevices.Length
            && _deviceStates.Select(state => state.DeviceIndex)
                .SequenceEqual(normalizedDevices))
        {
            return;
        }

        // A precision/device transition is outside the training hot path.
        // Preserve the authoritative optimizer state before replacing it.
        if (_deviceStates.Length != 0)
            SynchronizeHost(hostState);
        DisposeDeviceStates();

        var created = new DeviceState[normalizedDevices.Length];
        try
        {
            for (int slot = 0; slot < normalizedDevices.Length; slot++)
            {
                created[slot] = DeviceState.Create(
                    normalizedDevices[slot],
                    mode,
                    _parameters,
                    hostState,
                    _groupCount);
            }
            _deviceStates = created;
            _mode = mode;
        }
        catch
        {
            DisposeAll(created);
            throw;
        }
    }

    internal void Step(
        GainShareAdamWState state,
        IReadOnlyList<int> devices)
    {
        Prepare(state, devices);
        PrecisionMode mode = _mode
            ?? throw new InvalidOperationException(
                "GainShare CUDA precision was not prepared.");
        GainShareAdamWOptions options = state.Options;
        float inverseBiasCorrection1 = 1f
            / (1f - MathF.Pow(options.Beta1, state.Step));
        float inverseBiasCorrection2 = 1f
            / (1f - MathF.Pow(options.Beta2, state.Step));

        if (mode == PrecisionMode.Bfp8)
        {
            StepBfp8(
                options,
                inverseBiasCorrection1,
                inverseBiasCorrection2);
            return;
        }

        foreach (DeviceState device in _deviceStates)
        {
            device.BeginStep(mode);
            for (int parameterIndex = 0;
                parameterIndex < _parameters.Count;
                parameterIndex++)
            {
                PrepareDirection(
                    device,
                    mode,
                    parameterIndex,
                    options,
                    inverseBiasCorrection1,
                    inverseBiasCorrection2);
            }

            Check(
                CudaNativeGateway.GainShareComputeScales(
                    device.DeviceIndex,
                    device.GroupStats.NativePtr,
                    device.AlignmentEma.NativePtr,
                    device.Scales.NativePtr,
                    _groupCount,
                    options.Rho,
                    options.Gamma,
                    options.MinScale,
                    options.MaxScale,
                    options.Epsilon,
                    device.FinitePointer(mode),
                    device.Stream),
                "GainShare CUDA group scale");

            for (int parameterIndex = 0;
                parameterIndex < _parameters.Count;
                parameterIndex++)
            {
                ApplyParameter(
                    device,
                    mode,
                    parameterIndex,
                    options);
            }
        }

        int[] activeDevices = _deviceStates
            .Select(device => device.DeviceIndex)
            .ToArray();
        bool readFiniteStatus = mode is PrecisionMode.Bfp8
            or PrecisionMode.Mix8_32;
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            activeDevices,
            $"{mode} GainShareAdamW update",
            queueReadback: readFiniteStatus
                ? QueueFiniteStatusReadbacks
                : null,
            finalize: () => FinalizeStep(activeDevices, readFiniteStatus));
    }

    private void StepBfp8(
        GainShareAdamWOptions options,
        float inverseBiasCorrection1,
        float inverseBiasCorrection2)
    {
        foreach (DeviceState device in _deviceStates)
        {
            device.BeginStep(PrecisionMode.Bfp8);
            Bfp8MultiTensorPlan plan = device.GetBfp8Plan(
                _parameters,
                _parameterGroupIndices);
            Check(
                CudaNativeGateway.GainSharePrepareBfp8MultiTensor(
                    device.DeviceIndex,
                    plan.Descriptors.NativePtr,
                    plan.TensorCount,
                    plan.MaximumChunks,
                    plan.Reduction.NativePtr,
                    device.GroupStats.NativePtr,
                    options.Beta1,
                    options.Beta2,
                    inverseBiasCorrection1,
                    inverseBiasCorrection2,
                    options.Epsilon,
                    device.FiniteStatus.NativePtr,
                    device.Stream),
                "GainShare CUDA fused BFP8 moments/direction");
            Check(
                CudaNativeGateway.GainShareComputeScales(
                    device.DeviceIndex,
                    device.GroupStats.NativePtr,
                    device.AlignmentEma.NativePtr,
                    device.Scales.NativePtr,
                    _groupCount,
                    options.Rho,
                    options.Gamma,
                    options.MinScale,
                    options.MaxScale,
                    options.Epsilon,
                    device.FiniteStatus.NativePtr,
                    device.Stream),
                "GainShare CUDA BFP8 group scale");
            Check(
                CudaNativeGateway.GainShareApplyBfp8MultiTensor(
                    device.DeviceIndex,
                    plan.Descriptors.NativePtr,
                    plan.TensorCount,
                    plan.MaximumChunks,
                    plan.Reduction.NativePtr,
                    device.Scales.NativePtr,
                    options.LearningRate,
                    options.WeightDecay,
                    options.Decay1D,
                    device.FiniteStatus.NativePtr,
                    device.Stream),
                "GainShare CUDA fused BFP8 parameter update");
        }

        int[] activeDevices = _deviceStates
            .Select(device => device.DeviceIndex)
            .ToArray();
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            activeDevices,
            "Bfp8 GainShareAdamW update",
            queueReadback: QueueFiniteStatusReadbacks,
            finalize: () => FinalizeStep(activeDevices, readFiniteStatus: true));
    }

    internal void SynchronizeHost(GainShareAdamWState hostState)
    {
        ArgumentNullException.ThrowIfNull(hostState);
        if (_deviceStates.Length == 0)
            return;
        DeviceState primary = _deviceStates[0];
        primary.Accelerator.Synchronize(
            "GainShareAdamW checkpoint synchronization");
        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            ParameterDeviceState deviceState =
                primary.Parameters[parameterIndex];
            GainShareAdamWParameterState destination =
                hostState.ParameterStates[parameterIndex];
            deviceState.CopyMomentsToHost(
                destination.FirstMoment,
                destination.SecondMoment);
        }

        var alignment = new float[_groupCount];
        primary.AlignmentEma.CopyToCPU(alignment);
        for (int group = 0; group < _groupCount; group++)
        {
            float value = alignment[group];
            hostState.GroupStates[group] = hostState.GroupStates[group] with
            {
                AlignmentEma = float.IsNaN(value) ? null : value,
            };
        }
    }

    private void PrepareDirection(
        DeviceState device,
        PrecisionMode mode,
        int parameterIndex,
        GainShareAdamWOptions options,
        float inverseBiasCorrection1,
        float inverseBiasCorrection2)
    {
        Parameter parameter = _parameters[parameterIndex];
        ParameterDeviceState parameterState =
            device.Parameters[parameterIndex];
        int groupIndex = _parameterGroupIndices[parameterIndex];
        switch (mode)
        {
            case PrecisionMode.Float32:
            case PrecisionMode.Mix16_32:
            case PrecisionMode.Mix8_32:
            {
                NativeCudaBuffer<float> gradient =
                    parameter.T.EnsureCudaGradientBuffer(device.DeviceIndex);
                Check(
                    CudaNativeGateway.GainSharePrepareFloat32(
                        device.DeviceIndex,
                        gradient.NativePtr,
                        parameterState.FirstFloat!.NativePtr,
                        parameterState.SecondFloat!.NativePtr,
                        parameterState.Direction.NativePtr,
                        device.GroupStats.NativePtr,
                        groupIndex,
                        parameter.T.Numel,
                        options.Beta1,
                        options.Beta2,
                        inverseBiasCorrection1,
                        inverseBiasCorrection2,
                        options.Epsilon,
                        device.FinitePointer(mode),
                        device.Stream),
                    "GainShare CUDA FP32 direction/reduction");
                break;
            }
            case PrecisionMode.BFloat16:
            {
                if (!parameter.T.TryGetCudaBFloat16GradientBuffer(
                        device.DeviceIndex,
                        out NativeCudaBuffer<ushort>? gradient))
                {
                    throw new InvalidOperationException(
                        $"Pure BFloat16 GainShareAdamW requires a resident " +
                        $"BF16 gradient for parameter '{parameter.Name}' " +
                        $"on CUDA device {device.DeviceIndex}.");
                }
                Check(
                    CudaNativeGateway.GainSharePrepareBFloat16(
                        device.DeviceIndex,
                        gradient!.NativePtr,
                        parameterState.FirstBFloat16!.NativePtr,
                        parameterState.SecondBFloat16!.NativePtr,
                        parameterState.Direction.NativePtr,
                        device.GroupStats.NativePtr,
                        groupIndex,
                        parameter.T.Numel,
                        options.Beta1,
                        options.Beta2,
                        inverseBiasCorrection1,
                        inverseBiasCorrection2,
                        options.Epsilon,
                        0,
                        device.Stream),
                    "GainShare CUDA BF16 direction/reduction");
                break;
            }
            default:
                throw new InvalidOperationException(
                    $"Unsupported GainShare precision '{mode}'.");
        }
    }

    private void ApplyParameter(
        DeviceState device,
        PrecisionMode mode,
        int parameterIndex,
        GainShareAdamWOptions options)
    {
        Parameter parameter = _parameters[parameterIndex];
        ParameterDeviceState parameterState =
            device.Parameters[parameterIndex];
        int groupIndex = _parameterGroupIndices[parameterIndex];
        bool applyWeightDecay =
            parameter.WeightDecay == WeightDecayPolicy.Apply
            || (options.Decay1D && parameter.T.Rank == 1);
        switch (mode)
        {
            case PrecisionMode.Float32:
            {
                NativeCudaBuffer<float> data =
                    parameter.T.EnsureCudaMasterFloat32Buffer(
                        device.DeviceIndex);
                ApplyFloat32(
                    device, parameterState, data, 0, groupIndex,
                    parameter.T.Numel, options, applyWeightDecay, mode);
                break;
            }
            case PrecisionMode.Mix16_32:
            {
                NativeCudaBuffer<float> master =
                    parameter.T.EnsureCudaMasterFloat32Buffer(
                        device.DeviceIndex);
                NativeCudaBuffer<ushort> physical =
                    parameter.T.EnsureCudaBFloat16Buffer(device.DeviceIndex);
                ApplyFloat32(
                    device, parameterState, master, physical.NativePtr,
                    groupIndex, parameter.T.Numel, options,
                    applyWeightDecay, mode);
                break;
            }
            case PrecisionMode.Mix8_32:
            {
                NativeCudaBuffer<float> master =
                    parameter.T.EnsureCudaMasterFloat32Buffer(
                        device.DeviceIndex);
                ApplyFloat32(
                    device, parameterState, master, 0, groupIndex,
                    parameter.T.Numel, options, applyWeightDecay, mode);
                CudaOptimizerKernels.PublishMix8Master(
                    parameter.T,
                    device.DeviceIndex,
                    device.FiniteStatus);
                break;
            }
            case PrecisionMode.BFloat16:
            {
                NativeCudaBuffer<ushort> data =
                    parameter.T.EnsureCudaBFloat16Buffer(device.DeviceIndex);
                Check(
                    CudaNativeGateway.GainShareApplyBFloat16(
                        device.DeviceIndex,
                        data.NativePtr,
                        parameterState.Direction.NativePtr,
                        device.Scales.NativePtr,
                        groupIndex,
                        parameter.T.Numel,
                        options.LearningRate,
                        options.WeightDecay,
                        applyWeightDecay,
                        0,
                        device.Stream),
                    "GainShare CUDA BF16 apply");
                break;
            }
            default:
                throw new InvalidOperationException(
                    $"Unsupported GainShare precision '{mode}'.");
        }
    }

    private static void ApplyFloat32(
        DeviceState device,
        ParameterDeviceState parameterState,
        NativeCudaBuffer<float> data,
        nint bfloat16Output,
        int groupIndex,
        int length,
        GainShareAdamWOptions options,
        bool applyWeightDecay,
        PrecisionMode mode)
    {
        Check(
            CudaNativeGateway.GainShareApplyFloat32(
                device.DeviceIndex,
                data.NativePtr,
                parameterState.Direction.NativePtr,
                device.Scales.NativePtr,
                bfloat16Output,
                groupIndex,
                length,
                options.LearningRate,
                options.WeightDecay,
                applyWeightDecay,
                device.FinitePointer(mode),
                device.Stream),
            "GainShare CUDA FP32 apply");
    }

    private void QueueFiniteStatusReadbacks()
    {
        foreach (DeviceState state in _deviceStates)
            state.FiniteReadback.Begin(state.FiniteStatus);
    }

    private void FinalizeStep(
        IReadOnlyList<int> devices,
        bool readFiniteStatus)
    {
        if (readFiniteStatus)
        {
            foreach (DeviceState state in _deviceStates)
            {
                if (state.FiniteReadback.ReadAfterSynchronization() != 0)
                {
                    throw new InvalidOperationException(
                        "Non-finite CUDA value detected during BFP8 " +
                        $"GainShareAdamW update on device " +
                        $"{state.DeviceIndex}.");
                }
            }
        }
        foreach (Parameter parameter in _parameters)
            parameter.T.MarkCudaDataReplicasSynchronized(devices);
    }

    private PrecisionMode ResolveAndValidateMode()
    {
        PrecisionMode mode = TensorExecutionContext.ActivePrecisionPolicy
            ?.Mode
            ?? (_parameters.All(parameter =>
                    parameter.T.DType == TensorDType.Float32)
                ? PrecisionMode.Float32
                : throw new InvalidOperationException(
                    "Low-precision CUDA GainShareAdamW requires an active " +
                    "PrecisionPolicy."));
        foreach (Parameter parameter in _parameters)
        {
            bool valid = mode switch
            {
                PrecisionMode.Float32 =>
                    parameter.T.DType == TensorDType.Float32,
                PrecisionMode.BFloat16 or PrecisionMode.Mix16_32 =>
                    parameter.T.DType == TensorDType.BFloat16,
                PrecisionMode.Bfp8 =>
                    parameter.T.DType == TensorDType.Bfp8
                    && parameter.T.Bfp8Quantization
                        == Bfp8QuantizationDescriptor.TensorWide,
                PrecisionMode.Mix8_32 =>
                    parameter.T.DType == TensorDType.Bfp8
                    && parameter.T.Bfp8Quantization?.Granularity
                        == Bfp8ScaleGranularity.Block,
                _ => false,
            };
            if (!valid)
            {
                throw new InvalidOperationException(
                    $"GainShareAdamW precision '{mode}' is incompatible " +
                    $"with parameter '{parameter.Name}' storage " +
                    $"'{parameter.T.DType}'.");
            }
        }
        return mode;
    }

    private static int[] NormalizeDevices(IReadOnlyList<int> devices)
    {
        ArgumentNullException.ThrowIfNull(devices);
        if (devices.Count == 0)
            throw new ArgumentException("At least one CUDA device is required.");
        int[] result = devices.ToArray();
        if (result.Any(device => device < 0)
            || result.Distinct().Count() != result.Length)
        {
            throw new ArgumentException(
                "CUDA device indices must be unique and non-negative.");
        }
        return result;
    }

    private static void Check(int status, string operation)
        => NativeCudaRuntime.Check(status, operation);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        DisposeDeviceStates();
    }

    private void DisposeDeviceStates()
    {
        DeviceState[] states = _deviceStates;
        _deviceStates = [];
        _mode = null;
        DisposeAll(states);
    }

    private static void DisposeAll(IEnumerable<IDisposable?> resources)
    {
        List<Exception>? failures = null;
        foreach (IDisposable? resource in resources)
        {
            if (resource is null)
                continue;
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        if (failures is not null)
            throw new AggregateException(
                "GainShare CUDA resource cleanup failed.", failures);
    }

    private sealed class DeviceState : IDisposable
    {
        private Bfp8MultiTensorPlan? _bfp8Plan;
        private int _disposed;

        private DeviceState(
            int deviceIndex,
            NativeCudaDevice accelerator,
            ParameterDeviceState[] parameters,
            NativeCudaBuffer<float> groupStats,
            NativeCudaBuffer<float> alignmentEma,
            NativeCudaBuffer<float> scales,
            NativeCudaBuffer<int> finiteStatus,
            CudaOptimizerFiniteStatusReadback finiteReadback)
        {
            DeviceIndex = deviceIndex;
            Accelerator = accelerator;
            Parameters = parameters;
            GroupStats = groupStats;
            AlignmentEma = alignmentEma;
            Scales = scales;
            FiniteStatus = finiteStatus;
            FiniteReadback = finiteReadback;
        }

        internal int DeviceIndex { get; }
        internal NativeCudaDevice Accelerator { get; }
        internal nint Stream => Accelerator.DefaultStream;
        internal ParameterDeviceState[] Parameters { get; }
        internal NativeCudaBuffer<float> GroupStats { get; }
        internal NativeCudaBuffer<float> AlignmentEma { get; }
        internal NativeCudaBuffer<float> Scales { get; }
        internal NativeCudaBuffer<int> FiniteStatus { get; }
        internal CudaOptimizerFiniteStatusReadback FiniteReadback { get; }

        internal static DeviceState Create(
            int deviceIndex,
            PrecisionMode mode,
            IReadOnlyList<Parameter> parameters,
            GainShareAdamWState hostState,
            int groupCount)
        {
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
            accelerator.Bind();
            var owned = new List<IDisposable>();
            try
            {
                var parameterStates = new ParameterDeviceState[
                    parameters.Count];
                for (int index = 0; index < parameters.Count; index++)
                {
                    parameterStates[index] = ParameterDeviceState.Create(
                        accelerator,
                        mode,
                        hostState.ParameterStates[index],
                        parameters[index].T.Numel);
                    owned.Add(parameterStates[index]);
                    PrewarmParameterStorage(
                        parameters[index].T, deviceIndex, mode);
                }
                NativeCudaBuffer<float> groupStats =
                    accelerator.Allocate1D<float>(groupCount * 2);
                owned.Add(groupStats);
                groupStats.MemSetToZero();
                var emaHost = new float[groupCount];
                for (int group = 0; group < groupCount; group++)
                {
                    emaHost[group] = hostState.GroupStates[group].AlignmentEma
                        is { } value ? (float)value : float.NaN;
                }
                NativeCudaBuffer<float> alignmentEma =
                    accelerator.Allocate1D(emaHost);
                owned.Add(alignmentEma);
                NativeCudaBuffer<float> scales =
                    accelerator.Allocate1D<float>(groupCount);
                owned.Add(scales);
                NativeCudaBuffer<int> finiteStatus =
                    accelerator.Allocate1D<int>(1);
                owned.Add(finiteStatus);
                finiteStatus.MemSetToZero();
                var finiteReadback =
                    new CudaOptimizerFiniteStatusReadback(deviceIndex);
                owned.Add(finiteReadback);
                var result = new DeviceState(
                    deviceIndex,
                    accelerator,
                    parameterStates,
                    groupStats,
                    alignmentEma,
                    scales,
                    finiteStatus,
                    finiteReadback);
                owned.Clear();
                return result;
            }
            catch
            {
                DisposeAll(owned.AsEnumerable().Reverse());
                throw;
            }
        }

        internal void BeginStep(PrecisionMode mode)
        {
            Accelerator.Bind();
            GroupStats.MemSetToZero();
            if (mode is PrecisionMode.Bfp8 or PrecisionMode.Mix8_32)
                FiniteStatus.MemSetToZero();
        }

        internal nint FinitePointer(PrecisionMode mode)
            => mode is PrecisionMode.Bfp8 or PrecisionMode.Mix8_32
                ? FiniteStatus.NativePtr
                : 0;

        internal Bfp8MultiTensorPlan GetBfp8Plan(
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<int> groupIndices)
        {
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0, this);
            if (_bfp8Plan is not null
                && _bfp8Plan.HasCurrentBindings(parameters))
            {
                return _bfp8Plan;
            }

            _bfp8Plan?.Dispose();
            _bfp8Plan = Bfp8MultiTensorPlan.Create(
                Accelerator,
                DeviceIndex,
                parameters,
                Parameters,
                groupIndices);
            return _bfp8Plan;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            IEnumerable<IDisposable> resources = Parameters
                .Cast<IDisposable>()
                .Concat(new IDisposable[]
                {
                    GroupStats,
                    AlignmentEma,
                    Scales,
                    FiniteStatus,
                    FiniteReadback,
                });
            if (_bfp8Plan is not null)
                resources = resources.Append(_bfp8Plan);
            DisposeAll(resources.Reverse());
        }

        private static void PrewarmParameterStorage(
            Tensor parameter,
            int deviceIndex,
            PrecisionMode mode)
        {
            switch (mode)
            {
                case PrecisionMode.Float32:
                    _ = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                    break;
                case PrecisionMode.BFloat16:
                    _ = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
                    break;
                case PrecisionMode.Mix16_32:
                    _ = parameter.EnsureCudaBFloat16Buffer(deviceIndex);
                    _ = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                    break;
                case PrecisionMode.Bfp8:
                    _ = parameter.EnsureCudaBfp8Buffer(deviceIndex);
                    break;
                case PrecisionMode.Mix8_32:
                    _ = parameter.EnsureCudaBfp8Buffer(deviceIndex);
                    _ = parameter.EnsureCudaMasterFloat32Buffer(deviceIndex);
                    break;
            }
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    private readonly struct GainShareBfp8Tensor(
        nint dataPayload,
        nint dataScale,
        nint gradientPayload,
        nint gradientScale,
        nint firstPayload,
        nint firstScale,
        nint secondPayload,
        nint secondScale,
        nint direction,
        int length,
        int groupIndex,
        int applyWeightDecay,
        int rankOne)
    {
        internal readonly nint DataPayload = dataPayload;
        internal readonly nint DataScale = dataScale;
        internal readonly nint GradientPayload = gradientPayload;
        internal readonly nint GradientScale = gradientScale;
        internal readonly nint FirstPayload = firstPayload;
        internal readonly nint FirstScale = firstScale;
        internal readonly nint SecondPayload = secondPayload;
        internal readonly nint SecondScale = secondScale;
        internal readonly nint Direction = direction;
        internal readonly int Length = length;
        internal readonly int GroupIndex = groupIndex;
        internal readonly int ApplyWeightDecay = applyWeightDecay;
        internal readonly int RankOne = rankOne;
    }

    private sealed class Bfp8MultiTensorPlan : IDisposable
    {
        private readonly int _deviceIndex;
        private readonly nint[] _gradientPayloadPointers;
        private readonly nint[] _gradientScalePointers;
        private int _disposed;

        private Bfp8MultiTensorPlan(
            int deviceIndex,
            NativeCudaBuffer<GainShareBfp8Tensor> descriptors,
            NativeCudaBuffer<float> reduction,
            int maximumChunks,
            nint[] gradientPayloadPointers,
            nint[] gradientScalePointers)
        {
            _deviceIndex = deviceIndex;
            Descriptors = descriptors;
            Reduction = reduction;
            MaximumChunks = maximumChunks;
            _gradientPayloadPointers = gradientPayloadPointers;
            _gradientScalePointers = gradientScalePointers;
        }

        internal NativeCudaBuffer<GainShareBfp8Tensor> Descriptors { get; }
        internal NativeCudaBuffer<float> Reduction { get; }
        internal int MaximumChunks { get; }
        internal int TensorCount => Descriptors.Length;

        internal static Bfp8MultiTensorPlan Create(
            NativeCudaDevice accelerator,
            int deviceIndex,
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<ParameterDeviceState> states,
            IReadOnlyList<int> groupIndices)
        {
            if (parameters.Count == 0 || parameters.Count > 65_535
                || states.Count != parameters.Count
                || groupIndices.Count != parameters.Count)
            {
                throw new ArgumentException(
                    "GainShare BFP8 multi-tensor metadata is inconsistent.");
            }

            var descriptors = new GainShareBfp8Tensor[parameters.Count];
            var gradientPayloadPointers = new nint[parameters.Count];
            var gradientScalePointers = new nint[parameters.Count];
            int maximumLength = 0;
            for (int index = 0; index < parameters.Count; index++)
            {
                Parameter parameter = parameters[index];
                Tensor tensor = parameter.T;
                CudaBfp8BufferView data =
                    tensor.EnsureCudaBfp8Buffer(deviceIndex);
                CudaBfp8BufferView gradient =
                    tensor.EnsureCudaBfp8GradientBuffer(deviceIndex);
                ParameterDeviceState state = states[index];
                CudaBfp8BufferView first = state.FirstBfp8
                    ?? throw new InvalidOperationException(
                        "GainShare BFP8 first moment is not resident.");
                CudaBfp8BufferView second = state.SecondBfp8
                    ?? throw new InvalidOperationException(
                        "GainShare BFP8 second moment is not resident.");
                gradientPayloadPointers[index] = gradient.Payload.NativePtr;
                gradientScalePointers[index] = gradient.Scales.NativePtr;
                maximumLength = Math.Max(maximumLength, tensor.Numel);
                descriptors[index] = new GainShareBfp8Tensor(
                    data.Payload.NativePtr,
                    data.Scales.NativePtr,
                    gradient.Payload.NativePtr,
                    gradient.Scales.NativePtr,
                    first.Payload.NativePtr,
                    first.Scales.NativePtr,
                    second.Payload.NativePtr,
                    second.Scales.NativePtr,
                    state.Direction.NativePtr,
                    tensor.Numel,
                    groupIndices[index],
                    parameter.WeightDecay == WeightDecayPolicy.Apply ? 1 : 0,
                    tensor.Rank == 1 ? 1 : 0);
            }

            NativeCudaBuffer<GainShareBfp8Tensor>? descriptorBuffer = null;
            NativeCudaBuffer<float>? reduction = null;
            try
            {
                descriptorBuffer = accelerator.Allocate1D(descriptors);
                reduction = accelerator.Allocate1D<float>(
                    checked(parameters.Count * 6));
                int maximumChunks = Math.Min(
                    checked((maximumLength + 255) / 256),
                    1024);
                var result = new Bfp8MultiTensorPlan(
                    deviceIndex,
                    descriptorBuffer,
                    reduction,
                    maximumChunks,
                    gradientPayloadPointers,
                    gradientScalePointers);
                descriptorBuffer = null;
                reduction = null;
                return result;
            }
            finally
            {
                descriptorBuffer?.Dispose();
                reduction?.Dispose();
            }
        }

        internal bool HasCurrentBindings(IReadOnlyList<Parameter> parameters)
        {
            if (parameters.Count != _gradientPayloadPointers.Length)
                return false;
            for (int index = 0; index < parameters.Count; index++)
            {
                if (!parameters[index].T.TryGetCudaBfp8GradientBuffer(
                        _deviceIndex,
                        out CudaBfp8BufferView gradient)
                    || gradient.Payload.NativePtr
                        != _gradientPayloadPointers[index]
                    || gradient.Scales.NativePtr
                        != _gradientScalePointers[index])
                {
                    return false;
                }
            }
            return true;
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            DisposeAll(new IDisposable[] { Reduction, Descriptors });
        }
    }

    private sealed class ParameterDeviceState : IDisposable
    {
        private int _disposed;

        private ParameterDeviceState(
            NativeCudaBuffer<float> direction,
            NativeCudaBuffer<float>? firstFloat,
            NativeCudaBuffer<float>? secondFloat,
            NativeCudaBuffer<ushort>? firstBFloat16,
            NativeCudaBuffer<ushort>? secondBFloat16,
            CudaBfp8BufferView? firstBfp8,
            CudaBfp8BufferView? secondBfp8)
        {
            Direction = direction;
            FirstFloat = firstFloat;
            SecondFloat = secondFloat;
            FirstBFloat16 = firstBFloat16;
            SecondBFloat16 = secondBFloat16;
            FirstBfp8 = firstBfp8;
            SecondBfp8 = secondBfp8;
        }

        internal NativeCudaBuffer<float> Direction { get; }
        internal NativeCudaBuffer<float>? FirstFloat { get; }
        internal NativeCudaBuffer<float>? SecondFloat { get; }
        internal NativeCudaBuffer<ushort>? FirstBFloat16 { get; }
        internal NativeCudaBuffer<ushort>? SecondBFloat16 { get; }
        internal CudaBfp8BufferView? FirstBfp8 { get; }
        internal CudaBfp8BufferView? SecondBfp8 { get; }

        internal static ParameterDeviceState Create(
            NativeCudaDevice accelerator,
            PrecisionMode mode,
            GainShareAdamWParameterState host,
            int length)
        {
            NativeCudaBuffer<float>? direction = null;
            NativeCudaBuffer<float>? firstFloat = null;
            NativeCudaBuffer<float>? secondFloat = null;
            NativeCudaBuffer<ushort>? firstBFloat16 = null;
            NativeCudaBuffer<ushort>? secondBFloat16 = null;
            CudaBfp8BufferView? firstBfp8 = null;
            CudaBfp8BufferView? secondBfp8 = null;
            try
            {
                direction = accelerator.Allocate1D<float>(length);
                if (mode is PrecisionMode.Float32
                    or PrecisionMode.Mix16_32
                    or PrecisionMode.Mix8_32)
                {
                    firstFloat = AllocateFloat(
                        accelerator, host.FirstMoment);
                    secondFloat = AllocateFloat(
                        accelerator, host.SecondMoment);
                }
                else if (mode == PrecisionMode.BFloat16)
                {
                    firstBFloat16 = AllocateBFloat16(
                        accelerator, host.FirstMoment);
                    secondBFloat16 = AllocateBFloat16(
                        accelerator, host.SecondMoment);
                }
                else
                {
                    firstBfp8 = AllocateBfp8(
                        accelerator, host.FirstMoment);
                    secondBfp8 = AllocateBfp8(
                        accelerator, host.SecondMoment);
                }
                var result = new ParameterDeviceState(
                    direction,
                    firstFloat,
                    secondFloat,
                    firstBFloat16,
                    secondBFloat16,
                    firstBfp8,
                    secondBfp8);
                direction = null;
                firstFloat = null;
                secondFloat = null;
                firstBFloat16 = null;
                secondBFloat16 = null;
                firstBfp8 = null;
                secondBfp8 = null;
                return result;
            }
            finally
            {
                direction?.Dispose();
                firstFloat?.Dispose();
                secondFloat?.Dispose();
                firstBFloat16?.Dispose();
                secondBFloat16?.Dispose();
                DisposeBfp8(firstBfp8);
                DisposeBfp8(secondBfp8);
            }
        }

        internal void CopyMomentsToHost(
            float[] firstDestination,
            float[] secondDestination)
        {
            if (FirstFloat is not null && SecondFloat is not null)
            {
                FirstFloat.CopyToCPU(firstDestination);
                SecondFloat.CopyToCPU(secondDestination);
                return;
            }
            if (FirstBFloat16 is not null && SecondBFloat16 is not null)
            {
                var first = new ushort[firstDestination.Length];
                var second = new ushort[secondDestination.Length];
                FirstBFloat16.CopyToCPU(first);
                SecondBFloat16.CopyToCPU(second);
                TensorStorageCodec.DecodeBFloat16(first, firstDestination);
                TensorStorageCodec.DecodeBFloat16(second, secondDestination);
                return;
            }
            if (FirstBfp8 is { } firstBfp8
                && SecondBfp8 is { } secondBfp8)
            {
                DecodeBfp8(firstBfp8, firstDestination);
                DecodeBfp8(secondBfp8, secondDestination);
                return;
            }
            throw new InvalidOperationException(
                "GainShare CUDA moment representation is incomplete.");
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            DisposeAll(new IDisposable?[]
            {
                Direction,
                FirstFloat,
                SecondFloat,
                FirstBFloat16,
                SecondBFloat16,
                FirstBfp8?.Payload,
                FirstBfp8?.Scales,
                SecondBfp8?.Payload,
                SecondBfp8?.Scales,
            }.Reverse());
        }

        private static NativeCudaBuffer<float> AllocateFloat(
            NativeCudaDevice accelerator,
            float[] host)
        {
            if (host.All(value => value == 0f))
            {
                NativeCudaBuffer<float> result =
                    accelerator.Allocate1D<float>(host.Length);
                result.MemSetToZero();
                return result;
            }
            return accelerator.Allocate1D(host);
        }

        private static NativeCudaBuffer<ushort> AllocateBFloat16(
            NativeCudaDevice accelerator,
            float[] host)
        {
            var encoded = new ushort[host.Length];
            TensorStorageCodec.EncodeBFloat16(host, encoded);
            if (encoded.All(value => value == 0))
            {
                NativeCudaBuffer<ushort> result =
                    accelerator.Allocate1D<ushort>(encoded.Length);
                result.MemSetToZero();
                return result;
            }
            return accelerator.Allocate1D(encoded);
        }

        private static CudaBfp8BufferView AllocateBfp8(
            NativeCudaDevice accelerator,
            float[] host)
        {
            Bfp8EncodedStorage encoded = Bfp8QuantizationCodec.Default.Encode(
                host, Bfp8QuantizationDescriptor.TensorWide);
            NativeCudaBuffer<sbyte>? payload = null;
            NativeCudaBuffer<float>? scales = null;
            try
            {
                payload = accelerator.Allocate1D(encoded.Payload.Span);
                scales = accelerator.Allocate1D(encoded.Scales.Span);
                CudaBfp8BufferView result = new(
                    payload, scales, encoded.Descriptor);
                payload = null;
                scales = null;
                return result;
            }
            finally
            {
                payload?.Dispose();
                scales?.Dispose();
            }
        }

        private static void DecodeBfp8(
            CudaBfp8BufferView source,
            float[] destination)
        {
            var payload = new sbyte[source.Payload.Length];
            var scales = new float[source.Scales.Length];
            source.Payload.CopyToCPU(payload);
            source.Scales.CopyToCPU(scales);
            Bfp8QuantizationCodec.Default.Decode(
                payload, scales, source.Descriptor, destination);
        }

        private static void DisposeBfp8(CudaBfp8BufferView? view)
        {
            view?.Payload.Dispose();
            view?.Scales.Dispose();
        }
    }

}
