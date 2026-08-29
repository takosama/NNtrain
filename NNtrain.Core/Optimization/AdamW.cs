namespace NNtrain;

public partial class AdamW : IOptimizer, ILearningRateAdjustable
{
    private readonly List<Parameter> _parameters;
    private readonly long _totalElements;
    private readonly AdamWParameterRuntime[] _parameterRuntime;
    private readonly AdamWWorkItem[] _workItems;
    private readonly Dictionary<int, CudaOptimizerKernels.AdamWMultiTensorPlan>
        _cudaMultiTensorPlans = [];
    private readonly Dictionary<int,
        CudaOptimizerKernels.AdamWBfp8MultiTensorPlan>
        _cudaBfp8MultiTensorPlans = [];
    private readonly Dictionary<int, NativeCudaBuffer<int>>
        _cudaBfp8FiniteStatus = [];
    private readonly Dictionary<int, CudaOptimizerFiniteStatusReadback>
        _cudaBfp8FiniteReadbacks = [];
    private readonly Dictionary<int,
        CudaOptimizerKernels.AdamWBfp8DeviceScratch> _cudaBfp8Scratch = [];
    private readonly CudaDispatchPolicy _cudaDispatchPolicy;
    private AdamWOptions _options;
    private AdamWParameterState[] _parameterStates;
    private readonly Action<int> _updateWorkItemAction;
    private readonly Action<int> _clearWorkItemAction;
    private AdamWOptions _stepOptions = null!;
    private float _stepUpdateScale;
    private float _stepScaledEpsilon;
    private int _step;
    private int _cudaMultiTensorPlanBuildCount;
    // The execution device can change to CPU while optimizer state is still
    // resident. Keep the device that owns the last prepared/updated replica
    // instead of consulting the current global execution device at checkpoint
    // or CPU-continuation time.
    private int? _cudaStateAuthorityDevice;

    internal IReadOnlyList<Parameter> Parameters => _parameters;
    internal int CudaMultiTensorPlanBuildCount
        => _cudaMultiTensorPlanBuildCount;

    public AdamW(
        IEnumerable<Parameter> parameters,
        AdamWOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        _cudaDispatchPolicy = CudaDispatchPolicy.Current.Validate();

        _parameters = [];
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);

        foreach (Parameter parameter in parameters)
        {
            if (parameter is null)
            {
                throw new ArgumentException(
                    "Optimizer parameters cannot contain null.",
                    nameof(parameters));
            }

            if (!seenParameters.Add(parameter))
            {
                throw new ArgumentException(
                    $"Parameter '{parameter.Name}' was supplied to AdamW " +
                    "more than once.",
                    nameof(parameters));
            }

            _parameters.Add(parameter);
        }

        _totalElements = _parameters.Sum(
            parameter => (long)parameter.T.Numel);
        _workItems = CreateWorkItems(_parameters);

        AdamWOptions effectiveOptions = options ?? new AdamWOptions();
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16)
        {
            effectiveOptions = effectiveOptions with
            {
                UseBFloat16FirstMoment = true,
                UseBFloat16SecondMoment = true,
            };
        }
        ValidateOptions(effectiveOptions, nameof(options));
        AdamWState initialState = CreateInitialState(
            _parameters,
            effectiveOptions);
        _options = initialState.Options;
        _parameterStates = initialState.ParameterStates;
        _parameterRuntime = CreateParameterRuntime(
            _parameters,
            _parameterStates,
            _options);
        _updateWorkItemAction = UpdateWorkItem;
        _clearWorkItemAction = ClearWorkItem;
    }

    public AdamWState CaptureState()
    {
        AdamWState state = CaptureStateForStreaming();
        return _options.UseBFloat16FirstMoment
            || _options.UseBFloat16SecondMoment
            ? state
            : CloneState(state);
    }

    internal AdamWState CaptureStateForStreaming()
    {
        SynchronizeStateForStreaming();
        if (_options.UseBFloat16FirstMoment
            || _options.UseBFloat16SecondMoment)
        {
            var states = new AdamWParameterState[_parameters.Count];
            for (int index = 0; index < states.Length; index++)
            {
                Parameter parameter = _parameters[index];
                states[index] = new AdamWParameterState(
                    index,
                    parameter.Name,
                    parameter.T.Shape.ToArray(),
                    _parameterRuntime[index].FirstMomentBFloat16 is not null
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .FirstMomentBFloat16!)
                        : _parameterRuntime[index].FirstMoment.ToArray(),
                    _parameterRuntime[index].SecondMomentBFloat16 is not null
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .SecondMomentBFloat16!)
                        : _parameterRuntime[index].SecondMoment.ToArray());
            }
            return new AdamWState(
                AdamWState.CurrentFormatVersion,
                _step,
                _options with { },
                states);
        }

        return new AdamWState(
            AdamWState.CurrentFormatVersion,
            _step,
            _options,
            _parameterStates);
    }

    /// <summary>
    /// Makes the managed moment arrays authoritative before a streaming
    /// serializer reads them.  Unlike <see cref="CaptureStateForStreaming"/>,
    /// this does not decode every BF16 moment into a second full-size FP32
    /// object graph.
    /// </summary>
    internal void SynchronizeStateForStreaming()
    {
        if (_cudaStateAuthorityDevice is not int primaryDevice)
            return;
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            runtime.CudaState?.SynchronizeHost(primaryDevice);
            runtime.CudaBFloat16State?.SynchronizeHost(primaryDevice);
            runtime.CudaMixedState?.SynchronizeHost(primaryDevice);
            runtime.CudaBfp8State?.SynchronizeHost(primaryDevice);
        }
    }

    internal int StreamingStep => _step;

    internal AdamWOptions StreamingOptions => _options;

    internal int StreamingParameterCount => _parameterStates.Length;

    internal AdamWStreamingParameterState GetStreamingParameterState(int index)
    {
        AdamWParameterState state = _parameterStates[index];
        AdamWParameterRuntime runtime = _parameterRuntime[index];
        bool bfp8State = runtime.CudaBfp8State is not null;
        return new AdamWStreamingParameterState(
            state.Index,
            state.Name,
            state.Shape,
            runtime.FirstMoment,
            bfp8State ? null : runtime.FirstMomentBFloat16,
            runtime.SecondMoment,
            bfp8State ? null : runtime.SecondMomentBFloat16);
    }

    public float LearningRate => _options.LearningRate;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "AdamW learning rate must be finite and positive.");
        }

        _options = _options with { LearningRate = learningRate };
        RefreshWeightDecayFlags();
    }

    public void RestoreState(AdamWState state)
        => RestoreState(state, takeOwnership: false);

    private void RestoreState(
        AdamWState state,
        bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        AdamWState restored = takeOwnership ? state : CloneState(state);
        _step = restored.Step;
        _options = TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            ? restored.Options with
            {
                UseBFloat16FirstMoment = true,
                UseBFloat16SecondMoment = true,
            }
            : restored.Options;
        _parameterStates = restored.ParameterStates;
        DisposeCudaResources();
        for (int index = 0; index < _parameterRuntime.Length; index++)
        {
            bool bfp8State = UsesPureBfp8OptimizerState(
                _parameterRuntime[index].Parameter.T);
            bool mix8State = UsesMix8Parameter(
                _parameterRuntime[index].Parameter.T);
            if (_options.UseBFloat16FirstMoment
                && !bfp8State
                && !mix8State)
            {
                _parameterRuntime[index].FirstMoment = [];
                _parameterRuntime[index].FirstMomentBFloat16 =
                    EncodeBFloat16(_parameterStates[index].FirstMoment);
                _parameterStates[index] = _parameterStates[index] with
                {
                    FirstMoment = [],
                };
            }
            else
            {
                _parameterRuntime[index].FirstMoment =
                    _parameterStates[index].FirstMoment;
                _parameterRuntime[index].FirstMomentBFloat16 = null;
            }
            if (_options.UseBFloat16SecondMoment
                && !bfp8State
                && !mix8State)
            {
                _parameterRuntime[index].SecondMoment = [];
                _parameterRuntime[index].SecondMomentBFloat16 =
                    EncodeBFloat16(_parameterStates[index].SecondMoment);
                _parameterStates[index] = _parameterStates[index] with
                {
                    SecondMoment = [],
                };
            }
            else
            {
                _parameterRuntime[index].SecondMoment =
                    _parameterStates[index].SecondMoment;
                _parameterRuntime[index].SecondMomentBFloat16 = null;
            }
        }
        RefreshWeightDecayFlags();
    }

    internal void RestoreStateOwned(AdamWState state)
        => RestoreState(state, takeOwnership: true);

    internal void ZeroGrad()
    {
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
                runtime.Parameter.T.ClearGradient();
            return;
        }

        if (_workItems.Length > 1 && _totalElements >= 32_768)
        {
            Tensor.RunParallel(
                0,
                _workItems.Length,
                _clearWorkItemAction);
            return;
        }

        for (int index = 0; index < _workItems.Length; index++)
            ClearWorkItem(index);
    }

    public void zero_grad() => ZeroGrad();

    private void ClearWorkItem(int workItemIndex)
    {
        AdamWWorkItem workItem = _workItems[workItemIndex];
        _parameterRuntime[workItem.ParameterIndex]
            .Parameter
            .T
            .ClearGradientRange(workItem.Start, workItem.Length);
    }

    internal void Step()
    {
        if (_step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "AdamW cannot advance beyond Int32.MaxValue steps.");
        }

        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            CudaGradientOptimizerGuard.ValidateAndConsume(
                _parameters,
                Tensor.CudaDeviceIndices);
        }
        else
        {
            MakeCpuStateAuthoritative();
        }

        _step++;
        AdamWOptions options = _options;

        float bc1 = 1f - MathF.Pow(options.Beta1, _step);
        float bc2 = 1f - MathF.Pow(options.Beta2, _step);
        float sqrtBc2 = MathF.Sqrt(bc2);
        _stepOptions = options;
        _stepUpdateScale = options.LearningRate * sqrtBc2 / bc1;
        _stepScaledEpsilon = options.Epsilon * sqrtBc2;

        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            int[] devices = Tensor.CudaDeviceIndices.ToArray();
            if (devices.Length == 0)
            {
                throw new InvalidOperationException(
                    "CUDA AdamW requires at least one device.");
            }
            _cudaStateAuthorityDevice = devices[0];
            bool pureBFloat16 = UsesPureBFloat16OptimizerState();
            if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                    == NNtrain.Runtime.Execution.NumericFormat.BFloat16
                && !pureBFloat16)
            {
                throw new InvalidOperationException(
                    "The pure BFloat16 AdamW contract requires every " +
                    "parameter to use physical BFloat16 storage.");
            }
            bool pureBfp8 = _parameterRuntime.All(runtime =>
                UsesPureBfp8OptimizerState(runtime.Parameter.T));
            if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                    == NNtrain.Runtime.Execution.NumericFormat.Bfp8
                && !pureBfp8)
            {
                throw new InvalidOperationException(
                    "The pure BFP8 AdamW contract requires every parameter " +
                    "to use tensor-wide BFP8 storage.");
            }
            if (pureBfp8)
            {
                StepCudaBfp8(devices, options);
                return;
            }
            bool anyMix8 = _parameterRuntime.Any(runtime =>
                UsesMix8Parameter(runtime.Parameter.T));
            bool mix8 = _parameterRuntime.All(runtime =>
                UsesMix8Parameter(runtime.Parameter.T));
            if (anyMix8 && !mix8)
            {
                throw new InvalidOperationException(
                    "The mix8_32 AdamW contract cannot mix block-scaled " +
                    "BFP8 parameters with another storage format.");
            }
            if (anyMix8)
                ValidateMix8OptimizerContract();
            if (TensorExecutionContext.ActivePrecisionPolicy?.Mode
                    == NNtrain.Runtime.Execution.PrecisionMode.Mix8_32
                && !mix8)
            {
                throw new InvalidOperationException(
                    "The mix8_32 AdamW contract requires every parameter " +
                    "to use block-scaled BFP8 storage.");
            }
            if (mix8)
            {
                if (_cudaDispatchPolicy.EnableBlockBfp8OptimizerState)
                    StepCudaBfp8(devices, options, mixedBlockState: true);
                else
                    StepCudaMix8(devices, options);
                return;
            }
            for (int parameterIndex = 0;
                parameterIndex < _parameterRuntime.Length;
                parameterIndex++)
            {
                AdamWParameterRuntime runtime =
                    _parameterRuntime[parameterIndex];
                if (runtime.FirstMomentBFloat16 is null
                    && runtime.SecondMomentBFloat16 is null)
                {
                    runtime.CudaState ??=
                        new CudaOptimizerKernels.AdamWResidentState(
                            runtime.FirstMoment,
                            runtime.SecondMoment);
                    foreach (int deviceIndex in devices)
                        runtime.CudaState.GetOrCreate(deviceIndex);
                    continue;
                }

                if (runtime.FirstMomentBFloat16 is not null
                    && runtime.SecondMomentBFloat16 is not null)
                {
                    runtime.CudaBFloat16State ??=
                        new CudaOptimizerKernels.AdamWBFloat16ResidentState(
                            runtime.FirstMomentBFloat16,
                            runtime.SecondMomentBFloat16);
                    foreach (int deviceIndex in devices)
                        runtime.CudaBFloat16State.GetOrCreate(deviceIndex);
                    continue;
                }

                runtime.CudaMixedState ??=
                    new CudaOptimizerKernels.AdamWMixedResidentState(
                        runtime.FirstMoment,
                        runtime.FirstMomentBFloat16,
                        runtime.SecondMoment,
                        runtime.SecondMomentBFloat16);
                foreach (int deviceIndex in devices)
                    runtime.CudaMixedState.GetOrCreate(deviceIndex);
            }

            CudaOptimizerKernels.AdamWMultiTensorItem[] multiTensorItems =
                _parameterRuntime
                    .Where(runtime => runtime.CudaState is not null
                        || runtime.CudaBFloat16State is not null
                        || runtime.CudaMixedState is not null)
                    .Select(runtime => new CudaOptimizerKernels
                        .AdamWMultiTensorItem(
                            runtime.Parameter.T,
                            runtime.CudaState,
                            runtime.CudaBFloat16State,
                            runtime.ApplyWeightDecay,
                            pureBFloat16,
                            runtime.CudaMixedState))
                    .ToArray();
            if (multiTensorItems.Length > 0)
            {
                var plans = new CudaOptimizerKernels
                    .AdamWMultiTensorPlan[devices.Length];
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    ++deviceSlot)
                {
                    int deviceIndex = devices[deviceSlot];
                    if (!_cudaMultiTensorPlans.TryGetValue(
                        deviceIndex,
                        out CudaOptimizerKernels.AdamWMultiTensorPlan? plan)
                        || !plan.Matches(multiTensorItems))
                    {
                        if (plan is not null)
                        {
                            plan.Dispose();
                            _cudaMultiTensorPlans.Remove(deviceIndex);
                        }
                        plan = new CudaOptimizerKernels.AdamWMultiTensorPlan(
                            deviceIndex, multiTensorItems);
                        _cudaMultiTensorPlans.Add(deviceIndex, plan);
                        _cudaMultiTensorPlanBuildCount++;
                    }
                    plans[deviceSlot] = plan;
                }

                // One launch per GPU updates all AdamW tensors. Each worker
                // stays bound to one device, so both GPUs advance
                // concurrently. FP32, BF16, and asymmetric FP32/BF16 moment
                // pairs all use this same fully resident launch path.
                Parallel.For(0, devices.Length, deviceSlot =>
                {
                    plans[deviceSlot].Execute(
                        options.Beta1,
                        options.Beta2,
                        options.LearningRate,
                        options.WeightDecay,
                        _stepUpdateScale,
                        _stepScaledEpsilon);
                });
            }
            CudaOptimizerStepBatch.CompleteAfterSynchronization(
                devices,
                "AdamW update",
                queueReadback: null,
                finalize: () =>
                {
                    foreach (AdamWParameterRuntime runtime
                        in _parameterRuntime)
                    {
                        runtime.Parameter.T
                            .MarkCudaDataReplicasSynchronized(devices);
                    }
                });
            return;
        }

        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            AdamWParameterRuntime runtime =
                _parameterRuntime[parameterIndex];
            runtime.Gradient = runtime.Parameter.T.GradientBuffer;
        }

        if (_workItems.Length > 1 && _totalElements >= 32_768)
            Tensor.RunParallel(0, _workItems.Length, _updateWorkItemAction);
        else
            for (int index = 0; index < _workItems.Length; index++)
                UpdateWorkItem(index);

        // A parameter can span several parallel work items. Publish its
        // Float32 master weight to low-precision storage only after every
        // chunk has completed, and advance the data version exactly once.
        for (int parameterIndex = 0;
            parameterIndex < _parameterRuntime.Length;
            parameterIndex++)
        {
            _parameterRuntime[parameterIndex]
                .Parameter
                .CompleteUpdate();
        }
    }

    private bool UsesPureBFloat16OptimizerState()
        => TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            && _parameterRuntime.All(runtime =>
                runtime.Parameter.T.DType == TensorDType.BFloat16);

    public void step() => Step();

    private void StepCudaBfp8(
        int[] devices,
        AdamWOptions options,
        bool mixedBlockState = false)
    {
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "Pure BFP8 AdamW requires at least one CUDA device.");
        }

        // A precision transition invalidates plans which captured Float32 or
        // BF16 moment addresses. Preserve their latest values before making
        // the BFP8 payloads the only resident optimizer authority.
        bool transitioned = _parameterRuntime.Any(runtime =>
            runtime.CudaBfp8State is null);
        if (transitioned)
        {
            foreach (CudaOptimizerKernels.AdamWMultiTensorPlan plan
                in _cudaMultiTensorPlans.Values)
            {
                plan.Dispose();
            }
            _cudaMultiTensorPlans.Clear();
            DisposeCudaBfp8Plans();
        }

        int primaryDevice = devices[0];
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.CudaBfp8State is null)
            {
                runtime.CudaState?.SynchronizeHost(primaryDevice);
                runtime.CudaBFloat16State?.SynchronizeHost(primaryDevice);
                runtime.CudaMixedState?.SynchronizeHost(primaryDevice);
                runtime.CudaState?.Dispose();
                runtime.CudaState = null;
                runtime.CudaBFloat16State?.Dispose();
                runtime.CudaBFloat16State = null;
                runtime.CudaMixedState?.Dispose();
                runtime.CudaMixedState = null;

                if (runtime.FirstMomentBFloat16 is not null)
                {
                    runtime.FirstMoment = DecodeBFloat16(
                        runtime.FirstMomentBFloat16);
                    runtime.FirstMomentBFloat16 = null;
                }
                if (runtime.SecondMomentBFloat16 is not null)
                {
                    runtime.SecondMoment = DecodeBFloat16(
                        runtime.SecondMomentBFloat16);
                    runtime.SecondMomentBFloat16 = null;
                }
                if (runtime.FirstMoment.Length != runtime.Parameter.T.Numel)
                    runtime.FirstMoment = new float[runtime.Parameter.T.Numel];
                if (runtime.SecondMoment.Length != runtime.Parameter.T.Numel)
                    runtime.SecondMoment = new float[runtime.Parameter.T.Numel];

                runtime.CudaBfp8State =
                    new CudaOptimizerKernels.AdamWBfp8ResidentState(
                        runtime.FirstMoment,
                        runtime.SecondMoment,
                        mixedBlockState
                            ? Bfp8QuantizationDescriptor.Mix8_32
                            : Bfp8QuantizationDescriptor.TensorWide);
            }
            foreach (int deviceIndex in devices)
                runtime.CudaBfp8State.GetOrCreate(deviceIndex);
        }

        var statuses = new NativeCudaBuffer<int>[devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
        {
            int deviceIndex = devices[deviceSlot];
            statuses[deviceSlot] = GetOrCreateBfp8FiniteStatus(deviceIndex);
            statuses[deviceSlot].MemSetToZero();
        }

        CudaOptimizerKernels.AdamWBfp8MultiTensorPlan[]? purePlans = null;
        if (!mixedBlockState)
        {
            CudaOptimizerKernels.AdamWBfp8MultiTensorItem[] items =
                CreateCudaBfp8PlanItems();
            purePlans = PrepareCudaBfp8Plans(devices, items);
        }

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            NativeCudaBuffer<int> finiteStatus = statuses[deviceSlot];
            if (!mixedBlockState)
            {
                purePlans![deviceSlot].Execute(
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon,
                    finiteStatus);
                return;
            }
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
            {
                runtime.CudaBfp8State!.Execute(
                    runtime.Parameter.T,
                    deviceIndex,
                    finiteStatus,
                    scratch: null,
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon,
                    runtime.ApplyWeightDecay,
                    mixedBlockState: true);
            }
        });

        CudaOptimizerFiniteStatusReadback[] readbacks = devices
            .Select(GetOrCreateBfp8FiniteReadback)
            .ToArray();
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            mixedBlockState
                ? "block-BFP8-state mix8_32 AdamW update"
                : "pure BFP8 AdamW update",
            queueReadback: () =>
            {
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    readbacks[deviceSlot].Begin(statuses[deviceSlot]);
                }
            },
            finalize: () =>
            {
                int nonFiniteDevice = -1;
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    int finite =
                        readbacks[deviceSlot].ReadAfterSynchronization();
                    if (finite != 0 && nonFiniteDevice < 0)
                        nonFiniteDevice = devices[deviceSlot];
                }
                if (nonFiniteDevice >= 0)
                {
                    throw new InvalidOperationException(
                        $"Non-finite CUDA value detected while publishing " +
                        $"pure BFP8 AdamW state on device " +
                        $"{nonFiniteDevice} at optimizer step {_step}.");
                }

                foreach (AdamWParameterRuntime runtime in _parameterRuntime)
                {
                    runtime.Parameter.T
                        .MarkCudaBfp8DataReplicasSynchronized(devices);
                }
            });
    }

    private CudaOptimizerKernels.AdamWBfp8MultiTensorItem[]
        CreateCudaBfp8PlanItems()
        => _parameterRuntime
            .Select(runtime => new CudaOptimizerKernels
                .AdamWBfp8MultiTensorItem(
                    runtime.Parameter.T,
                    runtime.CudaBfp8State
                        ?? throw new InvalidOperationException(
                            "Pure BFP8 AdamW state was not prepared."),
                    runtime.ApplyWeightDecay))
            .ToArray();

    private CudaOptimizerKernels.AdamWBfp8MultiTensorPlan[]
        PrepareCudaBfp8Plans(
            IReadOnlyList<int> devices,
            IReadOnlyList<CudaOptimizerKernels.AdamWBfp8MultiTensorItem>
                items)
    {
        var plans = new CudaOptimizerKernels.AdamWBfp8MultiTensorPlan[
            devices.Count];
        for (int slot = 0; slot < devices.Count; slot++)
        {
            int deviceIndex = devices[slot];
            if (!_cudaBfp8MultiTensorPlans.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.AdamWBfp8MultiTensorPlan? plan)
                || !plan.Matches(items))
            {
                if (plan is not null)
                {
                    plan.Dispose();
                    _cudaBfp8MultiTensorPlans.Remove(deviceIndex);
                }
                plan = new CudaOptimizerKernels.AdamWBfp8MultiTensorPlan(
                    deviceIndex, items);
                _cudaBfp8MultiTensorPlans.Add(deviceIndex, plan);
                _cudaMultiTensorPlanBuildCount++;
            }
            plans[slot] = plan;
        }
        return plans;
    }

    private void DisposeCudaBfp8Plans()
    {
        foreach (CudaOptimizerKernels.AdamWBfp8MultiTensorPlan plan
            in _cudaBfp8MultiTensorPlans.Values)
        {
            plan.Dispose();
        }
        _cudaBfp8MultiTensorPlans.Clear();
    }

    private NativeCudaBuffer<int> GetOrCreateBfp8FiniteStatus(int deviceIndex)
    {
        if (_cudaBfp8FiniteStatus.TryGetValue(
                deviceIndex,
                out NativeCudaBuffer<int>? status))
        {
            return status;
        }
        status = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
            .Allocate1D<int>(1);
        _cudaBfp8FiniteStatus.Add(deviceIndex, status);
        return status;
    }

    private CudaOptimizerFiniteStatusReadback
        GetOrCreateBfp8FiniteReadback(int deviceIndex)
    {
        if (_cudaBfp8FiniteReadbacks.TryGetValue(
                deviceIndex,
                out CudaOptimizerFiniteStatusReadback? readback))
        {
            return readback;
        }
        readback = new CudaOptimizerFiniteStatusReadback(deviceIndex);
        _cudaBfp8FiniteReadbacks.Add(deviceIndex, readback);
        return readback;
    }

    private CudaOptimizerKernels.AdamWBfp8DeviceScratch
        GetOrCreateBfp8Scratch(int deviceIndex, int capacity)
    {
        if (_cudaBfp8Scratch.TryGetValue(
                deviceIndex,
                out CudaOptimizerKernels.AdamWBfp8DeviceScratch? scratch))
        {
            if (scratch.Capacity >= capacity)
                return scratch;
            scratch.Dispose();
            _cudaBfp8Scratch.Remove(deviceIndex);
        }
        scratch = new CudaOptimizerKernels.AdamWBfp8DeviceScratch(
            deviceIndex, capacity);
        _cudaBfp8Scratch.Add(deviceIndex, scratch);
        return scratch;
    }

    private void DisposeBfp8FiniteStatus()
    {
        List<Exception>? failures = null;
        foreach (CudaOptimizerFiniteStatusReadback readback
            in _cudaBfp8FiniteReadbacks.Values)
        {
            try
            {
                readback.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        _cudaBfp8FiniteReadbacks.Clear();
        foreach (NativeCudaBuffer<int> status
            in _cudaBfp8FiniteStatus.Values)
        {
            try
            {
                status.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        _cudaBfp8FiniteStatus.Clear();
        if (failures is not null)
        {
            throw new AggregateException(
                "AdamW BFP8 finite-status cleanup failed.", failures);
        }
    }

    private void DisposeBfp8Scratch()
    {
        List<Exception>? failures = null;
        foreach (CudaOptimizerKernels.AdamWBfp8DeviceScratch scratch
            in _cudaBfp8Scratch.Values)
        {
            try
            {
                scratch.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
        }
        _cudaBfp8Scratch.Clear();
        if (failures is not null)
        {
            throw new AggregateException(
                "AdamW BFP8 shared-scratch cleanup failed.", failures);
        }
    }

    internal void DisposeCudaResources()
    {
        List<Exception>? failures = null;
        foreach (CudaOptimizerKernels.AdamWMultiTensorPlan plan
            in _cudaMultiTensorPlans.Values)
        {
            TryDisposeCudaResource(plan, ref failures);
        }
        _cudaMultiTensorPlans.Clear();
        foreach (CudaOptimizerKernels.AdamWBfp8MultiTensorPlan plan
            in _cudaBfp8MultiTensorPlans.Values)
        {
            TryDisposeCudaResource(plan, ref failures);
        }
        _cudaBfp8MultiTensorPlans.Clear();
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            if (runtime.CudaState is not null)
                TryDisposeCudaResource(runtime.CudaState, ref failures);
            if (runtime.CudaBFloat16State is not null)
                TryDisposeCudaResource(runtime.CudaBFloat16State, ref failures);
            if (runtime.CudaMixedState is not null)
                TryDisposeCudaResource(runtime.CudaMixedState, ref failures);
            if (runtime.CudaBfp8State is not null)
                TryDisposeCudaResource(runtime.CudaBfp8State, ref failures);
            runtime.CudaState = null;
            runtime.CudaBFloat16State = null;
            runtime.CudaMixedState = null;
            runtime.CudaBfp8State = null;
        }
        try
        {
            DisposeBfp8FiniteStatus();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        try
        {
            DisposeBfp8Scratch();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        _cudaStateAuthorityDevice = null;
        if (failures is not null)
        {
            throw new AggregateException(
                "AdamW CUDA resource cleanup failed.", failures);
        }
    }

    private static void TryDisposeCudaResource(
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

    /// <summary>
    /// Transfers resident moments to their existing managed backing arrays,
    /// then drops plans and device buffers before a CPU update. This is a
    /// precision-preserving authority handoff; BF16 and BFP8 state is decoded
    /// only by the same resident-state codecs used for checkpoint streaming.
    /// </summary>
    private void MakeCpuStateAuthoritative()
    {
        if (_cudaStateAuthorityDevice is null)
            return;

        SynchronizeStateForStreaming();
        DisposeCudaResources();
    }

    internal (CudaBfp8BufferView First, CudaBfp8BufferView Second)
        GetCudaBfp8Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.AdamWBfp8ResidentState state =
            _parameterRuntime[parameterIndex].CudaBfp8State
            ?? throw new InvalidOperationException(
                "The AdamW parameter has no resident BFP8 state.");
        return (
            state.GetFirstMoment(deviceIndex),
            state.GetSecondMoment(deviceIndex));
    }

    internal (NativeCudaBuffer<short> First, NativeCudaBuffer<short> Second)
        GetCudaBFloat16Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.AdamWBFloat16ResidentState state =
            _parameterRuntime[parameterIndex].CudaBFloat16State
            ?? throw new InvalidOperationException(
                "The AdamW parameter has no resident BF16 state.");
        CudaOptimizerKernels.AdamWBFloat16ResidentState.Buffers buffers =
            state.GetOrCreate(deviceIndex);
        return (buffers.First, buffers.Second);
    }

    public OptimizerStateDictionary state_dict()
        => OptimizerStateDictionary.Create("AdamW", CaptureState());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        // The deserializer already created arrays owned by this restore. Do
        // not duplicate all first/second moments before assigning them.
        RestoreState(
            state.Read<AdamWState>("AdamW"),
            takeOwnership: true);
    }

}
