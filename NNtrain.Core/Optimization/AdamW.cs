namespace NNtrain;

public partial class AdamW : IOptimizer, ILearningRateAdjustable
{
    private readonly List<Parameter> _parameters;
    private readonly long _totalElements;
    private readonly AdamWParameterRuntime[] _parameterRuntime;
    private readonly AdamWWorkItem[] _workItems;
    private readonly Dictionary<int, CudaOptimizerKernels.AdamWMultiTensorPlan>
        _cudaMultiTensorPlans = [];
    private AdamWOptions _options;
    private AdamWParameterState[] _parameterStates;
    private readonly Action<int> _updateWorkItemAction;
    private readonly Action<int> _clearWorkItemAction;
    private AdamWOptions _stepOptions = null!;
    private float _stepUpdateScale;
    private float _stepScaledEpsilon;
    private int _step;

    internal IReadOnlyList<Parameter> Parameters => _parameters;

    public AdamW(
        IEnumerable<Parameter> parameters,
        AdamWOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);

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
                    _options.UseBFloat16FirstMoment
                        ? DecodeBFloat16(
                            _parameterRuntime[index]
                                .FirstMomentBFloat16!)
                        : _parameterRuntime[index].FirstMoment.ToArray(),
                    _options.UseBFloat16SecondMoment
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
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            return;

        int primaryDevice = Tensor.CudaDeviceIndex;
        foreach (AdamWParameterRuntime runtime in _parameterRuntime)
        {
            runtime.CudaState?.SynchronizeHost(primaryDevice);
            runtime.CudaBFloat16State?.SynchronizeHost(primaryDevice);
        }
    }

    internal int StreamingStep => _step;

    internal AdamWOptions StreamingOptions => _options;

    internal int StreamingParameterCount => _parameterStates.Length;

    internal AdamWStreamingParameterState GetStreamingParameterState(int index)
    {
        AdamWParameterState state = _parameterStates[index];
        AdamWParameterRuntime runtime = _parameterRuntime[index];
        return new AdamWStreamingParameterState(
            state.Index,
            state.Name,
            state.Shape,
            runtime.FirstMoment,
            runtime.FirstMomentBFloat16,
            runtime.SecondMoment,
            runtime.SecondMomentBFloat16);
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
        _options = restored.Options;
        _parameterStates = restored.ParameterStates;
        for (int index = 0; index < _parameterRuntime.Length; index++)
        {
            _parameterRuntime[index].CudaState?.Dispose();
            _parameterRuntime[index].CudaState = null;
            _parameterRuntime[index].CudaBFloat16State?.Dispose();
            _parameterRuntime[index].CudaBFloat16State = null;
            if (_options.UseBFloat16FirstMoment)
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
            if (_options.UseBFloat16SecondMoment)
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
        foreach (CudaOptimizerKernels.AdamWMultiTensorPlan plan
            in _cudaMultiTensorPlans.Values)
        {
            plan.Dispose();
        }
        _cudaMultiTensorPlans.Clear();
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

                float[] first = runtime.FirstMomentBFloat16 is null
                    ? runtime.FirstMoment
                    : DecodeBFloat16(runtime.FirstMomentBFloat16);
                float[] second = runtime.SecondMomentBFloat16 is null
                    ? runtime.SecondMoment
                    : DecodeBFloat16(runtime.SecondMomentBFloat16);
                CudaOptimizerKernels.AdamWUpdate(
                    runtime.Data,
                    runtime.Parameter.T.GradientBuffer,
                    first,
                    second,
                    options.Beta1,
                    options.Beta2,
                    options.LearningRate,
                    options.WeightDecay,
                    _stepUpdateScale,
                    _stepScaledEpsilon,
                    runtime.ApplyWeightDecay);
                if (runtime.FirstMomentBFloat16 is not null)
                    runtime.FirstMomentBFloat16 = EncodeBFloat16(first);
                if (runtime.SecondMomentBFloat16 is not null)
                    runtime.SecondMomentBFloat16 = EncodeBFloat16(second);
                if (runtime.FirstMomentBFloat16 is not null
                    || runtime.SecondMomentBFloat16 is not null)
                {
                    runtime.Parameter.CompleteUpdate();
                }
            }

            CudaOptimizerKernels.AdamWMultiTensorItem[] multiTensorItems =
                _parameterRuntime
                    .Where(runtime => runtime.CudaState is not null
                        || runtime.CudaBFloat16State is not null)
                    .Select(runtime => new CudaOptimizerKernels
                        .AdamWMultiTensorItem(
                            runtime.Parameter.T,
                            runtime.CudaState,
                            runtime.CudaBFloat16State,
                            runtime.ApplyWeightDecay))
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
                        out CudaOptimizerKernels.AdamWMultiTensorPlan? plan))
                    {
                        plan = new CudaOptimizerKernels.AdamWMultiTensorPlan(
                            deviceIndex, multiTensorItems);
                        _cudaMultiTensorPlans.Add(deviceIndex, plan);
                    }
                    plans[deviceSlot] = plan;
                }

                // One launch per GPU updates all AdamW tensors. Each worker
                // stays bound to one device, so both GPUs advance
                // concurrently. A checkpoint may contain a legacy mixed
                // moment representation; in that case the loop above has
                // already updated it and this array is intentionally empty.
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
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
            {
                runtime.Parameter.T.MarkCudaDataMutated(
                    Tensor.CudaDeviceIndex);
            }
            CudaOptimizerKernels.SynchronizeDevices(
                devices,
                "AdamW update");
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

    public void step() => Step();

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
