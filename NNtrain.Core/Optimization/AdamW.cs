namespace NNtrain;

public partial class AdamW : IOptimizer, ILearningRateAdjustable,
    IMix8QuantizationDiagnosticsProvider
{
    private readonly ArcOptimizerStateCache _arcState = new();
    private readonly bool _usesPackedBFloat16Checkpoint;
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
    private CudaMix8QuantizationDiagnostics? _cudaMix8Diagnostics;
    private int _cudaMix8DiagnosticsDevice = -1;
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
    internal bool UsesPackedBFloat16Checkpoint => _usesPackedBFloat16Checkpoint;
    internal int CudaMultiTensorPlanBuildCount
        => _cudaMultiTensorPlanBuildCount;

    public AdamW(
        IEnumerable<Parameter> parameters,
        AdamWOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        _cudaDispatchPolicy = CudaDispatchPolicy.Current.Validate();
        _usesPackedBFloat16Checkpoint = TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;

        _parameters = OptimizerParameters.Materialize(parameters, "AdamW");

        _totalElements = _parameters.Sum(
            parameter => (long)parameter.T.Numel);
        _workItems = CreateWorkItems(_parameters);

        AdamWOptions effectiveOptions = options ?? new AdamWOptions();
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16)
        {
            // The host checkpoint arrays remain float[] for compatibility;
            // Arc resident moments use physically packed BF16 buffers.
            effectiveOptions = effectiveOptions with
            {
                UseBFloat16FirstMoment = false,
                UseBFloat16SecondMoment = false,
            };
        }
        else if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
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

    public bool TryGetMix8QuantizationDiagnostics(
        out Mix8QuantizationDiagnostics diagnostics)
    {
        CudaMix8QuantizationDiagnostics? accumulator =
            _cudaMix8Diagnostics;
        int deviceIndex = _cudaMix8DiagnosticsDevice;
        if (accumulator is null || deviceIndex < 0)
        {
            diagnostics = default;
            return false;
        }
        return accumulator.TryRead(deviceIndex, out diagnostics);
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

    internal void ZeroGrad()
    {
        if (Tensor.ExecutionDevice is TensorDevice.Cuda or TensorDevice.Arc)
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
        if (!Tensor.ArcResident) _arcState.Dispose();
        if (_usesPackedBFloat16Checkpoint && Tensor.ExecutionDevice != TensorDevice.Arc)
            throw new NotSupportedException("An Arc mix8_16 AdamW optimizer can only step in an Arc session.");
        if (Tensor.ExecutionDevice == TensorDevice.Arc && !Tensor.ArcResident
            && TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16)
            throw new NotSupportedException("Arc mix8_16 AdamW requires resident tensors and physical BF16 optimizer state.");
        if (_step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "AdamW cannot advance beyond Int32.MaxValue steps.");
        }

        _cudaMix8DiagnosticsDevice = -1;
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

        if (Tensor.ArcResident)
        {
            bool bf16State = TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
            {
                if (runtime.FirstMomentBFloat16 is not null || runtime.SecondMomentBFloat16 is not null)
                    throw new NotSupportedException("Arc AdamW currently requires FP32 optimizer moments.");
                Tensor tensor = runtime.Parameter.T;
                if (bf16State)
                {
                    Tensor.ArcLane.Run("adam_bf16_packed", tensor.Numel, 0,
                        tensor.ArcBFloat16Gradient(),
                        _arcState.GetBFloat16(runtime.FirstMomentPacked!),
                        _arcState.GetBFloat16(runtime.SecondMomentPacked!),
                        tensor.ArcBFloat16Master(), tensor.Numel,
                        options.Beta1, options.Beta2, _stepUpdateScale, _stepScaledEpsilon,
                        runtime.ApplyWeightDecay ? 1f - options.LearningRate * options.WeightDecay : 1f);
                }
                else
                {
                    Tensor.ArcLane.Run("adam", tensor.Numel, 0, tensor.ArcGradient(),
                        _arcState.Get(runtime.FirstMoment), _arcState.Get(runtime.SecondMoment), tensor.ArcMaster(), tensor.Numel,
                        options.Beta1, options.Beta2, _stepUpdateScale, _stepScaledEpsilon,
                        runtime.ApplyWeightDecay ? 1f - options.LearningRate * options.WeightDecay : 1f);
                }
                tensor.CompleteArcUpdate();
            }
            return;
        }

        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            AdamWParameterRuntime runtime =
                _parameterRuntime[parameterIndex];
            // Loading BFP8 values or converting storage can replace the FP32
            // master after optimizer construction. Resolve its current owner
            // before updating, just as gradients are refreshed each step.
            runtime.Data = runtime.Parameter.DataBuffer;
            runtime.Gradient = runtime.Parameter.T.GradientBuffer;
        }

        if (Tensor.ExecutionDevice == TensorDevice.Arc)
        {
            foreach (AdamWParameterRuntime runtime in _parameterRuntime)
            {
                if (runtime.FirstMomentBFloat16 is not null || runtime.SecondMomentBFloat16 is not null)
                    throw new NotSupportedException("Arc AdamW currently requires FP32 optimizer moments.");
                Tensor.ArcLane.Run("adam", runtime.Data.Length, 0,
                    NNtrain.Arc.ArcExecutionLane.In(runtime.Gradient.Length == 0 ? new float[runtime.Data.Length] : runtime.Gradient),
                    NNtrain.Arc.ArcExecutionLane.InOut(runtime.FirstMoment),
                    NNtrain.Arc.ArcExecutionLane.InOut(runtime.SecondMoment),
                    NNtrain.Arc.ArcExecutionLane.InOut(runtime.Data), runtime.Data.Length,
                    options.Beta1, options.Beta2, _stepUpdateScale, _stepScaledEpsilon,
                    runtime.ApplyWeightDecay ? 1f - options.LearningRate * options.WeightDecay : 1f);
            }
        }
        else if (_workItems.Length > 1 && _totalElements >= 32_768)
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


}
