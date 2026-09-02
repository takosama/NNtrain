using System.Diagnostics;

namespace NNtrain;

public sealed partial class NekoMuon : IOptimizer, ILearningRateAdjustable
{
    private const float NewtonSchulzA = 3.4445f;
    private const float NewtonSchulzB = -4.7750f;
    private const float NewtonSchulzC = 2.0315f;

    private readonly List<Parameter> _parameters;
    private readonly long _totalElements;
    private readonly NekoMuonWorkspace?[] _workspaces;
    private readonly CudaOptimizerKernels.NekoMuonResidentState?[] _cudaStates;
    private readonly CudaOptimizerKernels.NekoMuonBFloat16ResidentState?[]
        _cudaBFloat16States;
    private readonly CudaOptimizerKernels.NekoMuonBfp8ResidentState?[]
        _cudaBfp8States;
    private readonly int _cudaBatchCapacity;
    private readonly long _cudaScratchBudgetBytes;
    private readonly Dictionary<int, CudaOptimizerKernels.NekoMuonDeviceScratch>
        _cudaScratch = [];
    private readonly Dictionary<int, CudaOptimizerKernels.NekoMuonStatsBatch>
        _cudaStatsBatches = [];
    private readonly Dictionary<int,
        CudaOptimizerKernels.NekoMuonBFloat16StatsBatch>
        _cudaBFloat16StatsBatches = [];
    private readonly Dictionary<int,
        CudaOptimizerKernels.NekoMuonBfp8StatsBatch> _cudaBfp8StatsBatches = [];
    private readonly Dictionary<int, NativeCudaBuffer<int>>
        _cudaBfp8FiniteStatus = [];
    private readonly Dictionary<int, CudaOptimizerFiniteStatusReadback>
        _cudaBfp8FiniteReadbacks = [];
    private readonly CudaDispatchPolicy _cudaDispatchPolicy;
    private NekoMuonState _state;
    private int? _cudaStateAuthorityDevice;

    internal IReadOnlyList<Parameter> Parameters => _parameters;

    public bool ProfilingEnabled { get; set; }

    public NekoMuonStepProfile LastStepProfile { get; private set; }

    /// <summary>
    /// Diagnostic-only switch used by the convergence profiler to compare
    /// adaptive depth with an exact full five-step Newton-Schulz update. It is
    /// intentionally not serialized and therefore cannot silently change a
    /// resumed training run.
    /// </summary>
    internal bool ForceFullNewtonSchulz { get; set; }

    public NekoMuon(
        IEnumerable<Parameter> parameters,
        NekoMuonOptions? options = null)
        : this(parameters, options, CudaDispatchPolicy.Current)
    {
    }

    internal NekoMuon(
        IEnumerable<Parameter> parameters,
        NekoMuonOptions? options,
        CudaDispatchPolicy cudaDispatchPolicy)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        _cudaDispatchPolicy = (cudaDispatchPolicy
            ?? throw new ArgumentNullException(nameof(cudaDispatchPolicy)))
            .Validate();

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
                    $"Parameter '{parameter.Name}' was supplied to " +
                    "NekoMuon more than once.",
                    nameof(parameters));
            }

            _parameters.Add(parameter);
        }

        _totalElements = _parameters.Sum(
            parameter => (long)parameter.T.Numel);

        NekoMuonOptions effectiveOptions = options ?? new NekoMuonOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        _state = CreateInitialState(_parameters, effectiveOptions);
        // CUDA never consumes the managed Newton-Schulz work arrays.  Keep
        // them lazy so constructing a CUDA optimizer does not retain another
        // six full-size host arrays for every matrix parameter.
        _workspaces = new NekoMuonWorkspace?[_parameters.Count];
        _cudaStates = new CudaOptimizerKernels.NekoMuonResidentState?[
            _parameters.Count];
        _cudaBFloat16States = new CudaOptimizerKernels
            .NekoMuonBFloat16ResidentState?[_parameters.Count];
        _cudaBfp8States = new CudaOptimizerKernels
            .NekoMuonBfp8ResidentState?[_parameters.Count];
        _cudaScratchBudgetBytes = ResolveCudaScratchBudgetBytes();
        _cudaBatchCapacity = ResolveCudaBatchCapacity(
            _cudaScratchBudgetBytes);
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
            CudaOptimizerKernels.PrewarmNekoMuon(Tensor.CudaDeviceIndices);
    }

    public NekoMuonState CaptureState()
        => CloneState(CaptureStateForStreaming());

    /// <summary>
    /// Returns scalar optimizer diagnostics without copying the large moment
    /// arrays. This is safe to query after <see cref="step"/> for training
    /// telemetry and makes adaptive Newton-Schulz depth observable.
    /// </summary>
    public NekoMuonDiagnostics GetDiagnostics()
    {
        NekoMuonParameterState[] states = _state.ParameterStates;
        if (states.Length == 0)
        {
            return new NekoMuonDiagnostics(
                _state.Step,
                0f,
                0f,
                0f,
                0f,
                _state.Options.MaxNewtonSchulzSteps);
        }

        float minimum = float.PositiveInfinity;
        float maximum = float.NegativeInfinity;
        double sum = 0d;
        double depthSum = 0d;
        bool runNewtonSchulz =
            _state.Step % _state.Options.NewtonSchulzInterval == 0;
        foreach (NekoMuonParameterState state in states)
        {
            minimum = MathF.Min(minimum, state.Confidence);
            maximum = MathF.Max(maximum, state.Confidence);
            sum += state.Confidence;
            depthSum += ForceFullNewtonSchulz && runNewtonSchulz
                ? _state.Options.MaxNewtonSchulzSteps
                : ResolveNewtonSchulzDepth(
                    _state.Options,
                    state.Confidence,
                    runNewtonSchulz);
        }
        float mean = (float)(sum / states.Length);
        return new NekoMuonDiagnostics(
            _state.Step,
            minimum,
            mean,
            maximum,
            (float)(depthSum / states.Length),
            _state.Options.MaxNewtonSchulzSteps);
    }

    internal NekoMuonState CaptureStateForStreaming()
    {
        if (_cudaStateAuthorityDevice is int primaryDevice)
        {
            for (int index = 0; index < _cudaStates.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonResidentState? state =
                    _cudaStates[index];
                state?.SynchronizeHost(primaryDevice);
                // Generic Float32/BF16 CUDA computes confidence from the
                // gathered host statistics and stores it in _state below the
                // step. Its device Confidence slot is only authoritative for
                // mix8_32's device-control path. Do not overwrite a current
                // host value with the generic slot's construction-time zero.
                if (state is not null
                    && state.IsDeviceConfidenceAuthoritative)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[index];
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
            }
            for (int index = 0; index < _cudaBFloat16States.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonBFloat16ResidentState? state =
                    _cudaBFloat16States[index];
                state?.SynchronizeHost(primaryDevice);
                if (state is not null
                    && state.IsDeviceConfidenceAuthoritative)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[index];
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
            }
            for (int index = 0; index < _cudaBfp8States.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonBfp8ResidentState? state =
                    _cudaBfp8States[index];
                state?.SynchronizeHost(primaryDevice);
                if (state is not null)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[index];
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
            }
        }
        return _state;
    }

    public float LearningRate => _state.Options.LearningRate;

    public NekoMuonNewtonSchulzDepthMode NewtonSchulzDepthMode =>
        _state.Options.NewtonSchulzDepthMode;

    public float NewtonSchulzDepth => _state.Options.NewtonSchulzDepth;

    public void SetLearningRate(float learningRate)
    {
        if (!float.IsFinite(learningRate) || learningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(learningRate),
                learningRate,
                "NekoMuon learning rate must be finite and positive.");
        }

        _state = _state with
        {
            Options = _state.Options with { LearningRate = learningRate },
        };
    }

    /// <summary>
    /// Selects how confidence controls Newton-Schulz depth. This intentionally
    /// updates optimizer state so the selected runtime policy is preserved by
    /// the next checkpoint.
    /// </summary>
    public void SetNewtonSchulzDepthPolicy(
        NekoMuonNewtonSchulzDepthMode mode,
        float depth = 0f)
    {
        NekoMuonOptions options = _state.Options with
        {
            NewtonSchulzDepthMode = mode,
            NewtonSchulzDepth = depth,
        };
        ValidateOptions(options, nameof(depth));
        _state = _state with { Options = options };
    }

    /// <summary>
    /// Reconfigures this optimizer to the ordinary Muon policy while
    /// preserving its current step, moments, confidence, and
    /// learning-rate/decay settings.
    /// </summary>
    /// <param name="momentum">
    /// Exponential momentum coefficient in the half-open interval [0, 1).
    /// </param>
    /// <param name="nesterov">
    /// When true, orthogonalizes beta * m_t + (1 - beta) * g_t. When false,
    /// orthogonalizes the momentum buffer directly.
    /// </param>
    public void SetOrdinaryMuonPolicy(
        float momentum = 0.95f,
        bool nesterov = true)
    {
        if (!float.IsFinite(momentum) || momentum < 0f || momentum >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(momentum),
                momentum,
                "Muon momentum must be finite and in [0, 1).");
        }

        NekoMuonOptions options = _state.Options with
        {
            BetaFast = momentum,
            BetaSlow = momentum,
            Nesterov = nesterov,
            Rho = 0f,
            MaxNewtonSchulzSteps = 5,
            NewtonSchulzInterval = 1,
            NewtonSchulzDepthMode =
                NekoMuonNewtonSchulzDepthMode.Fixed,
            NewtonSchulzDepth = 5f,
        };
        ValidateOptions(options, nameof(momentum));
        _state = _state with { Options = options };
    }

    public void RestoreState(NekoMuonState state)
        => RestoreState(state, takeOwnership: false);

    private void RestoreState(
        NekoMuonState state,
        bool takeOwnership)
    {
        ArgumentNullException.ThrowIfNull(state);
        ValidateState(state);
        DisposeCudaResources();
        _state = takeOwnership ? state : CloneState(state);
    }

    internal void RestoreStateOwned(NekoMuonState state)
        => RestoreState(state, takeOwnership: true);

    internal void ZeroGrad()
    {
        if (_parameters.Count > 1 && _totalElements >= 32_768)
        {
            Tensor.RunParallel(
                0,
                _parameters.Count,
                index => _parameters[index].ZeroGrad());
            return;
        }

        foreach (Parameter parameter in _parameters)
            parameter.ZeroGrad();
    }

    public void zero_grad() => ZeroGrad();

    internal void Step()
    {
        if (_state.Step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "NekoMuon cannot advance beyond Int32.MaxValue steps.");
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

        _state = _state with { Step = _state.Step + 1 };
        NekoMuonOptions options = _state.Options;
        float fastCorrection =
            1f - MathF.Pow(options.BetaFast, _state.Step);
        float slowCorrection =
            1f - MathF.Pow(options.BetaSlow, _state.Step);
        long[]? profileTicks = ProfilingEnabled ? new long[9] : null;

        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            StepCuda(
                options,
                fastCorrection,
                slowCorrection);
            LastStepProfile = default;
            return;
        }

        void UpdateParameter(int parameterIndex)
        {
            Parameter parameter = _parameters[parameterIndex];
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            float[] gradientBuffer = parameter.T.GradientBuffer;
            float[] fast = parameterState.FastMoment;
            float[] slow = parameterState.SlowMoment;
            NekoMuonWorkspace workspace = _workspaces[parameterIndex]
                ??= CreateWorkspace(parameter);
            float[] fastHat = workspace.FastHat;
            float[] slowHat = workspace.SlowHat;

            long phaseStart = profileTicks is null
                ? 0L
                : Stopwatch.GetTimestamp();
            UpdateMoments(
                gradientBuffer,
                fast,
                slow,
                fastHat,
                slowHat,
                options,
                fastCorrection,
                slowCorrection);
            AddProfileTicks(profileTicks, 0, phaseStart);

            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            float confidenceRaw = CalculateConfidenceRaw(
                fastHat,
                slowHat,
                options.Epsilon);
            float confidence = Math.Clamp(
                options.Rho * parameterState.Confidence
                    + (1f - options.Rho) * confidenceRaw,
                0f,
                1f);
            AddProfileTicks(profileTicks, 1, phaseStart);
            _state.ParameterStates[parameterIndex] =
                parameterState with { Confidence = confidence };

            bool runNewtonSchulz =
                _state.Step % options.NewtonSchulzInterval == 0;
            // Moments and weights still advance on the intervening steps.
            // They use the normalized current momentum matrix; only the
            // expensive orthogonalization is cadence-limited.
            float depth = ForceFullNewtonSchulz && runNewtonSchulz
                ? options.MaxNewtonSchulzSteps
                : ResolveNewtonSchulzDepth(
                    options,
                    confidence,
                    runNewtonSchulz);
            int wholeSteps = Math.Min(
                options.MaxNewtonSchulzSteps,
                (int)MathF.Floor(depth));
            float fraction = depth - wholeSteps;

            GetMatrixShape(
                parameter,
                out int originalRows,
                out int originalColumns);
            bool transpose = originalRows > originalColumns;
            int rows = Math.Min(originalRows, originalColumns);
            int columns = Math.Max(originalRows, originalColumns);
            float[] x = workspace.X;
            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            InitializeMuonMatrix(
                fastHat,
                x,
                originalRows,
                originalColumns,
                transpose,
                options.Epsilon);
            AddProfileTicks(profileTicks, 2, phaseStart);

            float[] next = workspace.Next;
            float[] gram = workspace.Gram;
            float[] gramSquared = workspace.GramSquared;
            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            for (int step = 0; step < wholeSteps; step++)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns,
                    parameter.T.DType == TensorDType.BFloat16,
                    profileTicks);
                (x, next) = (next, x);
            }

            if (fraction > 0f)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns,
                    parameter.T.DType == TensorDType.BFloat16,
                    profileTicks);
                InterpolateInPlace(x, next, fraction);
            }
            AddProfileTicks(profileTicks, 3, phaseStart);

            float finalScale = MathF.Sqrt(MathF.Max(
                1f,
                (float)originalRows / originalColumns));
            float[] update = transpose ? slowHat : x;
            if (transpose)
            {
                phaseStart = profileTicks is null
                    ? 0L
                    : Stopwatch.GetTimestamp();
                TransposeBack(
                    x,
                    update,
                    originalRows,
                    originalColumns);
                AddProfileTicks(profileTicks, 4, phaseStart);
            }

            phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
            ApplyUpdate(parameter, update, finalScale, options);
            AddProfileTicks(profileTicks, 5, phaseStart);
        }

        if (_parameters.Count > 1 && _totalElements >= 32_768)
            Tensor.RunParallel(0, _parameters.Count, UpdateParameter);
        else
            for (int index = 0; index < _parameters.Count; index++)
                UpdateParameter(index);

        if (profileTicks is not null)
        {
            LastStepProfile = new NekoMuonStepProfile(
                TicksToMilliseconds(profileTicks[0]),
                TicksToMilliseconds(profileTicks[1]),
                TicksToMilliseconds(profileTicks[2]),
                TicksToMilliseconds(profileTicks[3]),
                TicksToMilliseconds(profileTicks[4]),
                TicksToMilliseconds(profileTicks[5]),
                TicksToMilliseconds(profileTicks[6]),
                TicksToMilliseconds(profileTicks[7]),
                TicksToMilliseconds(profileTicks[8]));
        }
    }

    public void step() => Step();

    public OptimizerStateDictionary state_dict()
        => OptimizerStateDictionary.Create("NekoMuon", CaptureState());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        // Read<T> creates a private object graph from serialized JSON. Taking
        // ownership avoids immediately cloning every moment array during a
        // checkpoint restore, which otherwise doubles the resume peak.
        RestoreState(
            state.Read<NekoMuonState>("NekoMuon"),
            takeOwnership: true);
    }

    private void StepCuda(
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection)
    {
        if (_parameters.Count == 0)
            return;

        int[] devices = Tensor.CudaDeviceIndices.ToArray();
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "CUDA NekoMuon requires at least one device.");
        }
        _cudaStateAuthorityDevice = devices[0];
        bool pureBFloat16 = UsesPureBFloat16OptimizerState();
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.BFloat16
            && !pureBFloat16)
        {
            throw new InvalidOperationException(
                "The pure BFloat16 NekoMuon contract requires every " +
                "parameter to use physical BFloat16 storage.");
        }
        if (pureBFloat16)
        {
            StepCudaBFloat16(
                options, fastCorrection, slowCorrection, devices);
            return;
        }
        TransitionFromCudaBFloat16State(devices[0]);
        bool pureBfp8 = _parameters.All(parameter =>
            UsesPureBfp8OptimizerState(parameter.T));
        bool anyMix8 = _parameters.Any(parameter =>
            UsesMix8Parameter(parameter.T));
        bool mix8 = _parameters.All(parameter =>
            UsesMix8Parameter(parameter.T));
        if (anyMix8 && !mix8)
        {
            throw new InvalidOperationException(
                "The mix8_32 NekoMuon contract cannot mix block-scaled " +
                "BFP8 parameters with another storage format.");
        }
        if (anyMix8)
            ValidateMix8OptimizerContract();
        if (TensorExecutionContext.ActivePrecisionPolicy?.OptimizerState
                == NNtrain.Runtime.Execution.NumericFormat.Bfp8
            && !pureBfp8)
        {
            throw new InvalidOperationException(
                "The pure BFP8 NekoMuon contract requires every parameter " +
                "to use tensor-wide BFP8 storage.");
        }
        if (pureBfp8)
        {
            StepCudaBfp8(options, fastCorrection, slowCorrection, devices);
            return;
        }
        if (TensorExecutionContext.ActivePrecisionPolicy?.Mode
                == NNtrain.Runtime.Execution.PrecisionMode.Mix8_32
            && !mix8)
        {
            throw new InvalidOperationException(
                "The mix8_32 NekoMuon contract requires every parameter " +
                "to use block-scaled BFP8 storage.");
        }
        // Ordinary Muon recursively reuses m_t and feeds its Nesterov
        // direction to NS5 on every step. Requantizing those values to BFP8
        // can erase small momentum components and change the orthogonalized
        // update enough to stall convergence. Keep its recurrent state in
        // FP32 even when the optional low-memory NekoMuon state is enabled;
        // mix8_32 parameters and activations remain block-BFP8 and the
        // parameter update still accumulates in the resident FP32 master.
        if (mix8
            && _cudaDispatchPolicy.EnableBlockBfp8OptimizerState
            && !options.Nesterov)
        {
            StepCudaBfp8(
                options,
                fastCorrection,
                slowCorrection,
                devices,
                mixedBlockState: true);
            return;
        }
        bool runNewtonSchulz =
            _state.Step % options.NewtonSchulzInterval == 0;
        bool deviceOnlyFixedFive = runNewtonSchulz
            && options.MaxNewtonSchulzSteps == 5
            && (ForceFullNewtonSchulz
                || options.NewtonSchulzDepthMode
                    == NekoMuonNewtonSchulzDepthMode.Fixed
                && options.NewtonSchulzDepth == 5f);
        if (_cudaBfp8States.Any(state => state is not null))
        {
            int primaryDevice = devices[0];
            for (int index = 0; index < _cudaBfp8States.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonBfp8ResidentState? state =
                    _cudaBfp8States[index];
                if (state is null)
                    continue;
                state.SynchronizeHost(primaryDevice);
                NekoMuonParameterState parameterState =
                    _state.ParameterStates[index];
                _state.ParameterStates[index] = parameterState with
                {
                    Confidence = state.SynchronizeConfidence(primaryDevice),
                };
                state.Dispose();
                _cudaBfp8States[index] = null;
            }
            foreach (CudaOptimizerKernels.NekoMuonBfp8StatsBatch batch
                in _cudaBfp8StatsBatches.Values)
            {
                batch.Dispose();
            }
            _cudaBfp8StatsBatches.Clear();
        }
        int maximumLength = _parameters.Max(parameter => parameter.T.Numel);
        int maximumGramLength = 0;
        foreach (Parameter parameter in _parameters)
        {
            GetMatrixShape(parameter, out int shapeRows, out int shapeColumns);
            int rows = Math.Min(shapeRows, shapeColumns);
            maximumGramLength = Math.Max(
                maximumGramLength,
                checked(rows * rows));
        }

        CudaOptimizerKernels.NekoMuonDeviceScratch GetCudaScratch(
            int deviceIndex)
        {
            lock (_cudaScratch)
            {
                if (!_cudaScratch.TryGetValue(
                    deviceIndex,
                    out CudaOptimizerKernels.NekoMuonDeviceScratch? scratch))
                {
                    scratch = new CudaOptimizerKernels.NekoMuonDeviceScratch(
                        deviceIndex,
                        maximumLength,
                        maximumGramLength,
                        _cudaBatchCapacity,
                        !_cudaDispatchPolicy.DisableTensorCoreNekoMuon);
                    _cudaScratch.Add(deviceIndex, scratch);
                }
                return scratch;
            }
        }

        // Materialize one reusable work area per device before moment kernels
        // are queued.  Each device has a single ordered stream, so subsequent
        // parameter updates can safely reuse it without a host-side wait.
        var deviceScratch = new CudaOptimizerKernels.NekoMuonDeviceScratch[
            devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
            deviceScratch[deviceSlot] = GetCudaScratch(devices[deviceSlot]);
        NativeCudaBuffer<int>[]? deviceFiniteStatus = null;
        if (mix8 || deviceOnlyFixedFive)
        {
            deviceFiniteStatus = new NativeCudaBuffer<int>[devices.Length];
            for (int deviceSlot = 0;
                deviceSlot < devices.Length;
                deviceSlot++)
            {
                deviceFiniteStatus[deviceSlot] =
                    GetOrCreateBfp8FiniteStatus(devices[deviceSlot]);
                deviceFiniteStatus[deviceSlot].MemSetToZero();
            }
        }

        CudaOptimizerKernels.NekoMuonResidentState GetCudaState(
            int parameterIndex)
        {
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            return _cudaStates[parameterIndex] ??=
                new CudaOptimizerKernels.NekoMuonResidentState(
                    parameterState.FastMoment,
                    parameterState.SlowMoment,
                    parameterState.Confidence);
        }

        // Queue every moment/statistics update before waiting. Previously the
        // tiny D2H statistics copy serialized the stream once per parameter,
        // preventing kernels for later parameters from filling the GPU.
        void PrepareMomentsAndStatistics()
        {
            for (int index = 0; index < _parameters.Count; index++)
            {
                CudaOptimizerKernels.NekoMuonResidentState cudaState =
                    GetCudaState(index);
                for (int deviceSlot = 0;
                    deviceSlot < devices.Length;
                    deviceSlot++)
                {
                    int deviceIndex = devices[deviceSlot];
                    if (mix8)
                    {
                        CudaOptimizerKernels.NekoMuonPrepareMix8StatsResident(
                            _parameters[index].T,
                            deviceIndex,
                            cudaState,
                            deviceFiniteStatus![deviceSlot],
                            options.BetaFast,
                            options.BetaSlow,
                            fastCorrection,
                            slowCorrection,
                            options.Epsilon,
                            options.Rho,
                            options.Nesterov);
                    }
                    else if (deviceOnlyFixedFive)
                    {
                        CudaOptimizerKernels
                            .NekoMuonPrepareFixedFiveStatsResident(
                                _parameters[index].T,
                                deviceIndex,
                                cudaState,
                                deviceFiniteStatus![deviceSlot],
                                options.BetaFast,
                                options.BetaSlow,
                                fastCorrection,
                                slowCorrection,
                                options.Epsilon,
                                options.Rho,
                                options.Nesterov);
                    }
                    else
                    {
                        CudaOptimizerKernels.NekoMuonPrepareStatsResident(
                            _parameters[index].T,
                            deviceIndex,
                            cudaState,
                            options.BetaFast,
                            options.BetaSlow,
                            fastCorrection,
                            slowCorrection,
                            options.Nesterov);
                    }
                }
            }
        }
        if (CudaOperationProfiler.IsEnabled)
        {
            CudaOperationProfiler.MeasureDevices(
                "optimizer.nekomuon.moments_stats",
                devices,
                PrepareMomentsAndStatistics);
        }
        else
        {
            PrepareMomentsAndStatistics();
        }

        void ReadStatistics()
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                CudaOptimizerKernels.NekoMuonStatsBatch batch;
                lock (_cudaStatsBatches)
                {
                    if (!_cudaStatsBatches.TryGetValue(
                        deviceIndex,
                        out batch!))
                    {
                        var states = new CudaOptimizerKernels
                            .NekoMuonResidentState[_parameters.Count];
                        for (int index = 0; index < states.Length; ++index)
                            states[index] = GetCudaState(index);
                        batch = new CudaOptimizerKernels.NekoMuonStatsBatch(
                            deviceIndex, states);
                        _cudaStatsBatches[deviceIndex] = batch;
                    }
                }
                batch.GatherAndRead();
            });
        }
        if (!deviceOnlyFixedFive && CudaOperationProfiler.IsEnabled)
        {
            CudaOperationProfiler.MeasureDevices(
                "optimizer.nekomuon.stats_d2h",
                devices,
                ReadStatistics);
        }
        else if (!deviceOnlyFixedFive)
        {
            ReadStatistics();
        }

        // Give every CUDA device its own host dispatch loop. This avoids the
        // parameter-major GPU0/GPU1 alternation and keeps both default streams
        // populated during Newton-Schulz and weight publication.
        float[,]? confidences = deviceOnlyFixedFive
            ? null
            : new float[devices.Length, _parameters.Count];
        void FinishUpdates()
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                CudaOptimizerKernels.NekoMuonDeviceScratch scratch =
                    deviceScratch[deviceSlot];
                if (deviceOnlyFixedFive)
                {
                    var fixedBatchItems = new CudaOptimizerKernels
                        .NekoMuonBatchItem[_parameters.Count];
                    for (int parameterIndex = 0;
                        parameterIndex < _parameters.Count;
                        parameterIndex++)
                    {
                        Parameter parameter = _parameters[parameterIndex];
                        GetMatrixShape(
                            parameter,
                            out int originalRows,
                            out int originalColumns);
                        bool applyWeightDecay =
                            parameter.WeightDecay == WeightDecayPolicy.Apply
                            || (options.Decay1D && parameter.T.Rank == 1);
                        fixedBatchItems[parameterIndex] =
                            new CudaOptimizerKernels.NekoMuonBatchItem(
                                parameter.T,
                                GetCudaState(parameterIndex),
                                originalRows,
                                originalColumns,
                                PreviousConfidence: 0f,
                                ApplyWeightDecay: applyWeightDecay);
                    }
                    CudaOptimizerKernels
                        .NekoMuonFinishFixedFiveGroupedDeviceResident(
                            deviceIndex,
                            fixedBatchItems,
                            scratch,
                            deviceFiniteStatus![deviceSlot],
                            fastCorrection,
                            options.Epsilon,
                            NewtonSchulzA,
                            NewtonSchulzB,
                            NewtonSchulzC,
                            options.LearningRate,
                            options.WeightDecay,
                            publishMix8: mix8,
                            nesterov: options.Nesterov);
                    return;
                }
                var batchItems = new CudaOptimizerKernels
                    .NekoMuonBatchItem[_parameters.Count];
                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    Parameter parameter = _parameters[parameterIndex];
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[parameterIndex];
                    GetMatrixShape(
                        parameter,
                        out int originalRows,
                        out int originalColumns);
                    bool applyWeightDecay =
                        parameter.WeightDecay == WeightDecayPolicy.Apply
                        || (options.Decay1D && parameter.T.Rank == 1);
                    batchItems[parameterIndex] = new CudaOptimizerKernels
                        .NekoMuonBatchItem(
                            parameter.T,
                            GetCudaState(parameterIndex),
                            originalRows,
                            originalColumns,
                            parameterState.Confidence,
                            applyWeightDecay);
                }
                float[] deviceConfidences = CudaOptimizerKernels
                    .NekoMuonFinishStepGrouped(
                        deviceIndex,
                        batchItems,
                        scratch,
                        fastCorrection,
                        slowCorrection,
                        options.Epsilon,
                        options.Rho,
                        options.MaxNewtonSchulzSteps,
                        options.NewtonSchulzDepthMode,
                        options.NewtonSchulzDepth,
                        runNewtonSchulz,
                        NewtonSchulzA,
                        NewtonSchulzB,
                        NewtonSchulzC,
                        options.LearningRate,
                        options.WeightDecay,
                        ForceFullNewtonSchulz,
                        mix8 ? deviceFiniteStatus![deviceSlot] : null);
                for (int parameterIndex = 0;
                    parameterIndex < deviceConfidences.Length;
                    parameterIndex++)
                {
                    confidences![deviceSlot, parameterIndex] =
                        deviceConfidences[parameterIndex];
                }
                if (mix8)
                {
                    foreach (Parameter parameter in _parameters)
                    {
                        CudaOptimizerKernels.PublishMix8Master(
                            parameter.T,
                            deviceIndex,
                            deviceFiniteStatus![deviceSlot]);
                    }
                }
            });
        }
        if (CudaOperationProfiler.IsEnabled)
        {
            CudaOperationProfiler.MeasureDevices(
                "optimizer.nekomuon.initialize_ns_apply",
                devices,
                FinishUpdates);
        }
        else
        {
            FinishUpdates();
        }
        int primarySlot = Array.IndexOf(devices, Tensor.CudaDeviceIndex);
        if (primarySlot < 0)
            primarySlot = 0;
        int finalPrimarySlot = primarySlot;
        CudaOptimizerFiniteStatusReadback[]? mix8Readbacks = mix8
            ? devices.Select(GetOrCreateBfp8FiniteReadback).ToArray()
            : null;
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            "NekoMuon update",
            queueReadback: mix8
                ? () =>
                {
                    for (int deviceSlot = 0;
                        deviceSlot < devices.Length;
                        deviceSlot++)
                    {
                            mix8Readbacks![deviceSlot].Begin(
                            deviceFiniteStatus![deviceSlot]);
                    }
                }
                : null,
            finalize: () =>
            {
                if (mix8)
                {
                    ThrowIfMix8PublicationNonFinite(
                        mix8Readbacks!,
                        devices,
                        _state.Step);
                }

                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[parameterIndex];
                    if (!deviceOnlyFixedFive)
                    {
                        _state.ParameterStates[parameterIndex] =
                            parameterState with
                            {
                                Confidence = confidences![
                                    finalPrimarySlot,
                                    parameterIndex],
                            };
                    }
                    _parameters[parameterIndex].T
                        .MarkCudaDataReplicasSynchronized(devices);
                }
            });
    }

    private void StepCudaBfp8(
        NekoMuonOptions options,
        float fastCorrection,
        float slowCorrection,
        int[] devices,
        bool mixedBlockState = false)
    {
        if (devices.Length == 0)
        {
            throw new InvalidOperationException(
                "Pure BFP8 NekoMuon requires at least one CUDA device.");
        }

        int primaryDevice = devices[0];
        bool runNewtonSchulz =
            _state.Step % options.NewtonSchulzInterval == 0;
        bool deviceOnlyFixedFive = runNewtonSchulz
            && options.MaxNewtonSchulzSteps == 5
            && (ForceFullNewtonSchulz
                || options.NewtonSchulzDepthMode
                    == NekoMuonNewtonSchulzDepthMode.Fixed
                && options.NewtonSchulzDepth == 5f);
        if (_cudaStates.Any(state => state is not null))
        {
            for (int index = 0; index < _cudaStates.Length; index++)
            {
                CudaOptimizerKernels.NekoMuonResidentState? state =
                    _cudaStates[index];
                if (state is null)
                    continue;
                state.SynchronizeHost(primaryDevice);
                NekoMuonParameterState parameterState =
                    _state.ParameterStates[index];
                if (state.IsDeviceConfidenceAuthoritative)
                {
                    _state.ParameterStates[index] = parameterState with
                    {
                        Confidence = state.SynchronizeConfidence(primaryDevice),
                    };
                }
                state.Dispose();
                _cudaStates[index] = null;
            }
            foreach (CudaOptimizerKernels.NekoMuonStatsBatch batch
                in _cudaStatsBatches.Values)
            {
                batch.Dispose();
            }
            _cudaStatsBatches.Clear();
        }

        int maximumLength = _parameters.Max(parameter => parameter.T.Numel);
        int maximumGramLength = 0;
        foreach (Parameter parameter in _parameters)
        {
            GetMatrixShape(parameter, out int shapeRows, out int shapeColumns);
            int rows = Math.Min(shapeRows, shapeColumns);
            maximumGramLength = Math.Max(
                maximumGramLength,
                checked(rows * rows));
        }

        var scratch = new CudaOptimizerKernels
            .NekoMuonDeviceScratch[devices.Length];
        var statuses = new NativeCudaBuffer<int>[devices.Length];
        for (int deviceSlot = 0; deviceSlot < devices.Length; deviceSlot++)
        {
            int deviceIndex = devices[deviceSlot];
            lock (_cudaScratch)
            {
                if (!_cudaScratch.TryGetValue(
                        deviceIndex,
                        out CudaOptimizerKernels.NekoMuonDeviceScratch?
                            deviceScratch))
                {
                    deviceScratch = new CudaOptimizerKernels
                        .NekoMuonDeviceScratch(
                            deviceIndex,
                            maximumLength,
                            maximumGramLength,
                            _cudaBatchCapacity,
                            !_cudaDispatchPolicy.DisableTensorCoreNekoMuon);
                    _cudaScratch.Add(deviceIndex, deviceScratch);
                }
                scratch[deviceSlot] = deviceScratch;
            }
            statuses[deviceSlot] = GetOrCreateBfp8FiniteStatus(deviceIndex);
            statuses[deviceSlot].MemSetToZero();
        }

        var states = new CudaOptimizerKernels
            .NekoMuonBfp8ResidentState[_parameters.Count];
        for (int parameterIndex = 0;
            parameterIndex < _parameters.Count;
            parameterIndex++)
        {
            NekoMuonParameterState parameterState =
                _state.ParameterStates[parameterIndex];
            CudaOptimizerKernels.NekoMuonBfp8ResidentState state =
                _cudaBfp8States[parameterIndex] ??=
                    new CudaOptimizerKernels.NekoMuonBfp8ResidentState(
                        parameterState.FastMoment,
                        parameterState.SlowMoment,
                        parameterState.Confidence,
                        mixedBlockState
                            ? Bfp8QuantizationDescriptor.Mix8_32
                            : Bfp8QuantizationDescriptor.TensorWide);
            states[parameterIndex] = state;
            foreach (int deviceIndex in devices)
                state.GetOrCreate(deviceIndex);
        }

        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            for (int parameterIndex = 0;
                parameterIndex < _parameters.Count;
                parameterIndex++)
            {
                CudaOptimizerKernels.NekoMuonPrepareBfp8StatsResident(
                    _parameters[parameterIndex].T,
                    deviceIndex,
                    states[parameterIndex],
                    scratch[deviceSlot],
                    statuses[deviceSlot],
                    options.BetaFast,
                    options.BetaSlow,
                    fastCorrection,
                    slowCorrection,
                    options.Epsilon,
                    options.Rho,
                    mixedBlockState,
                    options.Nesterov);
            }
        });

        if (!deviceOnlyFixedFive)
        {
            Parallel.For(0, devices.Length, deviceSlot =>
            {
                int deviceIndex = devices[deviceSlot];
                CudaOptimizerKernels.NekoMuonBfp8StatsBatch batch;
                lock (_cudaBfp8StatsBatches)
                {
                    if (!_cudaBfp8StatsBatches.TryGetValue(
                            deviceIndex,
                            out batch!))
                    {
                        batch = new CudaOptimizerKernels
                            .NekoMuonBfp8StatsBatch(
                                deviceIndex,
                                states);
                        _cudaBfp8StatsBatches.Add(deviceIndex, batch);
                    }
                }
                batch.GatherAndRead();
            });
        }

        bool useGroupedPureBfp8 = deviceOnlyFixedFive
            && !mixedBlockState;
        float[,]? confidences = useGroupedPureBfp8
            ? null
            : new float[devices.Length, _parameters.Count];
        Parallel.For(0, devices.Length, deviceSlot =>
        {
            int deviceIndex = devices[deviceSlot];
            if (useGroupedPureBfp8)
            {
                var batchItems = new CudaOptimizerKernels
                    .NekoMuonBfp8BatchItem[_parameters.Count];
                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    Parameter parameter = _parameters[parameterIndex];
                    GetMatrixShape(
                        parameter,
                        out int originalRows,
                        out int originalColumns);
                    bool applyWeightDecay =
                        parameter.WeightDecay == WeightDecayPolicy.Apply
                        || (options.Decay1D && parameter.T.Rank == 1);
                    batchItems[parameterIndex] = new CudaOptimizerKernels
                        .NekoMuonBfp8BatchItem(
                            parameter.T,
                            states[parameterIndex],
                            originalRows,
                            originalColumns,
                            applyWeightDecay);
                }
                CudaOptimizerKernels
                    .NekoMuonFinishFixedFiveBfp8GroupedDeviceResident(
                        deviceIndex,
                        batchItems,
                        scratch[deviceSlot],
                        statuses[deviceSlot],
                        fastCorrection,
                        options.Epsilon,
                        NewtonSchulzA,
                        NewtonSchulzB,
                        NewtonSchulzC,
                        options.LearningRate,
                        options.WeightDecay,
                        options.Nesterov);
                return;
            }
            for (int parameterIndex = 0;
                parameterIndex < _parameters.Count;
                parameterIndex++)
            {
                Parameter parameter = _parameters[parameterIndex];
                NekoMuonParameterState parameterState =
                    _state.ParameterStates[parameterIndex];
                GetMatrixShape(
                    parameter,
                    out int originalRows,
                    out int originalColumns);
                bool applyWeightDecay =
                    parameter.WeightDecay == WeightDecayPolicy.Apply
                    || (options.Decay1D && parameter.T.Rank == 1);
                confidences![deviceSlot, parameterIndex] =
                    CudaOptimizerKernels.NekoMuonFinishBfp8StepResident(
                        parameter.T,
                        deviceIndex,
                        states[parameterIndex],
                        scratch[deviceSlot],
                        statuses[deviceSlot],
                        originalRows,
                        originalColumns,
                        fastCorrection,
                        options.Epsilon,
                        parameterState.Confidence,
                        options.Rho,
                        options.MaxNewtonSchulzSteps,
                        options.NewtonSchulzDepthMode,
                        options.NewtonSchulzDepth,
                        runNewtonSchulz,
                        NewtonSchulzA,
                        NewtonSchulzB,
                        NewtonSchulzC,
                        options.LearningRate,
                        options.WeightDecay,
                        applyWeightDecay,
                        deviceOnlyFixedFive,
                        mixedBlockState,
                        ForceFullNewtonSchulz,
                        options.Nesterov);
            }
        });

        int primarySlot = Array.IndexOf(devices, Tensor.CudaDeviceIndex);
        if (primarySlot < 0)
            primarySlot = 0;
        int finalPrimarySlot = primarySlot;
        CudaOptimizerFiniteStatusReadback[] readbacks = devices
            .Select(GetOrCreateBfp8FiniteReadback)
            .ToArray();
        CudaOptimizerStepBatch.CompleteAfterSynchronization(
            devices,
            mixedBlockState
                ? "block-BFP8-state mix8_32 NekoMuon update"
                : "pure BFP8 NekoMuon update",
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
                        $"pure BFP8 NekoMuon state on device " +
                        $"{nonFiniteDevice} at optimizer step " +
                        $"{_state.Step}.");
                }

                for (int parameterIndex = 0;
                    parameterIndex < _parameters.Count;
                    parameterIndex++)
                {
                    NekoMuonParameterState parameterState =
                        _state.ParameterStates[parameterIndex];
                    if (!deviceOnlyFixedFive)
                    {
                        _state.ParameterStates[parameterIndex] =
                            parameterState with
                            {
                                Confidence = confidences![
                                    finalPrimarySlot,
                                    parameterIndex],
                            };
                    }
                    _parameters[parameterIndex].T
                        .MarkCudaBfp8DataReplicasSynchronized(devices);
                }
            });
    }

    private static bool UsesPureBfp8OptimizerState(Tensor tensor)
    {
        NNtrain.Runtime.Execution.PrecisionPolicy? policy =
            TensorExecutionContext.ActivePrecisionPolicy;
        if (policy is not null
            && policy.OptimizerState
                != NNtrain.Runtime.Execution.NumericFormat.Bfp8)
        {
            return false;
        }
        return tensor.DType == TensorDType.Bfp8
            && tensor.Bfp8Quantization
                == Bfp8QuantizationDescriptor.TensorWide;
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

    internal (CudaBfp8BufferView Fast, CudaBfp8BufferView Slow)
        GetCudaBfp8Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.NekoMuonBfp8ResidentState state =
            _cudaBfp8States[parameterIndex]
            ?? throw new InvalidOperationException(
                "The NekoMuon parameter has no resident BFP8 state.");
        return (state.GetFast(deviceIndex), state.GetSlow(deviceIndex));
    }

    internal (NativeCudaBuffer<ushort> Fast, NativeCudaBuffer<ushort> Slow)
        GetCudaBFloat16Moments(int parameterIndex, int deviceIndex)
    {
        CudaOptimizerKernels.NekoMuonBFloat16ResidentState state =
            _cudaBFloat16States[parameterIndex]
            ?? throw new InvalidOperationException(
                "The NekoMuon parameter has no resident BF16 state.");
        return (state.GetFast(deviceIndex), state.GetSlow(deviceIndex));
    }

    internal void DisposeCudaResources()
    {
        List<Exception>? failures = null;
        for (int index = 0; index < _cudaStates.Length; index++)
        {
            if (_cudaStates[index] is IDisposable floatState)
                TryDisposeCudaResource(floatState, ref failures);
            if (_cudaBFloat16States[index] is IDisposable bfloat16State)
                TryDisposeCudaResource(bfloat16State, ref failures);
            if (_cudaBfp8States[index] is IDisposable bfp8State)
                TryDisposeCudaResource(bfp8State, ref failures);
            _cudaStates[index] = null;
            _cudaBFloat16States[index] = null;
            _cudaBfp8States[index] = null;
        }
        foreach (CudaOptimizerKernels.NekoMuonStatsBatch batch
            in _cudaStatsBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaStatsBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonBFloat16StatsBatch batch
            in _cudaBFloat16StatsBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaBFloat16StatsBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonBfp8StatsBatch batch
            in _cudaBfp8StatsBatches.Values)
        {
            TryDisposeCudaResource(batch, ref failures);
        }
        _cudaBfp8StatsBatches.Clear();
        foreach (CudaOptimizerKernels.NekoMuonDeviceScratch scratch
            in _cudaScratch.Values)
        {
            TryDisposeCudaResource(scratch, ref failures);
        }
        _cudaScratch.Clear();
        foreach (CudaOptimizerFiniteStatusReadback readback
            in _cudaBfp8FiniteReadbacks.Values)
        {
            TryDisposeCudaResource(readback, ref failures);
        }
        _cudaBfp8FiniteReadbacks.Clear();
        foreach (NativeCudaBuffer<int> status
            in _cudaBfp8FiniteStatus.Values)
        {
            TryDisposeCudaResource(status, ref failures);
        }
        _cudaBfp8FiniteStatus.Clear();
        _cudaStateAuthorityDevice = null;
        if (failures is not null)
        {
            throw new AggregateException(
                "NekoMuon CUDA resource cleanup failed.", failures);
        }
    }

    private void MakeCpuStateAuthoritative()
    {
        if (_cudaStateAuthorityDevice is null)
            return;

        _ = CaptureStateForStreaming();
        DisposeCudaResources();
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

    private static void AddProfileTicks(
        long[]? profileTicks,
        int index,
        long start)
    {
        if (profileTicks is null)
            return;
        Interlocked.Add(
            ref profileTicks[index],
            Stopwatch.GetTimestamp() - start);
    }

    private static double TicksToMilliseconds(long ticks)
        => ticks * 1000d / Stopwatch.Frequency;

    internal static float ResolveNewtonSchulzDepth(
        NekoMuonOptions options,
        float confidence,
        bool runNewtonSchulz)
    {
        ArgumentNullException.ThrowIfNull(options);
        return ResolveNewtonSchulzDepth(
            options.MaxNewtonSchulzSteps,
            options.NewtonSchulzDepthMode,
            options.NewtonSchulzDepth,
            confidence,
            runNewtonSchulz);
    }

    internal static float ResolveNewtonSchulzDepth(
        int maxNewtonSchulzSteps,
        NekoMuonNewtonSchulzDepthMode mode,
        float configuredDepth,
        float confidence,
        bool runNewtonSchulz)
    {
        if (!runNewtonSchulz)
            return 0f;

        float adaptiveDepth = maxNewtonSchulzSteps * confidence;
        float depth = mode switch
        {
            NekoMuonNewtonSchulzDepthMode.Adaptive => adaptiveDepth,
            NekoMuonNewtonSchulzDepthMode.Minimum =>
                MathF.Max(adaptiveDepth, configuredDepth),
            NekoMuonNewtonSchulzDepthMode.Fixed => configuredDepth,
            _ => throw new ArgumentOutOfRangeException(
                nameof(mode),
                mode,
                "NekoMuon Newton-Schulz depth mode is invalid."),
        };
        return Math.Clamp(depth, 0f, maxNewtonSchulzSteps);
    }

    private static void InitializeMuonMatrix(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        bool transpose,
        float epsilon)
    {
        double normSquared = 0d;
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && source.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = source.Length - source.Length % width;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> values =
                    Vector256.LoadUnsafe(ref source[index]);
                normSquared += Vector256.Sum(values * values);
            }
        }

        for (; index < source.Length; index++)
            normSquared += (double)source[index] * source[index];
        float inverseNorm = 1f / ((float)Math.Sqrt(normSquared) + epsilon);

        if (!transpose)
        {
            Scale(source, destination, inverseNorm);
            return;
        }

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                destination[column * rows + row] =
                    source[row * columns + column] * inverseNorm;
            }
        }
    }

    private static void NewtonSchulz(
        float[] source,
        float[] destination,
        float[] gram,
        float[] gramSquared,
        int rows,
        int columns,
        bool bfloat16MatrixOperands,
        long[]? profileTicks)
    {
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            long cudaStart = profileTicks is null
                ? 0L
                : Stopwatch.GetTimestamp();
            CudaOptimizerKernels.NekoMuonNewtonSchulz(
                source,
                destination,
                gram,
                gramSquared,
                rows,
                columns,
                NewtonSchulzA,
                NewtonSchulzB,
                NewtonSchulzC);
            AddProfileTicks(profileTicks, 8, cudaStart);
            return;
        }

        long phaseStart = profileTicks is null
            ? 0L
            : Stopwatch.GetTimestamp();
        ComputeSymmetricGram(
            source, gram, rows, columns, bfloat16MatrixOperands);
        AddProfileTicks(profileTicks, 6, phaseStart);

        phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
        ComputeSymmetricGram(
            gram, gramSquared, rows, rows, bfloat16MatrixOperands);
        AddProfileTicks(profileTicks, 7, phaseStart);

        phaseStart = profileTicks is null ? 0L : Stopwatch.GetTimestamp();
        if (bfloat16MatrixOperands)
        {
            // The CUDA Tensor Core path performs three BF16-operand/FP32-
            // accumulate GEMMs: X*X^T, G*G, and P*X.  Mirror those operand
            // boundaries on CPU while retaining FP32 reductions and output
            // storage, as required by PrecisionPolicy.
            for (int bfOutputRow = 0; bfOutputRow < rows; bfOutputRow++)
            {
                int coefficientOffset = bfOutputRow * rows;
                int destinationOffset = bfOutputRow * columns;
                for (int column = 0; column < columns; column++)
                {
                    float sum = 0f;
                    for (int inner = 0; inner < rows; inner++)
                    {
                        float coefficient = MathF.FusedMultiplyAdd(
                            NewtonSchulzC,
                            gramSquared[coefficientOffset + inner],
                            NewtonSchulzB
                                * gram[coefficientOffset + inner]);
                        if (bfOutputRow == inner)
                            coefficient += NewtonSchulzA;
                        coefficient = TensorStorageCodec.RoundToBFloat16(
                            coefficient);
                        float sourceValue = TensorStorageCodec.RoundToBFloat16(
                            source[inner * columns + column]);
                        sum = MathF.FusedMultiplyAdd(
                            coefficient, sourceValue, sum);
                    }
                    destination[destinationOffset + column] = sum;
                }
            }
            AddProfileTicks(profileTicks, 8, phaseStart);
            return;
        }

        Scale(source, destination, NewtonSchulzA);
        int outputRow = 0;
        for (; outputRow + 7 < rows; outputRow += 8)
        {
            int destination0 = outputRow * columns;
            int destination1 = destination0 + columns;
            int destination2 = destination1 + columns;
            int destination3 = destination2 + columns;
            int destination4 = destination3 + columns;
            int destination5 = destination4 + columns;
            int destination6 = destination5 + columns;
            int destination7 = destination6 + columns;
            int coefficient0 = outputRow * rows;
            int coefficient1 = coefficient0 + rows;
            int coefficient2 = coefficient1 + rows;
            int coefficient3 = coefficient2 + rows;
            int coefficient4 = coefficient3 + rows;
            int coefficient5 = coefficient4 + rows;
            int coefficient6 = coefficient5 + rows;
            int coefficient7 = coefficient6 + rows;

            for (int inner = 0; inner < rows; inner++)
            {
                AddScaledEightRows(
                    source,
                    inner * columns,
                    destination,
                    destination0,
                    destination1,
                    destination2,
                    destination3,
                    destination4,
                    destination5,
                    destination6,
                    destination7,
                    columns,
                    NewtonSchulzB * gram[coefficient0 + inner]
                        + NewtonSchulzC * gramSquared[coefficient0 + inner],
                    NewtonSchulzB * gram[coefficient1 + inner]
                        + NewtonSchulzC * gramSquared[coefficient1 + inner],
                    NewtonSchulzB * gram[coefficient2 + inner]
                        + NewtonSchulzC * gramSquared[coefficient2 + inner],
                    NewtonSchulzB * gram[coefficient3 + inner]
                        + NewtonSchulzC * gramSquared[coefficient3 + inner],
                    NewtonSchulzB * gram[coefficient4 + inner]
                        + NewtonSchulzC * gramSquared[coefficient4 + inner],
                    NewtonSchulzB * gram[coefficient5 + inner]
                        + NewtonSchulzC * gramSquared[coefficient5 + inner],
                    NewtonSchulzB * gram[coefficient6 + inner]
                        + NewtonSchulzC * gramSquared[coefficient6 + inner],
                    NewtonSchulzB * gram[coefficient7 + inner]
                        + NewtonSchulzC * gramSquared[coefficient7 + inner]);
            }
        }

        for (; outputRow + 3 < rows; outputRow += 4)
        {
            int destination0 = outputRow * columns;
            int destination1 = destination0 + columns;
            int destination2 = destination1 + columns;
            int destination3 = destination2 + columns;
            int coefficient0 = outputRow * rows;
            int coefficient1 = coefficient0 + rows;
            int coefficient2 = coefficient1 + rows;
            int coefficient3 = coefficient2 + rows;

            for (int inner = 0; inner < rows; inner++)
            {
                AddScaledFourRows(
                    source,
                    inner * columns,
                    destination,
                    destination0,
                    destination1,
                    destination2,
                    destination3,
                    columns,
                    NewtonSchulzB * gram[coefficient0 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient0 + inner],
                    NewtonSchulzB * gram[coefficient1 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient1 + inner],
                    NewtonSchulzB * gram[coefficient2 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient2 + inner],
                    NewtonSchulzB * gram[coefficient3 + inner]
                        + NewtonSchulzC
                            * gramSquared[coefficient3 + inner]);
            }
        }

        for (; outputRow < rows; outputRow++)
        {
            int destinationOffset = outputRow * columns;
            for (int inner = 0; inner < rows; inner++)
            {
                float coefficient =
                    NewtonSchulzB * gram[outputRow * rows + inner]
                    + NewtonSchulzC
                        * gramSquared[outputRow * rows + inner];
                AddScaled(
                    source,
                    inner * columns,
                    destination,
                    destinationOffset,
                    columns,
                    coefficient);
            }
        }
        AddProfileTicks(profileTicks, 8, phaseStart);
    }

    private static void ComputeSymmetricGram(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        bool bfloat16MatrixOperands = false)
    {
        if (bfloat16MatrixOperands)
        {
            for (int row = 0; row < rows; row++)
            {
                int rowOffset = row * columns;
                for (int other = 0; other <= row; other++)
                {
                    int otherOffset = other * columns;
                    float dot = 0f;
                    for (int column = 0; column < columns; column++)
                    {
                        float left = TensorStorageCodec.RoundToBFloat16(
                            source[rowOffset + column]);
                        float right = TensorStorageCodec.RoundToBFloat16(
                            source[otherOffset + column]);
                        dot = MathF.FusedMultiplyAdd(left, right, dot);
                    }
                    destination[row * rows + other] = dot;
                    destination[other * rows + row] = dot;
                }
            }
            return;
        }

        if (!Tensor.SimdEnabled
            || !Vector256.IsHardwareAccelerated
            || rows % 4 != 0
            || columns < Vector256<float>.Count)
        {
            for (int row = 0; row < rows; row++)
            {
                int rowOffset = row * columns;
                for (int other = 0; other <= row; other++)
                {
                    float dot = Dot(
                        source,
                        rowOffset,
                        other * columns,
                        columns);
                    destination[row * rows + other] = dot;
                    destination[other * rows + row] = dot;
                }
            }
            return;
        }

        // Four rows and two comparison rows share six vector loads across
        // eight dot products. This removes most repeated reads in the two
        // Gram products that dominate Newton-Schulz on wide FFN matrices.
        for (int rowBase = 0; rowBase < rows; rowBase += 4)
        {
            for (int otherBase = 0;
                otherBase <= rowBase + 2;
                otherBase += 2)
            {
                ComputeGramFourByTwo(
                    source,
                    destination,
                    rows,
                    columns,
                    rowBase,
                    otherBase);
            }
        }
    }

    private static void ComputeGramFourByTwo(
        float[] source,
        float[] destination,
        int rows,
        int columns,
        int rowBase,
        int otherBase)
    {
        int row0 = rowBase * columns;
        int row1 = row0 + columns;
        int row2 = row1 + columns;
        int row3 = row2 + columns;
        int other0 = otherBase * columns;
        int other1 = other0 + columns;
        int index = 0;
        int width = Vector256<float>.Count;
        int vectorizedLength = columns - columns % width;
        Vector256<float> sum00 = Vector256<float>.Zero;
        Vector256<float> sum01 = Vector256<float>.Zero;
        Vector256<float> sum10 = Vector256<float>.Zero;
        Vector256<float> sum11 = Vector256<float>.Zero;
        Vector256<float> sum20 = Vector256<float>.Zero;
        Vector256<float> sum21 = Vector256<float>.Zero;
        Vector256<float> sum30 = Vector256<float>.Zero;
        Vector256<float> sum31 = Vector256<float>.Zero;

        for (; index < vectorizedLength; index += width)
        {
            Vector256<float> a0 = Vector256.LoadUnsafe(ref source[row0 + index]);
            Vector256<float> a1 = Vector256.LoadUnsafe(ref source[row1 + index]);
            Vector256<float> a2 = Vector256.LoadUnsafe(ref source[row2 + index]);
            Vector256<float> a3 = Vector256.LoadUnsafe(ref source[row3 + index]);
            Vector256<float> b0 = Vector256.LoadUnsafe(
                ref source[other0 + index]);
            Vector256<float> b1 = Vector256.LoadUnsafe(
                ref source[other1 + index]);
            sum00 = Vector256.FusedMultiplyAdd(a0, b0, sum00);
            sum01 = Vector256.FusedMultiplyAdd(a0, b1, sum01);
            sum10 = Vector256.FusedMultiplyAdd(a1, b0, sum10);
            sum11 = Vector256.FusedMultiplyAdd(a1, b1, sum11);
            sum20 = Vector256.FusedMultiplyAdd(a2, b0, sum20);
            sum21 = Vector256.FusedMultiplyAdd(a2, b1, sum21);
            sum30 = Vector256.FusedMultiplyAdd(a3, b0, sum30);
            sum31 = Vector256.FusedMultiplyAdd(a3, b1, sum31);
        }

        float scalar00 = Vector256.Sum(sum00);
        float scalar01 = Vector256.Sum(sum01);
        float scalar10 = Vector256.Sum(sum10);
        float scalar11 = Vector256.Sum(sum11);
        float scalar20 = Vector256.Sum(sum20);
        float scalar21 = Vector256.Sum(sum21);
        float scalar30 = Vector256.Sum(sum30);
        float scalar31 = Vector256.Sum(sum31);
        for (; index < columns; index++)
        {
            float a0 = source[row0 + index];
            float a1 = source[row1 + index];
            float a2 = source[row2 + index];
            float a3 = source[row3 + index];
            float b0 = source[other0 + index];
            float b1 = source[other1 + index];
            scalar00 += a0 * b0;
            scalar01 += a0 * b1;
            scalar10 += a1 * b0;
            scalar11 += a1 * b1;
            scalar20 += a2 * b0;
            scalar21 += a2 * b1;
            scalar30 += a3 * b0;
            scalar31 += a3 * b1;
        }

        StoreSymmetricGram(destination, rows, rowBase, otherBase, scalar00);
        StoreSymmetricGram(destination, rows, rowBase, otherBase + 1, scalar01);
        StoreSymmetricGram(destination, rows, rowBase + 1, otherBase, scalar10);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 1,
            otherBase + 1,
            scalar11);
        StoreSymmetricGram(destination, rows, rowBase + 2, otherBase, scalar20);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 2,
            otherBase + 1,
            scalar21);
        StoreSymmetricGram(destination, rows, rowBase + 3, otherBase, scalar30);
        StoreSymmetricGram(
            destination,
            rows,
            rowBase + 3,
            otherBase + 1,
            scalar31);
    }

    private static void StoreSymmetricGram(
        float[] destination,
        int rows,
        int row,
        int column,
        float value)
    {
        destination[row * rows + column] = value;
        destination[column * rows + row] = value;
    }

    private static float Dot(
        float[] values,
        int firstOffset,
        int secondOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            int unrolledLength = vectorizedLength - 3 * width;
            Vector256<float> sum0 = Vector256<float>.Zero;
            Vector256<float> sum1 = Vector256<float>.Zero;
            Vector256<float> sum2 = Vector256<float>.Zero;
            Vector256<float> sum3 = Vector256<float>.Zero;
            for (; index < unrolledLength; index += 4 * width)
            {
                sum0 +=
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(ref values[secondOffset + index]);
                sum1 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + width]);
                sum2 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + 2 * width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + 2 * width]);
                sum3 +=
                    Vector256.LoadUnsafe(
                        ref values[firstOffset + index + 3 * width])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index + 3 * width]);
            }

            sum = Vector256.Sum((sum0 + sum1) + (sum2 + sum3));
            for (; index < vectorizedLength; index += width)
            {
                sum += Vector256.Sum(
                    Vector256.LoadUnsafe(ref values[firstOffset + index])
                    * Vector256.LoadUnsafe(
                        ref values[secondOffset + index]));
            }
        }

        for (; index < length; index++)
        {
            sum += values[firstOffset + index]
                * values[secondOffset + index];
        }

        return sum;
    }

    private static void Scale(
        float[] source,
        float[] destination,
        float scale)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && source.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = source.Length - source.Length % width;
            Vector256<float> vectorScale = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                (Vector256.LoadUnsafe(ref source[index]) * vectorScale)
                    .StoreUnsafe(ref destination[index]);
            }
        }

        for (; index < source.Length; index++)
            destination[index] = source[index] * scale;
    }

    private static void AddScaled(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destinationOffset,
        int length,
        float scale)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale = Vector256.Create(scale);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> result =
                    Vector256.LoadUnsafe(ref destination[
                        destinationOffset + index])
                    + vectorScale
                        * Vector256.LoadUnsafe(ref source[
                            sourceOffset + index]);
                result.StoreUnsafe(ref destination[
                    destinationOffset + index]);
            }
        }

        for (; index < length; index++)
        {
            destination[destinationOffset + index] +=
                scale * source[sourceOffset + index];
        }
    }

    private static void AddScaledFourRows(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destination0,
        int destination1,
        int destination2,
        int destination3,
        int length,
        float scale0,
        float scale1,
        float scale2,
        float scale3)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale0 = Vector256.Create(scale0);
            Vector256<float> vectorScale1 = Vector256.Create(scale1);
            Vector256<float> vectorScale2 = Vector256.Create(scale2);
            Vector256<float> vectorScale3 = Vector256.Create(scale3);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> sourceVector = Vector256.LoadUnsafe(
                    ref source[sourceOffset + index]);
                (Vector256.LoadUnsafe(ref destination[destination0 + index])
                    + vectorScale0 * sourceVector)
                    .StoreUnsafe(ref destination[destination0 + index]);
                (Vector256.LoadUnsafe(ref destination[destination1 + index])
                    + vectorScale1 * sourceVector)
                    .StoreUnsafe(ref destination[destination1 + index]);
                (Vector256.LoadUnsafe(ref destination[destination2 + index])
                    + vectorScale2 * sourceVector)
                    .StoreUnsafe(ref destination[destination2 + index]);
                (Vector256.LoadUnsafe(ref destination[destination3 + index])
                    + vectorScale3 * sourceVector)
                    .StoreUnsafe(ref destination[destination3 + index]);
            }
        }

        for (; index < length; index++)
        {
            float sourceValue = source[sourceOffset + index];
            destination[destination0 + index] += scale0 * sourceValue;
            destination[destination1 + index] += scale1 * sourceValue;
            destination[destination2 + index] += scale2 * sourceValue;
            destination[destination3 + index] += scale3 * sourceValue;
        }
    }

    private static void AddScaledEightRows(
        float[] source,
        int sourceOffset,
        float[] destination,
        int destination0,
        int destination1,
        int destination2,
        int destination3,
        int destination4,
        int destination5,
        int destination6,
        int destination7,
        int length,
        float scale0,
        float scale1,
        float scale2,
        float scale3,
        float scale4,
        float scale5,
        float scale6,
        float scale7)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> vectorScale0 = Vector256.Create(scale0);
            Vector256<float> vectorScale1 = Vector256.Create(scale1);
            Vector256<float> vectorScale2 = Vector256.Create(scale2);
            Vector256<float> vectorScale3 = Vector256.Create(scale3);
            Vector256<float> vectorScale4 = Vector256.Create(scale4);
            Vector256<float> vectorScale5 = Vector256.Create(scale5);
            Vector256<float> vectorScale6 = Vector256.Create(scale6);
            Vector256<float> vectorScale7 = Vector256.Create(scale7);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> sourceVector = Vector256.LoadUnsafe(
                    ref source[sourceOffset + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale0,
                    Vector256.LoadUnsafe(ref destination[destination0 + index]))
                    .StoreUnsafe(ref destination[destination0 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale1,
                    Vector256.LoadUnsafe(ref destination[destination1 + index]))
                    .StoreUnsafe(ref destination[destination1 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale2,
                    Vector256.LoadUnsafe(ref destination[destination2 + index]))
                    .StoreUnsafe(ref destination[destination2 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale3,
                    Vector256.LoadUnsafe(ref destination[destination3 + index]))
                    .StoreUnsafe(ref destination[destination3 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale4,
                    Vector256.LoadUnsafe(ref destination[destination4 + index]))
                    .StoreUnsafe(ref destination[destination4 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale5,
                    Vector256.LoadUnsafe(ref destination[destination5 + index]))
                    .StoreUnsafe(ref destination[destination5 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale6,
                    Vector256.LoadUnsafe(ref destination[destination6 + index]))
                    .StoreUnsafe(ref destination[destination6 + index]);
                Vector256.FusedMultiplyAdd(
                    sourceVector,
                    vectorScale7,
                    Vector256.LoadUnsafe(ref destination[destination7 + index]))
                    .StoreUnsafe(ref destination[destination7 + index]);
            }
        }

        for (; index < length; index++)
        {
            float sourceValue = source[sourceOffset + index];
            destination[destination0 + index] += scale0 * sourceValue;
            destination[destination1 + index] += scale1 * sourceValue;
            destination[destination2 + index] += scale2 * sourceValue;
            destination[destination3 + index] += scale3 * sourceValue;
            destination[destination4 + index] += scale4 * sourceValue;
            destination[destination5 + index] += scale5 * sourceValue;
            destination[destination6 + index] += scale6 * sourceValue;
            destination[destination7 + index] += scale7 * sourceValue;
        }
    }

    private static void InterpolateInPlace(
        float[] current,
        float[] next,
        float fraction)
    {
        int index = 0;
        if (Tensor.SimdEnabled
            && Vector256.IsHardwareAccelerated
            && current.Length >= Vector256<float>.Count)
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = current.Length - current.Length % width;
            Vector256<float> vectorFraction = Vector256.Create(fraction);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> currentValues =
                    Vector256.LoadUnsafe(ref current[index]);
                Vector256<float> nextValues =
                    Vector256.LoadUnsafe(ref next[index]);
                (currentValues
                    + vectorFraction * (nextValues - currentValues))
                    .StoreUnsafe(ref current[index]);
            }
        }

        for (; index < current.Length; index++)
        {
            current[index] += fraction * (next[index] - current[index]);
        }
    }

    private static void TransposeBack(
        float[] transposed,
        float[] destination,
        int rows,
        int columns)
    {
        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                destination[row * columns + column] =
                    transposed[column * rows + row];
            }
        }
    }

}

public readonly record struct NekoMuonDiagnostics(
    int Step,
    float MinimumConfidence,
    float MeanConfidence,
    float MaximumConfidence,
    float MeanNewtonSchulzDepth,
    int MaximumNewtonSchulzDepth);

public readonly record struct NekoMuonStepProfile(
    double UpdateMomentsMilliseconds,
    double ConfidenceMilliseconds,
    double InitializeMilliseconds,
    double NewtonSchulzMilliseconds,
    double TransposeMilliseconds,
    double ApplyUpdateMilliseconds,
    double FirstGramMilliseconds,
    double GramSquaredMilliseconds,
    double PolynomialMilliseconds)
{
    public double TotalCpuMilliseconds
        => UpdateMomentsMilliseconds
            + ConfidenceMilliseconds
            + InitializeMilliseconds
            + NewtonSchulzMilliseconds
            + TransposeMilliseconds
            + ApplyUpdateMilliseconds;
}
