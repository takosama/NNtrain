using System.Diagnostics;
using NNtrain.Cuda.Interop;

namespace NNtrain;

public sealed partial class NekoMuon : IOptimizer, ILearningRateAdjustable,
    IMix8QuantizationDiagnosticsProvider
{
    private const float NewtonSchulzA = 3.4445f;
    private const float NewtonSchulzB = -4.7750f;
    private const float NewtonSchulzC = 2.0315f;

    private readonly List<Parameter> _parameters;
    private readonly bool _usesPackedBFloat16Checkpoint;
    private readonly ushort[][] _arcPackedFastMoments;
    private readonly ushort[][] _arcPackedSlowMoments;
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
        CudaOptimizerKernels.NekoMuonConfidenceBatch>
        _cudaConfidenceBatches = [];
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
    private CudaMix8QuantizationDiagnostics? _cudaMix8Diagnostics;
    private int _cudaMix8DiagnosticsDevice = -1;
    private NekoMuonState _state;
    private int? _cudaStateAuthorityDevice;

    internal IReadOnlyList<Parameter> Parameters => _parameters;
    internal bool UsesPackedBFloat16Checkpoint => _usesPackedBFloat16Checkpoint;

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
        _usesPackedBFloat16Checkpoint = TensorExecutionContext.ActivePrecisionPolicy?.Mode
            == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16;
        _cudaDispatchPolicy = (cudaDispatchPolicy
            ?? throw new ArgumentNullException(nameof(cudaDispatchPolicy)))
            .Validate();

        _parameters = OptimizerParameters.Materialize(parameters, "NekoMuon");

        _totalElements = _parameters.Sum(
            parameter => (long)parameter.T.Numel);

        NekoMuonOptions effectiveOptions = options ?? new NekoMuonOptions();
        ValidateOptions(effectiveOptions, nameof(options));
        _state = CreateInitialState(_parameters, effectiveOptions,
            _usesPackedBFloat16Checkpoint);
        _arcPackedFastMoments = _usesPackedBFloat16Checkpoint
            ? _parameters.Select(p => new ushort[p.T.Numel]).ToArray()
            : [];
        _arcPackedSlowMoments = _usesPackedBFloat16Checkpoint
            ? _parameters.Select(p => new ushort[p.T.Numel]).ToArray()
            : [];
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
        if (!Tensor.ArcResident) _arcState.Dispose();
        if (_usesPackedBFloat16Checkpoint
            && (Tensor.ExecutionDevice != TensorDevice.Arc || !Tensor.ArcResident
                || !Tensor.ArcLane.Options.ResidentMuonIterations))
            throw new NotSupportedException("Arc mix8_16 NekoMuon requires resident Arc iteration kernels.");
        if (_state.Step == int.MaxValue)
        {
            throw new InvalidOperationException(
                "NekoMuon cannot advance beyond Int32.MaxValue steps.");
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

        NekoMuonOptions options = _state.Options;
        double fastDecayProduct =
            _state.FastDecayProduct!.Value * options.BetaFast;
        double slowDecayProduct =
            _state.SlowDecayProduct!.Value * options.BetaSlow;
        _state = _state with
        {
            Step = _state.Step + 1,
            FastDecayProduct = fastDecayProduct,
            SlowDecayProduct = slowDecayProduct,
        };
        float fastCorrection = (float)(1d - fastDecayProduct);
        float slowCorrection = (float)(1d - slowDecayProduct);
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

        if (Tensor.ExecutionDevice == TensorDevice.Arc && Tensor.ArcLane.Options.ResidentMuonIterations)
        {
            for (int index = 0; index < _parameters.Count; index++)
                UpdateArcParameter(index, options, fastCorrection, slowCorrection);
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
            bool residentArc = Tensor.ExecutionDevice == TensorDevice.Arc && Tensor.ArcLane.Options.ResidentMuonIterations;
            if (residentArc)
                ArcMuonMath.Orthogonalize(x, rows, columns, wholeSteps, fraction,
                    parameter.T.DType is TensorDType.BFloat16 or TensorDType.Bfp8,
                    NewtonSchulzA, NewtonSchulzB, NewtonSchulzC);
            for (int step = 0; !residentArc && step < wholeSteps; step++)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns,
                    parameter.T.DType == TensorDType.BFloat16
                        || (Tensor.ExecutionDevice == TensorDevice.Arc && parameter.T.DType == TensorDType.Bfp8),
                    profileTicks);
                (x, next) = (next, x);
            }

            if (!residentArc && fraction > 0f)
            {
                NewtonSchulz(
                    x,
                    next,
                    gram,
                    gramSquared,
                    rows,
                    columns,
                    parameter.T.DType == TensorDType.BFloat16
                        || (Tensor.ExecutionDevice == TensorDevice.Arc && parameter.T.DType == TensorDType.Bfp8),
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


}
