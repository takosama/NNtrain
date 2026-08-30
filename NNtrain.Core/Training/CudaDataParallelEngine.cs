using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

internal readonly record struct CudaTrainingGraphTelemetry(
    long CaptureCount,
    long ReplayCount,
    long FallbackCount,
    int CachedCompiledPlanCount,
    long GraphPinnedBytes,
    long CapturedReadyEventRecordCount,
    double CapturedReadyEventRecordMilliseconds);

/// <summary>
/// One language-model microbatch in a gradient-accumulation update.
/// Token and target arrays remain host-side batch inputs; model state,
/// activations, and accumulated gradients remain resident on their CUDA lane.
/// </summary>
public readonly record struct CudaLanguageModelMicroBatch(
    int[] Input,
    int[] Target,
    int BatchSize,
    int SequenceLength);

/// <summary>
/// Owns the reusable CUDA data-parallel resources for one language model.
/// Dispose the engine before ending a training session so gradient arenas,
/// reduction buffers, streams, and events are released deterministically.
/// </summary>
public sealed class CudaDataParallelEngine : IDisposable
{
    private readonly object _sync = new();
    private readonly LanguageModel _model;
    private readonly Parameter[] _parameters;
    private readonly int[] _cudaDeviceIndices;
    private readonly int[][] _devicePrefixes;
    private readonly CudaReplicaExecutor _replicaExecutor;
    private readonly CudaDispatchPolicy _dispatchPolicy;
    private readonly LinkedList<CudaTrainingShapePlan> _trainingShapePlans = [];
    private long _graphCacheBudgetBytes;
    private CudaAdaptiveShardScheduler _adaptiveShardScheduler;
    private TensorCudaKernels.FlatGradientPlan? _flatGradientPlan;
    private CudaBFloat16GradientAllReducePlan? _bfloat16GradientPlan;
    private CudaBfp8GradientAllReducePlan? _bfp8GradientPlan;
    private readonly Dictionary<int, NativeCudaBuffer<float>>
        _accumulatedLossBuffers = [];
    private int _flatGradientPlanBuildCount;
    private int _bfloat16GradientPlanBuildCount;
    private int _trainingShapePlanBuildCount;
    private long _trainingShapePlanEvictionCount;
    private long _implicitGlobalStep;
    private long _graphCaptureCount;
    private long _graphReplayCount;
    private long _graphFallbackCount;
    private Exception? _lastGraphFailure;
    private bool _parameterResidencyPrepared;
    private TensorPrecisionMode? _preflightPrecisionMode;
    private bool _preflightCompiledGraphRequested;
    private int _disposed;

    public CudaDataParallelEngine(
        LanguageModel model,
        CudaAdaptiveShardingOptions? adaptiveShardingOptions = null)
        : this(model, Tensor.CudaDeviceIndices, adaptiveShardingOptions)
    {
    }

    public CudaDataParallelEngine(
        LanguageModel model,
        IReadOnlyList<int> cudaDeviceIndices,
        CudaAdaptiveShardingOptions? adaptiveShardingOptions = null)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _dispatchPolicy = CudaDispatchPolicy.Current.Validate();
        _parameters = _model.Parameters().ToArray();
        ArgumentNullException.ThrowIfNull(cudaDeviceIndices);
        if (cudaDeviceIndices.Count == 0)
        {
            throw new ArgumentException(
                "At least one CUDA device must belong to the engine.",
                nameof(cudaDeviceIndices));
        }
        _cudaDeviceIndices = cudaDeviceIndices.ToArray();
        if (_cudaDeviceIndices.Any(index => index < 0)
            || _cudaDeviceIndices.Distinct().Count()
                != _cudaDeviceIndices.Length)
        {
            throw new ArgumentException(
                "CUDA device indices must be unique and non-negative.",
                nameof(cudaDeviceIndices));
        }
        _devicePrefixes = new int[_cudaDeviceIndices.Length][];
        for (int count = 1; count <= _cudaDeviceIndices.Length; count++)
            _devicePrefixes[count - 1] = _cudaDeviceIndices[..count];
        CudaAdaptiveShardingOptions options = adaptiveShardingOptions
            ?? new CudaAdaptiveShardingOptions();
        options.Validate();
        _graphCacheBudgetBytes = options.GraphCacheBudgetBytes;
        _adaptiveShardScheduler = new CudaAdaptiveShardScheduler(options);
        _replicaExecutor = new CudaReplicaExecutor(_cudaDeviceIndices);
    }

    private void PrepareParameterResidency()
    {
        PreflightCudaCapabilities();
        if (_parameterResidencyPrepared)
            return;
        foreach (int deviceIndex in _cudaDeviceIndices)
            _ = ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);

        var primary = new TorchDevice(
            TensorDevice.Cuda, _cudaDeviceIndices[0]);
        foreach (Parameter parameter in _model.Parameters())
        {
            // Moving the logical tensor once prevents CUDA result construction
            // from treating a lazily-resident parameter as a CPU autograd
            // parent.  Materialize every replica up front so the hot path never
            // routes parameter data or gradients through the shared host mirror.
            parameter.T.to(primary);
            for (int device = 1; device < _cudaDeviceIndices.Length; device++)
            {
                if (parameter.T.DType == TensorDType.BFloat16)
                {
                    parameter.T.EnsureCudaBFloat16Buffer(
                        _cudaDeviceIndices[device]);
                }
                else if (parameter.T.DType == TensorDType.Bfp8)
                {
                    parameter.T.EnsureCudaBfp8Buffer(
                        _cudaDeviceIndices[device]);
                }
                else
                {
                    parameter.T.EnsureCudaFloat32Buffer(
                        _cudaDeviceIndices[device]);
                }
            }
        }
        ForgetMemoryV2Cuda.GetAccelerator(_cudaDeviceIndices[0]).Bind();
        _parameterResidencyPrepared = true;
    }

    private void PreflightCudaCapabilities()
    {
        bool compiledGraphRequested = IsCompiledGraphPreflightRequested();
        TensorPrecisionMode precisionMode = _model.PrecisionMode;
        if (_preflightPrecisionMode == precisionMode
            && _preflightCompiledGraphRequested == compiledGraphRequested)
        {
            return;
        }

        CudaKernelFeature required = ResolveRequiredCudaFeatures(
            _model.GetType(),
            precisionMode,
            _cudaDeviceIndices.Length,
            compiledGraphRequested);
        ExecutionSession? session = ExecutionSession.Current;
        foreach (int deviceIndex in _cudaDeviceIndices)
        {
            bool hasLaneCapabilities = TryGetSessionCudaCapabilities(
                session,
                deviceIndex,
                out CudaKernelCapabilities? laneCapabilities);
            CudaKernelCapabilities capabilities = hasLaneCapabilities
                ? laneCapabilities!
                : NativeCudaRuntime.GetKernelCapabilities(deviceIndex);
            EnsureCudaCapabilities(
                _model.GetType(),
                precisionMode,
                deviceIndex,
                required,
                capabilities);
        }

        _preflightPrecisionMode = precisionMode;
        _preflightCompiledGraphRequested = compiledGraphRequested;
    }

    internal static void EnsureCudaCapabilities(
        Type modelType,
        TensorPrecisionMode precisionMode,
        int deviceIndex,
        CudaKernelFeature required,
        CudaKernelCapabilities capabilities)
    {
        ArgumentNullException.ThrowIfNull(modelType);
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        ArgumentNullException.ThrowIfNull(capabilities);
        CudaKernelFeature missing = required & ~capabilities.Features;
        if (missing == CudaKernelFeature.None)
            return;

        throw new NotSupportedException(
            $"CUDA training capability preflight failed for device " +
            $"{deviceIndex} (SM {capabilities.ComputeCapabilityMajor}." +
            $"{capabilities.ComputeCapabilityMinor}, model " +
            $"{modelType.Name}, precision " +
            $"{TensorPrecisionModeNames.Format(precisionMode)}). " +
            $"Missing required CUDA kernel capabilities: " +
            $"{FormatCudaFeatures(missing)}. CPU fallback is forbidden.");
    }

    private bool IsCompiledGraphPreflightRequested()
    {
        if (!_model.HasCheckpointableTrainingRandom
            || _dispatchPolicy.SynchronizeDataParallelPhases)
        {
            return false;
        }

        ExecutionSession? session = ExecutionSession.Current;
        if (session is null)
            return false;
        foreach (int deviceIndex in _cudaDeviceIndices)
        {
            if (!session.TryGetLane(
                    ExecutionDeviceKind.Cuda,
                    deviceIndex,
                    out IExecutionLane? executionLane)
                || executionLane is not CudaExecutionLane lane
                || !ReferenceEquals(
                    lane.Profiler,
                    NullExecutionProfiler.Instance))
            {
                return false;
            }
        }
        return true;
    }

    private static bool TryGetSessionCudaCapabilities(
        ExecutionSession? session,
        int deviceIndex,
        out CudaKernelCapabilities? capabilities)
    {
        capabilities = null;
        if (session is null
            || !session.TryGetLane(
                ExecutionDeviceKind.Cuda,
                deviceIndex,
                out IExecutionLane? executionLane)
            || executionLane is not CudaExecutionLane lane)
        {
            return false;
        }
        capabilities = lane.CudaCapabilities;
        return true;
    }

    internal static CudaKernelFeature ResolveRequiredCudaFeatures(
        Type modelType,
        TensorPrecisionMode precisionMode,
        int deviceCount,
        bool compiledGraphRequested)
    {
        ArgumentNullException.ThrowIfNull(modelType);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(deviceCount);
        if (!typeof(LanguageModel).IsAssignableFrom(modelType))
        {
            throw new ArgumentException(
                $"'{modelType.FullName}' is not a language model type.",
                nameof(modelType));
        }

        CudaKernelFeature required = CudaKernelFeature.None;
        switch (precisionMode)
        {
            case TensorPrecisionMode.Float32:
                break;
            case TensorPrecisionMode.BFloat16:
            case TensorPrecisionMode.Mix16_32:
                required |= CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16;
                break;
            case TensorPrecisionMode.Bfp8:
            case TensorPrecisionMode.Mix8_32:
                required |= CudaKernelFeature.TensorCores
                    | CudaKernelFeature.BFloat16
                    | CudaKernelFeature.Bfp8Quantization;
                break;
            default:
                throw new ArgumentOutOfRangeException(
                    nameof(precisionMode), precisionMode, null);
        }

        if (typeof(GptRinWikiJp).IsAssignableFrom(modelType))
        {
            required |= CudaKernelFeature.FlashAttention
                | CudaKernelFeature.FusedLayerNorm;
        }
        if (typeof(ForgetMemoryV2Gpt).IsAssignableFrom(modelType)
            || typeof(ForgetScanGpt).IsAssignableFrom(modelType))
        {
            required |= CudaKernelFeature.ForgetMemory;
        }
        if (deviceCount > 1)
            required |= CudaKernelFeature.AsynchronousGradientReduction;
        if (compiledGraphRequested)
            required |= CudaKernelFeature.CudaGraphs;
        return required;
    }

    internal static string FormatCudaFeatures(CudaKernelFeature features)
    {
        if (features == CudaKernelFeature.None)
            return nameof(CudaKernelFeature.None);
        return string.Join(
            ", ",
            Enum.GetValues<CudaKernelFeature>()
                .Where(feature => feature != CudaKernelFeature.None
                    && IsSingleCudaFeature(feature)
                    && features.HasFlag(feature))
                .Select(static feature => feature.ToString()));
    }

    private static bool IsSingleCudaFeature(CudaKernelFeature feature)
    {
        int value = (int)feature;
        return (value & (value - 1)) == 0;
    }

    /// <summary>
    /// Materializes parameter replicas and the gradient reduction plan before
    /// a transfer-guarded step. The operation is idempotent for a stable
    /// model/device set.
    /// </summary>
    public void PrepareForTraining(int batchSize)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        lock (_sync)
        {
            ThrowIfDisposed();
            PrepareParameterResidency();
            int[] devices = GetDevices(batchSize);
            if (devices.Length <= 1)
                return;

            Parameter[] parameters = _model.Parameters().ToArray();
            CudaBFloat16GradientAllReducePlan? bfloat16Plan =
                GetBFloat16GradientPlan(parameters, devices);
            CudaBfp8GradientAllReducePlan? bfp8Plan =
                GetBfp8GradientPlan(parameters, devices);
            if (bfloat16Plan is not null || bfp8Plan is not null)
                return;

            foreach (Parameter parameter in parameters)
                parameter.T.PrepareCudaGradientBuffers(devices);
            _ = GetFlatGradientPlan(parameters, devices);
        }
    }

    public LanguageModel Model => _model;

    public IReadOnlyList<int> CudaDeviceIndices
        => Array.AsReadOnly((int[])_cudaDeviceIndices.Clone());

    internal bool UsesCudaDevices(IReadOnlyList<int> cudaDeviceIndices)
        => _cudaDeviceIndices.SequenceEqual(cudaDeviceIndices);

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

    internal bool HasFlatGradientPlan => _flatGradientPlan is not null;

    internal int FlatGradientPlanBuildCount
        => Volatile.Read(ref _flatGradientPlanBuildCount);

    internal int BFloat16GradientPlanBuildCount
        => Volatile.Read(ref _bfloat16GradientPlanBuildCount);

    internal int TrainingShapePlanBuildCount
        => Volatile.Read(ref _trainingShapePlanBuildCount);

    internal int CachedTrainingShapePlanCount
    {
        get
        {
            lock (_sync)
                return _trainingShapePlans.Count;
        }
    }

    internal long TrainingShapePlanEvictionCount
        => Interlocked.Read(ref _trainingShapePlanEvictionCount);

    internal long GraphCacheBudgetBytes => _graphCacheBudgetBytes;

    internal CudaReplicaExecutorTelemetrySnapshot ReplicaExecutorTelemetry
        => _replicaExecutor.Telemetry;

    internal CudaTrainingGraphTelemetry TrainingGraphTelemetry
    {
        get
        {
            lock (_sync)
            {
                return new CudaTrainingGraphTelemetry(
                    Interlocked.Read(ref _graphCaptureCount),
                    Interlocked.Read(ref _graphReplayCount),
                    Interlocked.Read(ref _graphFallbackCount),
                    _trainingShapePlans.Count(static plan => plan.IsCompiled),
                    _trainingShapePlans.Sum(static plan => plan.GraphPinnedBytes),
                    (_bfloat16GradientPlan?
                        .CapturedReplayReadyEventRecordCount ?? 0)
                        + (_bfp8GradientPlan?
                            .CapturedReplayReadyEventRecordCount ?? 0),
                    (_bfloat16GradientPlan?
                        .CapturedReplayReadyEventRecordMilliseconds ?? 0d)
                        + (_bfp8GradientPlan?
                            .CapturedReplayReadyEventRecordMilliseconds ?? 0d));
            }
        }
    }

    internal Exception? LastGraphFailure
        => Volatile.Read(ref _lastGraphFailure);

    internal long LastBFloat16GradientTransportBytes
        => _bfloat16GradientPlan?.LastCompletedTransportBytes ?? 0;

    internal long BFloat16GradientTransportBytesPerStep
        => _bfloat16GradientPlan?.TransportBytesPerStep ?? 0;

    internal long BFloat16GradientTransportCompletedSteps
        => _bfloat16GradientPlan?.CompletedSteps ?? 0;

    internal long BFloat16GradientManagedLocalPackSubmissionCount
        => _bfloat16GradientPlan?.ManagedLocalPackSubmissionCount ?? 0;

    internal CudaGradientOverlapTelemetry? LastGradientOverlapTelemetry
        => _bfloat16GradientPlan?.LastOverlapTelemetry;

    /// <summary>Returns the most recent per-device batch allocation.</summary>
    public IReadOnlyList<int> LastShardBatchSizes
    {
        get
        {
            lock (_sync)
            {
                ThrowIfDisposed();
                return _adaptiveShardScheduler.LastAllocation;
            }
        }
    }

    /// <summary>
    /// Captures the adaptive CUDA shard scheduler state needed for an exact
    /// mid-run resume. The returned value owns independent array snapshots.
    /// </summary>
    public CudaAdaptiveShardState CaptureAdaptiveShardingState()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            return _adaptiveShardScheduler.CaptureState();
        }
    }

    /// <summary>
    /// Restores a previously committed adaptive shard scheduler state. Device
    /// identity and state integrity are validated by the scheduler.
    /// </summary>
    public void RestoreAdaptiveShardingState(CudaAdaptiveShardState state)
    {
        ArgumentNullException.ThrowIfNull(state);
        lock (_sync)
        {
            ThrowIfDisposed();
            _adaptiveShardScheduler.RestoreState(state, _cudaDeviceIndices);
            ReapplyCompiledGraphConstraint();
        }
    }

    /// <summary>Resets EMA history and applies new adaptive shard bounds.</summary>
    public void ConfigureAdaptiveSharding(
        CudaAdaptiveShardingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        lock (_sync)
        {
            ThrowIfDisposed();
            _graphCacheBudgetBytes = options.GraphCacheBudgetBytes;
            _adaptiveShardScheduler = new CudaAdaptiveShardScheduler(options);
            ReapplyCompiledGraphConstraint();
            if (_trainingShapePlans.First is not null)
                TrimTrainingShapePlans(_trainingShapePlans.First.Value);
        }
    }

    private void ReapplyCompiledGraphConstraint()
    {
        CudaTrainingShapePlan? compiled = _trainingShapePlans
            .FirstOrDefault(static plan => plan.IsCompiled);
        if (compiled is null)
            return;
        _adaptiveShardScheduler.ObserveCompiledGraph(
            compiled.Devices,
            compiled.ShardBatches,
            compiled.GraphPinnedBytes,
            _graphCacheBudgetBytes);
    }

    public float ForwardBackward(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
        => ForwardBackward(
            input,
            target,
            batchSize,
            sequenceLength,
            ignoreIndex,
            Interlocked.Increment(ref _implicitGlobalStep) - 1);

    /// <summary>
    /// Runs one CUDA training step with an explicit, checkpoint-stable global
    /// step. CUDA Graph dropout derives its replay counter from this value, so
    /// resuming the same step reproduces the same device masks exactly.
    /// </summary>
    public float ForwardBackward(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex,
        long globalStep)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(globalStep);
        lock (_sync)
        {
            ThrowIfDisposed();
            PrepareParameterResidency();
            using IDisposable precisionScope =
                TensorExecutionContext.PushPrecisionPolicy(
                    ResolvePrecisionPolicy(_model.PrecisionMode));
            return ForwardBackwardCore(
                input,
                target,
                batchSize,
                sequenceLength,
                ignoreIndex,
                globalStep);
        }
    }

    /// <summary>
    /// Accumulates several microbatches locally on every CUDA device and
    /// performs exactly one gradient reduction. Loss-root weights are based
    /// on the total number of valid targets, so the result is equivalent to
    /// one larger mean-loss batch even when the last microbatch is smaller.
    /// </summary>
    public float ForwardBackwardAccumulated(
        IReadOnlyList<CudaLanguageModelMicroBatch> microBatches,
        int ignoreIndex,
        long globalStep)
    {
        ArgumentNullException.ThrowIfNull(microBatches);
        ArgumentOutOfRangeException.ThrowIfNegative(globalStep);
        if (microBatches.Count == 0)
        {
            throw new ArgumentException(
                "At least one CUDA microbatch is required.",
                nameof(microBatches));
        }
        if (microBatches.Count == 1)
        {
            CudaLanguageModelMicroBatch single = microBatches[0];
            return ForwardBackward(
                single.Input,
                single.Target,
                single.BatchSize,
                single.SequenceLength,
                ignoreIndex,
                globalStep);
        }

        lock (_sync)
        {
            ThrowIfDisposed();
            PrepareParameterResidency();
            using IDisposable precisionScope =
                TensorExecutionContext.PushPrecisionPolicy(
                    ResolvePrecisionPolicy(_model.PrecisionMode));
            return ForwardBackwardAccumulatedCore(
                microBatches,
                ignoreIndex,
                globalStep);
        }
    }

    /// <summary>
    /// Diagnostic variant that synchronizes CUDA after every major phase.
    /// It deliberately calls the same ForwardLoss entry point as compiled
    /// training so BFP8 models retain their direct-BF16 loss head. The old
    /// decomposed Forward -> BFP8 logits -> CrossEntropy sequence is not a
    /// production path and must not be used for bottleneck attribution.
    /// </summary>
    internal CudaDataParallelProfile ForwardBackwardProfiled(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            // A profiled step deliberately executes the eager autograd path so
            // each phase can be synchronized independently. Keeping a compiled
            // training graph alive here pins its capture reservation while the
            // eager path asks the same lane for a second activation workspace.
            // At production shapes that unnecessary overlap can exhaust VRAM
            // after an otherwise successful benchmark. Retire only the cached
            // shape/graph resources; parameters, optimizer state and gradient
            // reduction plans remain resident and warm.
            ReleaseTrainingShapePlans();
            PrepareParameterResidency();
            using IDisposable precisionScope =
                TensorExecutionContext.PushPrecisionPolicy(
                    ResolvePrecisionPolicy(_model.PrecisionMode));
            ValidateBatch(input, target, batchSize, sequenceLength);

            int[] devices = GetDevices(batchSize);
            int[] shardBatches = AllocateShardBatches(batchSize, devices);
            int[] shardStarts = GetShardStarts(shardBatches);
            void SynchronizeAll()
            {
                foreach (int device in devices)
                    ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
            }

            SynchronizeAll();
            var totalTimer = System.Diagnostics.Stopwatch.StartNew();
            var prepareTimer = System.Diagnostics.Stopwatch.StartNew();
            Parameter[] parameters = _parameters;
            CudaBFloat16GradientAllReducePlan? bfloat16Plan =
                GetBFloat16GradientPlan(parameters, devices);
            CudaBfp8GradientAllReducePlan? bfp8Plan =
                GetBfp8GradientPlan(parameters, devices);
            ICudaGradientReductionPlan? reductionPlan =
                (ICudaGradientReductionPlan?)bfp8Plan ?? bfloat16Plan;
            if (reductionPlan is null)
            {
                foreach (Parameter parameter in parameters)
                    parameter.T.PrepareCudaGradientBuffers(devices);
            }
            int totalValid = target.Count(value => value != ignoreIndex);
            if (totalValid == 0)
            {
                throw new ArgumentException(
                    "At least one target must be valid.", nameof(target));
            }
            long reductionStepId = bfp8Plan?.BeginStep()
                ?? bfloat16Plan?.BeginStep()
                ?? 0;
            SynchronizeAll();
            prepareTimer.Stop();

            var weightedLosses = new double[devices.Length];
            var shards = new CudaShardProfile[devices.Length];
            try
            {
                Parallel.For(0, devices.Length, shard =>
                {
                    var shardTimer = System.Diagnostics.Stopwatch.StartNew();
                int batchStart = shardStarts[shard];
                int shardBatch = shardBatches[shard];
                int elementStart = batchStart * sequenceLength;
                int elementCount = shardBatch * sequenceLength;
                int[] shardInput = input.AsSpan(
                    elementStart, elementCount).ToArray();
                int[] shardTarget = target.AsSpan(
                    elementStart, elementCount).ToArray();
                int shardValid = shardTarget.Count(
                    value => value != ignoreIndex);
                double dataPreparation = shardTimer.Elapsed.TotalMilliseconds;

                using IDisposable shardPrecisionScope =
                    TensorExecutionContext.PushPrecisionPolicy(
                        ResolvePrecisionPolicy(_model.PrecisionMode));
                using IDisposable scope = TensorExecutionContext.Push(
                    new TorchDevice(TensorDevice.Cuda, devices[shard]));
                bfp8Plan?.BeginDeviceStep(
                    reductionStepId, devices[shard]);
                bfloat16Plan?.BeginDeviceStep(
                    reductionStepId, devices[shard]);
                using IDisposable? reductionScope = reductionPlan is null
                    ? null
                    : CudaGradientReductionContext.Push(
                        reductionPlan, devices[shard], reductionStepId);
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(devices[shard]);

                var phaseTimer = System.Diagnostics.Stopwatch.StartNew();
                Tensor loss = _model.ForwardLoss(
                    shardInput,
                    shardTarget,
                    shardBatch,
                    sequenceLength,
                    ignoreIndex);
                accelerator.Synchronize();
                double forward = phaseTimer.Elapsed.TotalMilliseconds;

                phaseTimer.Restart();
                weightedLosses[shard] = loss.item() * shardValid;
                accelerator.Synchronize();
                double lossMilliseconds = phaseTimer.Elapsed.TotalMilliseconds;

                phaseTimer.Restart();
                float weight = (float)shardValid / totalValid;
                loss.BackwardAndRelease([weight]);
                accelerator.Synchronize();
                double backward = phaseTimer.Elapsed.TotalMilliseconds;
                    shards[shard] = new CudaShardProfile(
                        devices[shard],
                        shardBatch,
                        dataPreparation,
                        forward,
                        lossMilliseconds,
                        backward);
                });
            }
            catch
            {
                bfp8Plan?.Abort(reductionStepId);
                bfloat16Plan?.Abort(reductionStepId);
                throw;
            }

            var allReduceTimer = System.Diagnostics.Stopwatch.StartNew();
            if (bfp8Plan is not null)
            {
                bfp8Plan.Complete(reductionStepId);
            }
            else if (bfloat16Plan is not null)
            {
                bfloat16Plan.Complete(reductionStepId);
            }
            else
            {
                TensorCudaKernels.FlatGradientPlan plan =
                    GetFlatGradientPlan(parameters, devices);
                TensorCudaKernels.AllReduceGradientsResident(
                    parameters, devices, plan);
            }
            SynchronizeAll();
            allReduceTimer.Stop();
            totalTimer.Stop();
            return new CudaDataParallelProfile(
                (float)(weightedLosses.Sum() / totalValid),
                prepareTimer.Elapsed.TotalMilliseconds,
                allReduceTimer.Elapsed.TotalMilliseconds,
                totalTimer.Elapsed.TotalMilliseconds,
                shards);
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            List<Exception>? failures = null;
            ReleaseTrainingShapePlans(ref failures);
            try
            {
                _bfp8GradientPlan?.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            finally
            {
                _bfp8GradientPlan = null;
            }
            try
            {
                _bfloat16GradientPlan?.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            finally
            {
                _bfloat16GradientPlan = null;
            }
            try
            {
                _flatGradientPlan?.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            finally
            {
                _flatGradientPlan = null;
            }
            foreach (NativeCudaBuffer<float> lossBuffer
                in _accumulatedLossBuffers.Values)
            {
                try
                {
                    lossBuffer.Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            _accumulatedLossBuffers.Clear();
            try
            {
                _replicaExecutor.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }

            if (failures is not null)
            {
                throw new AggregateException(
                    "One or more CUDA data-parallel resources failed to dispose.",
                    failures);
            }
        }
    }

    private void ReleaseTrainingShapePlans()
    {
        List<Exception>? failures = null;
        ReleaseTrainingShapePlans(ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                "CUDA training shape cleanup failed before the profiled " +
                "eager step.",
                failures);
        }
    }

    private void ReleaseTrainingShapePlans(ref List<Exception>? failures)
    {
        while (_trainingShapePlans.First is { } shapeNode)
        {
            _trainingShapePlans.RemoveFirst();
            DisposeTrainingShapePlan(shapeNode.Value, ref failures);
        }
    }

    private float ForwardBackwardCore(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex,
        long globalStep)
    {
        ValidateBatch(input, target, batchSize, sequenceLength);
        int[] devices = GetDevices(batchSize);
        int[] shardBatches = AllocateShardBatches(batchSize, devices);
        CudaTrainingShapePlan shapePlan = GetOrCreateTrainingShapePlan(
            batchSize,
            sequenceLength,
            devices,
            shardBatches);
        Parameter[] parameters = _parameters;
        CudaBFloat16GradientAllReducePlan? bfloat16Plan = devices.Length > 1
            ? GetBFloat16GradientPlan(parameters, devices)
            : null;
        CudaBfp8GradientAllReducePlan? bfp8Plan = devices.Length > 1
            ? GetBfp8GradientPlan(parameters, devices)
            : null;
        ICudaGradientReductionPlan? reductionPlan =
            (ICudaGradientReductionPlan?)bfp8Plan ?? bfloat16Plan;
        if (devices.Length > 1 && reductionPlan is null)
        {
            foreach (Parameter parameter in parameters)
                parameter.T.PrepareCudaGradientBuffers(devices);
        }
        int totalValid = target.Count(value => value != ignoreIndex);
        if (totalValid == 0)
        {
            throw new ArgumentException(
                "At least one target must be valid.", nameof(target));
        }
        shapePlan.PrepareBatch(input, target, ignoreIndex);
        double[] weightedLosses = shapePlan.WeightedLosses;
        double[] shardElapsed = shapePlan.ShardElapsed;
        bool graphEligible = IsCudaGraphEligible(
            shapePlan,
            totalValid,
            target.Length);
        bool capturedForThisBatch = false;

        if (graphEligible && !shapePlan.IsCompiled)
        {
            try
            {
                shapePlan.PrepareGraphInputs(ignoreIndex);
                ExecuteReplicas(shapePlan, devices.Length);
                RunGraphPrewarm(
                    shapePlan,
                    bfp8Plan,
                    bfloat16Plan,
                    reductionPlan,
                    parameters,
                    devices,
                    totalValid,
                    ignoreIndex,
                    globalStep);
            }
            catch (Exception prewarmFailure)
            {
                Volatile.Write(ref _lastGraphFailure, prewarmFailure);
                Exception? cleanupFailure = TryDisableAndDisposeGraph(
                    shapePlan,
                    prewarmFailure);
                Interlocked.Increment(ref _graphFallbackCount);
                graphEligible = false;
                if (cleanupFailure is not null)
                {
                    throw new AggregateException(
                        "CUDA Graph prewarm and rollback both failed.",
                        prewarmFailure,
                        cleanupFailure);
                }
            }
        }

        if (graphEligible && shapePlan.ShouldCapture)
        {
            long recordingStepId = BeginReductionStep(
                bfp8Plan,
                bfloat16Plan);
            bool recordingActive = recordingStepId != 0;
            try
            {
                shapePlan.PrepareCapture(
                    bfp8Plan,
                    bfloat16Plan,
                    reductionPlan,
                    recordingStepId,
                    totalValid,
                    ignoreIndex,
                    globalStep,
                    _model.TrainingRandomRootSeed);
                ExecuteReplicas(shapePlan, devices.Length);
                DiscardCapturedReductionStep(
                    bfp8Plan,
                    bfloat16Plan,
                    recordingStepId);
                recordingActive = false;
                shapePlan.CommitCapture();
                Interlocked.Increment(ref _graphCaptureCount);
                _adaptiveShardScheduler.ObserveCompiledGraph(
                    shapePlan.ShardBatches,
                    shapePlan.GraphPinnedBytes,
                    _graphCacheBudgetBytes);
                TrimTrainingShapePlans(shapePlan);
                capturedForThisBatch = true;
            }
            catch (Exception captureFailure)
            {
                Volatile.Write(ref _lastGraphFailure, captureFailure);
                if (recordingActive)
                    TryAbortReductionStep(
                        bfp8Plan,
                        bfloat16Plan,
                        recordingStepId);
                Exception? cleanupFailure = TryDisableAndDisposeGraph(
                    shapePlan,
                    captureFailure);
                Interlocked.Increment(ref _graphFallbackCount);
                if (cleanupFailure is not null)
                {
                    throw new AggregateException(
                        "CUDA Graph capture and rollback both failed.",
                        captureFailure,
                        cleanupFailure);
                }
            }
        }

        bool replayGraph = graphEligible && shapePlan.IsCompiled;
        long reductionStepId = BeginReductionStep(
            bfp8Plan,
            bfloat16Plan);
        try
        {
            if (replayGraph)
            {
                shapePlan.PrepareReplay(
                    bfp8Plan,
                    bfloat16Plan,
                    reductionStepId,
                    globalStep,
                    capturedForThisBatch);
                ExecuteReplicas(shapePlan, devices.Length);
                Interlocked.Increment(ref _graphReplayCount);
            }
            else
            {
                shapePlan.PrepareEager(
                    bfp8Plan,
                    bfloat16Plan,
                    reductionPlan,
                    reductionStepId,
                    totalValid,
                    ignoreIndex);
                ExecuteReplicas(shapePlan, devices.Length);
            }
        }
        catch
        {
            bfp8Plan?.Abort(reductionStepId);
            bfloat16Plan?.Abort(reductionStepId);
            throw;
        }

        bool canMeasureShardRuntime = bfp8Plan is null
            && (bfloat16Plan is null
                || bfloat16Plan.DefersExchangeUntilBackward);
        if (canMeasureShardRuntime && !replayGraph)
        {
            // The scalar readback is queued before backward. Synchronizing all
            // devices in parallel captures real shard runtime without adding
            // a serial wait to the non-peer exchange path.
            shapePlan.PrepareSynchronization();
            ExecuteReplicas(shapePlan, devices.Length);
        }

        if (canMeasureShardRuntime)
            _adaptiveShardScheduler.Observe(shardBatches, shardElapsed);

        if (bfp8Plan is not null)
        {
            bfp8Plan.Complete(reductionStepId);
        }
        else if (bfloat16Plan is not null)
        {
            bfloat16Plan.Complete(reductionStepId);
        }
        else if (devices.Length > 1)
        {
            TensorCudaKernels.FlatGradientPlan plan =
                GetFlatGradientPlan(parameters, devices);
            TensorCudaKernels.AllReduceGradientsResident(
                parameters, devices, plan);
        }
        return (float)(weightedLosses.Sum() / totalValid);
    }

    private float ForwardBackwardAccumulatedCore(
        IReadOnlyList<CudaLanguageModelMicroBatch> microBatches,
        int ignoreIndex,
        long globalStep)
    {
        CudaLanguageModelMicroBatch first = microBatches[0];
        ValidateBatch(
            first.Input,
            first.Target,
            first.BatchSize,
            first.SequenceLength);
        int[] devices = GetDevices(first.BatchSize);
        long totalValidLong = 0;
        foreach (CudaLanguageModelMicroBatch microBatch in microBatches)
        {
            ValidateBatch(
                microBatch.Input,
                microBatch.Target,
                microBatch.BatchSize,
                microBatch.SequenceLength);
            if (!GetDevices(microBatch.BatchSize).SequenceEqual(devices))
            {
                throw new ArgumentException(
                    "Every accumulated microbatch must use the same CUDA " +
                    "device set. Flush accumulation before a smaller tail " +
                    "batch changes the active device count.",
                    nameof(microBatches));
            }
            totalValidLong = checked(
                totalValidLong + microBatch.Target.Count(
                    value => value != ignoreIndex));
        }
        if (totalValidLong == 0)
        {
            throw new ArgumentException(
                "At least one accumulated target must be valid.",
                nameof(microBatches));
        }
        if (totalValidLong > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(microBatches),
                "An accumulated update has too many valid targets.");
        }
        int totalValid = (int)totalValidLong;

        Parameter[] parameters = _parameters;
        CudaBFloat16GradientAllReducePlan? bfloat16Plan = devices.Length > 1
            ? GetBFloat16GradientPlan(parameters, devices)
            : null;
        CudaBfp8GradientAllReducePlan? bfp8Plan = devices.Length > 1
            ? GetBfp8GradientPlan(parameters, devices)
            : null;
        ICudaGradientReductionPlan? reductionPlan =
            (ICudaGradientReductionPlan?)bfp8Plan ?? bfloat16Plan;
        if (devices.Length > 1 && reductionPlan is null)
        {
            foreach (Parameter parameter in parameters)
                parameter.T.PrepareCudaGradientBuffers(devices);
        }

        long reductionStepId = BeginReductionStep(
            bfp8Plan,
            bfloat16Plan);
        NativeCudaBuffer<float>[] accumulatedLossBuffers =
            GetAccumulatedLossBuffers(devices);
        double weightedLoss = 0d;
        try
        {
            for (int index = 0; index < microBatches.Count; index++)
            {
                CudaLanguageModelMicroBatch microBatch = microBatches[index];
                int[] shardBatches = AllocateShardBatches(
                    microBatch.BatchSize,
                    devices);
                CudaTrainingShapePlan shapePlan =
                    GetOrCreateTrainingShapePlan(
                        microBatch.BatchSize,
                        microBatch.SequenceLength,
                        devices,
                        shardBatches);
                shapePlan.PrepareBatch(
                    microBatch.Input,
                    microBatch.Target,
                    ignoreIndex);
                bool publishGradients = index == microBatches.Count - 1;
                ICudaGradientReductionPlan? activeReductionPlan =
                    publishGradients
                        ? reductionPlan
                        : DeferredGradientReductionPlan.Instance;
                shapePlan.PrepareEager(
                    bfp8Plan,
                    bfloat16Plan,
                    activeReductionPlan,
                    reductionStepId,
                    totalValid,
                    ignoreIndex,
                    beginReductionDeviceStep: index == 0,
                    accumulatedLossBuffers: accumulatedLossBuffers,
                    completeAccumulatedLoss: publishGradients);
                ExecuteReplicas(shapePlan, devices.Length);
                weightedLoss += shapePlan.WeightedLosses.Sum();

                bool canMeasureShardRuntime = bfp8Plan is null
                    && (bfloat16Plan is null
                        || bfloat16Plan.DefersExchangeUntilBackward);
                if (canMeasureShardRuntime)
                {
                    shapePlan.PrepareSynchronization();
                    ExecuteReplicas(shapePlan, devices.Length);
                    _adaptiveShardScheduler.Observe(
                        shardBatches,
                        shapePlan.ShardElapsed);
                }
            }

            if (bfp8Plan is not null)
                bfp8Plan.Complete(reductionStepId);
            else if (bfloat16Plan is not null)
                bfloat16Plan.Complete(reductionStepId);
            else if (devices.Length > 1)
            {
                TensorCudaKernels.FlatGradientPlan plan =
                    GetFlatGradientPlan(parameters, devices);
                TensorCudaKernels.AllReduceGradientsResident(
                    parameters,
                    devices,
                    plan);
            }
        }
        catch
        {
            bfp8Plan?.Abort(reductionStepId);
            bfloat16Plan?.Abort(reductionStepId);
            throw;
        }

        _ = globalStep;
        return (float)(weightedLoss / totalValid);
    }

    private NativeCudaBuffer<float>[] GetAccumulatedLossBuffers(
        IReadOnlyList<int> devices)
    {
        var result = new NativeCudaBuffer<float>[devices.Count];
        for (int index = 0; index < devices.Count; index++)
        {
            int device = devices[index];
            if (!_accumulatedLossBuffers.TryGetValue(
                    device,
                    out NativeCudaBuffer<float>? buffer)
                || !buffer.IsAlive)
            {
                buffer?.Dispose();
                buffer = ForgetMemoryV2Cuda.GetAccelerator(device)
                    .Allocate1D<float>(
                        1,
                        CudaMemoryKind.Persistent);
                _accumulatedLossBuffers[device] = buffer;
            }
            result[index] = buffer;
        }
        return result;
    }

    private sealed class DeferredGradientReductionPlan
        : ICudaGradientReductionPlan
    {
        internal static DeferredGradientReductionPlan Instance { get; } =
            new();

        private DeferredGradientReductionPlan()
        {
        }

        public void NotifyGradientReady(
            Tensor tensor,
            int deviceIndex,
            long stepId)
        {
            // Earlier microbatches accumulate into the reducer-owned local
            // gradient arena. The final backward publishes every leaf once,
            // which starts the single reduction for the whole update.
        }
    }

    private void ExecuteReplicas(
        ICudaReplicaWorkDescriptor work,
        int replicaCount)
    {
        try
        {
            _replicaExecutor.Execute(
                work,
                replicaCount,
                ExecutionSession.Current,
                ResolvePrecisionPolicy(_model.PrecisionMode));
        }
        catch (AggregateException exception)
            when (exception.InnerExceptions.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(exception.InnerExceptions[0])
                .Throw();
        }
    }

    private bool IsCudaGraphEligible(
        CudaTrainingShapePlan shapePlan,
        int totalValid,
        int targetLength)
    {
        if (shapePlan.GraphDisabled
            || totalValid != targetLength
            || !_model.HasCheckpointableTrainingRandom
            || _dispatchPolicy.SynchronizeDataParallelPhases)
        {
            return false;
        }

        ExecutionSession? session = ExecutionSession.Current;
        if (session is null)
            return false;
        foreach (int device in shapePlan.Devices)
        {
            if (!session.TryGetLane(
                    ExecutionDeviceKind.Cuda,
                    device,
                    out IExecutionLane? executionLane)
                || executionLane is not CudaExecutionLane lane
                || !lane.CudaCapabilities.Supports(
                    CudaKernelFeature.CudaGraphs)
                || !ReferenceEquals(
                    lane.Profiler,
                    NullExecutionProfiler.Instance))
            {
                return false;
            }
        }
        return true;
    }

    private void RunGraphPrewarm(
        CudaTrainingShapePlan shapePlan,
        CudaBfp8GradientAllReducePlan? bfp8Plan,
        CudaBFloat16GradientAllReducePlan? bfloat16Plan,
        ICudaGradientReductionPlan? reductionPlan,
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices,
        int totalValid,
        int ignoreIndex,
        long globalStep)
    {
        TrainingRandomState? randomState = _model.CaptureTrainingRandomState()
            ?? throw new InvalidOperationException(
                "CUDA Graph prewarm requires checkpointable training RNG state.");
        Exception? prewarmFailure = null;
        try
        {
            while (shapePlan.WarmupCount < 2)
            {
                long stepId = BeginReductionStep(bfp8Plan, bfloat16Plan);
                bool stepActive = stepId != 0;
                try
                {
                    shapePlan.PrepareEager(
                        bfp8Plan,
                        bfloat16Plan,
                        reductionPlan,
                        stepId,
                        totalValid,
                        ignoreIndex,
                        useFixedInputs: true);
                    ExecuteReplicas(shapePlan, devices.Count);
                    shapePlan.PrepareSynchronization();
                    ExecuteReplicas(shapePlan, devices.Count);
                    if (bfp8Plan is not null)
                    {
                        bfp8Plan.Abort(stepId);
                        stepActive = false;
                    }
                    else if (bfloat16Plan is not null)
                    {
                        bfloat16Plan.Abort(stepId);
                        stepActive = false;
                    }
                    shapePlan.RecordWarmup();
                }
                catch
                {
                    if (stepActive)
                    {
                        TryAbortReductionStep(
                            bfp8Plan,
                            bfloat16Plan,
                            stepId);
                    }
                    throw;
                }
                finally
                {
                    foreach (Parameter parameter in parameters)
                        parameter.ZeroGrad();
                }
            }

            try
            {
                long reservationStepId = BeginReductionStep(
                    bfp8Plan,
                    bfloat16Plan);
                bool reservationStepActive = reservationStepId != 0;
                try
                {
                    shapePlan.PrepareReservationPrewarm(
                        bfp8Plan,
                        bfloat16Plan,
                        reductionPlan,
                        reservationStepId,
                        globalStep,
                        _model.TrainingRandomRootSeed);
                    ExecuteReplicas(shapePlan, devices.Count);
                    DiscardCapturedReductionStep(
                        bfp8Plan,
                        bfloat16Plan,
                        reservationStepId);
                    reservationStepActive = false;
                }
                finally
                {
                    if (reservationStepActive)
                    {
                        TryAbortReductionStep(
                            bfp8Plan,
                            bfloat16Plan,
                            reservationStepId);
                    }
                }
            }
            finally
            {
                foreach (Parameter parameter in parameters)
                    parameter.ZeroGrad();
            }
        }
        catch (Exception exception)
        {
            prewarmFailure = exception;
        }

        try
        {
            _model.RestoreTrainingRandomState(randomState);
        }
        catch (Exception restoreFailure)
        {
            prewarmFailure = prewarmFailure is null
                ? restoreFailure
                : new AggregateException(prewarmFailure, restoreFailure);
        }
        if (prewarmFailure is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(prewarmFailure)
                .Throw();
        }
    }

    private static long BeginReductionStep(
        CudaBfp8GradientAllReducePlan? bfp8Plan,
        CudaBFloat16GradientAllReducePlan? bfloat16Plan)
        => bfp8Plan?.BeginStep() ?? bfloat16Plan?.BeginStep() ?? 0;

    private static void DiscardCapturedReductionStep(
        CudaBfp8GradientAllReducePlan? bfp8Plan,
        CudaBFloat16GradientAllReducePlan? bfloat16Plan,
        long stepId)
    {
        if (bfp8Plan is not null)
            bfp8Plan.DiscardCapturedBackwardRecordingStep(stepId);
        else if (bfloat16Plan is not null)
            bfloat16Plan.DiscardCapturedBackwardRecordingStep(stepId);
    }

    private static void TryAbortReductionStep(
        CudaBfp8GradientAllReducePlan? bfp8Plan,
        CudaBFloat16GradientAllReducePlan? bfloat16Plan,
        long stepId)
    {
        try
        {
            bfp8Plan?.Abort(stepId);
            bfloat16Plan?.Abort(stepId);
        }
        catch
        {
            // Preserve the capture failure. Reducer Abort is best-effort and
            // the subsequent BeginStep validates that no stale step survived.
        }
    }

    private Exception? TryDisableAndDisposeGraph(
        CudaTrainingShapePlan shapePlan,
        Exception reason)
    {
        shapePlan.DisableGraph(reason);
        shapePlan.PrepareGraphDisposal();
        try
        {
            ExecuteReplicas(shapePlan, shapePlan.Devices.Length);
            return null;
        }
        catch (Exception cleanupFailure)
        {
            return cleanupFailure;
        }
    }

    private void ValidateBatch(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(target);
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            throw new InvalidOperationException("CUDA execution must be selected.");
        if (input.Length != target.Length
            || input.Length != checked(batchSize * sequenceLength))
        {
            throw new ArgumentException(
                "Input and target must match the batch shape.");
        }
    }

    private int[] GetDevices(int batchSize)
        => _devicePrefixes[
            Math.Min(batchSize, _cudaDeviceIndices.Length) - 1];

    private int[] AllocateShardBatches(
        int batchSize,
        IReadOnlyList<int> devices)
    {
        return _adaptiveShardScheduler.Allocate(batchSize, devices);
    }

    private CudaTrainingShapePlan GetOrCreateTrainingShapePlan(
        int batchSize,
        int sequenceLength,
        IReadOnlyList<int> devices,
        IReadOnlyList<int> shardBatches)
    {
        LinkedListNode<CudaTrainingShapePlan>? node =
            _trainingShapePlans.First;
        while (node is not null)
        {
            LinkedListNode<CudaTrainingShapePlan>? next = node.Next;
            if (node.Value.Matches(
                    batchSize,
                    sequenceLength,
                    devices,
                    shardBatches))
            {
                if (!ReferenceEquals(node, _trainingShapePlans.First))
                {
                    _trainingShapePlans.Remove(node);
                    _trainingShapePlans.AddFirst(node);
                }
                return node.Value;
            }
            node = next;
        }

        var plan = new CudaTrainingShapePlan(
            this,
            batchSize,
            sequenceLength,
            devices,
            shardBatches);
        _trainingShapePlans.AddFirst(plan);
        Interlocked.Increment(ref _trainingShapePlanBuildCount);
        TrimTrainingShapePlans(plan);
        return plan;
    }

    private void TrimTrainingShapePlans(CudaTrainingShapePlan protectedPlan)
    {
        while (_trainingShapePlans.Count > 1
            && (_trainingShapePlans.Count > 3
                || _trainingShapePlans.Sum(
                    static candidate => candidate.GraphPinnedBytes)
                    > _graphCacheBudgetBytes))
        {
            LinkedListNode<CudaTrainingShapePlan> evicted =
                _trainingShapePlans.Last!;
            if (ReferenceEquals(evicted.Value, protectedPlan))
            {
                evicted = evicted.Previous
                    ?? throw new InvalidOperationException(
                        "A protected CUDA shape plan cannot be the only " +
                        "eviction candidate.");
            }
            _trainingShapePlans.Remove(evicted);
            List<Exception>? failures = null;
            DisposeTrainingShapePlan(evicted.Value, ref failures);
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA training shape eviction failed.",
                    failures);
            }
            Interlocked.Increment(ref _trainingShapePlanEvictionCount);
        }
    }

    private void DisposeTrainingShapePlan(
        CudaTrainingShapePlan plan,
        ref List<Exception>? failures)
    {
        plan.PrepareGraphDisposal();
        try
        {
            ExecuteReplicas(plan, plan.Devices.Length);
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
        try
        {
            plan.Dispose();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }

    private TensorCudaKernels.FlatGradientPlan GetFlatGradientPlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (_flatGradientPlan is null
            || !_flatGradientPlan.Matches(parameters, devices))
        {
            _flatGradientPlan?.Dispose();
            _flatGradientPlan = new TensorCudaKernels.FlatGradientPlan(
                parameters, devices);
            Interlocked.Increment(ref _flatGradientPlanBuildCount);
        }
        return _flatGradientPlan;
    }

    private CudaBFloat16GradientAllReducePlan? GetBFloat16GradientPlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (!UseBFloat16GradientBuckets(devices, _model.PrecisionMode))
        {
            _bfloat16GradientPlan?.Dispose();
            _bfloat16GradientPlan = null;
            return null;
        }
        _flatGradientPlan?.Dispose();
        _flatGradientPlan = null;
        bool useBlockBfp8Transport =
            _model.PrecisionMode == TensorPrecisionMode.Mix8_32;
        if (_bfloat16GradientPlan is null
            || !_bfloat16GradientPlan.Matches(parameters, devices)
            || _bfloat16GradientPlan.UsesBlockBfp8Transport
                != useBlockBfp8Transport)
        {
            _bfloat16GradientPlan?.Dispose();
            _bfloat16GradientPlan =
                new CudaBFloat16GradientAllReducePlan(
                    parameters,
                    devices,
                    _dispatchPolicy,
                    useBFloat16GradientStorage:
                        ResolvePrecisionPolicy(_model.PrecisionMode).Gradient
                            == NumericFormat.BFloat16,
                    useBlockBfp8Transport: useBlockBfp8Transport);
            Interlocked.Increment(ref _bfloat16GradientPlanBuildCount);
        }
        return _bfloat16GradientPlan;
    }

    private CudaBfp8GradientAllReducePlan? GetBfp8GradientPlan(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> devices)
    {
        if (devices.Count != 2
            || ResolvePrecisionPolicy(_model.PrecisionMode).Gradient
                != NumericFormat.Bfp8)
        {
            _bfp8GradientPlan?.Dispose();
            _bfp8GradientPlan = null;
            return null;
        }
        _flatGradientPlan?.Dispose();
        _flatGradientPlan = null;
        if (_bfp8GradientPlan is null
            || !_bfp8GradientPlan.Matches(parameters, devices))
        {
            _bfp8GradientPlan?.Dispose();
            _bfp8GradientPlan =
                new CudaBfp8GradientAllReducePlan(parameters, devices);
        }
        return _bfp8GradientPlan;
    }

    private static PrecisionPolicy ResolvePrecisionPolicy(
        TensorPrecisionMode precisionMode)
        => precisionMode switch
        {
            TensorPrecisionMode.Float32 => PrecisionPolicy.Float32,
            TensorPrecisionMode.BFloat16 => PrecisionPolicy.BFloat16,
            TensorPrecisionMode.Mix16_32 => PrecisionPolicy.Mix16_32,
            TensorPrecisionMode.Bfp8 => PrecisionPolicy.Bfp8,
            TensorPrecisionMode.Mix8_32 => PrecisionPolicy.Mix8_32,
            _ => throw new ArgumentOutOfRangeException(
                nameof(precisionMode), precisionMode, null),
        };

    private bool UseBFloat16GradientBuckets(
        IReadOnlyList<int> devices,
        TensorPrecisionMode precisionMode)
        => devices.Count == 2
            && (precisionMode == TensorPrecisionMode.Mix8_32
                || (ResolvePrecisionPolicy(precisionMode).Gradient
                        == NumericFormat.BFloat16
                    && !_dispatchPolicy.DisableBFloat16GradientBuckets));

    private static int[] GetShardStarts(
        IReadOnlyList<int> shardBatches)
    {
        var starts = new int[shardBatches.Count];
        int next = 0;
        for (int index = 0; index < starts.Length; index++)
        {
            starts[index] = next;
            next = checked(next + shardBatches[index]);
        }
        return starts;
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(IsDisposed, this);

    /// <summary>
    /// Reuses the managed part of a stable CUDA training shape.  CUDA tensor
    /// storage remains lane-owned; this plan removes per-step shard arrays and
    /// scratch result arrays while retaining only the three most recent
    /// adaptive shapes.
    /// </summary>
    private sealed class CudaTrainingShapePlan
        : ICudaReplicaWorkDescriptor, IDisposable
    {
        private readonly CudaDataParallelEngine _owner;
        private readonly CompiledDeviceGraph?[] _compiledGraphs;
        private readonly CudaGraphBatchInputs?[] _graphInputs;
        private readonly CudaGraphRngState?[] _graphRngStates;
        private readonly NativeCudaBuffer<float>?[] _graphLossSlots;
        private readonly CudaBfp8GraphGradientPublication?[]
            _graphBfp8Publications;
        private readonly CudaGraphCaptureScope?[] _captureMemoryScopes;
        private readonly long[] _capturePinnedBaselines;
        private CudaBfp8GradientAllReducePlan? _bfp8Plan;
        private CudaBFloat16GradientAllReducePlan? _bfloat16Plan;
        private ICudaGradientReductionPlan? _reductionPlan;
        private long _reductionStepId;
        private long _globalStep;
        private ulong _rootSeed;
        private int _totalValid;
        private int _ignoreIndex;
        private int _warmupCount;
        private bool _reuseCaptureInputs;
        private bool _useFixedInputs;
        private bool _readEagerLoss;
        private bool _beginReductionDeviceStep;
        private NativeCudaBuffer<float>[]? _accumulatedLossBuffers;
        private bool _completeAccumulatedLoss;
        private bool _graphDisabled;
        private int _disposed;
        private WorkPhase _phase;

        internal CudaTrainingShapePlan(
            CudaDataParallelEngine owner,
            int batchSize,
            int sequenceLength,
            IReadOnlyList<int> devices,
            IReadOnlyList<int> shardBatches)
        {
            _owner = owner;
            BatchSize = batchSize;
            SequenceLength = sequenceLength;
            Devices = devices.ToArray();
            ShardBatches = shardBatches.ToArray();
            ShardStarts = GetShardStarts(ShardBatches);
            ShardInputs = new int[Devices.Length][];
            ShardTargets = new int[Devices.Length][];
            ShardValidTargets = new int[Devices.Length];
            WeightedLosses = new double[Devices.Length];
            ShardElapsed = new double[Devices.Length];
            ShardStarted = new long[Devices.Length];
            _compiledGraphs = new CompiledDeviceGraph?[Devices.Length];
            _graphInputs = new CudaGraphBatchInputs?[Devices.Length];
            _graphRngStates = new CudaGraphRngState?[Devices.Length];
            _graphLossSlots = new NativeCudaBuffer<float>?[Devices.Length];
            _graphBfp8Publications =
                new CudaBfp8GraphGradientPublication?[Devices.Length];
            _captureMemoryScopes = new CudaGraphCaptureScope?[Devices.Length];
            _capturePinnedBaselines = new long[Devices.Length];
            for (int shard = 0; shard < Devices.Length; shard++)
            {
                int length = checked(ShardBatches[shard] * sequenceLength);
                ShardInputs[shard] = new int[length];
                ShardTargets[shard] = new int[length];
            }
        }

        internal int BatchSize { get; }
        internal int SequenceLength { get; }
        internal int[] Devices { get; }
        internal int[] ShardBatches { get; }
        internal int[] ShardStarts { get; }
        internal int[][] ShardInputs { get; }
        internal int[][] ShardTargets { get; }
        internal int[] ShardValidTargets { get; }
        internal double[] WeightedLosses { get; }
        internal double[] ShardElapsed { get; }
        internal long[] ShardStarted { get; }

        internal bool IsCompiled
            => !_graphDisabled
                && _compiledGraphs.All(static graph => graph is not null);

        internal long GraphPinnedBytes
            => _compiledGraphs.Sum(static graph => graph?.GraphPinnedBytes ?? 0);

        internal bool GraphDisabled => _graphDisabled;

        internal bool ShouldCapture
            => !_graphDisabled && !IsCompiled && _warmupCount >= 2;

        internal int WarmupCount => _warmupCount;

        internal void PrepareEager(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            long reductionStepId,
            int totalValid,
            int ignoreIndex,
            bool useFixedInputs = false,
            bool beginReductionDeviceStep = true,
            NativeCudaBuffer<float>[]? accumulatedLossBuffers = null,
            bool completeAccumulatedLoss = false)
        {
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = reductionPlan;
            _reductionStepId = reductionStepId;
            _totalValid = totalValid;
            _ignoreIndex = ignoreIndex;
            _useFixedInputs = useFixedInputs;
            _readEagerLoss = !useFixedInputs;
            _beginReductionDeviceStep = beginReductionDeviceStep;
            _accumulatedLossBuffers = accumulatedLossBuffers;
            _completeAccumulatedLoss = completeAccumulatedLoss;
            _phase = WorkPhase.Eager;
        }

        internal void PrepareGraphInputs(int ignoreIndex)
        {
            _ignoreIndex = ignoreIndex;
            _phase = WorkPhase.InitializeGraphInputs;
        }

        internal void PrepareReservationPrewarm(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            long reductionStepId,
            long globalStep,
            ulong rootSeed)
        {
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = reductionPlan;
            _reductionStepId = reductionStepId;
            _globalStep = globalStep;
            _rootSeed = rootSeed;
            _useFixedInputs = true;
            _phase = WorkPhase.ReservationPrewarm;
        }

        internal void PrepareCapture(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            long reductionStepId,
            int totalValid,
            int ignoreIndex,
            long globalStep,
            ulong rootSeed)
        {
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = reductionPlan;
            _reductionStepId = reductionStepId;
            _totalValid = totalValid;
            _ignoreIndex = ignoreIndex;
            _globalStep = globalStep;
            _rootSeed = rootSeed;
            _phase = WorkPhase.Capture;
        }

        internal void PrepareReplay(
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            long reductionStepId,
            long globalStep,
            bool reuseCaptureInputs)
        {
            if (!IsCompiled)
                throw new InvalidOperationException(
                    "The CUDA training graph is not compiled.");
            _bfp8Plan = bfp8Plan;
            _bfloat16Plan = bfloat16Plan;
            _reductionPlan = null;
            _reductionStepId = reductionStepId;
            _globalStep = globalStep;
            _reuseCaptureInputs = reuseCaptureInputs;
            _phase = WorkPhase.Replay;
        }

        internal void PrepareSynchronization()
            => _phase = WorkPhase.Synchronize;

        internal void PrepareGraphDisposal()
            => _phase = WorkPhase.DisposeGraph;

        internal void CommitCapture()
        {
            if (_compiledGraphs.Any(static graph => graph is null))
            {
                throw new InvalidOperationException(
                    "CUDA Graph capture did not publish every device graph.");
            }
        }

        internal void RecordWarmup()
        {
            if (_warmupCount < 2)
                _warmupCount++;
        }

        internal void DisableGraph(Exception reason)
        {
            ArgumentNullException.ThrowIfNull(reason);
            _graphDisabled = true;
        }

        public void Execute(
            int replicaIndex,
            CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            ObjectDisposedException.ThrowIf(
                Volatile.Read(ref _disposed) != 0,
                this);
            if (_phase == WorkPhase.Synchronize)
            {
                int deviceIndex = Devices[replicaIndex];
                ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                    .Synchronize(
                        $"data-parallel backward device {deviceIndex}");
                ShardElapsed[replicaIndex] =
                    System.Diagnostics.Stopwatch.GetElapsedTime(
                        ShardStarted[replicaIndex]).TotalMilliseconds;
                return;
            }
            if (_phase == WorkPhase.DisposeGraph)
            {
                DisposeDeviceGraph(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.InitializeGraphInputs)
            {
                InitializeDeviceGraphInputs(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.ReservationPrewarm)
            {
                ExecuteReservationPrewarm(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.Capture)
            {
                CaptureDeviceGraph(replicaIndex);
                return;
            }
            if (_phase == WorkPhase.Replay)
            {
                ReplayDeviceGraph(replicaIndex);
                return;
            }
            if (_phase != WorkPhase.Eager)
            {
                throw new InvalidOperationException(
                    "CUDA training shape work was not prepared.");
            }

            ExecuteEager(replicaIndex);
        }

        private void ExecuteEager(int replicaIndex)
        {
            ShardStarted[replicaIndex] =
                System.Diagnostics.Stopwatch.GetTimestamp();
            int device = Devices[replicaIndex];
            int shardBatch = ShardBatches[replicaIndex];
            int[] shardInput = ShardInputs[replicaIndex];
            int[] shardTarget = ShardTargets[replicaIndex];
            int shardValid = ShardValidTargets[replicaIndex];

            if (_beginReductionDeviceStep)
            {
                _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
                _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);
                _accumulatedLossBuffers?[replicaIndex].MemSetToZero();
            }
            using IDisposable? reductionScope = _reductionPlan is null
                ? null
                : CudaGradientReductionContext.Push(
                    _reductionPlan, device, _reductionStepId);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(device);
            CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope? inputScope = null;
            IDisposable? bfp8GraphPublicationScope = null;
            if (_useFixedInputs)
            {
                CudaGraphBatchInputs inputs = _graphInputs[replicaIndex]
                    ?? throw new InvalidOperationException(
                        "CUDA Graph inputs were not initialized.");
                inputScope = inputs.PushCaptureScope();
                bfp8GraphPublicationScope = _graphBfp8Publications[replicaIndex]
                    ?.PushRecording();
            }
            try
            {
                Tensor loss = _owner._model.ForwardLoss(
                    shardInput,
                    shardTarget,
                    shardBatch,
                    SequenceLength,
                    _ignoreIndex);
                if (_owner._dispatchPolicy.SynchronizeDataParallelPhases)
                {
                    accelerator.Synchronize(
                        $"data-parallel forward device {device}");
                }
                float weight = (float)shardValid / _totalValid;
                NativeCudaScalarReadback? readback = null;
                bool accumulatedLoss =
                    _accumulatedLossBuffers is not null;
                if (accumulatedLoss)
                {
                    NativeCudaBuffer<float> lossBuffer =
                        loss.EnsureCudaFloat32Buffer(device);
                    CudaTensorNative.Scale(
                        device,
                        lossBuffer.NativePtr,
                        1,
                        shardValid);
                    CudaTensorNative.Accumulate(
                        device,
                        lossBuffer.NativePtr,
                        _accumulatedLossBuffers![replicaIndex].NativePtr,
                        1);
                    if (_completeAccumulatedLoss)
                    {
                        readback = NativeCudaScalarReadback.Rent(device);
                        readback.Begin(
                            _accumulatedLossBuffers[replicaIndex].NativePtr,
                            accelerator.DefaultStream);
                    }
                }
                else if (_readEagerLoss)
                {
                    readback = NativeCudaScalarReadback.Rent(device);
                    readback.Begin(
                        loss.EnsureCudaFloat32Buffer(device).NativePtr,
                        accelerator.DefaultStream);
                }
                if (_owner._dispatchPolicy.SynchronizeDataParallelPhases)
                {
                    accelerator.Synchronize(
                        $"data-parallel loss device {device}");
                }
                loss.BackwardAndRelease([weight]);
                if (_owner._dispatchPolicy.SynchronizeDataParallelPhases)
                {
                    accelerator.Synchronize(
                        $"data-parallel backward device {device}");
                }
                WeightedLosses[replicaIndex] = readback is null
                    ? 0d
                    : readback.CompleteAndReturn()
                        * (accumulatedLoss ? 1 : shardValid);
            }
            finally
            {
                bfp8GraphPublicationScope?.Dispose();
                inputScope?.Dispose();
            }
            ShardElapsed[replicaIndex] =
                System.Diagnostics.Stopwatch.GetElapsedTime(
                    ShardStarted[replicaIndex]).TotalMilliseconds;
        }

        private void InitializeDeviceGraphInputs(int replicaIndex)
        {
            int device = Devices[replicaIndex];
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    device,
                    out IStreamExecutionLane streamLane)
                || streamLane is not CudaExecutionLane lane)
            {
                throw new NotSupportedException(
                    $"CUDA Graph inputs require a session lane for device {device}.");
            }
            if (_graphInputs[replicaIndex] is not null)
                return;
            CudaGraphBatchInputs inputs = CudaGraphBatchInputs.Create(
                lane,
                ShardInputs[replicaIndex].Length,
                _owner._model.VocabularySize,
                _ignoreIndex);
            CudaGraphRngState? rng = null;
            NativeCudaBuffer<float>? lossSlot = null;
            CudaBfp8GraphGradientPublication? bfp8Publication = null;
            try
            {
                rng = CudaGraphRngState.Create(lane);
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(device);
                lossSlot = accelerator.Allocate1D<float>(
                    1,
                    CudaMemoryKind.Persistent);
                if (_owner._model.PrecisionMode == TensorPrecisionMode.Bfp8
                    && Devices.Length == 1)
                {
                    bfp8Publication = new CudaBfp8GraphGradientPublication(
                        device,
                        _owner._parameters);
                }
                inputs.Update(
                    ShardInputs[replicaIndex],
                    ShardTargets[replicaIndex],
                    lane.ComputeStreamHandle);
                _graphInputs[replicaIndex] = inputs;
                _graphRngStates[replicaIndex] = rng;
                _graphLossSlots[replicaIndex] = lossSlot;
                _graphBfp8Publications[replicaIndex] = bfp8Publication;
            }
            catch (Exception initializationFailure)
            {
                var failures = new List<Exception> { initializationFailure };
                TryDispose(bfp8Publication, failures);
                TryDispose(lossSlot, failures);
                TryDispose(rng, failures);
                TryDispose(inputs, failures);
                if (failures.Count == 1)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo
                        .Capture(initializationFailure)
                        .Throw();
                }
                throw new AggregateException(
                    "CUDA Graph input-state initialization and cleanup failed.",
                    failures);
            }
        }

        private void ExecuteReservationPrewarm(int replicaIndex)
        {
            int device = Devices[replicaIndex];
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    device,
                    out IStreamExecutionLane streamLane)
                || streamLane is not CudaExecutionLane lane)
            {
                throw new NotSupportedException(
                    $"CUDA Graph reservation requires a session lane for device {device}.");
            }
            if (_captureMemoryScopes[replicaIndex] is not null)
            {
                throw new InvalidOperationException(
                    "CUDA Graph memory reservation prewarm was already started.");
            }
            CudaGraphBatchInputs inputs = _graphInputs[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph inputs were not initialized.");
            CudaGraphRngState rng = _graphRngStates[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph RNG was not initialized.");
            CudaGraphCaptureScope captureMemory =
                lane.Memory.BeginGraphCaptureReservation();
            _capturePinnedBaselines[replicaIndex] =
                lane.Memory.GraphPinnedBytes;
            _captureMemoryScopes[replicaIndex] = captureMemory;
            rng.SetCounter(DeriveCounter(_rootSeed, _globalStep, device));
            _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
            _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);
            using IDisposable? reductionScope = _reductionPlan is null
                ? null
                : CudaGradientReductionContext.Push(
                    _reductionPlan,
                    device,
                    _reductionStepId);
            using IDisposable? recording =
                BeginCapturedBackwardRecording(
                    device,
                    CudaCapturedBackwardRecordingMode.ReservationPrewarm);
            using CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope
                inputScope = inputs.PushCaptureScope();
            using CudaGraphDropoutCaptureScope dropoutScope =
                CudaGraphDropoutCaptureScope.Begin(
                    rng,
                    DeriveOperationSeed(_rootSeed, device));
            using IDisposable? bfp8PublicationScope =
                _graphBfp8Publications[replicaIndex]?.PushRecording();
            Tensor loss = _owner._model.ForwardLoss(
                ShardInputs[replicaIndex],
                ShardTargets[replicaIndex],
                ShardBatches[replicaIndex],
                SequenceLength,
                _ignoreIndex);
            float weight = (float)ShardValidTargets[replicaIndex]
                / _totalValid;
            loss.BackwardAndRelease([weight]);
            lane.SynchronizeComputeStream();
        }

        private void CaptureDeviceGraph(int replicaIndex)
        {
            int device = Devices[replicaIndex];
            if (!TensorExecutionContext.TryGetCudaStreamLane(
                    device,
                    out IStreamExecutionLane streamLane)
                || streamLane is not CudaExecutionLane lane)
            {
                throw new NotSupportedException(
                    $"CUDA Graph capture requires a session lane for device {device}.");
            }

            _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
            _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);

            CudaGraphBatchInputs inputs = _graphInputs[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph inputs were not initialized before prewarm.");
            CudaGraphRngState rng = _graphRngStates[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph RNG was not initialized before capture.");
            NativeCudaBuffer<float> lossSlot = _graphLossSlots[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph loss slot was not initialized before capture.");
            CudaGraphExecutable? graph = null;
            IDisposable? memoryReservation = null;
            CudaGraphCaptureScope captureMemory =
                _captureMemoryScopes[replicaIndex]
                ?? throw new InvalidOperationException(
                    "CUDA Graph capture memory was not reservation-prewarmed.");
            long pinnedBefore = _capturePinnedBaselines[replicaIndex];
            try
            {
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(device);
                rng.SetCounter(DeriveCounter(
                    _rootSeed,
                    _globalStep,
                    device));
                lane.SynchronizeComputeStream();

                using IDisposable? reductionScope = _reductionPlan is null
                    ? null
                    : CudaGradientReductionContext.Push(
                        _reductionPlan,
                        device,
                        _reductionStepId);
                graph = CudaGraphExecutable.Capture(lane, () =>
                {
                    using IDisposable? recording =
                        BeginCapturedBackwardRecording(
                            device,
                            CudaCapturedBackwardRecordingMode.StreamCapture);
                    CudaGraphBatchInputs.CudaGraphBatchInputCaptureScope
                        inputScope = inputs.PushCaptureScope();
                    try
                    {
                        using CudaGraphBfp8ParameterRefreshScope
                            parameterRefreshScope =
                                CudaGraphBfp8ParameterRefreshScope.Begin(
                                    device);
                        using CudaGraphDropoutCaptureScope dropoutScope =
                            CudaGraphDropoutCaptureScope.Begin(
                                rng,
                                DeriveOperationSeed(_rootSeed, device));
                        using IDisposable? bfp8PublicationScope =
                            _graphBfp8Publications[replicaIndex]
                                ?.PushRecording();
                        Tensor loss = _owner._model.ForwardLoss(
                            ShardInputs[replicaIndex],
                            ShardTargets[replicaIndex],
                            ShardBatches[replicaIndex],
                            SequenceLength,
                            _ignoreIndex);
                        nint lossPointer = loss
                            .EnsureCudaFloat32Buffer(device)
                            .NativePtr;
                        NativeCudaRuntime.Check(
                            NativeCudaRuntime.CopyDeviceToDeviceAsyncNative(
                                device,
                                lossSlot.NativePtr,
                                device,
                                lossPointer,
                                sizeof(float),
                                lane.ComputeStreamHandle),
                            "capture CUDA loss scalar snapshot");
                        float weight = (float)ShardValidTargets[replicaIndex]
                            / _totalValid;
                        loss.BackwardAndRelease([weight]);
                    }
                    catch (Exception recordFailure)
                    {
                        try
                        {
                            inputScope.Dispose();
                        }
                        catch (Exception scopeFailure)
                        {
                            throw new AggregateException(
                                "CUDA Graph record and input-scope cleanup failed.",
                                recordFailure,
                                scopeFailure);
                        }
                        throw;
                    }
                    inputScope.Dispose();
                });
                memoryReservation = captureMemory.Commit();
                captureMemory.Dispose();
                _captureMemoryScopes[replicaIndex] = null;
                long pinnedBytes = Math.Max(
                    0,
                    lane.Memory.GraphPinnedBytes - pinnedBefore);
                int[]? bfloat16BucketOrder = _bfloat16Plan
                    ?.SnapshotCapturedBucketOrderForGraph(device);
                _compiledGraphs[replicaIndex] = new CompiledDeviceGraph(
                    lane,
                    graph,
                    memoryReservation,
                    lossSlot,
                    rng,
                    inputs,
                    _graphBfp8Publications[replicaIndex],
                    bfloat16BucketOrder,
                    pinnedBytes);
                _graphInputs[replicaIndex] = null;
                _graphRngStates[replicaIndex] = null;
                _graphLossSlots[replicaIndex] = null;
                _graphBfp8Publications[replicaIndex] = null;
                graph = null;
                memoryReservation = null;
            }
            catch (Exception captureFailure)
            {
                var failures = new List<Exception> { captureFailure };
                TryDispose(graph, failures);
                TryDispose(memoryReservation, failures);
                if (failures.Count == 1)
                {
                    System.Runtime.ExceptionServices.ExceptionDispatchInfo
                        .Capture(captureFailure)
                        .Throw();
                }
                throw new AggregateException(
                    "CUDA Graph capture cleanup failed.",
                    failures);
            }
        }

        private IDisposable? BeginCapturedBackwardRecording(
            int device,
            CudaCapturedBackwardRecordingMode recordingMode)
        {
            if (_bfp8Plan is not null)
            {
                return _bfp8Plan.BeginCapturedBackwardRecording(
                    _reductionStepId,
                    device);
            }
            return _bfloat16Plan?.BeginCapturedBackwardRecording(
                _reductionStepId,
                device,
                recordingMode);
        }

        private void ReplayDeviceGraph(int replicaIndex)
        {
            CompiledDeviceGraph compiled = _compiledGraphs[replicaIndex]
                ?? throw new InvalidOperationException(
                    "The CUDA device graph is not compiled.");
            int device = Devices[replicaIndex];
            ShardStarted[replicaIndex] =
                System.Diagnostics.Stopwatch.GetTimestamp();
            _bfp8Plan?.BeginDeviceStep(_reductionStepId, device);
            _bfloat16Plan?.BeginDeviceStep(_reductionStepId, device);
            if (!_reuseCaptureInputs)
            {
                compiled.Inputs.Update(
                    ShardInputs[replicaIndex],
                    ShardTargets[replicaIndex],
                    compiled.Lane.ComputeStreamHandle);
            }
            compiled.Rng.SetCounter(DeriveCounter(
                _rootSeed,
                _globalStep,
                device));
            compiled.Graph.Launch();
            NativeCudaScalarReadback readback =
                NativeCudaScalarReadback.Rent(device);
            readback.Begin(
                compiled.LossSlot.NativePtr,
                compiled.Lane.ComputeStreamHandle);
            if (_bfp8Plan is not null)
            {
                _bfp8Plan.PublishCapturedDeviceGradientsAfterReplay(
                    _reductionStepId,
                    device);
            }
            else if (_bfloat16Plan is not null)
            {
                _bfloat16Plan.PublishCapturedDeviceGradientsForReplay(
                    _reductionStepId,
                    device,
                    compiled.BFloat16BucketOrder);
            }
            float lossValue = readback.CompleteAndReturn();
            compiled.Bfp8Publication?.PublishAfterReplay();
            WeightedLosses[replicaIndex] =
                lossValue * ShardValidTargets[replicaIndex];
            ShardElapsed[replicaIndex] =
                System.Diagnostics.Stopwatch.GetElapsedTime(
                    ShardStarted[replicaIndex]).TotalMilliseconds;
        }

        private void DisposeDeviceGraph(int replicaIndex)
        {
            CompiledDeviceGraph? graph = Interlocked.Exchange(
                ref _compiledGraphs[replicaIndex],
                null);
            var failures = new List<Exception>();
            TryDispose(graph, failures);
            CudaGraphCaptureScope? captureMemory = Interlocked.Exchange(
                ref _captureMemoryScopes[replicaIndex],
                null);
            TryDispose(captureMemory, failures);
            NativeCudaBuffer<float>? lossSlot = Interlocked.Exchange(
                ref _graphLossSlots[replicaIndex],
                null);
            TryDispose(lossSlot, failures);
            CudaBfp8GraphGradientPublication? bfp8Publication =
                Interlocked.Exchange(
                    ref _graphBfp8Publications[replicaIndex],
                    null);
            TryDispose(bfp8Publication, failures);
            CudaGraphRngState? rng = Interlocked.Exchange(
                ref _graphRngStates[replicaIndex],
                null);
            TryDispose(rng, failures);
            CudaGraphBatchInputs? inputs = Interlocked.Exchange(
                ref _graphInputs[replicaIndex],
                null);
            TryDispose(inputs, failures);
            if (failures.Count != 0)
            {
                throw new AggregateException(
                    "CUDA device graph cleanup failed.",
                    failures);
            }
        }

        internal bool Matches(
            int batchSize,
            int sequenceLength,
            IReadOnlyList<int> devices,
            IReadOnlyList<int> shardBatches)
        {
            if (BatchSize != batchSize
                || SequenceLength != sequenceLength
                || Devices.Length != devices.Count
                || ShardBatches.Length != shardBatches.Count)
            {
                return false;
            }
            for (int index = 0; index < Devices.Length; index++)
            {
                if (Devices[index] != devices[index]
                    || ShardBatches[index] != shardBatches[index])
                {
                    return false;
                }
            }
            return true;
        }

        internal void PrepareBatch(
            ReadOnlySpan<int> input,
            ReadOnlySpan<int> target,
            int ignoreIndex)
        {
            Array.Clear(WeightedLosses);
            Array.Clear(ShardElapsed);
            Array.Clear(ShardStarted);
            for (int shard = 0; shard < Devices.Length; shard++)
            {
                int elementStart = checked(
                    ShardStarts[shard] * SequenceLength);
                int elementCount = ShardInputs[shard].Length;
                input.Slice(elementStart, elementCount)
                    .CopyTo(ShardInputs[shard]);
                target.Slice(elementStart, elementCount)
                    .CopyTo(ShardTargets[shard]);
                int valid = 0;
                foreach (int value in ShardTargets[shard])
                {
                    if (value != ignoreIndex)
                        valid++;
                }
                ShardValidTargets[shard] = valid;
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;
            var failures = new List<Exception>();
            for (int device = 0; device < _compiledGraphs.Length; device++)
            {
                try
                {
                    DisposeDeviceGraph(device);
                }
                catch (Exception exception)
                {
                    failures.Add(exception);
                }
            }
            if (failures.Count != 0)
            {
                throw new AggregateException(
                    "CUDA training shape plan cleanup failed.",
                    failures);
            }
        }

        private static ulong DeriveCounter(
            ulong rootSeed,
            long globalStep,
            int deviceIndex)
            => Mix64(
                rootSeed
                ^ unchecked((ulong)globalStep * 0x9e3779b97f4a7c15UL)
                ^ unchecked((ulong)(uint)deviceIndex
                    * 0xbf58476d1ce4e5b9UL));

        private static ulong DeriveOperationSeed(
            ulong rootSeed,
            int deviceIndex)
            => Mix64(
                rootSeed
                ^ 0x4752_4150_4852_4e47UL
                ^ unchecked((ulong)(uint)deviceIndex
                    * 0x94d049bb133111ebUL));

        private static ulong Mix64(ulong value)
        {
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL;
            return value ^ (value >> 31);
        }

        private static void TryDispose(
            IDisposable? resource,
            List<Exception> failures)
        {
            if (resource is null)
                return;
            try
            {
                resource.Dispose();
            }
            catch (Exception exception)
            {
                failures.Add(exception);
            }
        }

        private sealed class CompiledDeviceGraph(
            CudaExecutionLane lane,
            CudaGraphExecutable graph,
            IDisposable memoryReservation,
            NativeCudaBuffer<float> lossSlot,
            CudaGraphRngState rng,
            CudaGraphBatchInputs inputs,
            CudaBfp8GraphGradientPublication? bfp8Publication,
            int[]? bfloat16BucketOrder,
            long graphPinnedBytes) : IDisposable
        {
            private int _disposed;

            internal CudaExecutionLane Lane { get; } = lane;
            internal CudaGraphExecutable Graph { get; } = graph;
            internal NativeCudaBuffer<float> LossSlot { get; } = lossSlot;
            internal CudaGraphRngState Rng { get; } = rng;
            internal CudaGraphBatchInputs Inputs { get; } = inputs;
            internal CudaBfp8GraphGradientPublication? Bfp8Publication
                { get; } = bfp8Publication;
            internal int[]? BFloat16BucketOrder { get; }
                = bfloat16BucketOrder;
            internal long GraphPinnedBytes { get; } = graphPinnedBytes;

            public void Dispose()
            {
                if (Interlocked.Exchange(ref _disposed, 1) != 0)
                    return;
                var failures = new List<Exception>();
                try
                {
                    Lane.SynchronizeComputeStream();
                }
                catch (Exception exception)
                {
                    failures.Add(exception);
                }
                TryDispose(Graph, failures);
                TryDispose(memoryReservation, failures);
                TryDispose(Bfp8Publication, failures);
                TryDispose(LossSlot, failures);
                TryDispose(Rng, failures);
                TryDispose(Inputs, failures);
                if (failures.Count != 0)
                {
                    throw new AggregateException(
                        "CUDA compiled training graph cleanup failed.",
                        failures);
                }
            }
        }

        private enum WorkPhase
        {
            None = 0,
            Eager = 1,
            Synchronize = 2,
            Capture = 3,
            Replay = 4,
            DisposeGraph = 5,
            InitializeGraphInputs = 6,
            ReservationPrewarm = 7,
        }
    }

}
