using NNtrain.Runtime.Execution;

namespace NNtrain;

/// <summary>
/// Owns the reusable CUDA data-parallel resources for one language model.
/// Dispose the engine before ending a training session so gradient arenas,
/// reduction buffers, streams, and events are released deterministically.
/// </summary>
public sealed class CudaDataParallelEngine : IDisposable
{
    private const string CudaSyncPhasesEnvironmentVariable =
        "NNTRAIN_CUDA_SYNC_PHASES";
    private static readonly bool SynchronizeCudaPhases = string.Equals(
        Environment.GetEnvironmentVariable(CudaSyncPhasesEnvironmentVariable),
        "1",
        StringComparison.Ordinal);

    private readonly object _sync = new();
    private readonly LanguageModel _model;
    private readonly int[] _cudaDeviceIndices;
    private CudaAdaptiveShardScheduler _adaptiveShardScheduler;
    private TensorCudaKernels.FlatGradientPlan? _flatGradientPlan;
    private CudaBFloat16GradientAllReducePlan? _bfloat16GradientPlan;
    private bool _parameterResidencyPrepared;
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
        CudaAdaptiveShardingOptions options = adaptiveShardingOptions
            ?? new CudaAdaptiveShardingOptions();
        options.Validate();
        _adaptiveShardScheduler = new CudaAdaptiveShardScheduler(options);
    }

    private void PrepareParameterResidency()
    {
        if (_parameterResidencyPrepared)
            return;
        if (_model.PrecisionMode == TensorPrecisionMode.Bfp8)
        {
            throw new NotSupportedException(
                "Pure BFP8 CUDA data-parallel training requires a BFP8 " +
                "gradient reducer, which is not implemented. Use mix8_32 " +
                "for FP32 gradient accumulation and reduction.");
        }
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

    public LanguageModel Model => _model;

    public IReadOnlyList<int> CudaDeviceIndices
        => Array.AsReadOnly((int[])_cudaDeviceIndices.Clone());

    internal bool UsesCudaDevices(IReadOnlyList<int> cudaDeviceIndices)
        => _cudaDeviceIndices.SequenceEqual(cudaDeviceIndices);

    public bool IsDisposed => Volatile.Read(ref _disposed) != 0;

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

    /// <summary>Resets EMA history and applies new adaptive shard bounds.</summary>
    public void ConfigureAdaptiveSharding(
        CudaAdaptiveShardingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        lock (_sync)
        {
            ThrowIfDisposed();
            _adaptiveShardScheduler = new CudaAdaptiveShardScheduler(options);
        }
    }

    public float ForwardBackward(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
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
                ignoreIndex);
        }
    }

    /// <summary>
    /// Diagnostic variant that synchronizes CUDA after every major phase.
    /// This is intentionally separate from the normal asynchronous path.
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
            Parameter[] parameters = _model.Parameters().ToArray();
            CudaBFloat16GradientAllReducePlan? bfloat16Plan =
                GetBFloat16GradientPlan(parameters, devices);
            if (bfloat16Plan is null)
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
            long bfloat16StepId = bfloat16Plan?.BeginStep() ?? 0;
            SynchronizeAll();
            prepareTimer.Stop();

            var weightedLosses = new double[devices.Length];
            var shards = new CudaShardProfile[devices.Length];
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
                bfloat16Plan?.BeginDeviceStep(
                    bfloat16StepId, devices[shard]);
                using IDisposable? reductionScope = bfloat16Plan is null
                    ? null
                    : CudaGradientReductionContext.Push(
                        bfloat16Plan, devices[shard], bfloat16StepId);
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(devices[shard]);

                var phaseTimer = System.Diagnostics.Stopwatch.StartNew();
                Tensor logits = _model.Forward(
                    shardInput, shardBatch, sequenceLength);
                accelerator.Synchronize();
                double forward = phaseTimer.Elapsed.TotalMilliseconds;

                phaseTimer.Restart();
                Tensor loss = logits.CrossEntropyWithLogits(
                    shardTarget,
                    ignoreIndex: ignoreIndex);
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

            var allReduceTimer = System.Diagnostics.Stopwatch.StartNew();
            if (bfloat16Plan is not null)
            {
                bfloat16Plan.Complete(bfloat16StepId);
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

            if (failures is not null)
            {
                throw new AggregateException(
                    "One or more CUDA data-parallel resources failed to dispose.",
                    failures);
            }
        }
    }

    private float ForwardBackwardCore(
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex)
    {
        ValidateBatch(input, target, batchSize, sequenceLength);
        int[] devices = GetDevices(batchSize);
        if (devices.Length == 1)
        {
            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[0]));
            Tensor logits = _model.Forward(input, batchSize, sequenceLength);
            Tensor loss = logits.CrossEntropyWithLogits(
                target, ignoreIndex: ignoreIndex);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(devices[0]);
            NativeCudaScalarReadback readback =
                NativeCudaScalarReadback.Rent(devices[0]);
            readback.Begin(
                loss.EnsureCudaFloat32Buffer(devices[0]).NativePtr,
                accelerator.DefaultStream);
            loss.BackwardAndRelease();
            accelerator.Synchronize(
                $"data-parallel backward device {devices[0]}");
            return readback.CompleteAndReturn();
        }

        int[] shardBatches = AllocateShardBatches(batchSize, devices);
        int[] shardStarts = GetShardStarts(shardBatches);
        Parameter[] parameters = _model.Parameters().ToArray();
        CudaBFloat16GradientAllReducePlan? bfloat16Plan =
            GetBFloat16GradientPlan(parameters, devices);
        if (bfloat16Plan is null)
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
        long bfloat16StepId = bfloat16Plan?.BeginStep() ?? 0;
        var weightedLosses = new double[devices.Length];
        var shardElapsed = new double[devices.Length];
        var shardStarted = new long[devices.Length];
        Parallel.For(0, devices.Length, shard =>
        {
            shardStarted[shard] =
                System.Diagnostics.Stopwatch.GetTimestamp();
            int batchStart = shardStarts[shard];
            int shardBatch = shardBatches[shard];
            int elementStart = batchStart * sequenceLength;
            int elementCount = shardBatch * sequenceLength;
            int[] shardInput = input.AsSpan(
                elementStart, elementCount).ToArray();
            int[] shardTarget = target.AsSpan(
                elementStart, elementCount).ToArray();
            int shardValid = shardTarget.Count(value => value != ignoreIndex);

            using IDisposable shardPrecisionScope =
                TensorExecutionContext.PushPrecisionPolicy(
                    ResolvePrecisionPolicy(_model.PrecisionMode));
            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[shard]));
            bfloat16Plan?.BeginDeviceStep(
                bfloat16StepId, devices[shard]);
            using IDisposable? reductionScope = bfloat16Plan is null
                ? null
                : CudaGradientReductionContext.Push(
                    bfloat16Plan, devices[shard], bfloat16StepId);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard]);
            Tensor logits = _model.Forward(
                shardInput,
                shardBatch,
                sequenceLength);
            if (SynchronizeCudaPhases)
            {
                accelerator.Synchronize(
                    $"data-parallel forward device {devices[shard]}");
            }
            Tensor loss = logits.CrossEntropyWithLogits(
                shardTarget,
                ignoreIndex: ignoreIndex);
            float weight = (float)shardValid / totalValid;
            NativeCudaScalarReadback readback =
                NativeCudaScalarReadback.Rent(devices[shard]);
            readback.Begin(
                loss.EnsureCudaFloat32Buffer(devices[shard]).NativePtr,
                accelerator.DefaultStream);
            if (SynchronizeCudaPhases)
            {
                accelerator.Synchronize(
                    $"data-parallel loss device {devices[shard]}");
            }
            loss.BackwardAndRelease([weight]);
            if (SynchronizeCudaPhases)
            {
                accelerator.Synchronize(
                    $"data-parallel backward device {devices[shard]}");
            }
            weightedLosses[shard] =
                readback.CompleteAndReturn() * shardValid;
            shardElapsed[shard] =
                System.Diagnostics.Stopwatch.GetElapsedTime(
                    shardStarted[shard]).TotalMilliseconds;
        });

        bool canMeasureShardRuntime = bfloat16Plan is null
            || bfloat16Plan.DefersExchangeUntilBackward;
        if (canMeasureShardRuntime)
        {
            // The scalar readback is queued before backward. Synchronizing all
            // devices in parallel captures real shard runtime without adding
            // a serial wait to the non-peer exchange path.
            Parallel.For(0, devices.Length, shard =>
            {
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard])
                    .Synchronize(
                        $"data-parallel backward device {devices[shard]}");
                shardElapsed[shard] =
                    System.Diagnostics.Stopwatch.GetElapsedTime(
                        shardStarted[shard]).TotalMilliseconds;
            });
        }

        if (canMeasureShardRuntime)
            _adaptiveShardScheduler.Observe(shardBatches, shardElapsed);

        if (bfloat16Plan is not null)
        {
            bfloat16Plan.Complete(bfloat16StepId);
        }
        else
        {
            TensorCudaKernels.FlatGradientPlan plan =
                GetFlatGradientPlan(parameters, devices);
            TensorCudaKernels.AllReduceGradientsResident(
                parameters, devices, plan);
        }
        return (float)(weightedLosses.Sum() / totalValid);
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
        => _cudaDeviceIndices
            .Take(Math.Min(batchSize, _cudaDeviceIndices.Length))
            .ToArray();

    private int[] AllocateShardBatches(
        int batchSize,
        IReadOnlyList<int> devices)
    {
        int[] previous = _adaptiveShardScheduler.LastAllocation;
        int[] current = _adaptiveShardScheduler.Allocate(batchSize, devices);
        if (previous.Length == current.Length
            && !previous.SequenceEqual(current))
        {
            // A shard transition changes activation lengths. Retire the old
            // exact-shape transient generation before it accumulates in VRAM.
            foreach (int device in devices)
                Tensor.ClearCudaFloatBufferPool(device);
        }
        return current;
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
        if (_bfloat16GradientPlan is null
            || !_bfloat16GradientPlan.Matches(parameters, devices))
        {
            _bfloat16GradientPlan?.Dispose();
            _bfloat16GradientPlan =
                new CudaBFloat16GradientAllReducePlan(parameters, devices);
        }
        return _bfloat16GradientPlan;
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

    private static bool UseBFloat16GradientBuckets(
        IReadOnlyList<int> devices,
        TensorPrecisionMode precisionMode)
        => devices.Count == 2
            && ResolvePrecisionPolicy(precisionMode).Gradient
                == NumericFormat.BFloat16
            && !string.Equals(
                Environment.GetEnvironmentVariable(
                    "NNTRAIN_DISABLE_BF16_GRADIENT_BUCKETS"),
                "1",
                StringComparison.Ordinal);

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
}
