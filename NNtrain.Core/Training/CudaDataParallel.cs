namespace NNtrain;

/// <summary>CUDA data-parallel forward/backward for language-model batches.</summary>
public static class CudaDataParallel
{
    private const string CudaSyncPhasesEnvironmentVariable =
        "NNTRAIN_CUDA_SYNC_PHASES";
    private static readonly bool SynchronizeCudaPhases = string.Equals(
        Environment.GetEnvironmentVariable(CudaSyncPhasesEnvironmentVariable),
        "1",
        StringComparison.Ordinal);
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        LanguageModel,
        FlatGradientPlanCache> FlatGradientPlans = new();
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        LanguageModel,
        BFloat16GradientPlanCache> BFloat16GradientPlans = new();
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        LanguageModel,
        CudaAdaptiveShardScheduler> AdaptiveShardSchedulers = new();
    private static CudaAdaptiveShardingOptions _adaptiveShardingOptions = new();

    /// <summary>Configures EMA-based CUDA batch balancing.</summary>
    public static void ConfigureAdaptiveSharding(
        CudaAdaptiveShardingOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        options.Validate();
        Volatile.Write(ref _adaptiveShardingOptions, options);
        AdaptiveShardSchedulers.Clear();
    }

    /// <summary>Returns the last per-device batch allocation for a model.</summary>
    public static IReadOnlyList<int> GetLastShardBatchSizes(LanguageModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        return AdaptiveShardSchedulers.TryGetValue(model, out var scheduler)
            ? scheduler.LastAllocation
            : [];
    }

    public static float ForwardBackward(
        LanguageModel model,
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(target);
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            throw new InvalidOperationException("CUDA execution must be selected.");
        if (input.Length != target.Length
            || input.Length != checked(batchSize * sequenceLength))
        {
            throw new ArgumentException("Input and target must match the batch shape.");
        }

        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(batchSize, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        if (devices.Length == 1)
        {
            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[0]));
            Tensor logits = model.Forward(input, batchSize, sequenceLength);
            Tensor loss = logits.CrossEntropyWithLogits(target, ignoreIndex: ignoreIndex);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(devices[0]);
            NativeCudaScalarReadback readback =
                NativeCudaScalarReadback.Rent(devices[0]);
            readback.Begin(
                loss.EnsureCudaFloat32Buffer(devices[0]).NativePtr,
                accelerator.DefaultStream);
            loss.BackwardAndRelease();
            return readback.CompleteAndReturn();
        }

        CudaAdaptiveShardScheduler shardScheduler =
            GetAdaptiveShardScheduler(model);
        int[] shardBatches = AllocateShardBatches(
            shardScheduler, batchSize, devices);
        int[] shardStarts = GetShardStarts(shardBatches);

        Parameter[] parameters = model.Parameters().ToArray();
        CudaBFloat16GradientAllReducePlan? bfloat16Plan = UseBFloat16GradientBuckets(
            devices, model.PrecisionMode)
            ? BFloat16GradientPlans.GetValue(
                model, _ => new BFloat16GradientPlanCache())
                .Get(parameters, devices)
            : null;
        if (bfloat16Plan is null)
        {
            foreach (Parameter parameter in parameters)
                parameter.T.PrepareCudaGradientBuffers(devices);
        }
        bfloat16Plan?.BeginStep();

        int totalValid = target.Count(value => value != ignoreIndex);
        if (totalValid == 0)
            throw new ArgumentException("At least one target must be valid.", nameof(target));
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
            int[] shardInput = input.AsSpan(elementStart, elementCount).ToArray();
            int[] shardTarget = target.AsSpan(elementStart, elementCount).ToArray();
            int shardValid = shardTarget.Count(value => value != ignoreIndex);

            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[shard]));
            using IDisposable? reductionScope = bfloat16Plan is null
                ? null
                : CudaGradientReductionContext.Push(
                    bfloat16Plan, devices[shard]);
            NativeCudaDevice accelerator =
                ForgetMemoryV2Cuda.GetAccelerator(devices[shard]);
            Tensor logits = model.Forward(
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
            shardElapsed[shard] = System.Diagnostics.Stopwatch.GetElapsedTime(
                shardStarted[shard]).TotalMilliseconds;
        });

        bool canMeasureShardRuntime = bfloat16Plan is null
            || bfloat16Plan.DefersExchangeUntilBackward;
        if (canMeasureShardRuntime)
        {
            // The scalar readback is intentionally queued before backward, so
            // CPU task duration measures enqueue time rather than GPU work.
            // Non-peer exchange already has to wait for both completed packs;
            // synchronize in parallel here to obtain real per-device runtimes
            // without adding another serialization point to that path.
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
            shardScheduler.Observe(shardBatches, shardElapsed);

        if (bfloat16Plan is not null)
        {
            bfloat16Plan.Complete();
        }
        else
        {
            FlatGradientPlanCache cache = FlatGradientPlans.GetValue(
                model, _ => new FlatGradientPlanCache());
            TensorCudaKernels.FlatGradientPlan plan = cache.Get(parameters, devices);
            TensorCudaKernels.AllReduceGradientsResident(parameters, devices, plan);
        }
        return (float)(weightedLosses.Sum() / totalValid);
    }

    /// <summary>
    /// Diagnostic variant that synchronizes CUDA after every major phase.
    /// This is intentionally separate from the normal asynchronous path.
    /// </summary>
    internal static CudaDataParallelProfile ForwardBackwardProfiled(
        LanguageModel model,
        int[] input,
        int[] target,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentNullException.ThrowIfNull(target);
        if (Tensor.ExecutionDevice != TensorDevice.Cuda)
            throw new InvalidOperationException("CUDA execution must be selected.");
        if (input.Length != target.Length
            || input.Length != checked(batchSize * sequenceLength))
        {
            throw new ArgumentException("Input and target must match the batch shape.");
        }

        int[] devices = Tensor.CudaDeviceIndices
            .Take(Math.Min(batchSize, Tensor.CudaDeviceIndices.Count))
            .ToArray();
        CudaAdaptiveShardScheduler shardScheduler =
            GetAdaptiveShardScheduler(model);
        int[] shardBatches = AllocateShardBatches(
            shardScheduler, batchSize, devices);
        int[] shardStarts = GetShardStarts(shardBatches);
        void SynchronizeAll()
        {
            foreach (int device in devices)
                ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
        }

        SynchronizeAll();
        var totalTimer = System.Diagnostics.Stopwatch.StartNew();
        var prepareTimer = System.Diagnostics.Stopwatch.StartNew();
        Parameter[] parameters = model.Parameters().ToArray();
        CudaBFloat16GradientAllReducePlan? bfloat16Plan = UseBFloat16GradientBuckets(
            devices, model.PrecisionMode)
            ? BFloat16GradientPlans.GetValue(
                model, _ => new BFloat16GradientPlanCache())
                .Get(parameters, devices)
            : null;
        if (bfloat16Plan is null)
        {
            foreach (Parameter parameter in parameters)
                parameter.T.PrepareCudaGradientBuffers(devices);
        }
        bfloat16Plan?.BeginStep();
        SynchronizeAll();
        prepareTimer.Stop();

        int totalValid = target.Count(value => value != ignoreIndex);
        if (totalValid == 0)
            throw new ArgumentException("At least one target must be valid.", nameof(target));

        var weightedLosses = new double[devices.Length];
        var shards = new CudaShardProfile[devices.Length];
        Parallel.For(0, devices.Length, shard =>
        {
            var shardTimer = System.Diagnostics.Stopwatch.StartNew();
            int batchStart = shardStarts[shard];
            int shardBatch = shardBatches[shard];
            int elementStart = batchStart * sequenceLength;
            int elementCount = shardBatch * sequenceLength;
            int[] shardInput = input.AsSpan(elementStart, elementCount).ToArray();
            int[] shardTarget = target.AsSpan(elementStart, elementCount).ToArray();
            int shardValid = shardTarget.Count(value => value != ignoreIndex);
            double dataPreparation = shardTimer.Elapsed.TotalMilliseconds;

            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[shard]));
            using IDisposable? reductionScope = bfloat16Plan is null
                ? null
                : CudaGradientReductionContext.Push(
                    bfloat16Plan, devices[shard]);
            var accelerator = ForgetMemoryV2Cuda.GetAccelerator(devices[shard]);

            var phaseTimer = System.Diagnostics.Stopwatch.StartNew();
            Tensor logits = model.Forward(shardInput, shardBatch, sequenceLength);
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
            bfloat16Plan.Complete();
        }
        else
        {
            FlatGradientPlanCache cache = FlatGradientPlans.GetValue(
                model, _ => new FlatGradientPlanCache());
            TensorCudaKernels.FlatGradientPlan plan = cache.Get(parameters, devices);
            TensorCudaKernels.AllReduceGradientsResident(parameters, devices, plan);
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

    private static CudaAdaptiveShardScheduler GetAdaptiveShardScheduler(
        LanguageModel model)
        => AdaptiveShardSchedulers.GetValue(
            model,
            _ => new CudaAdaptiveShardScheduler(
                Volatile.Read(ref _adaptiveShardingOptions)));

    private static int[] AllocateShardBatches(
        CudaAdaptiveShardScheduler scheduler,
        int batchSize,
        IReadOnlyList<int> devices)
    {
        int[] previous = scheduler.LastAllocation;
        int[] current = scheduler.Allocate(batchSize, devices);
        if (previous.Length == current.Length
            && !previous.SequenceEqual(current))
        {
            // A shard change changes every activation length. Keeping all
            // prior-shape blocks in the exact-size transient pools eventually
            // fills VRAM and forces a multi-gigabyte emergency flush. Retire
            // the old generation at the deliberate transition instead. Once
            // the EMA converges, the new fixed shape remains allocation-free.
            foreach (int device in devices)
                Tensor.ClearCudaFloatBufferPool(device);
        }
        return current;
    }

    private static int[] GetShardStarts(IReadOnlyList<int> shardBatches)
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

    private sealed class FlatGradientPlanCache
    {
        private readonly object _sync = new();
        private TensorCudaKernels.FlatGradientPlan? _plan;

        internal TensorCudaKernels.FlatGradientPlan Get(
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<int> devices)
        {
            lock (_sync)
            {
                if (_plan is null || !_plan.Matches(parameters, devices))
                {
                    _plan?.Dispose();
                    _plan = new TensorCudaKernels.FlatGradientPlan(
                        parameters, devices);
                }
                return _plan;
            }
        }
    }

    private static bool UseBFloat16GradientBuckets(
        IReadOnlyList<int> devices,
        TensorPrecisionMode precisionMode)
        => devices.Count == 2
            && precisionMode != TensorPrecisionMode.Float32
            && !string.Equals(
                Environment.GetEnvironmentVariable(
                    "NNTRAIN_DISABLE_BF16_GRADIENT_BUCKETS"),
                "1",
                StringComparison.Ordinal);

    private sealed class BFloat16GradientPlanCache
    {
        private readonly object _sync = new();
        private CudaBFloat16GradientAllReducePlan? _plan;

        internal CudaBFloat16GradientAllReducePlan Get(
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<int> devices)
        {
            lock (_sync)
            {
                if (_plan is null || !_plan.Matches(parameters, devices))
                {
                    _plan?.Dispose();
                    _plan = new CudaBFloat16GradientAllReducePlan(
                        parameters, devices);
                }
                return _plan;
            }
        }
    }
}

internal readonly record struct CudaShardProfile(
    int Device,
    int BatchSize,
    double DataPreparationMilliseconds,
    double ForwardMilliseconds,
    double LossMilliseconds,
    double BackwardMilliseconds);

internal readonly record struct CudaDataParallelProfile(
    float Loss,
    double GradientPreparationMilliseconds,
    double AllReduceMilliseconds,
    double TotalMilliseconds,
    IReadOnlyList<CudaShardProfile> Shards);
