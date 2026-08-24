namespace NNtrain;

/// <summary>CUDA data-parallel forward/backward for language-model batches.</summary>
public static class CudaDataParallel
{
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<
        LanguageModel,
        FlatGradientPlanCache> FlatGradientPlans = new();

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
            float value = loss.item();
            loss.BackwardAndRelease();
            return value;
        }

        Parameter[] parameters = model.Parameters().ToArray();
        foreach (Parameter parameter in parameters)
            parameter.T.PrepareCudaGradientBuffers(devices);

        int totalValid = target.Count(value => value != ignoreIndex);
        if (totalValid == 0)
            throw new ArgumentException("At least one target must be valid.", nameof(target));
        var weightedLosses = new double[devices.Length];
        Parallel.For(0, devices.Length, shard =>
        {
            int batchStart = batchSize * shard / devices.Length;
            int batchEnd = batchSize * (shard + 1) / devices.Length;
            int shardBatch = batchEnd - batchStart;
            int elementStart = batchStart * sequenceLength;
            int elementCount = shardBatch * sequenceLength;
            int[] shardInput = input.AsSpan(elementStart, elementCount).ToArray();
            int[] shardTarget = target.AsSpan(elementStart, elementCount).ToArray();
            int shardValid = shardTarget.Count(value => value != ignoreIndex);

            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[shard]));
            Tensor logits = model.Forward(
                shardInput,
                shardBatch,
                sequenceLength);
            Tensor loss = logits.CrossEntropyWithLogits(
                shardTarget,
                ignoreIndex: ignoreIndex);
            float weight = (float)shardValid / totalValid;
            weightedLosses[shard] = loss.item() * shardValid;
            loss.BackwardAndRelease([weight]);
        });

        FlatGradientPlanCache cache = FlatGradientPlans.GetValue(
            model, _ => new FlatGradientPlanCache());
        TensorCudaKernels.FlatGradientPlan plan = cache.Get(parameters, devices);
        TensorCudaKernels.AllReduceGradientsResident(parameters, devices, plan);
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
        void SynchronizeAll()
        {
            foreach (int device in devices)
                ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
        }

        SynchronizeAll();
        var totalTimer = System.Diagnostics.Stopwatch.StartNew();
        var prepareTimer = System.Diagnostics.Stopwatch.StartNew();
        Parameter[] parameters = model.Parameters().ToArray();
        foreach (Parameter parameter in parameters)
            parameter.T.PrepareCudaGradientBuffers(devices);
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
            int batchStart = batchSize * shard / devices.Length;
            int batchEnd = batchSize * (shard + 1) / devices.Length;
            int shardBatch = batchEnd - batchStart;
            int elementStart = batchStart * sequenceLength;
            int elementCount = shardBatch * sequenceLength;
            int[] shardInput = input.AsSpan(elementStart, elementCount).ToArray();
            int[] shardTarget = target.AsSpan(elementStart, elementCount).ToArray();
            int shardValid = shardTarget.Count(value => value != ignoreIndex);
            double dataPreparation = shardTimer.Elapsed.TotalMilliseconds;

            using IDisposable scope = TensorExecutionContext.Push(
                new TorchDevice(TensorDevice.Cuda, devices[shard]));
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
        FlatGradientPlanCache cache = FlatGradientPlans.GetValue(
            model, _ => new FlatGradientPlanCache());
        TensorCudaKernels.FlatGradientPlan plan = cache.Get(parameters, devices);
        TensorCudaKernels.AllReduceGradientsResident(parameters, devices, plan);
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
