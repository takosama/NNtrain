using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
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
            int[] devices = GetDevices(NeedsReducerTailPadding(batchSize)
                ? _cudaDeviceIndices.Length : batchSize);
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
            CudaLanguageModelMicroBatch batch = PadReducerTail(
                new(input, target, batchSize, sequenceLength), ignoreIndex);
            return ForwardBackwardCore(
                batch.Input,
                batch.Target,
                batch.BatchSize,
                batch.SequenceLength,
                ignoreIndex,
                globalStep);
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
        (bool replayGraph, bool capturedForThisBatch) =
            PrepareTrainingGraph(
                shapePlan,
                bfp8Plan,
                bfloat16Plan,
                reductionPlan,
                parameters,
                devices,
                totalValid,
                target.Length,
                ignoreIndex,
                globalStep);
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
                ResolvePrecisionPolicy(_model.PrecisionMode),
                dispatchPolicy: _dispatchPolicy);
        }
        catch (AggregateException exception)
            when (exception.InnerExceptions.Count == 1)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(exception.InnerExceptions[0])
                .Throw();
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

    private bool NeedsReducerTailPadding(int batchSize)
        => batchSize > 0 && batchSize < _cudaDeviceIndices.Length
            && UseBFloat16GradientBuckets(_cudaDeviceIndices, _model.PrecisionMode);

    private CudaLanguageModelMicroBatch PadReducerTail(
        CudaLanguageModelMicroBatch batch, int ignoreIndex)
    {
        ValidateBatch(batch.Input, batch.Target, batch.BatchSize, batch.SequenceLength);
        if (!NeedsReducerTailPadding(batch.BatchSize))
            return batch;
        // Keep the reducer/device set stable at an epoch tail. Extra rows
        // carry no targets and contribute zero; no samples are duplicated.
        int length = checked(_cudaDeviceIndices.Length * batch.SequenceLength);
        var input = new int[length];
        var target = new int[length];
        Array.Fill(target, ignoreIndex);
        batch.Input.CopyTo(input, 0);
        batch.Target.CopyTo(target, 0);
        return new(input, target, _cudaDeviceIndices.Length, batch.SequenceLength);
    }

    private int[] GetDevices(int batchSize)
        => _devicePrefixes[
            Math.Min(batchSize, _cudaDeviceIndices.Length) - 1];

}
