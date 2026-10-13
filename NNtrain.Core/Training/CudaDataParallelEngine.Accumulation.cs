using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
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
                microBatches.Any(batch => NeedsReducerTailPadding(batch.BatchSize))
                    ? microBatches.Select(batch => PadReducerTail(batch, ignoreIndex)).ToArray()
                    : microBatches,
                ignoreIndex,
                globalStep);
        }
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
        long totalTargetLong = 0;
        bool uniformGraphShape = true;
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
            totalTargetLong = checked(
                totalTargetLong + microBatch.Target.LongLength);
            uniformGraphShape &= microBatch.BatchSize == first.BatchSize
                && microBatch.SequenceLength == first.SequenceLength;
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

        CudaTrainingShapePlan? graphShapePlan = null;
        bool replayGraph = false;
        bool capturedForFirstBatch = false;
        if (uniformGraphShape && totalTargetLong <= int.MaxValue)
        {
            int[] graphShardBatches = AllocateShardBatches(
                first.BatchSize,
                devices);
            graphShapePlan = GetOrCreateTrainingShapePlan(
                first.BatchSize,
                first.SequenceLength,
                devices,
                graphShardBatches);
            graphShapePlan.PrepareBatch(
                first.Input,
                first.Target,
                ignoreIndex);
            (replayGraph, capturedForFirstBatch) = PrepareTrainingGraph(
                graphShapePlan,
                bfp8Plan,
                bfloat16Plan,
                reductionPlan,
                parameters,
                devices,
                totalValid,
                (int)totalTargetLong,
                ignoreIndex,
                globalStep);
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
                int[] shardBatches;
                CudaTrainingShapePlan shapePlan;
                if (replayGraph)
                {
                    shapePlan = graphShapePlan!;
                    shardBatches = shapePlan.ShardBatches;
                }
                else
                {
                    shardBatches = AllocateShardBatches(
                        microBatch.BatchSize,
                        devices);
                    shapePlan = GetOrCreateTrainingShapePlan(
                        microBatch.BatchSize,
                        microBatch.SequenceLength,
                        devices,
                        shardBatches);
                }
                shapePlan.PrepareBatch(
                    microBatch.Input,
                    microBatch.Target,
                    ignoreIndex);
                bool publishGradients = index == microBatches.Count - 1;
                if (replayGraph)
                {
                    shapePlan.PrepareReplay(
                        bfp8Plan,
                        bfloat16Plan,
                        reductionStepId,
                        globalStep,
                        reuseCaptureInputs:
                            capturedForFirstBatch && index == 0,
                        beginReductionDeviceStep: index == 0,
                        publishGradientsAfterReplay: publishGradients,
                        rngSequence: index,
                        accumulatedLossBuffers: accumulatedLossBuffers,
                        completeAccumulatedLoss: publishGradients);
                }
                else
                {
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
                }
                ExecuteReplicas(shapePlan, devices.Length);
                if (replayGraph)
                    Interlocked.Increment(ref _graphReplayCount);
                weightedLoss += shapePlan.WeightedLosses.Sum();

                bool canMeasureShardRuntime = bfp8Plan is null
                    && (bfloat16Plan is null
                        || bfloat16Plan.DefersExchangeUntilBackward);
                if (canMeasureShardRuntime && !replayGraph)
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

}
