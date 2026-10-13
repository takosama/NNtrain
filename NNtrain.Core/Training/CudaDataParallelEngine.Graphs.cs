using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    private bool IsCudaGraphEligible(
        CudaTrainingShapePlan shapePlan,
        int totalValid,
        int targetLength)
    {
        if (shapePlan.GraphDisabled
            || totalValid != targetLength
            || !_model.HasCheckpointableTrainingRandom
            || _dispatchPolicy.DisableCudaGraphs
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

    private (bool ReplayGraph, bool CapturedForThisBatch)
        PrepareTrainingGraph(
            CudaTrainingShapePlan shapePlan,
            CudaBfp8GradientAllReducePlan? bfp8Plan,
            CudaBFloat16GradientAllReducePlan? bfloat16Plan,
            ICudaGradientReductionPlan? reductionPlan,
            IReadOnlyList<Parameter> parameters,
            IReadOnlyList<int> devices,
            int totalValid,
            int targetLength,
            int ignoreIndex,
            long globalStep)
    {
        bool graphEligible = IsCudaGraphEligible(
            shapePlan,
            totalValid,
            targetLength);
        if (shapePlan.IsCompiled
            && !shapePlan.IsCompiledFor(totalValid))
        {
            // The backward root weight is captured as a constant. Running an
            // eager activation graph beside the multi-GiB compiled reservation
            // can OOM, so retire the incompatible graph before a partial or
            // padded accumulation flush. The next full update may recapture.
            shapePlan.PrepareGraphDisposal();
            ExecuteReplicas(shapePlan, devices.Count);
            _adaptiveShardScheduler.ObserveCompiledGraph(
                shapePlan.ShardBatches,
                graphPinnedBytes: 0,
                graphCacheBudgetBytes: _graphCacheBudgetBytes);
            graphEligible = false;
        }
        bool capturedForThisBatch = false;

        if (graphEligible && !shapePlan.IsCompiled)
        {
            try
            {
                shapePlan.PrepareGraphInputs(ignoreIndex);
                ExecuteReplicas(shapePlan, devices.Count);
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
                ExecuteReplicas(shapePlan, devices.Count);
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
                {
                    TryAbortReductionStep(
                        bfp8Plan,
                        bfloat16Plan,
                        recordingStepId);
                }
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

        return (
            graphEligible && shapePlan.IsCompiledFor(totalValid),
            capturedForThisBatch);
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

}
