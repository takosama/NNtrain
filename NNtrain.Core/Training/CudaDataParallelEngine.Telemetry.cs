using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
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

    private int[] AllocateShardBatches(
        int batchSize,
        IReadOnlyList<int> devices)
    {
        return _adaptiveShardScheduler.Allocate(batchSize, devices);
    }

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

}
