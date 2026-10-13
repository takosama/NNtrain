using NNtrain.Cuda.Execution;
using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain;

public sealed partial class CudaDataParallelEngine
{
    /// <summary>
    /// Retires compiled activation graphs and idle lane allocations after a
    /// checkpoint has been published. Checkpoint serialization itself is
    /// bounded, but keeping its synchronized step's graph reservation alive
    /// leaves too little headroom for the first eager step at near-capacity
    /// batch sizes. Parameters, optimizer state, gradient reducers, and shard
    /// EMA remain resident.
    /// </summary>
    internal int ReleaseCheckpointTransientMemory()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            int retiredPlanCount = _trainingShapePlans.Count;
            List<Exception>? failures = null;
            ReleaseTrainingShapePlans(ref failures);

            ExecutionSession? session = ExecutionSession.Current;
            foreach (int deviceIndex in _cudaDeviceIndices)
            {
                try
                {
                    if (session is not null
                        && session.TryGetLane(
                            ExecutionDeviceKind.Cuda,
                            deviceIndex,
                            out IExecutionLane? executionLane)
                        && executionLane is CudaExecutionLane lane)
                    {
                        lane.SynchronizeComputeStream();
                        lane.SynchronizeCommunicationStream();
                        Tensor.ClearCudaFloatBufferPool(deviceIndex);
                        lane.Memory.CollectCompleted();
                        lane.Memory.TrimReusableCache();
                    }
                    else
                    {
                        ForgetMemoryV2Cuda.GetAccelerator(deviceIndex)
                            .Synchronize();
                        Tensor.ClearCudaFloatBufferPool(deviceIndex);
                    }
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "CUDA checkpoint transient-memory cleanup failed.",
                    failures);
            }
            return retiredPlanCount;
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

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(IsDisposed, this);

}
