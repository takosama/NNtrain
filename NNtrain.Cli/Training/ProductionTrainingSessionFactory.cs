using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;

namespace NNtrain;

/// <summary>
/// Creates the authoritative step session independently from whether the task
/// happens to use CUDA data parallelism.
/// </summary>
internal static class ProductionTrainingSessionFactory
{
    internal static TrainingSession Create(
        TensorPrecisionMode precisionMode,
        long lastCommittedStep,
        Func<int, IExecutionLane>? cudaLaneFactory = null)
        => Create(
            precisionMode,
            lastCommittedStep,
            Tensor.ExecutionDevice,
            Tensor.CudaDeviceIndices,
            cudaLaneFactory);

    /// <summary>
    /// Creates a session from the canonical training specification.  The
    /// explicit device authority prevents unrelated legacy Tensor globals
    /// from silently changing a production run between configuration parsing
    /// and session creation.
    /// </summary>
    internal static TrainingSession Create(
        TensorPrecisionMode precisionMode,
        long lastCommittedStep,
        TensorDevice executionDevice,
        IReadOnlyList<int> cudaDeviceIndices,
        Func<int, IExecutionLane>? cudaLaneFactory = null)
    {
        if (lastCommittedStep < -1)
            throw new ArgumentOutOfRangeException(nameof(lastCommittedStep));
        ExecutionSession execution = CreateExecutionSession(
            precisionMode,
            executionDevice,
            cudaDeviceIndices,
            cudaLaneFactory);
        try
        {
            return new TrainingSession(
                execution,
                ownsExecutionSession: true,
                lastCommittedStep: lastCommittedStep);
        }
        catch (Exception creationFailure)
        {
            try
            {
                execution.Dispose();
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "Training-session creation and execution cleanup failed.",
                    creationFailure,
                    cleanupFailure);
            }
            throw;
        }
    }

    /// <summary>
    /// Creates the execution authority before model allocation or checkpoint
    /// restore. Callers that do not know the restored global step yet can
    /// enter this session, materialize resident CUDA state through its lanes,
    /// and then wrap it in a <see cref="TrainingSession"/> with the restored
    /// commit position.
    /// </summary>
    internal static ExecutionSession CreateExecutionSession(
        TensorPrecisionMode precisionMode,
        TensorDevice executionDevice,
        IReadOnlyList<int> cudaDeviceIndices,
        Func<int, IExecutionLane>? cudaLaneFactory = null)
    {
        ArgumentNullException.ThrowIfNull(cudaDeviceIndices);
        if (cudaDeviceIndices.Count == 0)
        {
            throw new ArgumentException(
                "At least one configured CUDA device index is required.",
                nameof(cudaDeviceIndices));
        }

        int[] configuredCudaDevices = cudaDeviceIndices.ToArray();
        var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = executionDevice == TensorDevice.Cuda
                ? ExecutionDeviceKind.Cuda
                : ExecutionDeviceKind.Cpu,
            CudaDevices = new DeviceSet(configuredCudaDevices),
            Precision = PrecisionPolicy.Parse(
                TensorPrecisionModeNames.Format(precisionMode)),
        });
        try
        {
            if (execution.Options.Device == ExecutionDeviceKind.Cuda)
            {
                Func<int, IExecutionLane> createLane =
                    cudaLaneFactory
                    ?? (static deviceIndex =>
                        CudaExecutionLaneFactory.Create(deviceIndex));
                foreach (int deviceIndex in execution.Options.CudaDevices)
                {
                    IExecutionLane lane = createLane(deviceIndex)
                        ?? throw new InvalidOperationException(
                            $"CUDA lane factory returned null for device {deviceIndex}.");
                    try
                    {
                        execution.AttachLane(lane);
                    }
                    catch (Exception attachFailure)
                    {
                        try
                        {
                            lane.Dispose();
                        }
                        catch (Exception cleanupFailure)
                        {
                            throw new AggregateException(
                                $"CUDA lane {deviceIndex} could not be attached or cleaned up.",
                                attachFailure,
                                cleanupFailure);
                        }
                        throw;
                    }
                }
            }

            return execution;
        }
        catch (Exception creationFailure)
        {
            try
            {
                execution.Dispose();
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "Training execution-session creation and cleanup failed.",
                    creationFailure,
                    cleanupFailure);
            }
            throw;
        }
    }

    internal static void EnsureCanPublishCheckpoint(
        TrainingSession session,
        long globalStep)
    {
        ArgumentNullException.ThrowIfNull(session);
        if (!session.CanPublishCheckpoint
            || session.LastCommittedStep != globalStep)
        {
            throw new InvalidOperationException(
                $"Training step {globalStep} has not completed its metrics commit.");
        }
    }
}
