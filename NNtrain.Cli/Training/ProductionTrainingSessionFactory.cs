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
        long lastCommittedStep)
    {
        if (lastCommittedStep < -1)
            throw new ArgumentOutOfRangeException(nameof(lastCommittedStep));

        IReadOnlyList<int> configuredCudaDevices = Tensor.CudaDeviceIndices;
        var execution = new ExecutionSession(new ExecutionOptions
        {
            Device = Tensor.ExecutionDevice == TensorDevice.Cuda
                ? ExecutionDeviceKind.Cuda
                : ExecutionDeviceKind.Cpu,
            CudaDevices = configuredCudaDevices.Count == 0
                ? DeviceSet.Default
                : new DeviceSet(configuredCudaDevices),
            Precision = PrecisionPolicy.Parse(
                TensorPrecisionModeNames.Format(precisionMode)),
        });
        try
        {
            return new TrainingSession(
                execution,
                ownsExecutionSession: true,
                lastCommittedStep: lastCommittedStep);
        }
        catch
        {
            execution.Dispose();
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
