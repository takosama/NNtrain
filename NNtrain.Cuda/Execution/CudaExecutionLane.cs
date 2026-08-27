using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Owns the compute stream, communication stream, memory manager,
/// capabilities and profiler for exactly one CUDA device.
/// </summary>
public sealed class CudaExecutionLane : IExecutionLane
{
    private readonly Action<int>? _synchronizeBeforeDispose;
    private int _disposed;

    public CudaExecutionLane(
        int deviceIndex,
        CudaStreamHandle computeStream,
        CudaStreamHandle communicationStream,
        CudaMemoryManager memoryManager,
        CudaKernelCapabilities capabilities,
        IExecutionProfiler? profiler = null,
        Action<int>? synchronizeBeforeDispose = null)
    {
        if (deviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        DeviceIndex = deviceIndex;
        ComputeStream = computeStream
            ?? throw new ArgumentNullException(nameof(computeStream));
        CommunicationStream = communicationStream
            ?? throw new ArgumentNullException(nameof(communicationStream));
        Memory = memoryManager
            ?? throw new ArgumentNullException(nameof(memoryManager));
        CudaCapabilities = capabilities
            ?? throw new ArgumentNullException(nameof(capabilities));
        Profiler = profiler ?? NullExecutionProfiler.Instance;
        _synchronizeBeforeDispose = synchronizeBeforeDispose;

        if (ComputeStream.DeviceIndex != deviceIndex
            || CommunicationStream.DeviceIndex != deviceIndex
            || Memory.DeviceIndex != deviceIndex)
        {
            throw new ArgumentException(
                "Every resource in a CUDA lane must belong to the lane's device.");
        }
    }

    public ExecutionDeviceKind DeviceKind => ExecutionDeviceKind.Cuda;
    public int DeviceIndex { get; }
    public CudaStreamHandle ComputeStream { get; }
    public CudaStreamHandle CommunicationStream { get; }
    public CudaMemoryManager Memory { get; }
    public CudaKernelCapabilities CudaCapabilities { get; }
    public IDeviceMemoryManager MemoryManager => Memory;
    public IKernelCapabilitySet Capabilities => CudaCapabilities;
    public IExecutionProfiler Profiler { get; }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        List<Exception>? failures = null;
        TryDispose(() => _synchronizeBeforeDispose?.Invoke(DeviceIndex), ref failures);
        TryDispose(Memory.Dispose, ref failures);
        TryDispose(CommunicationStream.Dispose, ref failures);
        TryDispose(ComputeStream.Dispose, ref failures);
        if (!ReferenceEquals(Profiler, NullExecutionProfiler.Instance))
            TryDispose(Profiler.Dispose, ref failures);

        if (failures is not null)
            throw new AggregateException(
                $"CUDA lane {DeviceIndex} failed to clean up completely.",
                failures);
    }

    private static void TryDispose(
        Action action,
        ref List<Exception>? failures)
    {
        try
        {
            action();
        }
        catch (Exception exception)
        {
            (failures ??= []).Add(exception);
        }
    }
}
