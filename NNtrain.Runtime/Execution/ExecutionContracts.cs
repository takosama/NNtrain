namespace NNtrain.Runtime.Execution;

/// <summary>Memory owned by an execution lane.</summary>
public interface IDeviceMemoryManager : IDisposable
{
    int DeviceIndex { get; }
    long AllocationCount { get; }
    long AllocatedBytes { get; }
}

/// <summary>Capability query supplied by a device backend.</summary>
public interface IKernelCapabilitySet
{
    bool Supports(string feature);
}

/// <summary>Profiler owned by an execution lane.</summary>
public interface IExecutionProfiler : IDisposable
{
    IDisposable Measure(string operation);
    void RecordCounter(string name, long value);
}

/// <summary>
/// A device-specific execution lane. Its owner must dispose it after all
/// work submitted to the lane has completed.
/// </summary>
public interface IExecutionLane : IDisposable
{
    ExecutionDeviceKind DeviceKind { get; }
    int DeviceIndex { get; }
    IDeviceMemoryManager MemoryManager { get; }
    IKernelCapabilitySet Capabilities { get; }
    IExecutionProfiler Profiler { get; }
}

/// <summary>
/// An execution lane with explicit compute and communication streams. Stream
/// handles are opaque to the runtime assembly; the owning backend remains
/// responsible for binding and synchronizing them.
/// </summary>
public interface IStreamExecutionLane : IExecutionLane
{
    nint ComputeStreamHandle { get; }
    nint CommunicationStreamHandle { get; }

    /// <summary>
    /// Makes this lane's device and compute stream current on the calling
    /// native thread. Implementations must not assume managed asynchronous
    /// context and native thread-local state are the same thing.
    /// </summary>
    void ActivateComputeStream();

    void SynchronizeComputeStream();

    void SynchronizeCommunicationStream();

    /// <summary>
    /// Transfers a backend resource to the lane. Owned resources are released
    /// after both streams complete and before the streams/memory owner close.
    /// </summary>
    T OwnResource<T>(T resource)
        where T : class, IDisposable;
}

/// <summary>Transactional transfer of a newly-created resource to a lane.</summary>
public static class ExecutionLaneResources
{
    public static T Attach<T>(IStreamExecutionLane lane, T resource)
        where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(lane);
        ArgumentNullException.ThrowIfNull(resource);
        try
        {
            return lane.OwnResource(resource);
        }
        catch (Exception attachFailure)
        {
            try
            {
                resource.Dispose();
            }
            catch (Exception cleanupFailure)
            {
                throw new AggregateException(
                    "An execution resource could not be attached or rolled back.",
                    attachFailure,
                    cleanupFailure);
            }
            throw;
        }
    }
}

/// <summary>A zero-overhead default for sessions without profiling enabled.</summary>
public sealed class NullExecutionProfiler : IExecutionProfiler
{
    private NullExecutionProfiler()
    {
    }

    public static NullExecutionProfiler Instance { get; } = new();

    public IDisposable Measure(string operation)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(operation);
        return EmptyScope.Instance;
    }

    public void RecordCounter(string name, long value)
        => ArgumentException.ThrowIfNullOrWhiteSpace(name);

    public void Dispose()
    {
    }

    private sealed class EmptyScope : IDisposable
    {
        internal static EmptyScope Instance { get; } = new();
        public void Dispose()
        {
        }
    }
}
