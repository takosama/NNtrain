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
