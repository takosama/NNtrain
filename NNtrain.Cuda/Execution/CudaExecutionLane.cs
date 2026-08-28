using NNtrain.Cuda.Memory;
using NNtrain.Runtime.Execution;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Owns the compute stream, communication stream, memory manager,
/// capabilities and profiler for exactly one CUDA device.
/// </summary>
public sealed class CudaExecutionLane : IStreamExecutionLane
{
    private readonly object _resourceSync = new();
    private readonly List<IDisposable> _ownedResources = [];
    private readonly Action<int, nint>? _activateComputeStream;
    private readonly Action<int, nint>? _synchronizeStream;
    private int _disposed;

    public CudaExecutionLane(
        int deviceIndex,
        CudaStreamHandle computeStream,
        CudaStreamHandle communicationStream,
        CudaMemoryManager memoryManager,
        CudaKernelCapabilities capabilities,
        IExecutionProfiler? profiler = null,
        Action<int, nint>? activateComputeStream = null,
        Action<int, nint>? synchronizeStream = null)
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
        _activateComputeStream = activateComputeStream;
        _synchronizeStream = synchronizeStream;

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
    public nint ComputeStreamHandle => ComputeStream.DangerousGetHandle();
    public nint CommunicationStreamHandle =>
        CommunicationStream.DangerousGetHandle();

    public void ActivateComputeStream()
    {
        ThrowIfDisposed();
        _activateComputeStream?.Invoke(DeviceIndex, ComputeStreamHandle);
    }

    public void SynchronizeComputeStream()
    {
        ThrowIfDisposed();
        _synchronizeStream?.Invoke(DeviceIndex, ComputeStreamHandle);
    }

    public void SynchronizeCommunicationStream()
    {
        ThrowIfDisposed();
        _synchronizeStream?.Invoke(DeviceIndex, CommunicationStreamHandle);
    }

    public T OwnResource<T>(T resource)
        where T : class, IDisposable
    {
        ArgumentNullException.ThrowIfNull(resource);
        lock (_resourceSync)
        {
            ThrowIfDisposed();
            _ownedResources.Add(resource);
        }
        return resource;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        List<Exception>? failures = null;
        TryDispose(SynchronizeBothStreams, ref failures);
        // Tensor kernels keep the selected stream in native thread-local
        // state. Never leave that TLS entry pointing at a stream that this
        // lane is about to destroy; the next legacy CUDA launch on the same
        // worker would otherwise dereference an invalid resource handle.
        TryDispose(RestoreLegacyDefaultStream, ref failures);
        IDisposable[] resources;
        lock (_resourceSync)
        {
            resources = _ownedResources.ToArray();
            _ownedResources.Clear();
        }
        for (int index = resources.Length - 1; index >= 0; index--)
            TryDispose(resources[index].Dispose, ref failures);
        TryDispose(DisposeMemoryChecked, ref failures);
        TryDispose(CommunicationStream.DisposeChecked, ref failures);
        TryDispose(ComputeStream.DisposeChecked, ref failures);
        if (!ReferenceEquals(Profiler, NullExecutionProfiler.Instance))
            TryDispose(Profiler.Dispose, ref failures);

        if (failures is not null)
            throw new AggregateException(
                $"CUDA lane {DeviceIndex} failed to clean up completely.",
                failures);
    }

    private void SynchronizeBothStreams()
    {
        if (_synchronizeStream is null)
            return;
        List<Exception>? failures = null;
        TryDispose(
            () => _synchronizeStream(DeviceIndex, ComputeStreamHandle),
            ref failures);
        TryDispose(
            () => _synchronizeStream(DeviceIndex, CommunicationStreamHandle),
            ref failures);
        if (failures is not null)
        {
            throw new AggregateException(
                $"CUDA lane {DeviceIndex} stream synchronization failed.",
                failures);
        }
    }

    private void RestoreLegacyDefaultStream()
        => _activateComputeStream?.Invoke(DeviceIndex, nint.Zero);

    private void DisposeMemoryChecked()
    {
        Memory.Dispose();
        if (Memory.ReleaseErrors.Count != 0)
        {
            throw new AggregateException(
                $"CUDA lane {DeviceIndex} memory cleanup failed.",
                Memory.ReleaseErrors);
        }
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);

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
