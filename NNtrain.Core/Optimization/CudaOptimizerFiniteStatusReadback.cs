namespace NNtrain;

/// <summary>
/// One persistent pinned integer used to queue an optimizer finite-status
/// readback before a consolidated compute-stream barrier.
/// </summary>
internal sealed unsafe class CudaOptimizerFiniteStatusReadback : IDisposable
{
    private readonly int _deviceIndex;
    private nint _host;
    private bool _pending;

    internal CudaOptimizerFiniteStatusReadback(int deviceIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        _deviceIndex = deviceIndex;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostAllocateNative(sizeof(int), out _host),
            "cudaMallocHost(optimizer finite status)");
    }

    internal void Begin(NativeCudaBuffer<int> source)
    {
        ArgumentNullException.ThrowIfNull(source);
        ObjectDisposedException.ThrowIf(_host == 0, this);
        if (_pending)
        {
            throw new InvalidOperationException(
                "An optimizer finite-status readback is already pending.");
        }
        if (source.Device.Index != _deviceIndex || source.Length < 1)
        {
            throw new ArgumentException(
                "Optimizer status source does not match the readback device.",
                nameof(source));
        }

        source.Device.Bind();
        nint stream = source.Device.DefaultStream;
        NativeCudaRuntime.Check(
            NativeCudaRuntime.CopyDeviceToHostAsyncNative(
                _deviceIndex,
                _host,
                source.NativePtr,
                sizeof(int),
                stream),
            "cudaMemcpyAsync(D2H optimizer finite status)");
        _pending = true;
    }

    internal int ReadAfterSynchronization()
    {
        ObjectDisposedException.ThrowIf(_host == 0, this);
        if (!_pending)
        {
            throw new InvalidOperationException(
                "Optimizer finite-status readback was not queued.");
        }
        _pending = false;
        return *(int*)_host;
    }

    public void Dispose()
    {
        nint host = Interlocked.Exchange(ref _host, 0);
        if (host == 0)
            return;
        if (_pending)
        {
            NativeCudaRuntime.SynchronizeDeviceComputeStream(_deviceIndex);
            _pending = false;
        }
        NativeCudaRuntime.Check(
            NativeCudaRuntime.HostFreeNative(host),
            "cudaFreeHost(optimizer finite status)");
    }
}
