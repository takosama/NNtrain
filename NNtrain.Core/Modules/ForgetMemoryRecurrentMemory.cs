using NNtrain.Cuda.Memory;

namespace NNtrain;

/// <summary>
/// Owns one layer's fixed-size recurrent memory and keeps the authoritative
/// copy on the execution device between incremental inference calls.
/// </summary>
/// <remarks>
/// Host storage remains available for the public diagnostics and CPU
/// reference path, but a CUDA continuation does not copy the state back to
/// the host after every token.  A host read synchronizes only on demand.
/// </remarks>
internal sealed class ForgetMemoryRecurrentMemory : IDisposable
{
    private readonly object _sync = new();
    private readonly float[] _host;
    private NativeCudaBuffer<float>? _cuda;
    private int _cudaDeviceIndex = -1;
    private bool _hostCurrent = true;
    private bool _cudaCurrent;
    private int _disposed;

    internal ForgetMemoryRecurrentMemory(int length)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        _host = new float[length];
    }

    internal int Length => _host.Length;

    /// <summary>
    /// Returns the host state for the CPU recurrence.  The caller must invoke
    /// <see cref="MarkHostMutated"/> after it updates the returned array.
    /// </summary>
    internal float[] HostForCpuMutation()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            SynchronizeHostLocked();
            return _host;
        }
    }

    internal IReadOnlyList<float> HostSnapshot()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            SynchronizeHostLocked();
            return _host;
        }
    }

    internal void MarkHostMutated()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            _hostCurrent = true;
            _cudaCurrent = false;
        }
    }

    internal NativeCudaBuffer<float> EnsureCudaBuffer(int deviceIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        lock (_sync)
        {
            ThrowIfDisposed();
            if (_cuda is null || _cudaDeviceIndex != deviceIndex)
            {
                // Moving a recurrent state between adapters is uncommon, but
                // it must preserve the last device-authoritative values.
                SynchronizeHostLocked();
                _cuda?.Dispose();
                NativeCudaDevice accelerator =
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex);
                _cuda = accelerator.Allocate1D<float>(
                    _host.Length,
                    CudaMemoryKind.Persistent);
                _cudaDeviceIndex = deviceIndex;
                _cudaCurrent = false;
            }
            if (!_cudaCurrent)
            {
                _cuda.CopyFromCPU(_host);
                _cudaCurrent = true;
            }
            return _cuda;
        }
    }

    internal void MarkCudaMutated(int deviceIndex)
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            if (_cuda is null || _cudaDeviceIndex != deviceIndex)
            {
                throw new InvalidOperationException(
                    "The recurrent CUDA state was not prepared on this device.");
            }
            _cudaCurrent = true;
            _hostCurrent = false;
        }
    }

    internal void Reset()
    {
        lock (_sync)
        {
            ThrowIfDisposed();
            Array.Clear(_host);
            _hostCurrent = true;
            if (_cuda is not null)
            {
                _cuda.MemSetToZero();
                _cudaCurrent = true;
            }
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        lock (_sync)
        {
            _cuda?.Dispose();
            _cuda = null;
            _cudaDeviceIndex = -1;
            _cudaCurrent = false;
        }
        GC.SuppressFinalize(this);
    }

    private void SynchronizeHostLocked()
    {
        if (_hostCurrent)
            return;
        if (_cuda is null || !_cudaCurrent)
        {
            throw new InvalidOperationException(
                "The recurrent state has no authoritative storage.");
        }
        _cuda.CopyToCPU(_host);
        _hostCurrent = true;
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
