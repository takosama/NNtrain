using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Safe ownership wrapper for a CUDA stream. A null release callback marks a
/// borrowed stream; owned stream cleanup never throws from finalization.
/// </summary>
public sealed class CudaStreamHandle : SafeHandle
{
    private readonly Action<int, nint>? _release;
    private Exception? _releaseFailure;

    public CudaStreamHandle(
        int deviceIndex,
        nint stream,
        Action<int, nint>? release = null)
        : base(nint.Zero, ownsHandle: true)
    {
        if (deviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        DeviceIndex = deviceIndex;
        _release = release;
        SetHandle(stream);
    }

    public int DeviceIndex { get; }

    public bool OwnsNativeStream => _release is not null;

    /// <summary>
    /// A native destroy failure captured by the non-throwing SafeHandle
    /// release path. Explicit owners can surface it with
    /// <see cref="DisposeChecked"/> while finalization remains safe.
    /// </summary>
    public Exception? ReleaseFailure => Volatile.Read(ref _releaseFailure);

    public override bool IsInvalid => handle == nint.Zero;

    public void DisposeChecked()
    {
        Dispose();
        if (ReleaseFailure is Exception failure)
        {
            throw new InvalidOperationException(
                $"CUDA stream cleanup failed on device {DeviceIndex}.",
                failure);
        }
    }

    protected override bool ReleaseHandle()
    {
        try
        {
            _release?.Invoke(DeviceIndex, handle);
            return true;
        }
        catch (Exception exception)
        {
            Interlocked.CompareExchange(
                ref _releaseFailure,
                exception,
                comparand: null);
            return false;
        }
    }
}
