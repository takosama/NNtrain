using System.Runtime.InteropServices;

namespace NNtrain.Cuda.Execution;

/// <summary>
/// Safe ownership wrapper for a CUDA stream. A null release callback marks a
/// borrowed stream; owned stream cleanup never throws from finalization.
/// </summary>
public sealed class CudaStreamHandle : SafeHandle
{
    private readonly Action<int, nint>? _release;

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

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        try
        {
            _release?.Invoke(DeviceIndex, handle);
            return true;
        }
        catch
        {
            return false;
        }
    }
}
