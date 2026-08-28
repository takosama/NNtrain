using NNtrain.Cuda.Execution;

namespace NNtrain;

/// <summary>
/// Typed capture-only dropout context. It records one device counter advance
/// at graph head, then hands each forward operation a stable seed token that
/// its backward closure retains verbatim.
/// </summary>
internal sealed class CudaGraphDropoutCaptureScope : IDisposable
{
    private static readonly AsyncLocal<CudaGraphDropoutCaptureScope?> Current
        = new();
    private readonly CudaGraphRngState _rngState;
    private readonly ulong _baseSeed;
    private long _nextOrdinal;
    private int _disposed;

    private CudaGraphDropoutCaptureScope(
        CudaGraphRngState rngState,
        ulong baseSeed)
    {
        if (Current.Value is not null)
        {
            throw new InvalidOperationException(
                "CUDA Graph dropout capture scopes cannot be nested; one " +
                "step must advance its RNG counter exactly once.");
        }
        _rngState = rngState;
        _baseSeed = baseSeed;
        Current.Value = this;
        try
        {
            _rngState.EnqueueAdvance();
        }
        catch
        {
            Current.Value = null;
            throw;
        }
    }

    internal static CudaGraphDropoutCaptureScope Begin(
        CudaGraphRngState rngState,
        ulong baseSeed)
    {
        ArgumentNullException.ThrowIfNull(rngState);
        return new CudaGraphDropoutCaptureScope(rngState, baseSeed);
    }

    internal static bool IsActiveFor(int deviceIndex)
        => Current.Value is { } scope
            && Volatile.Read(ref scope._disposed) == 0
            && scope._rngState.DeviceIndex == deviceIndex;

    internal static bool TryAcquire(
        int deviceIndex,
        out CudaGraphDropoutToken token)
    {
        CudaGraphDropoutCaptureScope? scope = Current.Value;
        if (scope is null || Volatile.Read(ref scope._disposed) != 0)
        {
            token = default;
            return false;
        }
        if (scope._rngState.DeviceIndex != deviceIndex)
        {
            throw new InvalidOperationException(
                "The active CUDA Graph dropout capture scope belongs to " +
                $"device {scope._rngState.DeviceIndex}, but dropout was " +
                $"dispatched to device {deviceIndex}.");
        }

        ulong ordinal = checked((ulong)Interlocked.Increment(
            ref scope._nextOrdinal) - 1UL);
        token = new CudaGraphDropoutToken(
            scope._rngState,
            MixSeed(scope._baseSeed, ordinal),
            ordinal);
        return true;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (ReferenceEquals(Current.Value, this))
            Current.Value = null;
    }

    private static ulong MixSeed(ulong baseSeed, ulong ordinal)
    {
        ulong value = baseSeed
            + (ordinal + 1UL) * 0x9e3779b97f4a7c15UL;
        value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9UL;
        value = (value ^ (value >> 27)) * 0x94d049bb133111ebUL;
        return value ^ (value >> 31);
    }
}

internal readonly record struct CudaGraphDropoutToken(
    CudaGraphRngState RngState,
    ulong OperationSeed,
    ulong OperationOrdinal)
{
    internal int DeviceIndex => RngState.DeviceIndex;
}
