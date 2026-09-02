namespace NNtrain;

/// <summary>
/// Records one BFP8-to-BF16 leaf-cache refresh per parameter while a CUDA
/// training graph is being captured. The resulting dequantization nodes run
/// on every replay, so graph-bound BF16 operand addresses always contain the
/// latest optimizer-published BFP8 weights.
/// </summary>
internal sealed class CudaGraphBfp8ParameterRefreshScope : IDisposable
{
    private static readonly AsyncLocal<CudaGraphBfp8ParameterRefreshScope?>
        Current = new();

    private readonly int _deviceIndex;
    private readonly HashSet<object> _registeredReplicas = new(
        ReferenceEqualityComparer.Instance);
    private int _disposed;

    private CudaGraphBfp8ParameterRefreshScope(int deviceIndex)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(deviceIndex);
        if (Current.Value is not null)
        {
            throw new InvalidOperationException(
                "CUDA Graph BFP8 parameter refresh scopes cannot be nested.");
        }

        _deviceIndex = deviceIndex;
        Current.Value = this;
    }

    internal static CudaGraphBfp8ParameterRefreshScope Begin(int deviceIndex)
        => new(deviceIndex);

    internal static bool Register(int deviceIndex, object replica)
    {
        ArgumentNullException.ThrowIfNull(replica);
        CudaGraphBfp8ParameterRefreshScope? scope = Current.Value;
        if (scope is null || Volatile.Read(ref scope._disposed) != 0)
            return false;
        if (scope._deviceIndex != deviceIndex)
        {
            throw new InvalidOperationException(
                $"The active BFP8 parameter refresh scope belongs to CUDA " +
                $"device {scope._deviceIndex}, but a replica on device " +
                $"{deviceIndex} was requested.");
        }
        return scope._registeredReplicas.Add(replica);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        if (!ReferenceEquals(Current.Value, this))
        {
            throw new InvalidOperationException(
                "A CUDA Graph BFP8 parameter refresh scope must be disposed " +
                "in its capture execution context.");
        }
        Current.Value = null;
        _registeredReplicas.Clear();
    }
}
