namespace NNtrain;

/// <summary>
/// Owns the logical state behind the public <see cref="Tensor"/> facade.
/// </summary>
/// <remarks>
/// Keeping logical value, host storage, device replicas, and autograd identity
/// behind one stable object prevents partial <c>Tensor</c> implementations from
/// accidentally acquiring independent ownership models.  The public tensor
/// remains API-compatible while each subsystem has one explicit owner.
/// </remarks>
internal sealed class TensorValue
{
    private AutogradNode? _autograd;

    internal TensorStorageOwner Storage { get; } = new();

    internal Tensor.DeviceReplicaSet Replicas { get; } = new();

    internal AutogradNode Autograd
    {
        get => _autograd
            ?? throw new InvalidOperationException(
                "Tensor autograd state has not been initialized.");
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            if (Interlocked.CompareExchange(
                    ref _autograd,
                    value,
                    comparand: null) is not null)
            {
                throw new InvalidOperationException(
                    "Tensor autograd ownership can only be initialized once.");
            }
        }
    }
}

/// <summary>
/// Owns a tensor's host/master payload and versioned CPU decode caches.
/// </summary>
internal sealed class TensorStorageOwner
{
    private TensorStorage? _data;

    internal TensorStorage Data
    {
        get => _data
            ?? throw new InvalidOperationException(
                "Tensor storage has not been initialized.");
        set => _data = value
            ?? throw new ArgumentNullException(nameof(value));
    }

    internal float[]? MasterData { get; set; }

    internal float[]? PhysicalFloat32Cache { get; set; }

    internal long PhysicalFloat32CacheDataVersion { get; set; } = -1;

    internal float[] Gradient { get; set; } = [];

    internal int[] Shape { get; set; } = [];

    internal long DataVersion { get; set; }

    internal object CacheSync { get; } = new();

    internal float[]? TransposedDataCache { get; set; }

    internal long TransposedDataVersion { get; set; } = -1;

    internal bool AllowInPlaceBFloat16Gradient { get; set; }
}
