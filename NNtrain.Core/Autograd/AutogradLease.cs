namespace NNtrain;

/// <summary>Describes whether an autograd node owns or only borrows a context.</summary>
internal enum AutogradLeaseOwnership
{
    Borrowed = 0,
    Owned = 1,
}

/// <summary>
/// Identifies the execution generation and storage contract of saved autograd
/// state.  Legacy resources may omit <see cref="DType"/>, but newly registered
/// CUDA workspaces always provide it.
/// </summary>
internal readonly record struct AutogradLeaseMetadata
{
    internal AutogradLeaseMetadata(
        TensorDevice device,
        int deviceIndex,
        TensorDType? dtype,
        long generation,
        AutogradLeaseOwnership ownership)
    {
        if (!Enum.IsDefined(device))
            throw new ArgumentOutOfRangeException(nameof(device));
        if (deviceIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(deviceIndex));
        if (device == TensorDevice.Cpu && deviceIndex != 0)
        {
            throw new ArgumentException(
                "A CPU autograd lease must use device index zero.",
                nameof(deviceIndex));
        }
        if (dtype.HasValue && !Enum.IsDefined(dtype.Value))
            throw new ArgumentOutOfRangeException(nameof(dtype));
        if (generation < 0)
            throw new ArgumentOutOfRangeException(nameof(generation));
        if (!Enum.IsDefined(ownership))
            throw new ArgumentOutOfRangeException(nameof(ownership));

        Device = device;
        DeviceIndex = deviceIndex;
        DType = dtype;
        Generation = generation;
        Ownership = ownership;
    }

    internal TensorDevice Device { get; }
    internal int DeviceIndex { get; }
    internal TensorDType? DType { get; }
    internal long Generation { get; }
    internal AutogradLeaseOwnership Ownership { get; }

    internal static AutogradLeaseMetadata LegacyOwned { get; } = new(
        TensorDevice.Cpu,
        deviceIndex: 0,
        dtype: null,
        generation: 0,
        AutogradLeaseOwnership.Owned);

    internal static AutogradLeaseMetadata CudaOwned(
        int deviceIndex,
        TensorDType dtype,
        long generation)
        => new(
            TensorDevice.Cuda,
            deviceIndex,
            dtype,
            generation,
            AutogradLeaseOwnership.Owned);
}

/// <summary>Non-generic ownership boundary used by <see cref="AutogradNode"/>.</summary>
internal interface IAutogradLease : IDisposable
{
    AutogradLeaseMetadata Metadata { get; }
    bool IsReleased { get; }
}

/// <summary>
/// Owns or borrows one strongly typed saved context.  The release callback is
/// invoked at most once and is deliberately supplied by the CUDA operation so
/// it can preserve that operation's stream/fence ordering contract.
/// </summary>
internal sealed class AutogradLease<TContext> : IAutogradLease
    where TContext : class
{
    private readonly object _sync = new();
    private TContext? _context;
    private Action<TContext>? _release;
    private int _released;

    private AutogradLease(
        TContext context,
        AutogradLeaseMetadata metadata,
        Action<TContext>? release)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (metadata.Ownership == AutogradLeaseOwnership.Owned
            && release is null)
        {
            throw new ArgumentNullException(
                nameof(release),
                "An owned autograd context requires an explicit release callback.");
        }
        if (metadata.Ownership == AutogradLeaseOwnership.Borrowed
            && release is not null)
        {
            throw new ArgumentException(
                "A borrowed autograd context cannot own a release callback.",
                nameof(release));
        }

        _context = context;
        _release = release;
        Metadata = metadata;
    }

    internal AutogradLeaseMetadata Metadata { get; }

    AutogradLeaseMetadata IAutogradLease.Metadata => Metadata;

    internal bool IsReleased => Volatile.Read(ref _released) != 0;

    bool IAutogradLease.IsReleased => IsReleased;

    internal static AutogradLease<TContext> Own(
        TContext context,
        AutogradLeaseMetadata metadata,
        Action<TContext> release)
    {
        if (metadata.Ownership != AutogradLeaseOwnership.Owned)
        {
            throw new ArgumentException(
                "Owned leases require owned metadata.",
                nameof(metadata));
        }
        return new AutogradLease<TContext>(context, metadata, release);
    }

    internal static AutogradLease<TContext> Borrow(
        TContext context,
        AutogradLeaseMetadata metadata)
    {
        if (metadata.Ownership != AutogradLeaseOwnership.Borrowed)
        {
            throw new ArgumentException(
                "Borrowed leases require borrowed metadata.",
                nameof(metadata));
        }
        return new AutogradLease<TContext>(context, metadata, release: null);
    }

    /// <summary>
    /// Runs an operation while release is excluded. Concurrent release waits
    /// for the operation to finish; operations starting after release fail.
    /// </summary>
    internal void Use(Action<TContext> action)
    {
        ArgumentNullException.ThrowIfNull(action);
        lock (_sync)
        {
            if (Volatile.Read(ref _released) != 0 || _context is null)
                throw new ObjectDisposedException(nameof(AutogradLease<TContext>));
            action(_context);
        }
    }

    public void Dispose()
    {
        TContext? context;
        Action<TContext>? release;
        lock (_sync)
        {
            if (_released != 0)
                return;
            Volatile.Write(ref _released, 1);
            context = _context;
            _context = null;
            release = _release;
            _release = null;
        }

        if (context is not null && release is not null)
            release(context);
    }
}
