namespace NNtrain;

/// <summary>
/// Owns one tensor's incoming graph edges and local reverse-mode operation.
/// </summary>
internal sealed class AutogradNode
{
    private static readonly Action NoBackward = static () => { };

    private Tensor[] _parents;
    private long[] _parentDataVersions;
    private readonly bool _isDetached;
    private Action _backwardAction = NoBackward;
    private bool _hasBackwardAction;
    private List<IAutogradLease>? _leases;

    internal AutogradNode(Tensor[]? parents = null)
        : this(parents, isDetached: false)
    {
    }

    private AutogradNode(Tensor[]? parents, bool isDetached)
    {
        _parents = parents is null ? [] : (Tensor[])parents.Clone();
        if (Array.Exists(_parents, static parent => parent is null))
        {
            throw new ArgumentException(
                "An autograd node cannot contain a null parent.",
                nameof(parents));
        }

        _parentDataVersions = new long[_parents.Length];
        for (int index = 0; index < _parents.Length; index++)
            _parentDataVersions[index] = _parents[index].DataVersion;
        _isDetached = isDetached;
        Parents = Array.AsReadOnly(_parents);
    }

    internal IReadOnlyList<Tensor> Parents { get; private set; }
    internal bool IsLeaf => _parents.Length == 0;
    internal bool IsDetached => _isDetached;

    internal static AutogradNode Detached()
        => new([], isDetached: true);

    internal Action BackwardAction
    {
        set
        {
            ArgumentNullException.ThrowIfNull(value);
            if (_isDetached)
                return;

            if (_hasBackwardAction)
            {
                throw new InvalidOperationException(
                    "The backward action for an autograd node can only be assigned once.");
            }

            _backwardAction = value;
            _hasBackwardAction = true;
        }
    }

    internal void RunBackward() => _backwardAction();

    /// <summary>
    /// Atomically associates a typed saved context with the only backward
    /// action that may consume it. The action captures the lease, never the
    /// context itself.
    /// </summary>
    internal void SetBackward<TContext>(
        AutogradLease<TContext> lease,
        Action<TContext> backward)
        where TContext : class
    {
        ArgumentNullException.ThrowIfNull(lease);
        ArgumentNullException.ThrowIfNull(backward);

        if (_isDetached)
        {
            lease.Dispose();
            return;
        }
        if (_hasBackwardAction)
        {
            DisposeRejectedLease(lease);
            throw new InvalidOperationException(
                "The backward action for an autograd node can only be assigned once.");
        }

        Action leasedBackward = () => lease.Use(backward);
        try
        {
            RegisterLease(lease);
            _backwardAction = leasedBackward;
            _hasBackwardAction = true;
        }
        catch (Exception registrationFailure)
        {
            try
            {
                lease.Dispose();
            }
            catch (Exception releaseFailure)
            {
                throw new AggregateException(
                    "Registering an autograd lease failed, and rolling it back also failed.",
                    registrationFailure,
                    releaseFailure);
            }
            throw;
        }
    }

    internal void RegisterLease(IAutogradLease lease)
    {
        ArgumentNullException.ThrowIfNull(lease);
        if (lease.IsReleased)
            throw new ObjectDisposedException(nameof(lease));
        (_leases ??= []).Add(lease);
    }

    internal void RegisterResource(IDisposable resource)
    {
        ArgumentNullException.ThrowIfNull(resource);
        RegisterLease(AutogradLease<IDisposable>.Own(
            resource,
            AutogradLeaseMetadata.LegacyOwned,
            static owned => owned.Dispose()));
    }

    internal bool HasLeases => _leases is { Count: > 0 };

    internal void ReleaseGraph()
    {
        List<Exception>? failures = null;
        List<IAutogradLease>? leases = Interlocked.Exchange(
            ref _leases,
            null);
        try
        {
            if (leases is not null)
            {
                foreach (IAutogradLease lease in leases)
                {
                    try
                    {
                        lease.Dispose();
                    }
                    catch (Exception exception)
                    {
                        AddFailure(ref failures, exception);
                    }
                }
            }
        }
        finally
        {
            _backwardAction = NoBackward;
            _parents = [];
            _parentDataVersions = [];
            Parents = Array.Empty<Tensor>();
        }

        if (failures is not null)
        {
            throw new AggregateException(
                "One or more autograd leases failed to release.",
                failures);
        }
    }

    private static void DisposeRejectedLease(IAutogradLease lease)
    {
        try
        {
            lease.Dispose();
        }
        catch (Exception releaseFailure)
        {
            throw new AggregateException(
                "The autograd node already has a backward action, and the rejected " +
                "lease failed to release.",
                releaseFailure);
        }
    }

    private static void AddFailure(
        ref List<Exception>? failures,
        Exception exception)
    {
        failures ??= [];
        if (exception is AggregateException aggregate)
            failures.AddRange(aggregate.Flatten().InnerExceptions);
        else
            failures.Add(exception);
    }

    internal void ValidateParentVersions()
    {
        for (int index = 0; index < _parents.Length; index++)
        {
            Tensor parent = _parents[index];
            if (parent.DataVersion == _parentDataVersions[index])
                continue;

            string identity = string.IsNullOrEmpty(parent.Name)
                ? $"with shape [{string.Join(", ", parent.Shape)}]"
                : $"'{parent.Name}'";

            throw new InvalidOperationException(
                $"Cannot run Backward because parent tensor {identity} changed after " +
                "the forward pass. Build a new forward graph before calling Backward again.");
        }
    }
}
