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
    private List<IDisposable>? _resources;

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

    internal void RegisterResource(IDisposable resource)
    {
        ArgumentNullException.ThrowIfNull(resource);
        (_resources ??= []).Add(resource);
    }

    internal void ReleaseGraph()
    {
        if (_resources is not null)
        {
            foreach (IDisposable resource in _resources)
                resource.Dispose();
            _resources = null;
        }
        _backwardAction = NoBackward;
        _parents = [];
        _parentDataVersions = [];
        Parents = Array.Empty<Tensor>();
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
