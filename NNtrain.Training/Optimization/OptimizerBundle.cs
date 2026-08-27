namespace NNtrain.Training.Optimization;

/// <summary>
/// A stable name attached to one optimizer group. Group order is significant
/// and is preserved when the bundle computes its checkpoint leaf order.
/// </summary>
public sealed record OptimizerGroup
{
    public OptimizerGroup(string name, IOptimizer optimizer)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Name = name;
        Optimizer = optimizer
            ?? throw new ArgumentNullException(nameof(optimizer));
    }

    public string Name { get; }

    public IOptimizer Optimizer { get; }
}

/// <summary>
/// One checkpoint leaf in its deterministic bundle position.
/// </summary>
public sealed record OptimizerLeaf(
    string Name,
    string GroupName,
    int Index,
    int IndexWithinGroup,
    IOptimizer Optimizer);

/// <summary>
/// Session-owned optimizer topology with stable group names and a frozen,
/// deterministic checkpoint-leaf order. Lifecycle and state-dictionary calls
/// are forwarded to the original optimizer, so wrapping an existing
/// <see cref="CompositeOptimizer"/> does not change its checkpoint format.
/// </summary>
public sealed class OptimizerBundle : IOptimizerContainer
{
    private readonly IReadOnlyList<OptimizerGroup> _groups;
    private readonly IReadOnlyList<OptimizerLeaf> _leaves;
    private readonly IReadOnlyList<IOptimizer> _leafOptimizers;

    /// <summary>
    /// Creates a bundle from explicitly named groups. More than one group is
    /// represented by the existing <see cref="CompositeOptimizer"/> type.
    /// </summary>
    public OptimizerBundle(IEnumerable<OptimizerGroup> groups)
        : this(CreateConstruction(groups))
    {
    }

    private OptimizerBundle(BundleConstruction construction)
    {
        RootOptimizer = construction.RootOptimizer;
        _groups = Array.AsReadOnly(construction.Groups);

        var leaves = new List<OptimizerLeaf>();
        var leafOptimizers = new List<IOptimizer>();
        var seenLeaves = new HashSet<IOptimizer>(
            ReferenceEqualityComparer.Instance);
        for (int groupIndex = 0;
            groupIndex < construction.Groups.Length;
            groupIndex++)
        {
            OptimizerGroup group = construction.Groups[groupIndex];
            IReadOnlyList<IOptimizer> groupLeaves =
                OptimizerStateStream.GetLeafOptimizers(group.Optimizer);
            if (groupLeaves.Count == 0)
            {
                throw new ArgumentException(
                    $"Optimizer group '{group.Name}' has no checkpoint " +
                    "leaves.",
                    nameof(construction));
            }
            for (int localIndex = 0;
                localIndex < groupLeaves.Count;
                localIndex++)
            {
                IOptimizer leaf = groupLeaves[localIndex];
                if (!seenLeaves.Add(leaf))
                {
                    throw new ArgumentException(
                        $"Optimizer leaf '{leaf.GetType().Name}' is present " +
                        "in more than one bundle position.",
                        nameof(construction));
                }
                int index = leaves.Count;
                leaves.Add(new OptimizerLeaf(
                    $"{group.Name}/{localIndex:D4}",
                    group.Name,
                    index,
                    localIndex,
                    leaf));
                leafOptimizers.Add(leaf);
            }
        }

        _leaves = leaves.AsReadOnly();
        _leafOptimizers = leafOptimizers.AsReadOnly();
    }

    /// <summary>
    /// The exact optimizer supplied to <see cref="Wrap(IOptimizer)"/>, or the
    /// compatible optimizer constructed for explicitly named groups.
    /// </summary>
    public IOptimizer RootOptimizer { get; }

    public IReadOnlyList<OptimizerGroup> Groups => _groups;

    public IReadOnlyList<OptimizerLeaf> Leaves => _leaves;

    public IReadOnlyList<IOptimizer> LeafOptimizers => _leafOptimizers;

    /// <summary>
    /// Implements <see cref="IOptimizerContainer"/> using the frozen leaf
    /// sequence, preventing a mutable child container from changing checkpoint
    /// slot order after bundle construction.
    /// </summary>
    public IReadOnlyList<IOptimizer> Optimizers => _leafOptimizers;

    /// <summary>
    /// Wraps an optimizer without replacing it. Composite children become
    /// stable positional groups named group-0000, group-0001, and so on.
    /// A single optimizer uses the name default.
    /// </summary>
    public static OptimizerBundle Wrap(IOptimizer optimizer)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        if (optimizer is OptimizerBundle bundle)
            return bundle;

        IOptimizer[] groupOptimizers = GetGroupOptimizers(optimizer);
        var groups = new OptimizerGroup[groupOptimizers.Length];
        for (int index = 0; index < groups.Length; index++)
        {
            string name = optimizer is IOptimizerContainer
                ? $"group-{index:D4}"
                : "default";
            groups[index] = new OptimizerGroup(
                name,
                groupOptimizers[index]);
        }
        return new OptimizerBundle(
            new BundleConstruction(optimizer, groups));
    }

    /// <summary>
    /// Wraps an optimizer while assigning explicit names to its top-level
    /// groups. The number of names must match the optimizer topology.
    /// </summary>
    public static OptimizerBundle Wrap(
        IOptimizer optimizer,
        IEnumerable<string> groupNames)
    {
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(groupNames);
        IOptimizer[] groupOptimizers = GetGroupOptimizers(optimizer);
        string[] names = groupNames.ToArray();
        if (names.Length != groupOptimizers.Length)
        {
            throw new ArgumentException(
                $"Expected {groupOptimizers.Length} optimizer group names, " +
                $"but received {names.Length}.",
                nameof(groupNames));
        }
        var groups = new OptimizerGroup[groupOptimizers.Length];
        for (int index = 0; index < groups.Length; index++)
            groups[index] = new OptimizerGroup(names[index], groupOptimizers[index]);
        ValidateUniqueNames(groups, nameof(groupNames));
        return new OptimizerBundle(
            new BundleConstruction(optimizer, groups));
    }

    public void zero_grad() => RootOptimizer.zero_grad();

    public void step() => RootOptimizer.step();

    public OptimizerStateDictionary state_dict()
        => RootOptimizer.state_dict();

    public void load_state_dict(OptimizerStateDictionary state)
        => RootOptimizer.load_state_dict(state);

    private static BundleConstruction CreateConstruction(
        IEnumerable<OptimizerGroup> groups)
    {
        ArgumentNullException.ThrowIfNull(groups);
        OptimizerGroup[] materialized = groups.ToArray();
        if (materialized.Length == 0)
        {
            throw new ArgumentException(
                "An optimizer bundle requires at least one named group.",
                nameof(groups));
        }
        for (int index = 0; index < materialized.Length; index++)
        {
            if (materialized[index] is null)
            {
                throw new ArgumentException(
                    "Optimizer groups cannot contain null.",
                    nameof(groups));
            }
        }
        ValidateUniqueNames(materialized, nameof(groups));
        IOptimizer root = materialized.Length == 1
            ? materialized[0].Optimizer
            : new CompositeOptimizer(
                materialized
                    .Select(group => group.Optimizer)
                    .ToArray());
        return new BundleConstruction(root, materialized);
    }

    private static IOptimizer[] GetGroupOptimizers(IOptimizer optimizer)
    {
        if (optimizer is not IOptimizerContainer container)
            return [optimizer];
        IOptimizer[] children = container.Optimizers.ToArray();
        if (children.Length == 0)
        {
            throw new ArgumentException(
                "An optimizer container must expose at least one child.",
                nameof(optimizer));
        }
        return children;
    }

    private static void ValidateUniqueNames(
        IReadOnlyList<OptimizerGroup> groups,
        string parameterName)
    {
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (OptimizerGroup group in groups)
        {
            if (!names.Add(group.Name))
            {
                throw new ArgumentException(
                    $"Optimizer group name '{group.Name}' is duplicated.",
                    parameterName);
            }
        }
    }

    private sealed record BundleConstruction(
        IOptimizer RootOptimizer,
        OptimizerGroup[] Groups);
}
