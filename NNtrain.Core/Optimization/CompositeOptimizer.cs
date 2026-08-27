namespace NNtrain;

public sealed class CompositeOptimizer : IOptimizer
{
    private readonly IOptimizer[] _optimizers;

    public CompositeOptimizer(params IOptimizer[] optimizers)
    {
        ArgumentNullException.ThrowIfNull(optimizers);
        if (optimizers.Length == 0)
        {
            throw new ArgumentException(
                "A composite optimizer requires at least one optimizer.",
                nameof(optimizers));
        }

        var seenOptimizers =
            new HashSet<IOptimizer>(ReferenceEqualityComparer.Instance);
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);
        var parameters = new List<Parameter>();
        for (int index = 0; index < optimizers.Length; index++)
        {
            IOptimizer? optimizer = optimizers[index];
            if (optimizer is null)
            {
                throw new ArgumentException(
                    "Composite optimizers cannot contain null.",
                    nameof(optimizers));
            }

            if (!seenOptimizers.Add(optimizer))
            {
                throw new ArgumentException(
                    "The same optimizer instance cannot be registered twice.",
                    nameof(optimizers));
            }

            foreach (Parameter parameter in GetParameters(optimizer))
            {
                if (!seenParameters.Add(parameter))
                {
                    throw new ArgumentException(
                        $"Parameter '{parameter.Name}' is managed by more " +
                        "than one optimizer in the composite and would be " +
                        "updated twice.",
                        nameof(optimizers));
                }
                parameters.Add(parameter);
            }
        }

        _optimizers = optimizers.ToArray();
        Optimizers = Array.AsReadOnly(_optimizers);
        Parameters = parameters.AsReadOnly();
    }

    public IReadOnlyList<IOptimizer> Optimizers { get; }

    internal IReadOnlyList<Parameter> Parameters { get; }

    internal void ZeroGrad()
    {
        if (Tensor.ExecutionDevice == TensorDevice.Cuda)
        {
            // The composite already owns a deduplicated parameter list. Avoid
            // starting a separate parallel traversal for NekoMuon followed by
            // a second AdamW traversal; CUDA gradient arenas collapse these
            // calls to one clear per bucket through their dirty gates.
            foreach (Parameter parameter in Parameters)
                parameter.T.ClearGradient();
            return;
        }
        foreach (IOptimizer optimizer in _optimizers)
            optimizer.zero_grad();
    }

    public void zero_grad() => ZeroGrad();

    internal void Step()
    {
        foreach (IOptimizer optimizer in _optimizers)
            optimizer.step();
    }

    public void step() => Step();

    public OptimizerStateDictionary state_dict()
        => new(
            "CompositeOptimizer",
            StateJson: null,
            _optimizers.Select(optimizer => optimizer.state_dict()).ToArray());

    public void load_state_dict(OptimizerStateDictionary state)
    {
        ArgumentNullException.ThrowIfNull(state);
        if (!string.Equals(
                state.OptimizerType,
                "CompositeOptimizer",
                StringComparison.Ordinal)
            || state.Children is null
            || state.Children.Length != _optimizers.Length)
        {
            throw new ArgumentException(
                "Composite optimizer state does not match its children.",
                nameof(state));
        }

        for (int index = 0; index < _optimizers.Length; index++)
            _optimizers[index].load_state_dict(state.Children[index]);
    }

    private static IReadOnlyList<Parameter> GetParameters(
        IOptimizer optimizer)
        => optimizer switch
        {
            AdamW value => value.Parameters,
            Lion value => value.Parameters,
            NekoMuon value => value.Parameters,
            GainShareAdamW value => value.Parameters,
            CompositeOptimizer value => value.Parameters,
            _ => Array.Empty<Parameter>(),
        };
}
