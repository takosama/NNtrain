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

    public void ZeroGrad()
    {
        foreach (IOptimizer optimizer in _optimizers)
            optimizer.ZeroGrad();
    }

    public void Step()
    {
        foreach (IOptimizer optimizer in _optimizers)
            optimizer.Step();
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
