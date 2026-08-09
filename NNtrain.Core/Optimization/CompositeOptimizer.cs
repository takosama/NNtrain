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
        }

        _optimizers = optimizers.ToArray();
        Optimizers = Array.AsReadOnly(_optimizers);
    }

    public IReadOnlyList<IOptimizer> Optimizers { get; }

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
}
