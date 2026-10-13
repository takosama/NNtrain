namespace NNtrain;

/// <summary>Preserves optimizer slot order and rejects repeated parameter ownership.</summary>
internal static class OptimizerParameters
{
    internal static List<Parameter> Materialize(
        IEnumerable<Parameter> parameters,
        string optimizerName)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        List<Parameter> result = [];
        var seenParameters =
            new HashSet<Parameter>(ReferenceEqualityComparer.Instance);

        foreach (Parameter parameter in parameters)
        {
            if (parameter is null)
            {
                throw new ArgumentException(
                    "Optimizer parameters cannot contain null.",
                    nameof(parameters));
            }

            if (!seenParameters.Add(parameter))
            {
                throw new ArgumentException(
                    $"Parameter '{parameter.Name}' was supplied to {optimizerName} " +
                    "more than once.",
                    nameof(parameters));
            }

            result.Add(parameter);
        }

        return result;
    }
}
