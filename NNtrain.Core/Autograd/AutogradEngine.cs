namespace NNtrain;

/// <summary>
/// Traverses an autograd graph and executes its nodes in reverse topological order.
/// </summary>
internal static class AutogradEngine
{
    internal static void Backward(Tensor output, float[]? seed)
    {
        ArgumentNullException.ThrowIfNull(output);
        ValidateSeed(output, seed);

        List<Tensor> topologicalOrder = BuildTopologicalOrder(output);

        ValidateGraphVersions(topologicalOrder);
        ClearIntermediateGradients(topologicalOrder);
        AccumulateOutputGradient(output, seed);

        for (int index = topologicalOrder.Count - 1; index >= 0; index--)
            topologicalOrder[index].Node.RunBackward();
    }

    private static List<Tensor> BuildTopologicalOrder(Tensor output)
    {
        var topologicalOrder = new List<Tensor>();
        var visited = new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        var pending = new Stack<TraversalFrame>();
        pending.Push(new TraversalFrame(output, false));

        while (pending.Count > 0)
        {
            TraversalFrame frame = pending.Pop();
            if (frame.ParentsExpanded)
            {
                topologicalOrder.Add(frame.Tensor);
                continue;
            }

            if (!visited.Add(frame.Tensor))
                continue;

            pending.Push(new TraversalFrame(frame.Tensor, true));

            IReadOnlyList<Tensor> parents = frame.Tensor.Node.Parents;
            for (int index = parents.Count - 1; index >= 0; index--)
                pending.Push(new TraversalFrame(parents[index], false));
        }

        return topologicalOrder;
    }

    private readonly record struct TraversalFrame(
        Tensor Tensor,
        bool ParentsExpanded);

    private static void ValidateSeed(Tensor output, float[]? seed)
    {
        if (seed is null)
        {
            if (output.Numel != 1)
            {
                throw new InvalidOperationException(
                    $"Backward requires a seed for non-scalar output " +
                    $"[{string.Join(", ", output.Shape)}].");
            }

            return;
        }

        if (seed.Length != output.Numel)
        {
            throw new ArgumentException(
                $"Seed length {seed.Length} does not match tensor element count {output.Numel}.",
                nameof(seed));
        }
    }

    private static void ClearIntermediateGradients(IEnumerable<Tensor> topologicalOrder)
    {
        foreach (Tensor tensor in topologicalOrder)
        {
            if (!tensor.Node.IsLeaf)
                tensor.ClearGradient();
        }
    }

    private static void ValidateGraphVersions(IEnumerable<Tensor> topologicalOrder)
    {
        foreach (Tensor tensor in topologicalOrder)
            tensor.Node.ValidateParentVersions();
    }

    private static void AccumulateOutputGradient(Tensor output, float[]? seed)
    {
        Span<float> outputGradient = output.MutableGrad;
        if (seed is null)
        {
            outputGradient[0] += 1f;
            return;
        }

        for (int index = 0; index < seed.Length; index++)
            outputGradient[index] += seed[index];
    }
}
