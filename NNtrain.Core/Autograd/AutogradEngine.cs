namespace NNtrain;

/// <summary>
/// Traverses an autograd graph and executes its nodes in reverse topological order.
/// </summary>
internal static class AutogradEngine
{
    // Data-parallel backward runs on thread-pool workers. Thread-static
    // traversal caches permanently stranded one large graph workspace on
    // every worker that happened to execute a shard, so long runs could grow
    // managed memory and GC cost as the pool selected new workers. A bounded-
    // by-concurrency shared pool reuses the same two workspaces for two GPUs.
    private static readonly System.Collections.Concurrent.ConcurrentBag<
        List<Tensor>> OrderPool = [];
    private static readonly System.Collections.Concurrent.ConcurrentBag<
        HashSet<Tensor>> VisitedPool = [];
    private static readonly System.Collections.Concurrent.ConcurrentBag<
        Stack<TraversalFrame>> PendingPool = [];

    internal static void Backward(Tensor output, float[]? seed)
        => Backward(output, seed, releaseGraph: false);

    internal static void Backward(
        Tensor output,
        float[]? seed,
        bool releaseGraph)
    {
        ArgumentNullException.ThrowIfNull(output);
        ValidateSeed(output, seed);

        List<Tensor> topologicalOrder = BuildTopologicalOrder(output);
        try
        {
            ValidateGraphVersions(topologicalOrder);
            ClearIntermediateGradients(topologicalOrder);
            AccumulateOutputGradient(output, seed);

            for (int index = topologicalOrder.Count - 1; index >= 0; index--)
            {
                Tensor tensor = topologicalOrder[index];
                tensor.Node.RunBackward();
                if (tensor.Node.IsLeaf)
                    CudaGradientReductionContext.NotifyLeaf(tensor);
                if (releaseGraph && !tensor.Node.IsLeaf)
                {
                    // Work is ordered on the device stream. The allocation can
                    // be reused by an earlier graph node after its final use is
                    // queued, without retaining the entire graph until the end.
                    tensor.ReleaseCudaGraphBuffers();
                    tensor.Node.ReleaseGraph();
                }
            }

            // Resident backward operations share one ordered stream per
            // device.  A single graph-level barrier replaces the former
            // barrier after every individual operation and also guarantees
            // that buffers are no longer in flight before graph release or
            // test/application code changes execution contexts.
            if (Tensor.ExecutionDevice == TensorDevice.Cuda && !releaseGraph)
            {
                foreach (int deviceIndex in Tensor.CudaDeviceIndices)
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
            }

        }
        finally
        {
            topologicalOrder.Clear();
            OrderPool.Add(topologicalOrder);
        }
    }

    private static List<Tensor> BuildTopologicalOrder(Tensor output)
    {
        List<Tensor> topologicalOrder = OrderPool.TryTake(out var order)
            ? order
            : [];
        HashSet<Tensor> visited = VisitedPool.TryTake(out var visitedSet)
            ? visitedSet
            : new HashSet<Tensor>(ReferenceEqualityComparer.Instance);
        Stack<TraversalFrame> pending = PendingPool.TryTake(out var pendingStack)
            ? pendingStack
            : [];
        visited.Clear();
        pending.Clear();
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

        visited.Clear();
        pending.Clear();
        VisitedPool.Add(visited);
        PendingPool.Add(pending);
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
