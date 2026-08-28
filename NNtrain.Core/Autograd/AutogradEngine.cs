namespace NNtrain;

/// <summary>
/// Traverses an autograd graph and executes its nodes in reverse topological order.
/// </summary>
internal static class AutogradEngine
{
    internal const int MaximumCachedTraversalWorkspaces = 8;

    [ThreadStatic]
    private static int _releaseGraphDepth;

    internal static bool IsReleasingGraph => _releaseGraphDepth > 0;
    // Data-parallel backward runs on thread-pool workers. Thread-static
    // traversal caches permanently stranded one large graph workspace on
    // every worker that happened to execute a shard, so long runs could grow
    // managed memory and GC cost as the pool selected new workers. A bounded-
    // by-concurrency shared pool reuses the same two workspaces for two GPUs.
    private static readonly BoundedWorkspacePool<List<Tensor>> OrderPool =
        new(MaximumCachedTraversalWorkspaces);
    private static readonly BoundedWorkspacePool<HashSet<Tensor>> VisitedPool =
        new(MaximumCachedTraversalWorkspaces);
    private static readonly BoundedWorkspacePool<Stack<TraversalFrame>>
        PendingPool = new(MaximumCachedTraversalWorkspaces);
    private static readonly BoundedWorkspacePool<Dictionary<Tensor, int>>
        LeafConsumerPool = new(MaximumCachedTraversalWorkspaces);

    internal static AutogradTraversalWorkspaceTelemetry
        TraversalWorkspaceTelemetry => new(
            OrderPool.Count,
            VisitedPool.Count,
            PendingPool.Count);

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
        Dictionary<Tensor, int>? remainingLeafConsumers = null;
        bool notifyReducerLeavesAtLastConsumer =
            CudaGradientReductionContext.HasActivePlan;
        Exception? backwardFailure = null;
        List<Exception>? cleanupFailures = releaseGraph ? [] : null;
        CudaBfp8GradientPublicationScope? bfp8PublicationScope = null;
        try
        {
            if (notifyReducerLeavesAtLastConsumer)
            {
                remainingLeafConsumers = LeafConsumerPool.Rent(static () =>
                    new Dictionary<Tensor, int>(
                        ReferenceEqualityComparer.Instance));
                BuildLeafConsumerCounts(
                    topologicalOrder,
                    remainingLeafConsumers);
            }
            if (releaseGraph)
                _releaseGraphDepth++;
            bfp8PublicationScope =
                CudaBfp8GradientPublicationScope.TryCreate(topologicalOrder);
            ValidateGraphVersions(topologicalOrder);
            ClearIntermediateGradients(topologicalOrder);
            AccumulateOutputGradient(output, seed);
            if (notifyReducerLeavesAtLastConsumer && output.Node.IsLeaf)
                CudaGradientReductionContext.NotifyLeaf(output);

            for (int index = topologicalOrder.Count - 1; index >= 0; index--)
            {
                Tensor tensor = topologicalOrder[index];
                tensor.Node.RunBackward();
                if (notifyReducerLeavesAtLastConsumer)
                {
                    NotifyLeafParentsAtLastConsumer(
                        tensor.Node.Parents,
                        remainingLeafConsumers!);
                }
                else if (tensor.Node.IsLeaf)
                {
                    CudaGradientReductionContext.NotifyLeaf(tensor);
                }
                if (releaseGraph)
                    ReleaseGraphNode(tensor, cleanupFailures!);
            }

            // Resident backward operations share one ordered stream per
            // device.  A single graph-level barrier replaces the former
            // barrier after every individual operation and also guarantees
            // that buffers are no longer in flight before graph release or
            // test/application code changes execution contexts.
            if (Tensor.ExecutionDevice == TensorDevice.Cuda
                && !releaseGraph
                && bfp8PublicationScope is null)
            {
                foreach (int deviceIndex in Tensor.CudaDeviceIndices)
                    ForgetMemoryV2Cuda.GetAccelerator(deviceIndex).Synchronize();
            }
        }
        catch (Exception exception)
        {
            backwardFailure = exception;
        }
        finally
        {
            if (bfp8PublicationScope is not null)
            {
                try
                {
                    // Quantization launches for every completed leaf share the
                    // lane compute stream. Validate them once at the graph
                    // boundary, but publish no partial gradient state when
                    // backward itself failed.
                    bfp8PublicationScope.Complete(
                        publish: backwardFailure is null);
                }
                catch (Exception exception)
                {
                    if (backwardFailure is null)
                        backwardFailure = exception;
                    else
                    {
                        cleanupFailures ??= [];
                        AddFailures(cleanupFailures, exception);
                    }
                }

                try
                {
                    bfp8PublicationScope.Dispose();
                }
                catch (Exception exception)
                {
                    cleanupFailures ??= [];
                    AddFailures(cleanupFailures, exception);
                }
            }

            if (releaseGraph)
            {
                // RunBackward can fail at any point. Revisit the complete
                // topology so both processed and not-yet-processed nodes drop
                // every saved context. Already released leases are idempotent.
                foreach (Tensor tensor in topologicalOrder)
                    ReleaseGraphNode(tensor, cleanupFailures!);
            }
            topologicalOrder.Clear();
            OrderPool.Return(topologicalOrder);
            if (remainingLeafConsumers is not null)
            {
                remainingLeafConsumers.Clear();
                LeafConsumerPool.Return(remainingLeafConsumers);
            }
            if (releaseGraph)
                _releaseGraphDepth--;
        }

        ThrowAfterCleanup(backwardFailure, cleanupFailures);
    }

    private static void ReleaseGraphNode(
        Tensor tensor,
        List<Exception> cleanupFailures)
    {
        bool wasNonLeaf = !tensor.Node.IsLeaf;
        bool hasLeases = tensor.Node.HasLeases;
        if (!wasNonLeaf && !hasLeases)
            return;

        if (wasNonLeaf)
        {
            try
            {
                // Existing context callbacks preserve the operation's stream
                // ordering, while Tensor returns only intermediate buffers.
                tensor.ReleaseCudaGraphBuffers();
            }
            catch (Exception exception)
            {
                AddFailures(cleanupFailures, exception);
            }
        }

        try
        {
            tensor.Node.ReleaseGraph();
        }
        catch (Exception exception)
        {
            AddFailures(cleanupFailures, exception);
        }
    }

    private static void ThrowAfterCleanup(
        Exception? backwardFailure,
        List<Exception>? cleanupFailures)
    {
        if (backwardFailure is null
            && (cleanupFailures is null || cleanupFailures.Count == 0))
        {
            return;
        }

        if (cleanupFailures is null || cleanupFailures.Count == 0)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo
                .Capture(backwardFailure!)
                .Throw();
            return;
        }

        if (backwardFailure is not null)
            cleanupFailures.Insert(0, backwardFailure);
        throw new AggregateException(
            backwardFailure is null
                ? "One or more autograd resources failed to release."
                : "Backward failed, and one or more autograd resources also " +
                    "failed to release.",
            cleanupFailures);
    }

    private static void AddFailures(
        List<Exception> failures,
        Exception exception)
    {
        if (exception is AggregateException aggregate)
            failures.AddRange(aggregate.Flatten().InnerExceptions);
        else
            failures.Add(exception);
    }

    private static void BuildLeafConsumerCounts(
        IReadOnlyList<Tensor> topologicalOrder,
        Dictionary<Tensor, int> remainingLeafConsumers)
    {
        remainingLeafConsumers.Clear();
        foreach (Tensor tensor in topologicalOrder)
        {
            foreach (Tensor parent in tensor.Node.Parents)
            {
                if (!parent.Node.IsLeaf)
                    continue;
                remainingLeafConsumers.TryGetValue(
                    parent,
                    out int consumers);
                remainingLeafConsumers[parent] = checked(consumers + 1);
            }
        }
    }

    private static void NotifyLeafParentsAtLastConsumer(
        IReadOnlyList<Tensor> parents,
        Dictionary<Tensor, int> remainingLeafConsumers)
    {
        foreach (Tensor parent in parents)
        {
            if (!parent.Node.IsLeaf)
                continue;
            if (!remainingLeafConsumers.TryGetValue(
                    parent,
                    out int consumers)
                || consumers <= 0)
            {
                throw new InvalidOperationException(
                    "Autograd leaf-consumer accounting became inconsistent.");
            }
            if (consumers == 1)
            {
                remainingLeafConsumers.Remove(parent);
                CudaGradientReductionContext.NotifyLeaf(parent);
            }
            else
            {
                remainingLeafConsumers[parent] = consumers - 1;
            }
        }
    }

    private static List<Tensor> BuildTopologicalOrder(Tensor output)
    {
        List<Tensor> topologicalOrder = OrderPool.Rent(static () => []);
        HashSet<Tensor> visited = VisitedPool.Rent(
            static () => new HashSet<Tensor>(
                ReferenceEqualityComparer.Instance));
        Stack<TraversalFrame> pending = PendingPool.Rent(static () => []);
        visited.Clear();
        pending.Clear();
        pending.Push(new TraversalFrame(output, false));
        try
        {
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
        catch
        {
            topologicalOrder.Clear();
            OrderPool.Return(topologicalOrder);
            throw;
        }
        finally
        {
            visited.Clear();
            pending.Clear();
            VisitedPool.Return(visited);
            PendingPool.Return(pending);
        }
    }

    private readonly record struct TraversalFrame(
        Tensor Tensor,
        bool ParentsExpanded);

    private sealed class BoundedWorkspacePool<T>(int capacity)
        where T : class
    {
        private readonly System.Collections.Concurrent.ConcurrentBag<T>
            _items = [];
        private int _count;

        internal int Count => Volatile.Read(ref _count);

        internal T Rent(Func<T> factory)
        {
            ArgumentNullException.ThrowIfNull(factory);
            if (_items.TryTake(out T? item))
            {
                Interlocked.Decrement(ref _count);
                return item;
            }
            return factory();
        }

        internal void Return(T item)
        {
            ArgumentNullException.ThrowIfNull(item);
            while (true)
            {
                int count = Volatile.Read(ref _count);
                if (count >= capacity)
                    return;
                if (Interlocked.CompareExchange(
                        ref _count,
                        count + 1,
                        count) != count)
                {
                    continue;
                }

                try
                {
                    _items.Add(item);
                    return;
                }
                catch
                {
                    Interlocked.Decrement(ref _count);
                    throw;
                }
            }
        }
    }

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
        if (output.TryAccumulateCudaOutputGradient(seed))
            return;

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

internal readonly record struct AutogradTraversalWorkspaceTelemetry(
    int TopologicalOrderCount,
    int VisitedSetCount,
    int PendingStackCount);
