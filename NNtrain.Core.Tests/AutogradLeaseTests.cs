using NNtrain;
using Xunit;

public sealed class AutogradLeaseTests
{
    [Fact]
    public void OwnedLeasePublishesMetadataAndReleasesExactlyOnce()
    {
        var context = new TrackedContext();
        var metadata = new AutogradLeaseMetadata(
            TensorDevice.Cuda,
            deviceIndex: 1,
            TensorDType.BFloat16,
            generation: 17,
            AutogradLeaseOwnership.Owned);
        AutogradLease<TrackedContext> lease =
            AutogradLease<TrackedContext>.Own(
                context,
                metadata,
                static saved => saved.Release());

        Assert.Equal(TensorDevice.Cuda, lease.Metadata.Device);
        Assert.Equal(1, lease.Metadata.DeviceIndex);
        Assert.Equal(TensorDType.BFloat16, lease.Metadata.DType);
        Assert.Equal(17, lease.Metadata.Generation);
        Assert.Equal(AutogradLeaseOwnership.Owned, lease.Metadata.Ownership);

        Parallel.For(0, 128, _ => lease.Dispose());

        Assert.True(lease.IsReleased);
        Assert.Equal(1, context.ReleaseCount);
        Assert.Throws<ObjectDisposedException>(
            () => lease.Use(static _ => { }));
    }

    [Fact]
    public void BorrowedLeaseDropsReferenceWithoutReleasingContext()
    {
        var context = new TrackedContext();
        var metadata = new AutogradLeaseMetadata(
            TensorDevice.Cpu,
            deviceIndex: 0,
            TensorDType.Float32,
            generation: 3,
            AutogradLeaseOwnership.Borrowed);
        AutogradLease<TrackedContext> lease =
            AutogradLease<TrackedContext>.Borrow(context, metadata);
        int uses = 0;

        lease.Use(_ => uses++);
        lease.Dispose();
        lease.Dispose();

        Assert.Equal(1, uses);
        Assert.Equal(0, context.ReleaseCount);
        Assert.True(lease.IsReleased);
    }

    [Fact]
    public void NodeReleaseAttemptsEveryLeaseAndAggregatesFailures()
    {
        var first = new TrackedContext(throwOnRelease: true);
        var second = new TrackedContext();
        var third = new TrackedContext(throwOnRelease: true);
        var node = new AutogradNode(
            [Tensor.Scalar(1f), Tensor.Scalar(2f)]);
        node.RegisterLease(Owned(first));
        node.RegisterLease(Owned(second));
        node.RegisterLease(Owned(third));

        AggregateException exception = Assert.Throws<AggregateException>(
            node.ReleaseGraph);

        Assert.Equal(2, exception.InnerExceptions.Count);
        Assert.Equal(1, first.ReleaseCount);
        Assert.Equal(1, second.ReleaseCount);
        Assert.Equal(1, third.ReleaseCount);
        Assert.Empty(node.Parents);
        Assert.False(node.HasLeases);

        node.ReleaseGraph();
        Assert.Equal(1, first.ReleaseCount);
        Assert.Equal(1, second.ReleaseCount);
        Assert.Equal(1, third.ReleaseCount);
    }

    [Fact]
    public void RetainedBackwardKeepsLeaseUntilBackwardAndRelease()
    {
        var context = new TrackedContext();
        Tensor leaf = Tensor.Scalar(2f);
        leaf.Node.SetBackward(
            Owned(context),
            static saved => saved.RecordBackward());

        leaf.Backward();
        leaf.Backward();

        Assert.Equal(2, context.BackwardCount);
        Assert.Equal(0, context.ReleaseCount);

        leaf.BackwardAndRelease();

        Assert.Equal(3, context.BackwardCount);
        Assert.Equal(1, context.ReleaseCount);
        Assert.False(leaf.Node.HasLeases);
    }

    [Fact]
    public void BackwardFailureReleasesProcessedAndUnprocessedNodes()
    {
        var processed = new TrackedContext();
        var failing = new TrackedContext();
        var unprocessed = new TrackedContext();

        Tensor left = Tensor.Scalar(2f);
        Tensor right = Tensor.Scalar(3f);
        Tensor sibling = left + right;
        sibling.Node.RegisterLease(Owned(unprocessed));

        Tensor failingLeaf = Tensor.Scalar(4f);
        failingLeaf.Node.SetBackward(
            Owned(failing),
            static _ => throw new InvalidOperationException("backward failed"));

        // Topological order visits sibling before failingLeaf; reverse-mode
        // therefore fails at failingLeaf before sibling has run.
        Tensor root = sibling + failingLeaf;
        root.Node.RegisterLease(Owned(processed));

        InvalidOperationException exception =
            Assert.Throws<InvalidOperationException>(
                () => root.BackwardAndRelease());

        Assert.Equal("backward failed", exception.Message);
        Assert.Equal(1, processed.ReleaseCount);
        Assert.Equal(1, failing.ReleaseCount);
        Assert.Equal(1, unprocessed.ReleaseCount);
        Assert.False(root.Node.HasLeases);
        Assert.False(sibling.Node.HasLeases);
        Assert.False(failingLeaf.Node.HasLeases);
    }

    [Fact]
    public void BackwardAndCleanupFailuresAreReportedAfterEveryRelease()
    {
        var failing = new TrackedContext(throwOnRelease: true);
        var unprocessed = new TrackedContext(throwOnRelease: true);
        var processed = new TrackedContext();

        Tensor sibling = Tensor.Scalar(1f) + Tensor.Scalar(2f);
        sibling.Node.RegisterLease(Owned(unprocessed));
        Tensor failingLeaf = Tensor.Scalar(3f);
        failingLeaf.Node.SetBackward(
            Owned(failing),
            static _ => throw new InvalidOperationException("primary"));
        Tensor root = sibling + failingLeaf;
        root.Node.RegisterLease(Owned(processed));

        AggregateException exception = Assert.Throws<AggregateException>(
            () => root.BackwardAndRelease());

        Assert.Collection(
            exception.InnerExceptions,
            error => Assert.Equal("primary", error.Message),
            error => Assert.Equal("release failed", error.Message),
            error => Assert.Equal("release failed", error.Message));
        Assert.Equal(1, processed.ReleaseCount);
        Assert.Equal(1, failing.ReleaseCount);
        Assert.Equal(1, unprocessed.ReleaseCount);
    }

    [Fact]
    public void RejectedTypedRegistrationRollsBackOwnedLease()
    {
        var context = new TrackedContext();
        var node = new AutogradNode();
        node.BackwardAction = static () => { };

        Assert.Throws<InvalidOperationException>(
            () => node.SetBackward(Owned(context), static _ => { }));

        Assert.Equal(1, context.ReleaseCount);
        Assert.False(node.HasLeases);
    }

    [Fact]
    public void LegacyResourceAdapterSurvivesConcurrentGraphRelease()
    {
        var resource = new TrackedDisposable();
        var node = new AutogradNode(
            [Tensor.Scalar(1f), Tensor.Scalar(2f)]);
        node.RegisterResource(resource);

        Parallel.For(0, 128, _ => node.ReleaseGraph());

        Assert.Equal(1, resource.DisposeCount);
        Assert.False(node.HasLeases);
        Assert.Empty(node.Parents);
    }

    private static AutogradLease<TrackedContext> Owned(
        TrackedContext context)
        => AutogradLease<TrackedContext>.Own(
            context,
            new AutogradLeaseMetadata(
                TensorDevice.Cpu,
                deviceIndex: 0,
                TensorDType.Float32,
                generation: 0,
                AutogradLeaseOwnership.Owned),
            static saved => saved.Release());

    private sealed class TrackedContext(bool throwOnRelease = false)
    {
        private int _backwardCount;
        private int _releaseCount;

        internal int BackwardCount => Volatile.Read(ref _backwardCount);
        internal int ReleaseCount => Volatile.Read(ref _releaseCount);

        internal void RecordBackward()
            => Interlocked.Increment(ref _backwardCount);

        internal void Release()
        {
            Interlocked.Increment(ref _releaseCount);
            if (throwOnRelease)
                throw new InvalidOperationException("release failed");
        }
    }

    private sealed class TrackedDisposable : IDisposable
    {
        private int _disposeCount;

        internal int DisposeCount => Volatile.Read(ref _disposeCount);

        public void Dispose() => Interlocked.Increment(ref _disposeCount);
    }
}
