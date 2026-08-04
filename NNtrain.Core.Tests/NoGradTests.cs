using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class NoGradTests
{
    [Fact]
    public void NoGradPreservesForwardValuesAndDetachesHistory()
    {
        var input = Tensor.From1D([2f, 3f]);
        Tensor output;

        using (AutogradContext.NoGrad())
            output = input * input + input;

        AssertClose([6f, 12f], output.Data);
        Assert.True(output.Node.IsDetached);
        Assert.True(output.Node.IsLeaf);
        Assert.Empty(output.Node.Parents);

        output.Backward([1f, 1f]);

        AssertClose([0f, 0f], input.Grad);
        AssertClose([1f, 1f], output.Grad);
    }

    [Fact]
    public void NoGradAllocatesGradientStorageOnlyWhenItIsNeeded()
    {
        var input = Tensor.From1D([2f, 3f]);
        Tensor detached;

        using (AutogradContext.NoGrad())
            detached = input * input;

        Assert.False(detached.HasGradientBuffer);
        AssertClose([0f, 0f], detached.Grad);
        Assert.False(detached.HasGradientBuffer);

        Tensor tracked = detached * Tensor.Scalar(2f);
        tracked.Sum().Backward();

        Assert.True(detached.HasGradientBuffer);
        AssertClose([2f, 2f], detached.Grad);
    }

    [Fact]
    public void RecordingResumesWhenScopeIsDisposed()
    {
        var input = Tensor.Scalar(3f);

        using (AutogradContext.NoGrad())
            _ = input * input;

        Tensor tracked = input * input;
        tracked.Backward();

        Assert.False(tracked.Node.IsDetached);
        Assert.Collection(
            tracked.Node.Parents,
            parent => Assert.Same(input, parent),
            parent => Assert.Same(input, parent));
        AssertClose([6f], input.Grad);
    }

    [Fact]
    public void NestedScopesKeepRecordingDisabledUntilOutermostDispose()
    {
        var input = Tensor.Scalar(2f);
        Tensor inner;
        Tensor stillInsideOuter;

        using (AutogradContext.NoGrad())
        {
            using (AutogradContext.NoGrad())
                inner = input + Tensor.Scalar(1f);

            stillInsideOuter = input + Tensor.Scalar(2f);
            Assert.False(AutogradContext.IsRecordingEnabled);
        }

        Tensor tracked = input + Tensor.Scalar(3f);

        Assert.True(inner.Node.IsDetached);
        Assert.True(stillInsideOuter.Node.IsDetached);
        Assert.False(tracked.Node.IsDetached);
        Assert.True(AutogradContext.IsRecordingEnabled);
    }

    [Fact]
    public void ScopeRestoresRecordingAfterException()
    {
        var input = Tensor.Scalar(2f);

        Action action = () =>
        {
            using (AutogradContext.NoGrad())
            {
                Assert.True((input * input).Node.IsDetached);
                throw new InvalidOperationException("expected");
            }
        };
        Assert.Throws<InvalidOperationException>(action);

        Tensor tracked = input * input;
        Assert.True(AutogradContext.IsRecordingEnabled);
        Assert.False(tracked.Node.IsDetached);
    }

    [Fact]
    public async Task NoGradFlowsAcrossAwait()
    {
        var input = Tensor.Scalar(2f);
        Tensor output;

        using (AutogradContext.NoGrad())
        {
            await Task.Yield();
            output = input * Tensor.Scalar(4f);
        }

        Assert.True(output.Node.IsDetached);
        Assert.True(AutogradContext.IsRecordingEnabled);
    }

    [Fact]
    public void DetachedResultStopsGradientWhenUsedByTrackedOperation()
    {
        var input = Tensor.Scalar(2f);
        Tensor detached;

        using (AutogradContext.NoGrad())
            detached = input * Tensor.Scalar(3f);

        Tensor tracked = detached * Tensor.Scalar(4f);
        tracked.Backward();

        AssertClose([0f], input.Grad);
        AssertClose([4f], detached.Grad);
    }

    [Fact]
    public void CompositeModuleForwardDoesNotRetainParameterHistory()
    {
        var module = new Linear(2, 2, new Random(29));
        var input = Tensor.From1D([1f, -1f]);
        Tensor output;

        using (AutogradContext.NoGrad())
            output = module.Forward(input);

        output.Sum().Backward();

        Assert.True(output.Node.IsDetached);
        Assert.All(
            module.Parameters().SelectMany(parameter => parameter.T.Grad),
            gradient => Assert.Equal(0f, gradient));
    }

    [Fact]
    public void NoGradDoesNotDisableBackwardOnExistingGraph()
    {
        var input = Tensor.Scalar(3f);
        Tensor tracked = input.Pow(2f);

        using (AutogradContext.NoGrad())
            tracked.Backward();

        AssertClose([6f], input.Grad);
    }

    [Fact]
    public async Task IndependentAsyncContextsDoNotShareNoGradState()
    {
        var input = Tensor.Scalar(2f);
        var noGradActive = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);
        var continueNoGrad = new TaskCompletionSource(
            TaskCreationOptions.RunContinuationsAsynchronously);

        Task<Tensor> detachedTask = Task.Run(async () =>
        {
            using (AutogradContext.NoGrad())
            {
                noGradActive.SetResult();
                await continueNoGrad.Task;
                return input * Tensor.Scalar(2f);
            }
        });

        await noGradActive.Task;
        Task<Tensor> trackedTask = Task.Run(
            () => input * Tensor.Scalar(3f));
        continueNoGrad.SetResult();

        Tensor[] results = await Task.WhenAll(detachedTask, trackedTask);

        Assert.True(results[0].Node.IsDetached);
        Assert.False(results[1].Node.IsDetached);
        Assert.True(AutogradContext.IsRecordingEnabled);
    }
}
