using System.Reflection;
using NNtrain;
using Xunit;

public sealed class AutogradGradientReductionNotificationTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SharedLeafIsPublishedOnceAfterItsLastConsumer(
        bool releaseGraph)
    {
        Tensor unrelated = Tensor.Scalar(7f, "unrelated");
        Tensor weight = Tensor.Scalar(4f, "shared.weight");
        Tensor sharedBranch = weight * Tensor.Scalar(2f)
            + weight * Tensor.Scalar(3f);
        Tensor output = unrelated * Tensor.Scalar(11f) + sharedBranch;
        float? unrelatedGradientWhenWeightWasPublished = null;
        var plan = new RecordingReductionPlan(tensor =>
        {
            if (ReferenceEquals(tensor, weight))
            {
                unrelatedGradientWhenWeightWasPublished =
                    Assert.Single(unrelated.Grad);
            }
        });

        using (CudaGradientReductionContext.Push(plan, 1, 37))
        {
            if (releaseGraph)
                output.BackwardAndRelease();
            else
                output.Backward();
        }

        Notification notification = Assert.Single(
            plan.ForTensor(weight));
        Assert.Equal(1, notification.DeviceIndex);
        Assert.Equal(37, notification.StepId);
        Assert.Equal(5f, Assert.Single(notification.Gradient));
        Assert.Equal(5f, Assert.Single(weight.Grad));
        Assert.Equal(0f, unrelatedGradientWhenWeightWasPublished);
        Assert.Equal(11f, Assert.Single(unrelated.Grad));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DuplicateParentEdgePublishesOnlyAfterBothContributions(
        bool releaseGraph)
    {
        Tensor weight = Tensor.Scalar(3f, "tied.weight");
        Tensor output = weight * weight;
        var plan = new RecordingReductionPlan();

        using (CudaGradientReductionContext.Push(plan, 0, 91))
        {
            if (releaseGraph)
                output.BackwardAndRelease();
            else
                output.Backward();
        }

        Notification notification = Assert.Single(
            plan.ForTensor(weight));
        Assert.Equal(6f, Assert.Single(notification.Gradient));
        Assert.Equal(6f, Assert.Single(weight.Grad));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FailedConsumerDoesNotPublishItsLeaf(
        bool releaseGraph)
    {
        Tensor weight = Tensor.Scalar(2f, "failed.weight");
        Tensor output = CreateFailingConsumer(weight);
        var plan = new RecordingReductionPlan();

        using (CudaGradientReductionContext.Push(plan, 0, 123))
        {
            InvalidOperationException exception = Assert.Throws<
                InvalidOperationException>(() =>
                {
                    if (releaseGraph)
                        output.BackwardAndRelease();
                    else
                        output.Backward();
                });
            Assert.Equal("injected backward failure", exception.Message);
        }

        Assert.Empty(plan.ForTensor(weight));

        // A failed traversal must return a cleared consumer-count workspace.
        // Reusing a pooled dictionary cannot suppress the next notification.
        Tensor recovery = weight * Tensor.Scalar(4f);
        using (CudaGradientReductionContext.Push(plan, 0, 124))
            recovery.Backward();

        Notification recovered = Assert.Single(plan.ForTensor(weight));
        Assert.Equal(124, recovered.StepId);
        Assert.Equal(4f, Assert.Single(recovered.Gradient));
    }

    private static Tensor CreateFailingConsumer(Tensor parent)
    {
        ConstructorInfo constructor = typeof(Tensor).GetConstructor(
            BindingFlags.Instance | BindingFlags.NonPublic,
            binder: null,
            types:
            [
                typeof(float[]),
                typeof(int[]),
                typeof(Tensor[]),
                typeof(string),
                typeof(TensorDType?),
            ],
            modifiers: null)
            ?? throw new InvalidOperationException(
                "Tensor autograd-result constructor was not found.");
        var result = (Tensor)constructor.Invoke(
        [
            new float[] { 0f },
            new int[] { 1 },
            new Tensor[] { parent },
            "failing.consumer",
            null,
        ]);
        result.Node.BackwardAction = static () =>
            throw new InvalidOperationException("injected backward failure");
        return result;
    }

    private sealed class RecordingReductionPlan(
        Action<Tensor>? onNotify = null) : ICudaGradientReductionPlan
    {
        private readonly List<Notification> _notifications = [];

        public void NotifyGradientReady(
            Tensor tensor,
            int deviceIndex,
            long stepId)
        {
            onNotify?.Invoke(tensor);
            _notifications.Add(new Notification(
                tensor,
                deviceIndex,
                stepId,
                tensor.Grad.ToArray()));
        }

        internal IEnumerable<Notification> ForTensor(Tensor tensor)
            => _notifications.Where(notification =>
                ReferenceEquals(notification.Tensor, tensor));
    }

    private sealed record Notification(
        Tensor Tensor,
        int DeviceIndex,
        long StepId,
        float[] Gradient);
}
