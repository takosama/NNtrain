using NNtrain;
using Xunit;

public sealed class AutogradNodeTests
{
    [Fact]
    public void LeafNodeHasNoParentsAndNoOpBackward()
    {
        var node = new AutogradNode();

        Assert.True(node.IsLeaf);
        Assert.Empty(node.Parents);
        node.RunBackward();
    }

    [Fact]
    public void NodeCopiesAndProtectsItsParentEdges()
    {
        var first = Tensor.Scalar(1f);
        var second = Tensor.Scalar(2f);
        Tensor[] parents = [first, second];
        var node = new AutogradNode(parents);

        parents[0] = Tensor.Scalar(3f);

        Assert.False(node.IsLeaf);
        Assert.Same(first, node.Parents[0]);
        Assert.Same(second, node.Parents[1]);

        var readOnlyParents = Assert.IsAssignableFrom<IList<Tensor>>(node.Parents);
        Assert.Throws<NotSupportedException>(
            () => readOnlyParents[0] = Tensor.Scalar(4f));
    }

    [Fact]
    public void NodeOwnsAndRunsItsBackwardAction()
    {
        var node = new AutogradNode();
        int executionCount = 0;

        node.BackwardAction = () => executionCount++;
        node.RunBackward();

        Assert.Equal(1, executionCount);
        Assert.Throws<InvalidOperationException>(
            () => node.BackwardAction = () => { });
    }

    [Fact]
    public void TensorOperationsBuildExpectedParentEdges()
    {
        var left = Tensor.Scalar(2f);
        var right = Tensor.Scalar(3f);
        var sum = left + right;
        var product = sum * left;

        Assert.True(left.Node.IsLeaf);
        Assert.True(right.Node.IsLeaf);

        Assert.Collection(
            sum.Node.Parents,
            parent => Assert.Same(left, parent),
            parent => Assert.Same(right, parent));

        Assert.Collection(
            product.Node.Parents,
            parent => Assert.Same(sum, parent),
            parent => Assert.Same(left, parent));
    }
}
