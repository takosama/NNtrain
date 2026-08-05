using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class TensorContractTests
{
    [Fact]
    public void ConstructorCopiesInputDataAndShape()
    {
        float[] data = [1f, 2f];
        int[] shape = [2];
        var tensor = new Tensor(data, shape);

        data[0] = 99f;
        shape[0] = 1;

        AssertClose([1f, 2f], tensor.Data);
        Assert.Equal<int>([2], tensor.Shape);
    }

    [Fact]
    public void PublicViewsCannotBeMutated()
    {
        var tensor = Tensor.From1D([1f, 2f]);

        var data = Assert.IsAssignableFrom<IList<float>>(tensor.Data);
        var grad = Assert.IsAssignableFrom<IList<float>>(tensor.Grad);
        var shape = Assert.IsAssignableFrom<IList<int>>(tensor.Shape);

        Assert.Throws<NotSupportedException>(() => data[0] = 9f);
        Assert.Throws<NotSupportedException>(() => grad[0] = 9f);
        Assert.Throws<NotSupportedException>(() => shape[0] = 9);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ConstructorRejectsNonPositiveDimensions(int dimension)
    {
        Assert.Throws<ArgumentOutOfRangeException>(
            () => new Tensor([], [dimension]));
    }

    [Fact]
    public void ConstructorRejectsEmptyAndOverflowingShapes()
    {
        Assert.Throws<ArgumentException>(() => new Tensor([1f], []));
        Assert.Throws<ArgumentException>(
            () => new Tensor([1f], [int.MaxValue, 2]));
    }

    [Fact]
    public void ConstructorRejectsMismatchedDataLength()
    {
        var exception = Assert.Throws<ArgumentException>(
            () => new Tensor([1f, 2f], [3]));

        Assert.Contains("Data length 2", exception.Message);
        Assert.Contains("[3]", exception.Message);
    }

    [Fact]
    public void ConstructorRejectsNullArguments()
    {
        Assert.Throws<ArgumentNullException>(() => new Tensor(null!, [1]));
        Assert.Throws<ArgumentNullException>(() => new Tensor([1f], null!));
        Assert.Throws<ArgumentNullException>(() => new Tensor([1f], [1], null!));
        Assert.Throws<ArgumentNullException>(() => Tensor.From1D(null!));
        Assert.Throws<ArgumentNullException>(() => Tensor.From2D(null!));
    }

    [Fact]
    public void ScalarUsesOneElementShape()
    {
        var scalar = Tensor.Scalar(3f);

        Assert.Equal(1, scalar.Rank);
        Assert.Equal(1, scalar.Numel);
        Assert.Equal<int>([1], scalar.Shape);
    }

    [Fact]
    public void AnySingleElementTensorBroadcastsAsScalar()
    {
        var scalarLike = new Tensor([2f], [1, 1]);
        var vector = Tensor.From1D([1f, 2f, 3f]);

        var result = scalarLike + vector;

        Assert.Equal<int>([3], result.Shape);
        AssertClose([3f, 4f, 5f], result.Data);
    }

    [Fact]
    public void ElementWiseShapeMismatchReportsBothShapes()
    {
        var left = new Tensor([1f, 2f], [2]);
        var right = new Tensor([1f, 2f, 3f], [3]);

        var exception = Assert.Throws<ArgumentException>(() => left + right);

        Assert.Contains("[2]", exception.Message);
        Assert.Contains("[3]", exception.Message);
    }

    [Fact]
    public void ReshapeValidatesDimensionsAndElementCount()
    {
        var tensor = Tensor.From1D([1f, 2f, 3f, 4f]);

        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Reshape(2, 0));
        Assert.Throws<ArgumentException>(() => tensor.Reshape(3, 2));
    }

    [Fact]
    public void SliceUsesArgumentOutOfRangeForInvalidCoordinates()
    {
        var tensor = Tensor.From1D([1f, 2f, 3f]);

        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Slice(1, 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Slice(0, -1, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Slice(0, 0, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => tensor.Slice(0, 2, 2));
    }

    [Fact]
    public void LayerNormRequiresFinitePositiveEpsilon()
    {
        var tensor = Tensor.From1D([1f, 2f]);
        var gamma = Tensor.From1D([1f, 1f]);
        var beta = Tensor.From1D([0f, 0f]);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => tensor.LayerNormLastDim(gamma, beta, 0f));
        Assert.Throws<ArgumentOutOfRangeException>(
            () => tensor.LayerNormLastDim(gamma, beta, float.NaN));
    }

    [Theory]
    [InlineData(-0.01f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void CrossEntropyRejectsInvalidLabelSmoothing(float smoothing)
    {
        Tensor logits = Tensor.From1D([1f, 0f]);

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => logits.CrossEntropyWithLogits([0], smoothing));

        Assert.Equal("labelSmoothing", exception.ParamName);
    }

    [Fact]
    public void ToStringUsesDataFormatting()
    {
        var tensor = Tensor.From1D([1f, 2.5f]);
        Assert.Equal(tensor.DataString(), tensor.ToString());
    }
}
