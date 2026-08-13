using NNtrain;
using Xunit;

public sealed class TensorFloat16ActivationAndLossTests
{
    [Fact]
    public void ActivationPipelinePreservesFloat16AndFloat32Gradients()
    {
        float[] values =
        [
            -1.5f, -0.75f, 0.25f, 1.25f,
            0.5f, -0.25f, 1.75f, -1f,
        ];
        float[] seed =
        [
            0.5f, -0.25f, 0.75f, 0.125f,
            -0.5f, 0.25f, -0.75f, 1f,
        ];

        var float32 = new Tensor(values, [2, 4]);
        var float16 = new Tensor(
            values,
            [2, 4],
            dtype: TensorDType.Float16);
        var gamma32 = new Tensor([1f, 0.75f, 1.25f, 0.5f], [4]);
        var beta32 = new Tensor([0.1f, -0.2f, 0.3f, -0.4f], [4]);
        var gamma16 = new Tensor(
            [1f, 0.75f, 1.25f, 0.5f],
            [4],
            dtype: TensorDType.Float16);
        var beta16 = new Tensor(
            [0.1f, -0.2f, 0.3f, -0.4f],
            [4],
            dtype: TensorDType.Float16);

        Tensor output32 = float32
            .LayerNormLastDim(gamma32, beta32)
            .Relu()
            .SoftmaxLastDim();
        Tensor output16 = float16
            .LayerNormLastDim(gamma16, beta16)
            .Relu()
            .SoftmaxLastDim();
        output32.Backward(seed);
        output16.Backward(seed);

        Assert.Equal(TensorDType.Float16, output16.DType);
        Assert.Equal(TensorDType.Float32, output32.DType);
        AssertClose(output32.Data, output16.Data, 1.5e-3f);
        AssertClose(float32.Grad, float16.Grad, 3e-3f);
        AssertClose(gamma32.Grad, gamma16.Grad, 3e-3f);
        AssertClose(beta32.Grad, beta16.Grad, 3e-3f);
    }

    [Fact]
    public void CrossEntropyReducesFloat16LogitsToFloat32()
    {
        float[] values =
        [
            1.25f, -0.5f, 0.75f, 2f,
            -1f, 1.5f, 0.25f, -0.75f,
        ];
        int[] labels = [3, 1];
        var float32 = new Tensor(values, [2, 4]);
        var float16 = new Tensor(
            values,
            [2, 4],
            dtype: TensorDType.Float16);

        Tensor loss32 = float32.CrossEntropyWithLogits(
            labels,
            labelSmoothing: 0.1f);
        Tensor loss16 = float16.CrossEntropyWithLogits(
            labels,
            labelSmoothing: 0.1f);
        loss32.Backward();
        loss16.Backward();

        Assert.Equal(TensorDType.Float32, loss16.DType);
        AssertClose(loss32.Data, loss16.Data, 2e-4f);
        AssertClose(float32.Grad, float16.Grad, 4e-4f);
    }

    [Fact]
    public void ElementwiseAndMaskedActivationsPreserveFloat16()
    {
        float[] values =
        [
            -1.25f, 0.5f, 1.5f,
            0.25f, -0.75f, 2f,
            1f, -0.5f, 0.75f,
        ];
        float[] rowValues = [0.125f, -0.25f, 0.5f];
        float[] seed =
        [
            0.25f, -0.5f, 0.75f,
            -0.25f, 0.5f, -0.75f,
            1f, -1f, 0.125f,
        ];
        var input32 = new Tensor(values, [3, 3]);
        var input16 = new Tensor(
            values,
            [3, 3],
            dtype: TensorDType.Float16);
        var row32 = new Tensor(rowValues, [3]);
        var row16 = new Tensor(
            rowValues,
            [3],
            dtype: TensorDType.Float16);

        Tensor output32 = input32
            .Sin()
            .AddRowWise(row32)
            .LogSoftmaxLastDim()
            .CausalMask(-32f);
        Tensor output16 = input16
            .Sin()
            .AddRowWise(row16)
            .LogSoftmaxLastDim()
            .CausalMask(-32f);
        output32.Backward(seed);
        output16.Backward(seed);

        Assert.Equal(TensorDType.Float16, output16.DType);
        AssertClose(output32.Data, output16.Data, 1.5e-3f);
        AssertClose(input32.Grad, input16.Grad, 3e-3f);
        AssertClose(row32.Grad, row16.Grad, 3e-3f);
    }

    [Fact]
    public void EmbeddingAndDropoutPreserveFloat16Storage()
    {
        var table = new Tensor(
            [
                0.25f, 0.5f, 0.75f, 1f,
                1.25f, 1.5f, 1.75f, 2f,
                -0.25f, -0.5f, -0.75f, -1f,
            ],
            [3, 4],
            dtype: TensorDType.Float16);

        Tensor embedding = table.EmbeddingLookup([2, 0, 2], 3);
        Tensor dropped = embedding.Dropout(0.5f, new Random(73));
        dropped.Backward(Enumerable.Repeat(1f, dropped.Numel).ToArray());

        Assert.Equal(TensorDType.Float16, embedding.DType);
        Assert.Equal(TensorDType.Float16, dropped.DType);
        Assert.All(table.Grad, gradient => Assert.True(float.IsFinite(gradient)));
        Assert.Contains(table.Grad, gradient => gradient != 0f);
    }

    [Fact]
    public void PositionalEmbeddingAndResidualDropoutPreserveFloat16()
    {
        float[] tokenValues =
        [
            0.25f, -0.5f, 0.75f, 1f,
            -1.25f, 1.5f, -1.75f, 2f,
            0.5f, 0.25f, -0.25f, -0.5f,
        ];
        float[] positionValues =
        [
            0.125f, 0.25f, 0.375f, 0.5f,
            -0.125f, -0.25f, -0.375f, -0.5f,
        ];
        int[] tokens = [2, 0, 1, 2];
        var tokenTable32 = new Tensor(tokenValues, [3, 4]);
        var tokenTable16 = new Tensor(
            tokenValues,
            [3, 4],
            dtype: TensorDType.Float16);
        var positionTable32 = new Tensor(positionValues, [2, 4]);
        var positionTable16 = new Tensor(
            positionValues,
            [2, 4],
            dtype: TensorDType.Float16);

        Tensor embedding32 = tokenTable32.EmbeddingLookupWithPositions(
            positionTable32,
            tokens,
            batchSize: 2,
            sequenceLength: 2);
        Tensor embedding16 = tokenTable16.EmbeddingLookupWithPositions(
            positionTable16,
            tokens,
            batchSize: 2,
            sequenceLength: 2);
        Tensor output32 = embedding32.AddDropout(
            embedding32,
            0.25f,
            new Random(89));
        Tensor output16 = embedding16.AddDropout(
            embedding16,
            0.25f,
            new Random(89));
        float[] seed = Enumerable.Repeat(0.5f, output32.Numel).ToArray();
        output32.Backward(seed);
        output16.Backward(seed);

        Assert.Equal(TensorDType.Float16, embedding16.DType);
        Assert.Equal(TensorDType.Float16, output16.DType);
        AssertClose(output32.Data, output16.Data, 4e-3f);
        AssertClose(tokenTable32.Grad, tokenTable16.Grad, 4e-3f);
        AssertClose(positionTable32.Grad, positionTable16.Grad, 4e-3f);
    }

    private static void AssertClose(
        IReadOnlyList<float> expected,
        IReadOnlyList<float> actual,
        float tolerance)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int index = 0; index < expected.Count; index++)
        {
            Assert.True(
                MathF.Abs(expected[index] - actual[index]) <= tolerance,
                $"Mismatch at {index}: expected {expected[index]}, " +
                $"actual {actual[index]}, tolerance {tolerance}.");
        }
    }
}
