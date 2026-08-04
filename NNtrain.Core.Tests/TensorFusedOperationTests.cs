using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class TensorFusedOperationTests
{
    [Fact]
    public void FusedMultiHeadAttentionMatchesComposedRankTwoAttention()
    {
        const int sequence = 4;
        const int modelWidth = 8;
        const int heads = 2;
        float[] values = Pattern(sequence * 3 * modelWidth, 29, 0.017f);
        float[] seed = Pattern(sequence * modelWidth, 17, 0.021f);
        var fusedInput = new Tensor(values, [sequence, 3 * modelWidth]);
        var referenceInput = new Tensor(values, [sequence, 3 * modelWidth]);

        Tensor fused = fusedInput.FusedMultiHeadAttention(
            heads,
            causal: true);
        Tensor reference = ReferenceAttention(
            referenceInput,
            heads,
            causal: true);
        fused.Backward(seed);
        reference.Backward(seed);

        AssertClose(reference.Data, fused.Data, 2e-5f);
        AssertClose(referenceInput.Grad, fusedInput.Grad, 3e-4f);
    }

    [Fact]
    public void FusedMultiHeadAttentionMatchesComposedBatchedAttention()
    {
        const int batch = 3;
        const int sequence = 4;
        const int modelWidth = 8;
        const int heads = 2;
        float[] values = Pattern(
            batch * sequence * 3 * modelWidth,
            31,
            0.013f);
        float[] seed = Pattern(
            batch * sequence * modelWidth,
            19,
            0.016f);
        var fusedInput = new Tensor(
            values,
            [batch, sequence, 3 * modelWidth]);
        var referenceInput = new Tensor(
            values,
            [batch, sequence, 3 * modelWidth]);

        Tensor fused = fusedInput.FusedMultiHeadAttention(heads);
        Tensor reference = ReferenceAttention(referenceInput, heads, false);
        fused.Backward(seed);
        reference.Backward(seed);

        AssertClose(reference.Data, fused.Data, 2e-5f);
        AssertClose(referenceInput.Grad, fusedInput.Grad, 3e-4f);
    }

    [Fact]
    public void FusedLinearReluMatchesSeparateOperations()
    {
        const int rows = 5;
        const int inputWidth = 8;
        const int outputWidth = 7;
        float[] inputValues = Pattern(rows * inputWidth, 23, 0.025f);
        float[] weightValues = Pattern(
            outputWidth * inputWidth,
            17,
            0.019f);
        float[] biasValues = Pattern(outputWidth, 11, 0.01f);
        float[] seed = Pattern(rows * outputWidth, 13, 0.031f);
        var fusedInput = new Tensor(inputValues, [rows, inputWidth]);
        var fusedWeight = new Tensor(
            weightValues,
            [outputWidth, inputWidth]);
        Tensor fusedBias = Tensor.From1D(biasValues);
        var referenceInput = new Tensor(inputValues, [rows, inputWidth]);
        var referenceWeight = new Tensor(
            weightValues,
            [outputWidth, inputWidth]);
        Tensor referenceBias = Tensor.From1D(biasValues);

        Tensor fused = fusedInput.MatMulTransposedRightAddRowRelu(
            fusedWeight,
            fusedBias);
        Tensor reference = referenceInput
            .MatMulTransposedRightAddRow(referenceWeight, referenceBias)
            .Relu();
        fused.Backward(seed);
        reference.Backward(seed);

        AssertClose(reference.Data, fused.Data);
        AssertClose(referenceInput.Grad, fusedInput.Grad, 2e-5f);
        AssertClose(referenceWeight.Grad, fusedWeight.Grad, 2e-5f);
        AssertClose(referenceBias.Grad, fusedBias.Grad, 2e-5f);
    }

    [Fact]
    public void FusedResidualLayerNormMatchesSeparateOperations()
    {
        const int batch = 2;
        const int rows = 3;
        const int columns = 8;
        float[] leftValues = Pattern(batch * rows * columns, 23, 0.02f);
        float[] rightValues = Pattern(batch * rows * columns, 19, 0.017f);
        float[] gammaValues = Enumerable.Range(0, columns)
            .Select(index => 0.8f + index * 0.03f)
            .ToArray();
        float[] betaValues = Pattern(columns, 9, 0.011f);
        float[] seed = Pattern(batch * rows * columns, 17, 0.014f);
        var fusedLeft = new Tensor(leftValues, [batch, rows, columns]);
        var fusedRight = new Tensor(rightValues, [batch, rows, columns]);
        Tensor fusedGamma = Tensor.From1D(gammaValues);
        Tensor fusedBeta = Tensor.From1D(betaValues);
        var referenceLeft = new Tensor(leftValues, [batch, rows, columns]);
        var referenceRight = new Tensor(rightValues, [batch, rows, columns]);
        Tensor referenceGamma = Tensor.From1D(gammaValues);
        Tensor referenceBeta = Tensor.From1D(betaValues);

        Tensor fused = fusedLeft.AddLayerNormLastDim(
            fusedRight,
            fusedGamma,
            fusedBeta);
        Tensor reference = (referenceLeft + referenceRight)
            .LayerNormLastDim(referenceGamma, referenceBeta);
        fused.Backward(seed);
        reference.Backward(seed);

        AssertClose(reference.Data, fused.Data, 2e-5f);
        AssertClose(referenceLeft.Grad, fusedLeft.Grad, 3e-4f);
        AssertClose(referenceRight.Grad, fusedRight.Grad, 3e-4f);
        AssertClose(referenceGamma.Grad, fusedGamma.Grad, 3e-4f);
        AssertClose(referenceBeta.Grad, fusedBeta.Grad, 2e-5f);
    }

    [Fact]
    public void FusedCrossEntropyMatchesOneHotComposition()
    {
        const int batch = 4;
        const int classes = 7;
        float[] values = Pattern(batch * classes, 19, 0.12f);
        int[] labels = [0, 3, 6, 2];
        float[] targets = new float[batch * classes];
        for (int row = 0; row < batch; row++)
            targets[row * classes + labels[row]] = 1f;
        var fusedInput = new Tensor(values, [batch, classes]);
        var referenceInput = new Tensor(values, [batch, classes]);

        Tensor fused = fusedInput.CrossEntropyWithLogits(labels);
        Tensor reference = (Tensor.Scalar(0f)
            - (new Tensor(targets, [batch, classes])
                * referenceInput.LogSoftmaxLastDim()).Sum())
            / Tensor.Scalar(batch);
        fused.Backward();
        reference.Backward();

        AssertClose(reference.Data, fused.Data, 2e-5f);
        AssertClose(referenceInput.Grad, fusedInput.Grad, 2e-5f);
    }

    [Fact]
    public void OwnedTensorSkipsCopyAndAllocatesGradientLazily()
    {
        float[] values = [1f, 2f, 3f, 4f];
        Tensor owned = Tensor.FromOwnedData(values, [2, 2]);

        Assert.False(owned.HasGradientBuffer);
        values[2] = 9f;
        Assert.Equal(9f, owned.Data[2]);

        Tensor tracked = owned * Tensor.Scalar(2f);
        Assert.True(owned.HasGradientBuffer);
        tracked.Sum().Backward();
        AssertClose([2f, 2f, 2f, 2f], owned.Grad);
    }

    private static Tensor ReferenceAttention(
        Tensor projected,
        int numHeads,
        bool causal)
    {
        int featureDimension = projected.Rank - 1;
        int modelWidth = projected.Shape[^1] / 3;
        int headWidth = modelWidth / numHeads;
        float scale = 1f / MathF.Sqrt(headWidth);
        Tensor query = projected.Slice(featureDimension, 0, modelWidth);
        Tensor key = projected.Slice(
            featureDimension,
            modelWidth,
            modelWidth);
        Tensor value = projected.Slice(
            featureDimension,
            2 * modelWidth,
            modelWidth);
        var parts = new Tensor[numHeads];

        for (int head = 0; head < numHeads; head++)
        {
            int offset = head * headWidth;
            Tensor headQuery = query.Slice(
                featureDimension,
                offset,
                headWidth);
            Tensor headKey = key.Slice(
                featureDimension,
                offset,
                headWidth);
            Tensor headValue = value.Slice(
                featureDimension,
                offset,
                headWidth);
            Tensor scores = projected.Rank == 2
                ? headQuery.MatMulTransposedRight(headKey)
                : headQuery.BatchedMatMulTransposedRight(headKey);
            scores *= Tensor.Scalar(scale);
            if (causal)
                scores = scores.CausalMask();
            Tensor probabilities = scores.SoftmaxLastDim();
            parts[head] = projected.Rank == 2
                ? probabilities.MatMul(headValue)
                : probabilities.BatchedMatMul(headValue);
        }

        return Tensor.Concat(featureDimension, parts);
    }

    private static float[] Pattern(int length, int modulus, float scale)
        => Enumerable.Range(0, length)
            .Select(index => (index % modulus - modulus / 2) * scale)
            .ToArray();
}
