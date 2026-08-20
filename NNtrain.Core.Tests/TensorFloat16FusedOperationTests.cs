using NNtrain;
using Xunit;

[Collection(TensorSimdCollection.Name)]
public sealed class TensorFloat16FusedOperationTests
{
    [Fact]
    public void AttentionUsesPackedFloat16InputsAndFloat32Gradients()
    {
        const int batch = 2;
        const int sequence = 3;
        const int modelWidth = 16;
        const int heads = 2;
        float[] source = Pattern(batch * sequence * 3 * modelWidth, 0.021f);
        float[] seed = Pattern(batch * sequence * modelWidth, 0.017f);
        Tensor halfInput = HalfTensor(
            source,
            [batch, sequence, 3 * modelWidth]);
        Tensor referenceInput = FloatReference(halfInput);

        Tensor actual = halfInput.FusedMultiHeadAttention(heads, causal: true);
        Tensor expected = referenceInput.FusedMultiHeadAttention(
            heads,
            causal: true);
        actual.Backward(seed);
        expected.Backward(seed);

        Assert.Equal(TensorDType.Float16, actual.DType);
        Assert.Equal(TensorDType.Float32, actual.AccumulationDType);
        AssertCloseQuantized(expected.Data, actual.Data, 2e-5f);
        AssertClose(referenceInput.Grad, halfInput.Grad, 4e-4f);
    }

    [Fact]
    public void FusedLinearReluAndResidualLayerNormUseFloat16Storage()
    {
        const int rows = 4;
        const int inputWidth = 8;
        const int outputWidth = 9;
        Tensor halfInput = HalfTensor(
            Pattern(rows * inputWidth, 0.027f, offset: 0.35f),
            [rows, inputWidth]);
        Tensor halfWeight = HalfTensor(
            Pattern(outputWidth * inputWidth, 0.013f, offset: 0.2f),
            [outputWidth, inputWidth]);
        Tensor halfBias = HalfTensor(
            Enumerable.Range(0, outputWidth)
                .Select(index => 0.3f + 0.01f * index)
                .ToArray(),
            [outputWidth]);
        Tensor referenceInput = FloatReference(halfInput);
        Tensor referenceWeight = FloatReference(halfWeight);
        Tensor referenceBias = FloatReference(halfBias);
        float[] linearSeed = Pattern(rows * outputWidth, 0.011f);

        Tensor actualLinear = halfInput.MatMulTransposedRightAddRowRelu(
            halfWeight,
            halfBias);
        Tensor expectedLinear = referenceInput
            .MatMulTransposedRightAddRowRelu(
                referenceWeight,
                referenceBias);
        actualLinear.Backward(linearSeed);
        expectedLinear.Backward(linearSeed);

        Assert.Equal(TensorDType.Float16, actualLinear.DType);
        AssertCloseQuantized(expectedLinear.Data, actualLinear.Data, 2e-5f);
        AssertClose(referenceInput.Grad, halfInput.Grad, 8e-4f);
        AssertClose(referenceWeight.Grad, halfWeight.Grad, 8e-4f);
        AssertClose(referenceBias.Grad, halfBias.Grad, 2e-5f);

        const int layerNormRows = 3;
        const int columns = 8;
        Tensor halfLeft = HalfTensor(
            Pattern(layerNormRows * columns, 0.019f),
            [layerNormRows, columns]);
        Tensor halfRight = HalfTensor(
            Pattern(layerNormRows * columns, 0.015f, offset: 0.07f),
            [layerNormRows, columns]);
        Tensor halfGamma = HalfTensor(
            Enumerable.Range(0, columns)
                .Select(index => 0.8f + index * 0.025f)
                .ToArray(),
            [columns]);
        Tensor halfBeta = HalfTensor(Pattern(columns, 0.009f), [columns]);
        Tensor referenceLeft = FloatReference(halfLeft);
        Tensor referenceRight = FloatReference(halfRight);
        Tensor referenceGamma = FloatReference(halfGamma);
        Tensor referenceBeta = FloatReference(halfBeta);
        float[] layerNormSeed = Pattern(layerNormRows * columns, 0.013f);

        Tensor actualLayerNorm = halfLeft.AddLayerNormLastDim(
            halfRight,
            halfGamma,
            halfBeta);
        Tensor expectedLayerNorm = referenceLeft.AddLayerNormLastDim(
            referenceRight,
            referenceGamma,
            referenceBeta);
        actualLayerNorm.Backward(layerNormSeed);
        expectedLayerNorm.Backward(layerNormSeed);

        Assert.Equal(TensorDType.Float16, actualLayerNorm.DType);
        AssertCloseQuantized(
            expectedLayerNorm.Data,
            actualLayerNorm.Data,
            2e-5f);
        AssertClose(referenceLeft.Grad, halfLeft.Grad, 8e-4f);
        AssertClose(referenceRight.Grad, halfRight.Grad, 8e-4f);
        AssertClose(referenceGamma.Grad, halfGamma.Grad, 8e-4f);
        AssertClose(referenceBeta.Grad, halfBeta.Grad, 2e-5f);
    }

    [Fact]
    public void ForgetScanKeepsRecurrenceAndBackwardAccumulationInFloat32()
    {
        const int batch = 2;
        const int sequence = 5;
        const int width = 12;
        Tensor halfInput = HalfTensor(
            Pattern(batch * sequence * 3 * width, 0.018f),
            [batch, sequence, 3 * width]);
        Tensor referenceInput = FloatReference(halfInput);
        float[] seed = Pattern(batch * sequence * width, 0.014f);

        Tensor actual = halfInput.FusedForgetScan();
        Tensor expected = referenceInput.FusedForgetScan();
        actual.Backward(seed);
        expected.Backward(seed);

        Assert.Equal(TensorDType.Float16, actual.DType);
        AssertCloseQuantized(expected.Data, actual.Data, 3e-5f);
        AssertClose(referenceInput.Grad, halfInput.Grad, 8e-4f);

        Tensor detached;
        using (AutogradContext.NoGrad())
        {
            detached = HalfTensor(
                    halfInput.Data.ToArray(),
                    [batch, sequence, 3 * width])
                .FusedForgetScan();
        }
        Assert.Equal(TensorDType.Float16, detached.DType);
        AssertClose(actual.Data, detached.Data, 3e-5f);
    }

    [Fact]
    public void ForgetMemoryV2KeepsMatrixStateAndGradientsInFloat32()
    {
        const int batch = 2;
        const int sequence = 4;
        const int keyWidth = 8;
        const int valueWidth = 5;
        const int projectionWidth = 2 * keyWidth + 3 * valueWidth;
        Tensor halfInput = HalfTensor(
            Pattern(batch * sequence * projectionWidth, 0.016f),
            [batch, sequence, projectionWidth]);
        Tensor referenceInput = FloatReference(halfInput);
        float[] seed = Pattern(batch * sequence * valueWidth, 0.012f);

        Tensor actual = halfInput.ForgetMemoryV2(
            keyWidth,
            valueWidth,
            retentionFloor: 0.25f);
        Tensor expected = referenceInput.ForgetMemoryV2(
            keyWidth,
            valueWidth,
            retentionFloor: 0.25f);
        actual.Backward(seed);
        expected.Backward(seed);

        Assert.Equal(TensorDType.Float16, actual.DType);
        AssertCloseQuantized(expected.Data, actual.Data, 3e-5f);
        AssertClose(referenceInput.Grad, halfInput.Grad, 9e-4f);
    }

    [Theory]
    [InlineData(HyenaConvolutionAlgorithm.Direct)]
    [InlineData(HyenaConvolutionAlgorithm.Fft)]
    public void HyenaDirectAndFftReadPackedFloat16Filters(
        HyenaConvolutionAlgorithm algorithm)
    {
        const int batch = 2;
        const int sequence = 8;
        const int width = 8;
        Tensor halfProjected = HalfTensor(
            Pattern(batch * sequence * 3 * width, 0.011f),
            [batch, sequence, 3 * width]);
        Tensor halfShortFilter = HalfTensor(
            Pattern(3 * 3 * width, 0.008f, offset: 0.04f),
            [3, 3 * width]);
        Tensor halfLongFilter = HalfTensor(
            Pattern(sequence * width, 0.006f, offset: 0.02f),
            [sequence, width]);
        Tensor halfDiagonal = HalfTensor(
            Enumerable.Range(0, width)
                .Select(index => 0.35f + 0.015f * index)
                .ToArray(),
            [width]);
        Tensor referenceProjected = FloatReference(halfProjected);
        Tensor referenceShortFilter = FloatReference(halfShortFilter);
        Tensor referenceLongFilter = FloatReference(halfLongFilter);
        Tensor referenceDiagonal = FloatReference(halfDiagonal);
        float[] seed = Pattern(batch * sequence * width, 0.01f);

        Tensor actual = halfProjected.FusedCausalHyenaOrder2(
            halfShortFilter,
            halfLongFilter,
            halfDiagonal,
            algorithm);
        Tensor expected = referenceProjected.FusedCausalHyenaOrder2(
            referenceShortFilter,
            referenceLongFilter,
            referenceDiagonal,
            algorithm);
        actual.Backward(seed);
        expected.Backward(seed);

        Assert.Equal(TensorDType.Float16, actual.DType);
        AssertCloseQuantized(expected.Data, actual.Data, 4e-5f);
        AssertClose(referenceProjected.Grad, halfProjected.Grad, 8e-4f);
        AssertClose(referenceShortFilter.Grad, halfShortFilter.Grad, 8e-4f);
        AssertClose(referenceLongFilter.Grad, halfLongFilter.Grad, 8e-4f);
        AssertClose(referenceDiagonal.Grad, halfDiagonal.Grad, 8e-4f);
    }

    private static Tensor HalfTensor(float[] values, int[] shape)
        => new(values, shape, dtype: TensorDType.Float16);

    private static Tensor FloatReference(Tensor half)
        => new(half.Data.ToArray(), half.Shape.ToArray());

    private static float[] Pattern(
        int length,
        float scale,
        float offset = 0f)
        => Enumerable.Range(0, length)
            .Select(index => offset + ((index * 37) % 29 - 14) * scale)
            .ToArray();

    private static void AssertCloseQuantized(
        IEnumerable<float> expected,
        IEnumerable<float> actual,
        float tolerance)
        => AssertClose(
            expected.Select(value => (float)(Half)value),
            actual,
            tolerance);

    private static void AssertClose(
        IEnumerable<float> expected,
        IEnumerable<float> actual,
        float tolerance)
        => TensorCharacterizationTests.AssertClose(
            expected,
            actual,
            tolerance);
}
