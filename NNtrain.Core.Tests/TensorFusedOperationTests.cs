using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class TensorFusedOperationTests
{
    [Fact]
    public void CudaFusedAttentionMatchesCpuForwardBackward()
    {
        if (!Tensor.IsCudaAvailable())
            return;
        const int batch = 2;
        const int sequence = 5;
        const int modelWidth = 8;
        const int heads = 2;
        float[] values = Pattern(batch * sequence * 3 * modelWidth, 31, 0.013f);
        float[] seed = Pattern(batch * sequence * modelWidth, 19, 0.016f);
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var cpuInput = new Tensor(values, [batch, sequence, 3 * modelWidth]);
            Tensor cpu = cpuInput.FusedMultiHeadAttention(heads, causal: true);
            cpu.Backward(seed);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            var cudaInput = new Tensor(values, [batch, sequence, 3 * modelWidth]);
            Tensor cuda = cudaInput.FusedMultiHeadAttention(heads, causal: true);
            float[] cudaOutput = cuda.Data.ToArray();
            cuda.BackwardAndRelease(seed);

            AssertClose(cpu.Data, cudaOutput, 3e-5f);
            AssertClose(cpuInput.Grad, cudaInput.Grad, 5e-4f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Fact]
    public void CudaBFloat16TiledAttentionMatchesCpuForwardBackward()
    {
        if (!Tensor.IsCudaAvailable())
            return;
        const int batch = 2;
        const int sequence = 9;
        const int modelWidth = 24;
        const int heads = 1;
        float[] values = Pattern(
            batch * sequence * 3 * modelWidth,
            47,
            0.009f);
        float[] seed = Pattern(
            batch * sequence * modelWidth,
            29,
            0.011f);
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var cpuInput = new Tensor(
                values,
                [batch, sequence, 3 * modelWidth],
                dtype: TensorDType.BFloat16);
            Tensor cpu = cpuInput.FusedMultiHeadAttention(heads, causal: true);
            cpu.Backward(seed);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            var cudaInput = new Tensor(
                values,
                [batch, sequence, 3 * modelWidth],
                dtype: TensorDType.BFloat16);
            Tensor cuda = cudaInput.FusedMultiHeadAttention(heads, causal: true);
            float[] cudaOutput = cuda.Data.ToArray();
            cuda.BackwardAndRelease(seed);

            // Tensor Core attention rounds the 16x16 probability tile to
            // BF16 before the P*V MMA. Allow two BF16 output ULPs while the
            // softmax statistics and accumulators remain Float32.
            AssertClose(cpu.Data, cudaOutput, 5e-4f);
            AssertClose(cpuInput.Grad, cudaInput.Grad, 8e-4f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

    [Theory]
    [InlineData(8)]
    [InlineData(16)]
    [InlineData(24)]
    [InlineData(40)]
    [InlineData(64)]
    [InlineData(96)]
    [InlineData(128)]
    public void CudaBFloat16TensorCoreAttentionSupportsGeneralHeadWidths(
        int headWidth)
    {
        if (!Tensor.IsCudaAvailable())
            return;
        const int batch = 1;
        const int sequence = 7;
        const int heads = 2;
        int modelWidth = checked(headWidth * heads);
        float[] values = Pattern(
            batch * sequence * 3 * modelWidth,
            61 + headWidth,
            0.006f);
        float[] seed = Pattern(
            batch * sequence * modelWidth,
            43 + headWidth,
            0.008f);
        TensorDevice previous = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var cpuInput = new Tensor(
                values,
                [batch, sequence, 3 * modelWidth],
                dtype: TensorDType.BFloat16);
            Tensor cpu = cpuInput.FusedMultiHeadAttention(
                heads,
                causal: true);
            cpu.Backward(seed);

            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndex = 0;
            var cudaInput = new Tensor(
                values,
                [batch, sequence, 3 * modelWidth],
                dtype: TensorDType.BFloat16);
            Tensor cuda = cudaInput.FusedMultiHeadAttention(
                heads,
                causal: true);
            float[] cudaOutput = cuda.Data.ToArray();
            cuda.BackwardAndRelease(seed);

            // Absolute BF16 ULP size grows with magnitude. The widest heads
            // in this pattern can differ by one output ULP (0.001953125).
            AssertClose(cpu.Data, cudaOutput, 4e-3f);
            AssertClose(cpuInput.Grad, cudaInput.Grad, 5e-3f);
        }
        finally
        {
            Tensor.ExecutionDevice = previous;
        }
    }

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
    public void CrossEntropyFlattensAllLeadingDimensions()
    {
        const int batch = 2;
        const int sequence = 3;
        const int classes = 5;
        float[] values = Pattern(batch * sequence * classes, 29, 0.09f);
        int[] labels = [0, 2, 4, 1, 3, 0];
        var shapedInput = new Tensor(
            values,
            [batch, sequence, classes]);
        var flatInput = new Tensor(
            values,
            [batch * sequence, classes]);

        Tensor shapedLoss = shapedInput.CrossEntropyWithLogits(labels);
        Tensor flatLoss = flatInput.CrossEntropyWithLogits(labels);
        shapedLoss.Backward();
        flatLoss.Backward();

        AssertClose(flatLoss.Data, shapedLoss.Data, 2e-5f);
        AssertClose(flatInput.Grad, shapedInput.Grad, 2e-5f);
    }

    [Fact]
    public void CrossEntropyIgnoresMinusOneLabelsByDefault()
    {
        const int classes = 4;
        float[] values = Pattern(3 * classes, 23, 0.13f);
        var paddedInput = new Tensor(values, [3, classes]);
        var compactInput = new Tensor(
            values[..classes].Concat(values[(2 * classes)..]).ToArray(),
            [2, classes]);

        Tensor paddedLoss = paddedInput.CrossEntropyWithLogits([1, -1, 3]);
        Tensor compactLoss = compactInput.CrossEntropyWithLogits([1, 3]);
        paddedLoss.Backward();
        compactLoss.Backward();
        float[] paddedGradient = paddedInput.Grad.ToArray();
        float[] compactGradient = compactInput.Grad.ToArray();

        AssertClose(compactLoss.Data, paddedLoss.Data, 2e-5f);
        AssertClose(compactGradient[..classes], paddedGradient[..classes]);
        AssertClose(
            new float[classes],
            paddedGradient[classes..(2 * classes)]);
        AssertClose(
            compactGradient[classes..],
            paddedGradient[(2 * classes)..]);
    }

    [Fact]
    public void LabelSmoothedCrossEntropyMatchesUniformTargetMixture()
    {
        const int batch = 3;
        const int classes = 5;
        const float smoothing = 0.1f;
        float[] values = Pattern(batch * classes, 17, 0.11f);
        int[] labels = [0, 3, 1];
        float[] smoothedTargets = new float[batch * classes];
        for (int row = 0; row < batch; row++)
        {
            for (int column = 0; column < classes; column++)
            {
                smoothedTargets[row * classes + column] =
                    smoothing / classes;
            }

            smoothedTargets[row * classes + labels[row]] +=
                1f - smoothing;
        }

        var fusedInput = new Tensor(values, [batch, classes]);
        var referenceInput = new Tensor(values, [batch, classes]);

        Tensor fused = fusedInput.CrossEntropyWithLogits(
            labels,
            smoothing);
        Tensor reference = (Tensor.Scalar(0f)
            - (new Tensor(smoothedTargets, [batch, classes])
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
