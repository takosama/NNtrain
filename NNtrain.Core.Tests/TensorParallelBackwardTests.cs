using NNtrain;
using Xunit;
using static TensorCharacterizationTests;

public sealed class TensorParallelBackwardTests
{
    [Fact]
    public void LargeMatMulBackwardMatchesReferenceGradients()
    {
        const int rows = 48;
        const int inner = 32;
        const int columns = 32;
        float[] leftValues = Pattern(rows * inner, 23, 0.011f);
        float[] rightValues = Pattern(inner * columns, 19, 0.013f);
        float[] seed = Pattern(rows * columns, 17, 0.017f);
        float[] expectedLeft = new float[leftValues.Length];
        float[] expectedRight = new float[rightValues.Length];

        for (int row = 0; row < rows; row++)
        {
            for (int index = 0; index < inner; index++)
            {
                for (int column = 0; column < columns; column++)
                {
                    expectedLeft[row * inner + index] +=
                        rightValues[index * columns + column]
                        * seed[row * columns + column];
                    expectedRight[index * columns + column] +=
                        leftValues[row * inner + index]
                        * seed[row * columns + column];
                }
            }
        }

        for (int run = 0; run < 3; run++)
        {
            var left = new Tensor(leftValues, [rows, inner]);
            var right = new Tensor(rightValues, [inner, columns]);

            left.MatMul(right).Backward(seed);

            AssertClose(expectedLeft, left.Grad, 2e-4f);
            AssertClose(expectedRight, right.Grad, 2e-4f);
        }
    }

    [Fact]
    public void LargeMatrixVectorBackwardMatchesReferenceGradients()
    {
        const int rows = 512;
        const int columns = 128;
        float[] matrixValues = Pattern(rows * columns, 29, 0.007f);
        float[] vectorValues = Pattern(columns, 13, 0.019f);
        float[] seed = Pattern(rows, 11, 0.023f);
        float[] expectedMatrix = new float[matrixValues.Length];
        float[] expectedVector = new float[vectorValues.Length];

        for (int row = 0; row < rows; row++)
        {
            for (int column = 0; column < columns; column++)
            {
                expectedMatrix[row * columns + column] =
                    seed[row] * vectorValues[column];
                expectedVector[column] +=
                    matrixValues[row * columns + column] * seed[row];
            }
        }

        var matrix = new Tensor(matrixValues, [rows, columns]);
        Tensor vector = Tensor.From1D(vectorValues);

        matrix.MatMul(vector).Backward(seed);

        AssertClose(expectedMatrix, matrix.Grad, 2e-5f);
        AssertClose(expectedVector, vector.Grad, 2e-4f);
    }

    [Fact]
    public void LargeFusedLinearBackwardMatchesReferenceGradients()
    {
        const int rows = 48;
        const int inputWidth = 32;
        const int outputWidth = 24;
        float[] inputValues = Pattern(rows * inputWidth, 31, 0.009f);
        float[] weightValues = Pattern(outputWidth * inputWidth, 21, 0.012f);
        float[] seed = Pattern(rows * outputWidth, 15, 0.016f);
        float[] expectedInput = new float[inputValues.Length];
        float[] expectedWeight = new float[weightValues.Length];
        float[] expectedBias = new float[outputWidth];

        for (int row = 0; row < rows; row++)
        {
            for (int output = 0; output < outputWidth; output++)
            {
                float gradient = seed[row * outputWidth + output];
                expectedBias[output] += gradient;
                for (int input = 0; input < inputWidth; input++)
                {
                    expectedInput[row * inputWidth + input] +=
                        weightValues[output * inputWidth + input]
                        * gradient;
                    expectedWeight[output * inputWidth + input] +=
                        inputValues[row * inputWidth + input]
                        * gradient;
                }
            }
        }

        for (int run = 0; run < 3; run++)
        {
            var input = new Tensor(inputValues, [rows, inputWidth]);
            var weight = new Tensor(
                weightValues,
                [outputWidth, inputWidth]);
            Tensor bias = Tensor.From1D(new float[outputWidth]);

            input.MatMulTransposedRightAddRow(weight, bias).Backward(seed);

            AssertClose(expectedInput, input.Grad, 2e-4f);
            AssertClose(expectedWeight, weight.Grad, 2e-4f);
            AssertClose(expectedBias, bias.Grad, 2e-5f);
        }
    }

    [Fact]
    public void LargeSoftmaxBackwardsMatchReferenceGradients()
    {
        const int rows = 256;
        const int columns = 128;
        float[] values = Pattern(rows * columns, 37, 0.021f);
        float[] seed = Pattern(rows * columns, 25, 0.014f);

        var softmaxInput = new Tensor(values, [rows, columns]);
        Tensor softmax = softmaxInput.SoftmaxLastDim();
        float[] expectedSoftmax = new float[values.Length];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * columns;
            float dot = 0f;
            for (int column = 0; column < columns; column++)
            {
                dot += seed[offset + column]
                    * softmax.Data[offset + column];
            }

            for (int column = 0; column < columns; column++)
            {
                expectedSoftmax[offset + column] =
                    softmax.Data[offset + column]
                    * (seed[offset + column] - dot);
            }
        }

        softmax.Backward(seed);
        AssertClose(expectedSoftmax, softmaxInput.Grad, 2e-5f);

        var logSoftmaxInput = new Tensor(values, [rows, columns]);
        Tensor logSoftmax = logSoftmaxInput.LogSoftmaxLastDim();
        float[] expectedLogSoftmax = new float[values.Length];
        for (int row = 0; row < rows; row++)
        {
            int offset = row * columns;
            float gradientSum = 0f;
            for (int column = 0; column < columns; column++)
                gradientSum += seed[offset + column];

            for (int column = 0; column < columns; column++)
            {
                expectedLogSoftmax[offset + column] =
                    seed[offset + column]
                    - MathF.Exp(logSoftmax.Data[offset + column])
                        * gradientSum;
            }
        }

        logSoftmax.Backward(seed);
        AssertClose(expectedLogSoftmax, logSoftmaxInput.Grad, 2e-5f);
    }

    [Fact]
    public void LargeLayerNormBackwardMatchesReferenceGradients()
    {
        const int rows = 256;
        const int columns = 128;
        const float epsilon = 1e-5f;
        float[] values = Pattern(rows * columns, 41, 0.018f);
        float[] gammaValues = Enumerable.Range(0, columns)
            .Select(index => 0.7f + index * 0.003f)
            .ToArray();
        float[] seed = Pattern(rows * columns, 27, 0.015f);
        float[] expectedInput = new float[values.Length];
        float[] expectedGamma = new float[columns];
        float[] expectedBeta = new float[columns];

        for (int row = 0; row < rows; row++)
        {
            int offset = row * columns;
            float mean = 0f;
            for (int column = 0; column < columns; column++)
                mean += values[offset + column];
            mean /= columns;

            float variance = 0f;
            for (int column = 0; column < columns; column++)
            {
                float difference = values[offset + column] - mean;
                variance += difference * difference;
            }
            variance /= columns;

            float inverse = 1f / MathF.Sqrt(variance + epsilon);
            float sumDxhat = 0f;
            float sumDxhatXhat = 0f;
            for (int column = 0; column < columns; column++)
            {
                int index = offset + column;
                float normalized = (values[index] - mean) * inverse;
                float dxhat = seed[index] * gammaValues[column];
                expectedBeta[column] += seed[index];
                expectedGamma[column] += seed[index] * normalized;
                sumDxhat += dxhat;
                sumDxhatXhat += dxhat * normalized;
            }

            for (int column = 0; column < columns; column++)
            {
                int index = offset + column;
                float normalized = (values[index] - mean) * inverse;
                float dxhat = seed[index] * gammaValues[column];
                expectedInput[index] = inverse / columns
                    * (columns * dxhat - sumDxhat
                        - normalized * sumDxhatXhat);
            }
        }

        var input = new Tensor(values, [rows, columns]);
        Tensor gamma = Tensor.From1D(gammaValues);
        Tensor beta = Tensor.From1D(new float[columns]);

        input.LayerNormLastDim(gamma, beta, epsilon).Backward(seed);

        AssertClose(expectedInput, input.Grad, 3e-4f);
        AssertClose(expectedGamma, gamma.Grad, 3e-4f);
        AssertClose(expectedBeta, beta.Grad, 2e-4f);
    }

    private static float[] Pattern(int length, int modulus, float scale)
        => Enumerable.Range(0, length)
            .Select(index => (index % modulus - modulus / 2) * scale)
            .ToArray();
}
