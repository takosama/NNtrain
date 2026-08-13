namespace NNtrain;

partial class Tensor
{
    public Tensor MatMulTransposedRightAddRowRelu(
        Tensor other,
        Tensor rowBias)
    {
        ArgumentNullException.ThrowIfNull(other);
        ArgumentNullException.ThrowIfNull(rowBias);
        CheckRank(2);
        other.CheckRank(2);
        rowBias.CheckRank(1);

        int rows = _shape[0];
        int inputWidth = _shape[1];
        int outputWidth = other._shape[0];
        if (other._shape[1] != inputWidth
            || rowBias._shape[0] != outputWidth)
        {
            throw new ArgumentException(
                "MatMulTransposedRightAddRowRelu requires shapes " +
                "[rows, input], [output, input], and [output].");
        }

        float[] output = new float[checked(rows * outputWidth)];
        // A Float16 weight stays packed. Expanding the full matrix into the
        // Float32 transpose cache would erase its storage/cache advantage, so
        // use the contiguous dtype-aware dot-product kernel for that case.
        bool useOutputVectorization = other.DType == TensorDType.Float32
            && CanUseTransposedRightKernel(inputWidth, outputWidth);
        float[]? transposedOther = useOutputVectorization
            ? other.GetTransposedData2D()
            : null;
        void ForwardRow(int row)
        {
            int inputOffset = row * inputWidth;
            int outputOffset = row * outputWidth;
            if (transposedOther is not null)
            {
                rowBias._data.CopyRangeTo(
                    0,
                    output.AsSpan(outputOffset, outputWidth));
                for (int inner = 0; inner < inputWidth; inner++)
                {
                    AddScaledValues(
                        output,
                        outputOffset,
                        transposedOther,
                        inner * outputWidth,
                        _data[inputOffset + inner],
                        outputWidth);
                }
                ReluValuesInPlace(output, outputOffset, outputWidth);
                return;
            }

            for (int column = 0; column < outputWidth; column++)
            {
                float value = rowBias._data[column] + DotProduct(
                    _data,
                    inputOffset,
                    other._data,
                    column * inputWidth,
                    inputWidth);
                output[outputOffset + column] = value > 0f ? value : 0f;
            }
        }

        RunBatches(
            rows,
            (long)inputWidth * outputWidth,
            ForwardRow);

        var result = new Tensor(
            output,
            [rows, outputWidth],
            [this, other, rowBias]);
        result.Node.BackwardAction = () =>
        {
            void BackwardInputRow(int row)
            {
                int inputOffset = row * inputWidth;
                int outputOffset = row * outputWidth;
                if (transposedOther is not null)
                {
                    for (int inner = 0; inner < inputWidth; inner++)
                    {
                        _grad[inputOffset + inner] +=
                            DotProductMaskedByPositiveStoredMask(
                                result._grad,
                                outputOffset,
                                result._data,
                                outputOffset,
                                transposedOther,
                                inner * outputWidth,
                                outputWidth);
                    }
                    return;
                }

                for (int column = 0; column < outputWidth; column++)
                {
                    if (result._data[outputOffset + column] <= 0f)
                        continue;

                    AddScaledValues(
                        _grad,
                        inputOffset,
                        other._data,
                        column * inputWidth,
                        result._grad[outputOffset + column],
                        inputWidth);
                }
            }

            RunBatches(
                rows,
                (long)inputWidth * outputWidth,
                BackwardInputRow);

            void BackwardWeightRow(int column)
            {
                int weightOffset = column * inputWidth;
                float biasGradient = 0f;
                for (int row = 0; row < rows; row++)
                {
                    int outputIndex = row * outputWidth + column;
                    if (result._data[outputIndex] <= 0f)
                        continue;

                    float gradient = result._grad[outputIndex];
                    biasGradient += gradient;
                    AddScaledValues(
                        other._grad,
                        weightOffset,
                        _data,
                        row * inputWidth,
                        gradient,
                        inputWidth);
                }

                rowBias._grad[column] += biasGradient;
            }

            RunBatches(
                outputWidth,
                (long)rows * inputWidth,
                BackwardWeightRow);
        };

        return result;
    }

    public Tensor AddLayerNormLastDim(
        Tensor residual,
        Tensor gamma,
        Tensor beta,
        float eps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(residual);
        ArgumentNullException.ThrowIfNull(gamma);
        ArgumentNullException.ThrowIfNull(beta);
        if (eps <= 0f || !float.IsFinite(eps))
        {
            throw new ArgumentOutOfRangeException(
                nameof(eps),
                eps,
                "Epsilon must be finite and positive.");
        }

        if (!_shape.AsSpan().SequenceEqual(residual._shape))
            throw ShapeMismatch(this, residual, "Residual addition");
        gamma.CheckRank(1);
        beta.CheckRank(1);

        int columns = _shape[^1];
        int rows = Numel / columns;
        if (gamma._shape[0] != columns || beta._shape[0] != columns)
        {
            throw new ArgumentException(
                $"LayerNorm parameters must have shape [{columns}].");
        }

        float[] output = new float[Numel];
        float[] normalized = new float[Numel];
        float[] inverses = new float[rows];

        void ForwardRow(int row)
        {
            int offset = row * columns;
            float mean = SumAddedStoredValues(
                _data,
                residual._data,
                offset,
                columns) / columns;
            float variance = SumSquaredAddedStoredDifferences(
                _data,
                residual._data,
                offset,
                columns,
                mean) / columns;
            float inverse = 1f / MathF.Sqrt(variance + eps);
            inverses[row] = inverse;
            NormalizeAddedStoredAffineValues(
                _data,
                residual._data,
                offset,
                gamma._data,
                beta._data,
                mean,
                inverse,
                normalized,
                output,
                columns);
        }

        RunBatches(rows, columns, ForwardRow);

        var result = new Tensor(
            output,
            _shape,
            [this, residual, gamma, beta]);
        result.Node.BackwardAction = () =>
        {
            void BackwardInputRow(int row)
            {
                int offset = row * columns;
                ComputeStoredLayerNormGradientSums(
                    result._grad,
                    offset,
                    gamma._data,
                    normalized,
                    offset,
                    columns,
                    out float sumDxhat,
                    out float sumDxhatXhat);
                AccumulateStoredLayerNormInputGradientPair(
                    _grad,
                    residual._grad,
                    offset,
                    result._grad,
                    offset,
                    gamma._data,
                    normalized,
                    offset,
                    columns,
                    inverses[row],
                    sumDxhat,
                    sumDxhatXhat);
            }

            RunBatches(rows, columns, BackwardInputRow);

            void BackwardParameter(int column)
            {
                float gammaGradient = 0f;
                float betaGradient = 0f;
                for (int row = 0; row < rows; row++)
                {
                    int index = row * columns + column;
                    float gradient = result._grad[index];
                    betaGradient += gradient;
                    gammaGradient += gradient * normalized[index];
                }

                gamma._grad[column] += gammaGradient;
                beta._grad[column] += betaGradient;
            }

            RunBatches(columns, rows, BackwardParameter);
        };

        return result;
    }

    private static float DotProductMaskedByPositiveStoredMask(
        float[] gradient,
        int gradientOffset,
        TensorStorage activation,
        int activationOffset,
        float[] weight,
        int weightOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> zero = Vector256<float>.Zero;
            Vector256<float> sumVector = zero;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> activationVector = LoadVector256(
                    activation,
                    activationOffset + index);
                Vector256<float> maskedGradient =
                    Vector256.ConditionalSelect(
                        Vector256.GreaterThan(activationVector, zero),
                        LoadVector256(gradient, gradientOffset + index),
                        zero);
                sumVector += maskedGradient
                    * LoadVector256(weight, weightOffset + index);
            }
            sum = Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            if (activation[activationOffset + index] > 0f)
            {
                sum += gradient[gradientOffset + index]
                    * weight[weightOffset + index];
            }
        }
        return sum;
    }

    private static float SumAddedStoredValues(
        TensorStorage left,
        TensorStorage right,
        int offset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                sumVector += LoadVector256(left, offset + index)
                    + LoadVector256(right, offset + index);
            }
            sum += Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
            sum += left[offset + index] + right[offset + index];
        return sum;
    }

    private static float SumSquaredAddedStoredDifferences(
        TensorStorage left,
        TensorStorage right,
        int offset,
        int length,
        float mean)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> sumVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> difference =
                    LoadVector256(left, offset + index)
                    + LoadVector256(right, offset + index)
                    - meanVector;
                sumVector += difference * difference;
            }
            sum += Vector256.Sum(sumVector);
        }

        for (; index < length; index++)
        {
            float difference =
                left[offset + index] + right[offset + index] - mean;
            sum += difference * difference;
        }
        return sum;
    }

    private static void NormalizeAddedStoredAffineValues(
        TensorStorage left,
        TensorStorage right,
        int offset,
        TensorStorage gamma,
        TensorStorage beta,
        float mean,
        float inverseStandardDeviation,
        float[] normalized,
        float[] output,
        int length)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> meanVector = Vector256.Create(mean);
            Vector256<float> inverseVector =
                Vector256.Create(inverseStandardDeviation);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> normalizedVector =
                    (LoadVector256(left, offset + index)
                        + LoadVector256(right, offset + index)
                        - meanVector)
                    * inverseVector;
                StoreVector256(
                    normalizedVector,
                    normalized,
                    offset + index);
                StoreVector256(
                    normalizedVector * LoadVector256(gamma, index)
                        + LoadVector256(beta, index),
                    output,
                    offset + index);
            }
        }

        for (; index < length; index++)
        {
            float normalizedValue =
                (left[offset + index] + right[offset + index] - mean)
                * inverseStandardDeviation;
            normalized[offset + index] = normalizedValue;
            output[offset + index] =
                normalizedValue * gamma[index] + beta[index];
        }
    }

    private static void ComputeStoredLayerNormGradientSums(
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        out float sumGradientToNormalized,
        out float sumGradientToNormalizedTimesNormalized)
    {
        int index = 0;
        float firstSum = 0f;
        float secondSum = 0f;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> firstSumVector = Vector256<float>.Zero;
            Vector256<float> secondSumVector = Vector256<float>.Zero;
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> normalizedVector = LoadVector256(
                    normalized,
                    normalizedOffset + index);
                firstSumVector += gradientToNormalized;
                secondSumVector +=
                    gradientToNormalized * normalizedVector;
            }
            firstSum += Vector256.Sum(firstSumVector);
            secondSum += Vector256.Sum(secondSumVector);
        }

        for (; index < length; index++)
        {
            float gradientToNormalized =
                gradient[gradientOffset + index] * gamma[index];
            firstSum += gradientToNormalized;
            secondSum += gradientToNormalized
                * normalized[normalizedOffset + index];
        }

        sumGradientToNormalized = firstSum;
        sumGradientToNormalizedTimesNormalized = secondSum;
    }

    private static void AccumulateStoredLayerNormInputGradientPair(
        float[] firstDestination,
        float[] secondDestination,
        int destinationOffset,
        float[] gradient,
        int gradientOffset,
        TensorStorage gamma,
        float[] normalized,
        int normalizedOffset,
        int length,
        float inverseStandardDeviation,
        float sumGradientToNormalized,
        float sumGradientToNormalizedTimesNormalized)
    {
        int index = 0;
        float factor = inverseStandardDeviation / length;
        bool sameDestination = ReferenceEquals(
            firstDestination,
            secondDestination);
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> lengthVector = Vector256.Create((float)length);
            Vector256<float> firstSumVector =
                Vector256.Create(sumGradientToNormalized);
            Vector256<float> secondSumVector =
                Vector256.Create(sumGradientToNormalizedTimesNormalized);
            Vector256<float> factorVector = Vector256.Create(factor);
            Vector256<float> destinationScale =
                Vector256.Create(sameDestination ? 2f : 1f);

            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradientToNormalized =
                    LoadVector256(gradient, gradientOffset + index)
                    * LoadVector256(gamma, index);
                Vector256<float> contribution = factorVector
                    * (lengthVector * gradientToNormalized
                        - firstSumVector
                        - LoadVector256(normalized, normalizedOffset + index)
                            * secondSumVector);
                StoreVector256(
                    LoadVector256(
                        firstDestination,
                        destinationOffset + index)
                        + contribution * destinationScale,
                    firstDestination,
                    destinationOffset + index);
                if (!sameDestination)
                {
                    StoreVector256(
                        LoadVector256(
                            secondDestination,
                            destinationOffset + index)
                            + contribution,
                        secondDestination,
                        destinationOffset + index);
                }
            }
        }

        for (; index < length; index++)
        {
            float gradientToNormalized =
                gradient[gradientOffset + index] * gamma[index];
            float contribution = factor
                * (length * gradientToNormalized
                    - sumGradientToNormalized
                    - normalized[normalizedOffset + index]
                        * sumGradientToNormalizedTimesNormalized);
            firstDestination[destinationOffset + index] +=
                sameDestination ? 2f * contribution : contribution;
            if (!sameDestination)
            {
                secondDestination[destinationOffset + index] +=
                    contribution;
            }
        }
    }
}
