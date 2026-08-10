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
        bool useOutputVectorization = CanUseTransposedRightKernel(inputWidth, outputWidth);
        float[]? transposedOther = useOutputVectorization
            ? other.GetTransposedData2D()
            : null;
        void ForwardRow(int row)
        {
            int inputOffset = row * inputWidth;
            int outputOffset = row * outputWidth;
            if (transposedOther is not null)
            {
                Array.Copy(rowBias._data, 0, output, outputOffset, outputWidth);
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
                            DotProductMaskedByPositive(
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
            float mean = SumAddedValues(
                _data,
                residual._data,
                offset,
                columns) / columns;
            float variance = SumSquaredAddedDifferences(
                _data,
                residual._data,
                offset,
                columns,
                mean) / columns;
            float inverse = 1f / MathF.Sqrt(variance + eps);
            inverses[row] = inverse;
            NormalizeAddedAffineValues(
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
                ComputeLayerNormGradientSums(
                    result._grad,
                    offset,
                    gamma._data,
                    normalized,
                    offset,
                    columns,
                    out float sumDxhat,
                    out float sumDxhatXhat);
                AccumulateLayerNormInputGradientPair(
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
}
