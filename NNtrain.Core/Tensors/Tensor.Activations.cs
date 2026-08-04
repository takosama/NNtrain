namespace NNtrain;

partial class Tensor
{
    public Tensor Relu()
    {
        float[] y = new float[Numel];
        int i = 0;
        if (CanUseSimd(Numel))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = Numel - Numel % vectorWidth;
            Vector256<float> zero = Vector256<float>.Zero;

            for (; i < vectorizedLength; i += vectorWidth)
            {
                Vector256<float> value = LoadVector256(_data, i);
                StoreVector256(
                    Vector256.ConditionalSelect(
                        Vector256.GreaterThan(value, zero),
                        value,
                        zero),
                    y,
                    i);
            }
        }

        for (; i < Numel; i++)
            y[i] = _data[i] > 0f ? _data[i] : 0f;

        var t = new Tensor(y, _shape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            int index = 0;
            if (CanUseSimd(Numel))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorizedLength = Numel - Numel % vectorWidth;
                Vector256<float> zero = Vector256<float>.Zero;

                for (; index < vectorizedLength; index += vectorWidth)
                {
                    Vector256<float> value = LoadVector256(
                        _data,
                        index);
                    Vector256<float> gradient = LoadVector256(
                        t._grad,
                        index);
                    Vector256<float> contribution =
                        Vector256.ConditionalSelect(
                            Vector256.GreaterThan(value, zero),
                            gradient,
                            zero);
                    StoreVector256(
                        LoadVector256(_grad, index) + contribution,
                        _grad,
                        index);
                }
            }

            for (; index < Numel; index++)
            {
                _grad[index] +=
                    (_data[index] > 0f ? 1f : 0f) * t._grad[index];
            }
        };

        return t;
    }
    public Tensor AddRowWise(Tensor rowVec)
    {
        ArgumentNullException.ThrowIfNull(rowVec);
        CheckRank(2);
        rowVec.CheckRank(1);

        int rows = _shape[0];
        int cols = _shape[1];

        if (rowVec._shape[0] != cols)
            throw ShapeMismatch(this, rowVec, "Row-wise addition");

        float[] y = new float[Numel];
        for (int r = 0; r < rows; r++)
        {
            int rowOffset = r * cols;
            int c = 0;
            if (CanUseSimd(cols))
            {
                int vectorWidth = Vector256<float>.Count;
                int vectorizedLength = cols - cols % vectorWidth;
                for (; c < vectorizedLength; c += vectorWidth)
                {
                    StoreVector256(
                        LoadVector256(_data, rowOffset + c)
                            + LoadVector256(rowVec._data, c),
                        y,
                        rowOffset + c);
                }
            }

            for (; c < cols; c++)
            {
                y[rowOffset + c] =
                    _data[rowOffset + c] + rowVec._data[c];
            }
        }

        var t = new Tensor(y, _shape, new[] { this, rowVec });

        t.Node.BackwardAction = () =>
        {
            for (int r = 0; r < rows; r++)
            {
                int rowOffset = r * cols;
                AddScaledValues(
                    _grad,
                    rowOffset,
                    t._grad,
                    rowOffset,
                    1f,
                    cols);
                AddScaledValues(
                    rowVec._grad,
                    0,
                    t._grad,
                    rowOffset,
                    1f,
                    cols);
            }
        };

        return t;
    }

    public Tensor SoftmaxLastDim()
    {
        if (Rank == 1)
        {
            int n = _shape[0];
            float max = _data[0];
            for (int i = 1; i < n; i++)
                if (_data[i] > max) max = _data[i];

            float[] y = new float[n];
            float sum = 0f;
            for (int i = 0; i < n; i++)
            {
                y[i] = MathF.Exp(_data[i] - max);
                sum += y[i];
            }

            MultiplyValues(y, 0, 1f / sum, y, 0, n);

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                float dot = DotProduct(t._grad, 0, t._data, 0, n);
                AccumulateSoftmaxGradient(
                    _grad,
                    0,
                    t._data,
                    0,
                    t._grad,
                    0,
                    n,
                    dot);
            };

            return t;
        }

        if (Rank >= 2)
        {
            int cols = _shape[^1];
            int rows = Numel / cols;
            float[] y = new float[Numel];

            for (int r = 0; r < rows; r++)
            {
                float max = _data[r * cols];
                for (int c = 1; c < cols; c++)
                {
                    float v = _data[r * cols + c];
                    if (v > max) max = v;
                }

                float sum = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float e = MathF.Exp(_data[r * cols + c] - max);
                    y[r * cols + c] = e;
                    sum += e;
                }

                MultiplyValues(
                    y,
                    r * cols,
                    1f / sum,
                    y,
                    r * cols,
                    cols);
            }

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                void BackwardRow(int r)
                {
                    int rowOffset = r * cols;
                    float dot = DotProduct(
                        t._grad,
                        rowOffset,
                        t._data,
                        rowOffset,
                        cols);
                    AccumulateSoftmaxGradient(
                        _grad,
                        rowOffset,
                        t._data,
                        rowOffset,
                        t._grad,
                        rowOffset,
                        cols,
                        dot);
                }

                RunBatches(rows, cols, BackwardRow);
            };

            return t;
        }

        throw new NotSupportedException(
            "SoftmaxLastDim requires a tensor with at least one dimension.");
    }

    public Tensor LogSoftmaxLastDim()
    {
        if (Rank == 1)
        {
            int n = _shape[0];

            float max = _data[0];
            for (int i = 1; i < n; i++)
                if (_data[i] > max) max = _data[i];

            float sumExp = 0f;
            for (int i = 0; i < n; i++)
                sumExp += MathF.Exp(_data[i] - max);

            float logSumExpOfShiftedValues = MathF.Log(sumExp);

            float[] y = new float[n];
            float[] softmax = new float[n];

            for (int i = 0; i < n; i++)
            {
                y[i] = (_data[i] - max) - logSumExpOfShiftedValues;
                softmax[i] = MathF.Exp(y[i]);
            }

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                float gradSum = SumValues(t._grad, 0, n);
                AccumulateLogSoftmaxGradient(
                    _grad,
                    0,
                    softmax,
                    0,
                    t._grad,
                    0,
                    n,
                    gradSum);
            };

            return t;
        }

        if (Rank >= 2)
        {
            int cols = _shape[^1];
            int rows = Numel / cols;

            float[] y = new float[Numel];
            float[] softmax = new float[Numel];

            for (int r = 0; r < rows; r++)
            {
                float max = _data[r * cols];
                for (int c = 1; c < cols; c++)
                {
                    float v = _data[r * cols + c];
                    if (v > max) max = v;
                }

                float sumExp = 0f;
                for (int c = 0; c < cols; c++)
                    sumExp += MathF.Exp(_data[r * cols + c] - max);

                float logSumExpOfShiftedValues = MathF.Log(sumExp);

                for (int c = 0; c < cols; c++)
                {
                    int idx = r * cols + c;
                    y[idx] =
                        (_data[idx] - max) - logSumExpOfShiftedValues;
                    softmax[idx] = MathF.Exp(y[idx]);
                }
            }

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                void BackwardRow(int r)
                {
                    int rowOffset = r * cols;
                    float gradSum = SumValues(
                        t._grad,
                        rowOffset,
                        cols);
                    AccumulateLogSoftmaxGradient(
                        _grad,
                        rowOffset,
                        softmax,
                        rowOffset,
                        t._grad,
                        rowOffset,
                        cols,
                        gradSum);
                }

                RunBatches(rows, cols, BackwardRow);
            };

            return t;
        }

        throw new NotSupportedException(
            "LogSoftmaxLastDim requires a tensor with at least one dimension.");
    }
    public Tensor LayerNormLastDim(Tensor gamma, Tensor beta, float eps = 1e-5f)
    {
        ArgumentNullException.ThrowIfNull(gamma);
        ArgumentNullException.ThrowIfNull(beta);
        if (eps <= 0f || !float.IsFinite(eps))
            throw new ArgumentOutOfRangeException(nameof(eps), eps, "Epsilon must be finite and positive.");

        gamma.CheckRank(1);
        beta.CheckRank(1);

        if (Rank == 1)
        {
            int n = _shape[0];
            if (gamma._shape[0] != n || beta._shape[0] != n)
                throw new ArgumentException(
                    $"LayerNorm parameters must have shape [{n}], but gamma is {ShapeText(gamma)} " +
                    $"and beta is {ShapeText(beta)}.");

            float mean = SumValues(_data, 0, n) / n;
            float var = SumSquaredDifferences(_data, 0, n, mean) / n;

            float inv = 1f / MathF.Sqrt(var + eps);
            float[] xhat = new float[n];
            float[] y = new float[n];

            NormalizeAffineValues(
                _data,
                0,
                gamma._data,
                beta._data,
                mean,
                inv,
                xhat,
                0,
                y,
                0,
                n);

            var t = new Tensor(y, _shape, new[] { this, gamma, beta });

            t.Node.BackwardAction = () =>
            {
                AccumulateLayerNormParameterGradients(
                    t._grad,
                    0,
                    gamma._data,
                    xhat,
                    0,
                    gamma._grad,
                    beta._grad,
                    n,
                    out float sumDxhat,
                    out float sumDxhatXhat);
                AccumulateLayerNormInputGradient(
                    _grad,
                    0,
                    t._grad,
                    0,
                    gamma._data,
                    xhat,
                    0,
                    n,
                    inv,
                    sumDxhat,
                    sumDxhatXhat);
            };

            return t;
        }

        if (Rank >= 2)
        {
            int cols = _shape[^1];
            int rows = Numel / cols;

            if (gamma._shape[0] != cols || beta._shape[0] != cols)
                throw new ArgumentException(
                    $"LayerNorm parameters must have shape [{cols}], but gamma is {ShapeText(gamma)} " +
                    $"and beta is {ShapeText(beta)}.");

            float[] y = new float[Numel];
            float[] xhat = new float[Numel];
            float[] invs = new float[rows];

            for (int r = 0; r < rows; r++)
            {
                int rowOffset = r * cols;
                float mean = SumValues(_data, rowOffset, cols) / cols;
                float var = SumSquaredDifferences(
                    _data,
                    rowOffset,
                    cols,
                    mean) / cols;

                float inv = 1f / MathF.Sqrt(var + eps);
                invs[r] = inv;

                NormalizeAffineValues(
                    _data,
                    rowOffset,
                    gamma._data,
                    beta._data,
                    mean,
                    inv,
                    xhat,
                    rowOffset,
                    y,
                    rowOffset,
                    cols);
            }

            var t = new Tensor(y, _shape, new[] { this, gamma, beta });

            t.Node.BackwardAction = () =>
            {
                void BackwardInputRow(int r)
                {
                    int rowOffset = r * cols;
                    ComputeLayerNormGradientSums(
                        t._grad,
                        rowOffset,
                        gamma._data,
                        xhat,
                        rowOffset,
                        cols,
                        out float sumDxhat,
                        out float sumDxhatXhat);
                    AccumulateLayerNormInputGradient(
                        _grad,
                        rowOffset,
                        t._grad,
                        rowOffset,
                        gamma._data,
                        xhat,
                        rowOffset,
                        cols,
                        invs[r],
                        sumDxhat,
                        sumDxhatXhat);
                }

                RunBatches(rows, cols, BackwardInputRow);

                void BackwardParameter(int c)
                {
                    float gammaGradient = 0f;
                    float betaGradient = 0f;
                    for (int r = 0; r < rows; r++)
                    {
                        int index = r * cols + c;
                        float gradient = t._grad[index];
                        betaGradient += gradient;
                        gammaGradient += gradient * xhat[index];
                    }

                    gamma._grad[c] += gammaGradient;
                    beta._grad[c] += betaGradient;
                }

                RunBatches(cols, rows, BackwardParameter);
            };

            return t;
        }

        throw new NotSupportedException(
            "LayerNormLastDim requires a tensor with at least one dimension.");
    }

    public Tensor CausalMask(float fillValue = -1e9f)
    {
        if (Rank < 2)
        {
            throw new InvalidOperationException(
                "CausalMask requires a tensor with at least two " +
                "dimensions.");
        }

        int rows = _shape[^2];
        int cols = _shape[^1];
        int matrixCount = Numel / (rows * cols);
        float[] y = (float[])_data.Clone();

        for (int matrix = 0; matrix < matrixCount; matrix++)
        {
            int matrixOffset = matrix * rows * cols;
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    if (c > r)
                        y[matrixOffset + r * cols + c] = fillValue;
                }
            }
        }

        var t = new Tensor(y, _shape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int matrix = 0; matrix < matrixCount; matrix++)
            {
                int matrixOffset = matrix * rows * cols;
                for (int r = 0; r < rows; r++)
                {
                    int copiedColumns = Math.Min(r + 1, cols);
                    AddScaledValues(
                        _grad,
                        matrixOffset + r * cols,
                        t._grad,
                        matrixOffset + r * cols,
                        1f,
                        copiedColumns);
                }
            }
        };

        return t;
    }
}
