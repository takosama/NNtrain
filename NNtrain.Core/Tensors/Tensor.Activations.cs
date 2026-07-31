namespace NNtrain;

partial class Tensor
{
    public Tensor Relu()
    {
        float[] y = new float[Numel];
        for (int i = 0; i < Numel; i++)
            y[i] = _data[i] > 0f ? _data[i] : 0f;

        var t = new Tensor(y, _shape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int i = 0; i < Numel; i++)
                _grad[i] += (_data[i] > 0f ? 1f : 0f) * t._grad[i];
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
            for (int c = 0; c < cols; c++)
                y[r * cols + c] = _data[r * cols + c] + rowVec._data[c];

        var t = new Tensor(y, _shape, new[] { this, rowVec });

        t.Node.BackwardAction = () =>
        {
            for (int r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++)
                {
                    float g = t._grad[r * cols + c];
                    _grad[r * cols + c] += g;
                    rowVec._grad[c] += g;
                }
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

            for (int i = 0; i < n; i++)
                y[i] /= sum;

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                float dot = 0f;
                for (int i = 0; i < n; i++)
                    dot += t._grad[i] * t._data[i];

                for (int i = 0; i < n; i++)
                    _grad[i] += t._data[i] * (t._grad[i] - dot);
            };

            return t;
        }

        if (Rank == 2)
        {
            int rows = _shape[0];
            int cols = _shape[1];
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

                for (int c = 0; c < cols; c++)
                    y[r * cols + c] /= sum;
            }

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                for (int r = 0; r < rows; r++)
                {
                    float dot = 0f;
                    for (int c = 0; c < cols; c++)
                        dot += t._grad[r * cols + c] * t._data[r * cols + c];

                    for (int c = 0; c < cols; c++)
                        _grad[r * cols + c] += t._data[r * cols + c] * (t._grad[r * cols + c] - dot);
                }
            };

            return t;
        }

        throw new NotSupportedException("SoftmaxLastDim supports rank1/rank2 only");
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

            float logSumExp = MathF.Log(sumExp) + max;

            float[] y = new float[n];
            float[] softmax = new float[n];

            for (int i = 0; i < n; i++)
            {
                y[i] = _data[i] - logSumExp;
                softmax[i] = MathF.Exp(y[i]);
            }

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                float gradSum = 0f;
                for (int i = 0; i < n; i++)
                    gradSum += t._grad[i];

                for (int i = 0; i < n; i++)
                    _grad[i] += t._grad[i] - softmax[i] * gradSum;
            };

            return t;
        }

        if (Rank == 2)
        {
            int rows = _shape[0];
            int cols = _shape[1];

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

                float logSumExp = MathF.Log(sumExp) + max;

                for (int c = 0; c < cols; c++)
                {
                    int idx = r * cols + c;
                    y[idx] = _data[idx] - logSumExp;
                    softmax[idx] = MathF.Exp(y[idx]);
                }
            }

            var t = new Tensor(y, _shape, new[] { this });

            t.Node.BackwardAction = () =>
            {
                for (int r = 0; r < rows; r++)
                {
                    float gradSum = 0f;
                    for (int c = 0; c < cols; c++)
                        gradSum += t._grad[r * cols + c];

                    for (int c = 0; c < cols; c++)
                    {
                        int idx = r * cols + c;
                        _grad[idx] += t._grad[idx] - softmax[idx] * gradSum;
                    }
                }
            };

            return t;
        }

        throw new NotSupportedException("LogSoftmaxLastDim supports rank1/rank2 only");
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

            float mean = 0f;
            for (int i = 0; i < n; i++) mean += _data[i];
            mean /= n;

            float var = 0f;
            for (int i = 0; i < n; i++)
            {
                float d = _data[i] - mean;
                var += d * d;
            }
            var /= n;

            float inv = 1f / MathF.Sqrt(var + eps);
            float[] xhat = new float[n];
            float[] y = new float[n];

            for (int i = 0; i < n; i++)
            {
                xhat[i] = (_data[i] - mean) * inv;
                y[i] = xhat[i] * gamma._data[i] + beta._data[i];
            }

            var t = new Tensor(y, _shape, new[] { this, gamma, beta });

            t.Node.BackwardAction = () =>
            {
                float sumDxhat = 0f;
                float sumDxhatXhat = 0f;

                for (int i = 0; i < n; i++)
                {
                    float g = t._grad[i];
                    beta._grad[i] += g;
                    gamma._grad[i] += g * xhat[i];

                    float dxhat = g * gamma._data[i];
                    sumDxhat += dxhat;
                    sumDxhatXhat += dxhat * xhat[i];
                }

                for (int i = 0; i < n; i++)
                {
                    float dxhat = t._grad[i] * gamma._data[i];
                    _grad[i] += (inv / n) * (n * dxhat - sumDxhat - xhat[i] * sumDxhatXhat);
                }
            };

            return t;
        }

        if (Rank == 2)
        {
            int rows = _shape[0];
            int cols = _shape[1];

            if (gamma._shape[0] != cols || beta._shape[0] != cols)
                throw new ArgumentException(
                    $"LayerNorm parameters must have shape [{cols}], but gamma is {ShapeText(gamma)} " +
                    $"and beta is {ShapeText(beta)}.");

            float[] y = new float[Numel];
            float[] xhat = new float[Numel];
            float[] invs = new float[rows];

            for (int r = 0; r < rows; r++)
            {
                float mean = 0f;
                for (int c = 0; c < cols; c++)
                    mean += _data[r * cols + c];
                mean /= cols;

                float var = 0f;
                for (int c = 0; c < cols; c++)
                {
                    float d = _data[r * cols + c] - mean;
                    var += d * d;
                }
                var /= cols;

                float inv = 1f / MathF.Sqrt(var + eps);
                invs[r] = inv;

                for (int c = 0; c < cols; c++)
                {
                    int idx = r * cols + c;
                    xhat[idx] = (_data[idx] - mean) * inv;
                    y[idx] = xhat[idx] * gamma._data[c] + beta._data[c];
                }
            }

            var t = new Tensor(y, _shape, new[] { this, gamma, beta });

            t.Node.BackwardAction = () =>
            {
                for (int r = 0; r < rows; r++)
                {
                    float sumDxhat = 0f;
                    float sumDxhatXhat = 0f;

                    for (int c = 0; c < cols; c++)
                    {
                        int idx = r * cols + c;
                        float g = t._grad[idx];
                        beta._grad[c] += g;
                        gamma._grad[c] += g * xhat[idx];

                        float dxhat = g * gamma._data[c];
                        sumDxhat += dxhat;
                        sumDxhatXhat += dxhat * xhat[idx];
                    }

                    for (int c = 0; c < cols; c++)
                    {
                        int idx = r * cols + c;
                        float dxhat = t._grad[idx] * gamma._data[c];
                        _grad[idx] += (invs[r] / cols) * (cols * dxhat - sumDxhat - xhat[idx] * sumDxhatXhat);
                    }
                }
            };

            return t;
        }

        throw new NotSupportedException("LayerNormLastDim supports rank1/rank2 only");
    }

    public Tensor CausalMask(float fillValue = -1e9f)
    {
        CheckRank(2);

        int rows = _shape[0];
        int cols = _shape[1];
        float[] y = (float[])_data.Clone();

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                if (c > r)
                    y[r * cols + c] = fillValue;

        var t = new Tensor(y, _shape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    if (c <= r)
                        _grad[r * cols + c] += t._grad[r * cols + c];
        };

        return t;
    }
}
