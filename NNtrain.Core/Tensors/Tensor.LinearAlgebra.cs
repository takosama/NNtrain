namespace NNtrain;

partial class Tensor
{
    // rank1@rank1 -> scalar
    // rank2@rank1 -> rank1
    // rank2@rank2 -> rank2
    public Tensor MatMul(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);

        if (Rank == 1 && other.Rank == 1)
        {
            int n = _shape[0];
            if (other._shape[0] != n)
                throw ShapeMismatch(this, other, "MatMul");

            float s = 0f;
            for (int i = 0; i < n; i++)
                s += _data[i] * other._data[i];

            var t = new Tensor(new[] { s }, new[] { 1 }, new[] { this, other });

            t.Node.BackwardAction = () =>
            {
                float g = t._grad[0];
                for (int i = 0; i < n; i++)
                {
                    _grad[i] += other._data[i] * g;
                    other._grad[i] += _data[i] * g;
                }
            };

            return t;
        }

        if (Rank == 2 && other.Rank == 1)
        {
            int m = _shape[0];
            int k = _shape[1];
            if (other._shape[0] != k)
                throw ShapeMismatch(this, other, "MatMul");

            float[] y = new float[m];
            for (int r = 0; r < m; r++)
            {
                float s = 0f;
                for (int i = 0; i < k; i++)
                    s += _data[r * k + i] * other._data[i];
                y[r] = s;
            }

            var t = new Tensor(y, new[] { m }, new[] { this, other });

            t.Node.BackwardAction = () =>
            {
                for (int r = 0; r < m; r++)
                {
                    float g = t._grad[r];
                    for (int i = 0; i < k; i++)
                    {
                        _grad[r * k + i] += other._data[i] * g;
                        other._grad[i] += _data[r * k + i] * g;
                    }
                }
            };

            return t;
        }

        if (Rank == 2 && other.Rank == 2)
        {
            int m = _shape[0];
            int k = _shape[1];
            if (other._shape[0] != k)
                throw ShapeMismatch(this, other, "MatMul");
            int n = other._shape[1];

            float[] y = new float[m * n];

            for (int r = 0; r < m; r++)
            {
                for (int c = 0; c < n; c++)
                {
                    float s = 0f;
                    for (int i = 0; i < k; i++)
                        s += _data[r * k + i] * other._data[i * n + c];
                    y[r * n + c] = s;
                }
            }

            var t = new Tensor(y, new[] { m, n }, new[] { this, other });

            t.Node.BackwardAction = () =>
            {
                for (int r = 0; r < m; r++)
                {
                    for (int c = 0; c < n; c++)
                    {
                        float g = t._grad[r * n + c];
                        for (int i = 0; i < k; i++)
                        {
                            _grad[r * k + i] += other._data[i * n + c] * g;
                            other._grad[i * n + c] += _data[r * k + i] * g;
                        }
                    }
                }
            };

            return t;
        }

        throw new NotSupportedException("MatMul supports rank1@rank1, rank2@rank1, rank2@rank2");
    }
}
