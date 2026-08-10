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

            float s = DotProduct(_data, 0, other._data, 0, n);

            var t = new Tensor(new[] { s }, new[] { 1 }, new[] { this, other });

            t.Node.BackwardAction = () =>
            {
                float g = t._grad[0];
                AddScaledValues(_grad, 0, other._data, 0, g, n);
                AddScaledValues(other._grad, 0, _data, 0, g, n);
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
            void ForwardRow(int r)
            {
                y[r] = DotProduct(_data, r * k, other._data, 0, k);
            }

            RunBatches(m, k, ForwardRow);

            var t = new Tensor(y, new[] { m }, new[] { this, other });

            t.Node.BackwardAction = () =>
            {
                void BackwardInputRow(int r)
                {
                    float g = t._grad[r];
                    AddScaledValues(
                        _grad,
                        r * k,
                        other._data,
                        0,
                        g,
                        k);
                }

                RunBatches(m, k, BackwardInputRow);

                void BackwardVectorElement(int i)
                {
                    float gradient = 0f;
                    for (int r = 0; r < m; r++)
                    {
                        gradient += _data[r * k + i] * t._grad[r];
                    }

                    other._grad[i] += gradient;
                }

                RunBatches(k, m, BackwardVectorElement);
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

            void ForwardRow(int r)
            {
                int leftRow = r * k;
                int outputRow = r * n;
                if (CanUseSimd(n))
                {
                    for (int i = 0; i < k; i++)
                    {
                        AddScaledValues(
                            y,
                            outputRow,
                            other._data,
                            i * n,
                            _data[leftRow + i],
                            n);
                    }
                }
                else
                {
                    for (int i = 0; i < k; i++)
                    {
                        float leftValue = _data[leftRow + i];
                        int rightRow = i * n;
                        for (int c = 0; c < n; c++)
                        {
                            y[outputRow + c] +=
                                leftValue * other._data[rightRow + c];
                        }
                    }
                }
            }

            RunBatches(m, (long)k * n, ForwardRow);

            var t = new Tensor(y, new[] { m, n }, new[] { this, other });

            t.Node.BackwardAction = () =>
            {
                void BackwardLeftRow(int r)
                {
                    int leftRow = r * k;
                    int outputRow = r * n;
                    for (int i = 0; i < k; i++)
                    {
                        int rightRow = i * n;
                        float leftGradient = DotProduct(
                            other._data,
                            rightRow,
                            t._grad,
                            outputRow,
                            n);
                        _grad[leftRow + i] += leftGradient;
                    }
                }

                RunBatches(m, (long)k * n, BackwardLeftRow);

                void BackwardRightRow(int i)
                {
                    int rightRow = i * n;
                    for (int r = 0; r < m; r++)
                    {
                        AddScaledValues(
                            other._grad,
                            rightRow,
                            t._grad,
                            r * n,
                            _data[r * k + i],
                            n);
                    }
                }

                RunBatches(k, (long)m * n, BackwardRightRow);
            };

            return t;
        }

        throw new NotSupportedException("MatMul supports rank1@rank1, rank2@rank1, rank2@rank2");
    }

    /// <summary>
    /// Multiplies two rank-2 tensors while treating the right operand as
    /// transposed, without allocating a transposed tensor.
    /// </summary>
    /// <remarks>
    /// A tensor shaped [m, k] multiplied by a tensor shaped [n, k] produces
    /// [m, n]. This is equivalent to <c>left.MatMul(right.Transpose())</c>.
    /// </remarks>
    public Tensor MatMulTransposedRight(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);
        CheckRank(2);
        other.CheckRank(2);

        int m = _shape[0];
        int k = _shape[1];
        int n = other._shape[0];
        if (other._shape[1] != k)
            throw ShapeMismatch(this, other, "MatMulTransposedRight");

        float[] y = new float[m * n];

        void ForwardRow(int r)
        {
            int leftRow = r * k;
            int outputRow = r * n;
            for (int c = 0; c < n; c++)
            {
                int rightRow = c * k;
                y[outputRow + c] = DotProduct(
                    _data,
                    leftRow,
                    other._data,
                    rightRow,
                    k);
            }
        }

        RunBatches(m, (long)n * k, ForwardRow);

        var result = new Tensor(y, [m, n], [this, other]);
        result.Node.BackwardAction = () =>
        {
            void BackwardInputRow(int r)
            {
                int leftRow = r * k;
                int outputRow = r * n;
                for (int c = 0; c < n; c++)
                {
                    int rightRow = c * k;
                    float gradient = result._grad[outputRow + c];
                    AddScaledValues(
                        _grad,
                        leftRow,
                        other._data,
                        rightRow,
                        gradient,
                        k);
                }
            }

            RunBatches(m, (long)n * k, BackwardInputRow);

            void BackwardWeightRow(int c)
            {
                int rightRow = c * k;
                for (int r = 0; r < m; r++)
                {
                    float gradient = result._grad[r * n + c];
                    AddScaledValues(
                        other._grad,
                        rightRow,
                        _data,
                        r * k,
                        gradient,
                        k);
                }
            }

            RunBatches(n, (long)m * k, BackwardWeightRow);
        };

        return result;
    }

    /// <summary>
    /// Computes a rank-2 multiplication with a transposed right operand and
    /// adds a rank-1 bias to every output row in one operation.
    /// </summary>
    public Tensor MatMulTransposedRightAddRow(
        Tensor other,
        Tensor rowBias)
    {
        ArgumentNullException.ThrowIfNull(other);
        ArgumentNullException.ThrowIfNull(rowBias);
        CheckRank(2);
        other.CheckRank(2);
        rowBias.CheckRank(1);

        int m = _shape[0];
        int k = _shape[1];
        int n = other._shape[0];
        if (other._shape[1] != k || rowBias._shape[0] != n)
        {
            throw new ArgumentException(
                "MatMulTransposedRightAddRow requires shapes [m, k], " +
                "[n, k], and [n].");
        }

        float[] y = new float[m * n];
        bool useOutputVectorization = CanUseTransposedRightKernel(k, n);
        float[]? transposedOther = useOutputVectorization
            ? other.GetTransposedData2D()
            : null;

        void ForwardRow(int r)
        {
            int leftRow = r * k;
            int outputRow = r * n;
            if (transposedOther is not null)
            {
                Array.Copy(rowBias._data, 0, y, outputRow, n);
                for (int inner = 0; inner < k; inner++)
                {
                    AddScaledValues(
                        y,
                        outputRow,
                        transposedOther,
                        inner * n,
                        _data[leftRow + inner],
                        n);
                }
                return;
            }

            for (int c = 0; c < n; c++)
            {
                y[outputRow + c] = rowBias._data[c] + DotProduct(
                    _data,
                    leftRow,
                    other._data,
                    c * k,
                    k);
            }
        }

        RunBatches(m, (long)n * k, ForwardRow);

        var result = new Tensor(y, [m, n], [this, other, rowBias]);
        result.Node.BackwardAction = () =>
        {
            void BackwardInputRow(int r)
            {
                int leftRow = r * k;
                int outputRow = r * n;
                if (transposedOther is not null)
                {
                    for (int inner = 0; inner < k; inner++)
                    {
                        _grad[leftRow + inner] += DotProduct(
                            result._grad,
                            outputRow,
                            transposedOther,
                            inner * n,
                            n);
                    }
                    return;
                }

                for (int c = 0; c < n; c++)
                {
                    int rightRow = c * k;
                    float gradient = result._grad[outputRow + c];
                    AddScaledValues(
                        _grad,
                        leftRow,
                        other._data,
                        rightRow,
                        gradient,
                        k);
                }
            }

            RunBatches(m, (long)n * k, BackwardInputRow);

            void BackwardWeightRow(int c)
            {
                int rightRow = c * k;
                float biasGradient = 0f;
                for (int r = 0; r < m; r++)
                {
                    float gradient = result._grad[r * n + c];
                    biasGradient += gradient;
                    AddScaledValues(
                        other._grad,
                        rightRow,
                        _data,
                        r * k,
                        gradient,
                        k);
                }

                rowBias._grad[c] += biasGradient;
            }

            RunBatches(n, (long)m * k, BackwardWeightRow);
        };

        return result;
    }

    private static float DotProduct(
        float[] left,
        int leftOffset,
        float[] right,
        int rightOffset,
        int length)
    {
        int index = 0;
        float sum = 0f;
        if (CanUseSimd(length))
        {
            int vectorWidth = Vector256<float>.Count;
            int vectorizedLength = length - length % vectorWidth;
            Vector256<float> sumVector = Vector256<float>.Zero;

            for (; index < vectorizedLength; index += vectorWidth)
            {
                Vector256<float> leftVector = LoadVector256(
                    left,
                    leftOffset + index);
                Vector256<float> rightVector = LoadVector256(
                    right,
                    rightOffset + index);
                sumVector += leftVector * rightVector;
            }
            sum = Vector256.Sum(sumVector);
        }

        if (CanUseVector128(length - index))
        {
            sum += Vector128.Sum(
                LoadVector128(left, leftOffset + index)
                    * LoadVector128(right, rightOffset + index));
            index += Vector128<float>.Count;
        }

        for (; index < length; index++)
            sum += left[leftOffset + index] * right[rightOffset + index];

        return sum;
    }
}
