namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Multiplies rank-3 tensors shaped [batch, m, k] and [batch, k, n].
    /// </summary>
    public Tensor BatchedMatMul(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);
        CheckRank(3);
        other.CheckRank(3);

        int batch = _shape[0];
        int m = _shape[1];
        int k = _shape[2];
        if (other._shape[0] != batch || other._shape[1] != k)
            throw ShapeMismatch(this, other, "BatchedMatMul");
        int n = other._shape[2];

        float[] output = new float[checked(batch * m * n)];

        void ForwardBatch(int batchIndex)
        {
            int leftBatchOffset = batchIndex * m * k;
            int rightBatchOffset = batchIndex * k * n;
            int outputBatchOffset = batchIndex * m * n;

            for (int row = 0; row < m; row++)
            {
                int leftRow = leftBatchOffset + row * k;
                int outputRow = outputBatchOffset + row * n;
                for (int inner = 0; inner < k; inner++)
                {
                    AddScaledValues(
                        output,
                        outputRow,
                        other._data,
                        rightBatchOffset + inner * n,
                        _data[leftRow + inner],
                        n);
                }
            }
        }

        RunBatches(batch, (long)m * k * n, ForwardBatch);

        var result = new Tensor(
            output,
            [batch, m, n],
            [this, other]);

        result.Node.BackwardAction = () =>
        {
            void BackwardBatch(int batchIndex)
            {
                int leftBatchOffset = batchIndex * m * k;
                int rightBatchOffset = batchIndex * k * n;
                int outputBatchOffset = batchIndex * m * n;

                for (int row = 0; row < m; row++)
                {
                    int leftRow = leftBatchOffset + row * k;
                    int outputRow = outputBatchOffset + row * n;
                    for (int inner = 0; inner < k; inner++)
                    {
                        int rightRow = rightBatchOffset + inner * n;
                        float leftValue = _data[leftRow + inner];
                        _grad[leftRow + inner] += DotProduct(
                            other._data,
                            rightRow,
                            result._grad,
                            outputRow,
                            n);
                        AddScaledValues(
                            other._grad,
                            rightRow,
                            result._grad,
                            outputRow,
                            leftValue,
                            n);
                    }
                }
            }

            RunBatches(batch, (long)m * k * n, BackwardBatch);
        };

        return result;
    }

    /// <summary>
    /// Multiplies rank-3 tensors shaped [batch, m, k] and [batch, n, k]
    /// while treating the right operand as transposed.
    /// </summary>
    public Tensor BatchedMatMulTransposedRight(Tensor other)
    {
        ArgumentNullException.ThrowIfNull(other);
        CheckRank(3);
        other.CheckRank(3);

        int batch = _shape[0];
        int m = _shape[1];
        int k = _shape[2];
        if (other._shape[0] != batch || other._shape[2] != k)
        {
            throw ShapeMismatch(
                this,
                other,
                "BatchedMatMulTransposedRight");
        }
        int n = other._shape[1];

        float[] output = new float[checked(batch * m * n)];

        void ForwardBatch(int batchIndex)
        {
            int leftBatchOffset = batchIndex * m * k;
            int rightBatchOffset = batchIndex * n * k;
            int outputBatchOffset = batchIndex * m * n;

            for (int row = 0; row < m; row++)
            {
                int leftRow = leftBatchOffset + row * k;
                int outputRow = outputBatchOffset + row * n;
                for (int column = 0; column < n; column++)
                {
                    output[outputRow + column] = DotProduct(
                        _data,
                        leftRow,
                        other._data,
                        rightBatchOffset + column * k,
                        k);
                }
            }
        }

        RunBatches(batch, (long)m * n * k, ForwardBatch);

        var result = new Tensor(
            output,
            [batch, m, n],
            [this, other]);

        result.Node.BackwardAction = () =>
        {
            void BackwardBatch(int batchIndex)
            {
                int leftBatchOffset = batchIndex * m * k;
                int rightBatchOffset = batchIndex * n * k;
                int outputBatchOffset = batchIndex * m * n;

                for (int row = 0; row < m; row++)
                {
                    int leftRow = leftBatchOffset + row * k;
                    int outputRow = outputBatchOffset + row * n;
                    for (int column = 0; column < n; column++)
                    {
                        int rightRow = rightBatchOffset + column * k;
                        float gradient = result._grad[outputRow + column];
                        AddScaledValues(
                            _grad,
                            leftRow,
                            other._data,
                            rightRow,
                            gradient,
                            k);
                        AddScaledValues(
                            other._grad,
                            rightRow,
                            _data,
                            leftRow,
                            gradient,
                            k);
                    }
                }
            }

            RunBatches(batch, (long)m * n * k, BackwardBatch);
        };

        return result;
    }

    private static void RunBatches(
        int batchCount,
        long workPerBatch,
        Action<int> action)
    {
        const long ParallelWorkThreshold = 32_768;

        if (batchCount > 1
            && workPerBatch * batchCount >= ParallelWorkThreshold)
        {
            Parallel.For(0, batchCount, action);
            return;
        }

        for (int batch = 0; batch < batchCount; batch++)
            action(batch);
    }
}
