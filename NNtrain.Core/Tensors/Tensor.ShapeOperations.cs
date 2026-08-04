namespace NNtrain;

partial class Tensor
{
    public Tensor Reshape(params int[] newShape)
    {
        if (NumelOf(newShape) != Numel)
        {
            throw new ArgumentException(
                $"Cannot reshape {ShapeText(this)} with {Numel} elements to " +
                $"[{string.Join(", ", newShape)}].",
                nameof(newShape));
        }

        var t = new Tensor(_data, newShape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            AddScaledValues(_grad, 0, t._grad, 0, 1f, Numel);
        };

        return t;
    }

    public Tensor Slice(int dim, int start, int length)
    {
        if (Rank == 1)
        {
            if (dim != 0)
                throw new ArgumentOutOfRangeException(nameof(dim), dim, "Rank-1 tensors only have dimension 0.");
            ValidateSliceRange(_shape[0], start, length);

            float[] y = new float[length];
            Array.Copy(_data, start, y, 0, length);

            var t = new Tensor(y, new[] { length }, new[] { this });

            t.Node.BackwardAction = () =>
            {
                AddScaledValues(
                    _grad,
                    start,
                    t._grad,
                    0,
                    1f,
                    length);
            };

            return t;
        }

        if (Rank == 2)
        {
            int rows = _shape[0];
            int cols = _shape[1];

            if (dim == 0)
            {
                ValidateSliceRange(rows, start, length);

                float[] y = new float[length * cols];
                for (int r = 0; r < length; r++)
                    Array.Copy(_data, (start + r) * cols, y, r * cols, cols);

                var t = new Tensor(y, new[] { length, cols }, new[] { this });

                t.Node.BackwardAction = () =>
                {
                    for (int r = 0; r < length; r++)
                    {
                        AddScaledValues(
                            _grad,
                            (start + r) * cols,
                            t._grad,
                            r * cols,
                            1f,
                            cols);
                    }
                };

                return t;
            }

            if (dim == 1)
            {
                ValidateSliceRange(cols, start, length);

                float[] y = new float[rows * length];
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < length; c++)
                        y[r * length + c] = _data[r * cols + (start + c)];

                var t = new Tensor(y, new[] { rows, length }, new[] { this });

                t.Node.BackwardAction = () =>
                {
                    for (int r = 0; r < rows; r++)
                    {
                        AddScaledValues(
                            _grad,
                            r * cols + start,
                            t._grad,
                            r * length,
                            1f,
                            length);
                    }
                };

                return t;
            }

            throw new ArgumentOutOfRangeException(
                nameof(dim),
                dim,
                "Rank-2 tensors only have dimensions 0 and 1.");
        }

        if (Rank == 3)
        {
            int batch = _shape[0];
            int rows = _shape[1];
            int cols = _shape[2];

            if (dim == 0)
            {
                ValidateSliceRange(batch, start, length);
                int batchLength = rows * cols;
                float[] y = new float[length * batchLength];
                Array.Copy(
                    _data,
                    start * batchLength,
                    y,
                    0,
                    y.Length);

                var t = new Tensor(
                    y,
                    [length, rows, cols],
                    [this]);
                t.Node.BackwardAction = () =>
                    AddScaledValues(
                        _grad,
                        start * batchLength,
                        t._grad,
                        0,
                        1f,
                        y.Length);
                return t;
            }

            if (dim == 1)
            {
                ValidateSliceRange(rows, start, length);
                float[] y = new float[batch * length * cols];
                for (int batchIndex = 0;
                    batchIndex < batch;
                    batchIndex++)
                {
                    Array.Copy(
                        _data,
                        (batchIndex * rows + start) * cols,
                        y,
                        batchIndex * length * cols,
                        length * cols);
                }

                var t = new Tensor(
                    y,
                    [batch, length, cols],
                    [this]);
                t.Node.BackwardAction = () =>
                {
                    for (int batchIndex = 0;
                        batchIndex < batch;
                        batchIndex++)
                    {
                        AddScaledValues(
                            _grad,
                            (batchIndex * rows + start) * cols,
                            t._grad,
                            batchIndex * length * cols,
                            1f,
                            length * cols);
                    }
                };
                return t;
            }

            if (dim == 2)
            {
                ValidateSliceRange(cols, start, length);
                float[] y = new float[batch * rows * length];
                for (int batchIndex = 0;
                    batchIndex < batch;
                    batchIndex++)
                {
                    for (int row = 0; row < rows; row++)
                    {
                        Array.Copy(
                            _data,
                            (batchIndex * rows + row) * cols + start,
                            y,
                            (batchIndex * rows + row) * length,
                            length);
                    }
                }

                var t = new Tensor(
                    y,
                    [batch, rows, length],
                    [this]);
                t.Node.BackwardAction = () =>
                {
                    for (int batchIndex = 0;
                        batchIndex < batch;
                        batchIndex++)
                    {
                        for (int row = 0; row < rows; row++)
                        {
                            AddScaledValues(
                                _grad,
                                (batchIndex * rows + row) * cols + start,
                                t._grad,
                                (batchIndex * rows + row) * length,
                                1f,
                                length);
                        }
                    }
                };
                return t;
            }

            throw new ArgumentOutOfRangeException(
                nameof(dim),
                dim,
                "Rank-3 tensors only have dimensions 0, 1, and 2.");
        }

        throw new NotSupportedException("Slice supports rank1/rank2/rank3 only");
    }

    public static Tensor Concat(int dim, params Tensor[] xs)
    {
        ArgumentNullException.ThrowIfNull(xs);
        if (xs.Length == 0)
            throw new ArgumentException("Concat requires at least one tensor.", nameof(xs));
        if (xs.Any(static tensor => tensor is null))
            throw new ArgumentException("Concat does not accept null tensors.", nameof(xs));

        int rank = xs[0].Rank;
        for (int i = 1; i < xs.Length; i++)
            if (xs[i].Rank != rank)
                throw new ArgumentException("All tensors passed to Concat must have the same rank.", nameof(xs));

        if (rank == 1)
        {
            if (dim != 0)
                throw new ArgumentOutOfRangeException(nameof(dim), dim, "Rank-1 tensors can only concatenate on dimension 0.");

            int total = 0;
            for (int i = 0; i < xs.Length; i++) total += xs[i]._shape[0];

            float[] y = new float[total];
            int offset = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                Array.Copy(xs[i]._data, 0, y, offset, xs[i].Numel);
                offset += xs[i].Numel;
            }

            var t = new Tensor(y, new[] { total }, xs);

            t.Node.BackwardAction = () =>
            {
                int off = 0;
                for (int k = 0; k < xs.Length; k++)
                {
                    AddScaledValues(
                        xs[k]._grad,
                        0,
                        t._grad,
                        off,
                        1f,
                        xs[k].Numel);
                    off += xs[k].Numel;
                }
            };

            return t;
        }

        if (rank == 2)
        {
            int rows0 = xs[0]._shape[0];
            int cols0 = xs[0]._shape[1];

            if (dim == 0)
            {
                for (int i = 1; i < xs.Length; i++)
                    if (xs[i]._shape[1] != cols0)
                        throw new ArgumentException(
                            "All rank-2 tensors concatenated on dimension 0 must have the same column count.",
                            nameof(xs));

                int totalRows = 0;
                for (int i = 0; i < xs.Length; i++) totalRows += xs[i]._shape[0];

                float[] y = new float[totalRows * cols0];
                int rowOffset = 0;

                for (int k = 0; k < xs.Length; k++)
                {
                    int rows = xs[k]._shape[0];
                    int cols = xs[k]._shape[1];
                    for (int r = 0; r < rows; r++)
                        Array.Copy(xs[k]._data, r * cols, y, (rowOffset + r) * cols0, cols);
                    rowOffset += rows;
                }

                var t = new Tensor(y, new[] { totalRows, cols0 }, xs);

                t.Node.BackwardAction = () =>
                {
                    int ro = 0;
                    for (int k = 0; k < xs.Length; k++)
                    {
                        int rows = xs[k]._shape[0];
                        int cols = xs[k]._shape[1];
                        for (int r = 0; r < rows; r++)
                        {
                            AddScaledValues(
                                xs[k]._grad,
                                r * cols,
                                t._grad,
                                (ro + r) * cols0,
                                1f,
                                cols);
                        }
                        ro += rows;
                    }
                };

                return t;
            }

            if (dim == 1)
            {
                for (int i = 1; i < xs.Length; i++)
                    if (xs[i]._shape[0] != rows0)
                        throw new ArgumentException(
                            "All rank-2 tensors concatenated on dimension 1 must have the same row count.",
                            nameof(xs));

                int totalCols = 0;
                for (int i = 0; i < xs.Length; i++) totalCols += xs[i]._shape[1];

                float[] y = new float[rows0 * totalCols];

                for (int r = 0; r < rows0; r++)
                {
                    int colOffset = 0;
                    for (int k = 0; k < xs.Length; k++)
                    {
                        int cols = xs[k]._shape[1];
                        Array.Copy(xs[k]._data, r * cols, y, r * totalCols + colOffset, cols);
                        colOffset += cols;
                    }
                }

                var t = new Tensor(y, new[] { rows0, totalCols }, xs);

                t.Node.BackwardAction = () =>
                {
                    for (int r = 0; r < rows0; r++)
                    {
                        int colOffset = 0;
                        for (int k = 0; k < xs.Length; k++)
                        {
                            int cols = xs[k]._shape[1];
                            AddScaledValues(
                                xs[k]._grad,
                                r * cols,
                                t._grad,
                                r * totalCols + colOffset,
                                1f,
                                cols);
                            colOffset += cols;
                        }
                    }
                };

                return t;
            }

            throw new ArgumentOutOfRangeException(
                nameof(dim),
                dim,
                "Rank-2 tensors can only concatenate on dimensions 0 and 1.");
        }

        if (rank == 3)
        {
            int batch0 = xs[0]._shape[0];
            int rows0 = xs[0]._shape[1];
            int cols0 = xs[0]._shape[2];

            if (dim == 0)
            {
                for (int index = 1; index < xs.Length; index++)
                {
                    if (xs[index]._shape[1] != rows0
                        || xs[index]._shape[2] != cols0)
                    {
                        throw new ArgumentException(
                            "All rank-3 tensors concatenated on dimension " +
                            "0 must have the same row and column counts.",
                            nameof(xs));
                    }
                }

                int totalBatch = xs.Sum(tensor => tensor._shape[0]);
                float[] y = new float[totalBatch * rows0 * cols0];
                int offset = 0;
                foreach (Tensor tensor in xs)
                {
                    Array.Copy(tensor._data, 0, y, offset, tensor.Numel);
                    offset += tensor.Numel;
                }

                var t = new Tensor(
                    y,
                    [totalBatch, rows0, cols0],
                    xs);
                t.Node.BackwardAction = () =>
                {
                    int gradientOffset = 0;
                    foreach (Tensor tensor in xs)
                    {
                        AddScaledValues(
                            tensor._grad,
                            0,
                            t._grad,
                            gradientOffset,
                            1f,
                            tensor.Numel);
                        gradientOffset += tensor.Numel;
                    }
                };
                return t;
            }

            if (dim == 1)
            {
                for (int index = 1; index < xs.Length; index++)
                {
                    if (xs[index]._shape[0] != batch0
                        || xs[index]._shape[2] != cols0)
                    {
                        throw new ArgumentException(
                            "All rank-3 tensors concatenated on dimension " +
                            "1 must have the same batch and column counts.",
                            nameof(xs));
                    }
                }

                int totalRows = xs.Sum(tensor => tensor._shape[1]);
                float[] y = new float[batch0 * totalRows * cols0];
                for (int batchIndex = 0;
                    batchIndex < batch0;
                    batchIndex++)
                {
                    int rowOffset = 0;
                    foreach (Tensor tensor in xs)
                    {
                        int rows = tensor._shape[1];
                        Array.Copy(
                            tensor._data,
                            batchIndex * rows * cols0,
                            y,
                            (batchIndex * totalRows + rowOffset) * cols0,
                            rows * cols0);
                        rowOffset += rows;
                    }
                }

                var t = new Tensor(
                    y,
                    [batch0, totalRows, cols0],
                    xs);
                t.Node.BackwardAction = () =>
                {
                    for (int batchIndex = 0;
                        batchIndex < batch0;
                        batchIndex++)
                    {
                        int rowOffset = 0;
                        foreach (Tensor tensor in xs)
                        {
                            int rows = tensor._shape[1];
                            AddScaledValues(
                                tensor._grad,
                                batchIndex * rows * cols0,
                                t._grad,
                                (batchIndex * totalRows + rowOffset) * cols0,
                                1f,
                                rows * cols0);
                            rowOffset += rows;
                        }
                    }
                };
                return t;
            }

            if (dim == 2)
            {
                for (int index = 1; index < xs.Length; index++)
                {
                    if (xs[index]._shape[0] != batch0
                        || xs[index]._shape[1] != rows0)
                    {
                        throw new ArgumentException(
                            "All rank-3 tensors concatenated on dimension " +
                            "2 must have the same batch and row counts.",
                            nameof(xs));
                    }
                }

                int totalCols = xs.Sum(tensor => tensor._shape[2]);
                float[] y = new float[batch0 * rows0 * totalCols];
                for (int batchIndex = 0;
                    batchIndex < batch0;
                    batchIndex++)
                {
                    for (int row = 0; row < rows0; row++)
                    {
                        int columnOffset = 0;
                        foreach (Tensor tensor in xs)
                        {
                            int cols = tensor._shape[2];
                            Array.Copy(
                                tensor._data,
                                (batchIndex * rows0 + row) * cols,
                                y,
                                (batchIndex * rows0 + row) * totalCols
                                    + columnOffset,
                                cols);
                            columnOffset += cols;
                        }
                    }
                }

                var t = new Tensor(
                    y,
                    [batch0, rows0, totalCols],
                    xs);
                t.Node.BackwardAction = () =>
                {
                    for (int batchIndex = 0;
                        batchIndex < batch0;
                        batchIndex++)
                    {
                        for (int row = 0; row < rows0; row++)
                        {
                            int columnOffset = 0;
                            foreach (Tensor tensor in xs)
                            {
                                int cols = tensor._shape[2];
                                AddScaledValues(
                                    tensor._grad,
                                    (batchIndex * rows0 + row) * cols,
                                    t._grad,
                                    (batchIndex * rows0 + row) * totalCols
                                        + columnOffset,
                                    1f,
                                    cols);
                                columnOffset += cols;
                            }
                        }
                    }
                };
                return t;
            }

            throw new ArgumentOutOfRangeException(
                nameof(dim),
                dim,
                "Rank-3 tensors can only concatenate on dimensions 0, 1, " +
                "and 2.");
        }

        throw new NotSupportedException("Concat supports rank1/rank2/rank3 only");
    }

    public Tensor Transpose()
    {
        CheckRank(2);

        int rows = _shape[0];
        int cols = _shape[1];
        float[] y = new float[Numel];

        const int blockSize = 32;
        for (int rowBlock = 0; rowBlock < rows; rowBlock += blockSize)
        {
            int rowEnd = Math.Min(rowBlock + blockSize, rows);
            for (int columnBlock = 0;
                columnBlock < cols;
                columnBlock += blockSize)
            {
                int columnEnd = Math.Min(columnBlock + blockSize, cols);
                for (int r = rowBlock; r < rowEnd; r++)
                {
                    int sourceRow = r * cols;
                    for (int c = columnBlock; c < columnEnd; c++)
                        y[c * rows + r] = _data[sourceRow + c];
                }
            }
        }

        var t = new Tensor(y, new[] { cols, rows }, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int rowBlock = 0; rowBlock < rows; rowBlock += blockSize)
            {
                int rowEnd = Math.Min(rowBlock + blockSize, rows);
                for (int columnBlock = 0;
                    columnBlock < cols;
                    columnBlock += blockSize)
                {
                    int columnEnd = Math.Min(columnBlock + blockSize, cols);
                    for (int r = rowBlock; r < rowEnd; r++)
                    {
                        int destinationRow = r * cols;
                        for (int c = columnBlock; c < columnEnd; c++)
                        {
                            _grad[destinationRow + c] +=
                                t._grad[c * rows + r];
                        }
                    }
                }
            }
        };

        return t;
    }

    private static void ValidateSliceRange(int dimensionSize, int start, int length)
    {
        if (start < 0 || start >= dimensionSize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(start),
                start,
                $"Slice start must be between 0 and {dimensionSize - 1}.");
        }

        if (length <= 0 || length > dimensionSize - start)
        {
            throw new ArgumentOutOfRangeException(
                nameof(length),
                length,
                $"Slice length must be positive and fit within dimension size {dimensionSize} from start {start}.");
        }
    }
}
