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

        if (ExecutionDevice == TensorDevice.Cuda)
        {
            var cudaBuffer = TensorCudaKernels.CopyForwardResident(this);
            Tensor cudaResult = FromCudaResult(
                cudaBuffer,
                CudaDeviceIndex,
                newShape,
                [this],
                DType);
            cudaResult.Node.BackwardAction = () =>
                TensorCudaKernels.AccumulateGradientResident(
                    cudaResult,
                    this);
            return cudaResult;
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

            TensorStorage y = TensorStorage.CreateUninitialized(length, DType);
            _data.CopyRangeTo(start, y, 0, length);

            var t = FromStorageResult(y, [length], [this]);

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

                int resultLength = length * cols;
                TensorStorage y = TensorStorage.CreateUninitialized(
                    resultLength,
                    DType);
                _data.CopyRangeTo(
                    start * cols,
                    y,
                    0,
                    resultLength);

                var t = FromStorageResult(y, [length, cols], [this]);

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

                TensorStorage y = TensorStorage.CreateUninitialized(
                    rows * length,
                    DType);
                for (int r = 0; r < rows; r++)
                {
                    _data.CopyRangeTo(
                        r * cols + start,
                        y,
                        r * length,
                        length);
                }

                var t = FromStorageResult(y, [rows, length], [this]);

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
                int resultLength = length * batchLength;
                TensorStorage y = TensorStorage.CreateUninitialized(
                    resultLength,
                    DType);
                _data.CopyRangeTo(
                    start * batchLength,
                    y,
                    0,
                    resultLength);

                var t = FromStorageResult(
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
                        y.Count);
                return t;
            }

            if (dim == 1)
            {
                ValidateSliceRange(rows, start, length);
                TensorStorage y = TensorStorage.CreateUninitialized(
                    batch * length * cols,
                    DType);
                for (int batchIndex = 0;
                    batchIndex < batch;
                    batchIndex++)
                {
                    int copyLength = length * cols;
                    _data.CopyRangeTo(
                        (batchIndex * rows + start) * cols,
                        y,
                        batchIndex * copyLength,
                        copyLength);
                }

                var t = FromStorageResult(
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
                TensorStorage y = TensorStorage.CreateUninitialized(
                    batch * rows * length,
                    DType);
                for (int batchIndex = 0;
                    batchIndex < batch;
                    batchIndex++)
                {
                    for (int row = 0; row < rows; row++)
                    {
                        _data.CopyRangeTo(
                            (batchIndex * rows + row) * cols + start,
                            y,
                            (batchIndex * rows + row) * length,
                            length);
                    }
                }

                var t = FromStorageResult(
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

        TensorDType resultDType = TensorDTypeContract.Promote(xs);

        if (rank == 1)
        {
            if (dim != 0)
                throw new ArgumentOutOfRangeException(nameof(dim), dim, "Rank-1 tensors can only concatenate on dimension 0.");

            int total = 0;
            for (int i = 0; i < xs.Length; i++) total += xs[i]._shape[0];

            TensorStorage y = TensorStorage.CreateUninitialized(
                total,
                resultDType);
            int offset = 0;
            for (int i = 0; i < xs.Length; i++)
            {
                xs[i]._data.CopyRangeTo(
                    0,
                    y,
                    offset,
                    xs[i].Numel);
                offset += xs[i].Numel;
            }

            var t = FromStorageResult(y, [total], xs);

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

                TensorStorage y = TensorStorage.CreateUninitialized(
                    totalRows * cols0,
                    resultDType);
                int rowOffset = 0;

                for (int k = 0; k < xs.Length; k++)
                {
                    int rows = xs[k]._shape[0];
                    int cols = xs[k]._shape[1];
                    for (int r = 0; r < rows; r++)
                    {
                        xs[k]._data.CopyRangeTo(
                            r * cols,
                            y,
                            (rowOffset + r) * cols0,
                            cols);
                    }
                    rowOffset += rows;
                }

                var t = FromStorageResult(y, [totalRows, cols0], xs);

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

                TensorStorage y = TensorStorage.CreateUninitialized(
                    rows0 * totalCols,
                    resultDType);

                for (int r = 0; r < rows0; r++)
                {
                    int colOffset = 0;
                    for (int k = 0; k < xs.Length; k++)
                    {
                        int cols = xs[k]._shape[1];
                        xs[k]._data.CopyRangeTo(
                            r * cols,
                            y,
                            r * totalCols + colOffset,
                            cols);
                        colOffset += cols;
                    }
                }

                var t = FromStorageResult(y, [rows0, totalCols], xs);

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
                TensorStorage y = TensorStorage.CreateUninitialized(
                    totalBatch * rows0 * cols0,
                    resultDType);
                int offset = 0;
                foreach (Tensor tensor in xs)
                {
                    tensor._data.CopyRangeTo(
                        0,
                        y,
                        offset,
                        tensor.Numel);
                    offset += tensor.Numel;
                }

                var t = FromStorageResult(
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
                TensorStorage y = TensorStorage.CreateUninitialized(
                    batch0 * totalRows * cols0,
                    resultDType);
                for (int batchIndex = 0;
                    batchIndex < batch0;
                    batchIndex++)
                {
                    int rowOffset = 0;
                    foreach (Tensor tensor in xs)
                    {
                        int rows = tensor._shape[1];
                        int copyLength = rows * cols0;
                        tensor._data.CopyRangeTo(
                            batchIndex * copyLength,
                            y,
                            (batchIndex * totalRows + rowOffset) * cols0,
                            copyLength);
                        rowOffset += rows;
                    }
                }

                var t = FromStorageResult(
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
                TensorStorage y = TensorStorage.CreateUninitialized(
                    batch0 * rows0 * totalCols,
                    resultDType);
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
                            tensor._data.CopyRangeTo(
                                (batchIndex * rows0 + row) * cols,
                                y,
                                (batchIndex * rows0 + row) * totalCols
                                    + columnOffset,
                                cols);
                            columnOffset += cols;
                        }
                    }
                }

                var t = FromStorageResult(
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
        TensorStorage y = TensorStorage.CreateUninitialized(Numel, DType);
        _data.Transpose2DTo(y, rows, cols);

        var t = FromStorageResult(y, [cols, rows], [this]);

        t.Node.BackwardAction = () =>
        {
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
