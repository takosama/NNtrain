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

        float[] y = (float[])_data.Clone();
        var t = new Tensor(y, newShape, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int i = 0; i < Numel; i++)
                _grad[i] += t._grad[i];
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
                for (int i = 0; i < length; i++)
                    _grad[start + i] += t._grad[i];
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
                        for (int c = 0; c < cols; c++)
                            _grad[(start + r) * cols + c] += t._grad[r * cols + c];
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
                        for (int c = 0; c < length; c++)
                            _grad[r * cols + (start + c)] += t._grad[r * length + c];
                };

                return t;
            }

            throw new ArgumentOutOfRangeException(
                nameof(dim),
                dim,
                "Rank-2 tensors only have dimensions 0 and 1.");
        }

        throw new NotSupportedException("Slice supports rank1/rank2 only");
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
                    for (int i = 0; i < xs[k].Numel; i++)
                        xs[k]._grad[i] += t._grad[off + i];
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
                            for (int c = 0; c < cols; c++)
                                xs[k]._grad[r * cols + c] += t._grad[(ro + r) * cols0 + c];
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
                            for (int c = 0; c < cols; c++)
                                xs[k]._grad[r * cols + c] += t._grad[r * totalCols + colOffset + c];
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

        throw new NotSupportedException("Concat supports rank1/rank2 only");
    }

    public Tensor Transpose()
    {
        CheckRank(2);

        int rows = _shape[0];
        int cols = _shape[1];
        float[] y = new float[Numel];

        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                y[c * rows + r] = _data[r * cols + c];

        var t = new Tensor(y, new[] { cols, rows }, new[] { this });

        t.Node.BackwardAction = () =>
        {
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    _grad[r * cols + c] += t._grad[c * rows + r];
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
