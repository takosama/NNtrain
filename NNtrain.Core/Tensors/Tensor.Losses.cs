namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Computes mean cross entropy directly from rank-1 or rank-2 logits
    /// and integer class labels.
    /// </summary>
    public Tensor CrossEntropyWithLogits(int[] labels)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (Rank is not 1 and not 2)
        {
            throw new InvalidOperationException(
                "CrossEntropyWithLogits requires rank-1 or rank-2 logits.");
        }

        int rows = Rank == 1 ? 1 : _shape[0];
        int columns = Rank == 1 ? _shape[0] : _shape[1];
        if (labels.Length != rows)
        {
            throw new ArgumentException(
                $"Expected {rows} labels, but received {labels.Length}.",
                nameof(labels));
        }

        int[] retainedLabels = (int[])labels.Clone();
        for (int row = 0; row < rows; row++)
        {
            if ((uint)retainedLabels[row] >= (uint)columns)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(labels),
                    retainedLabels[row],
                    $"Label at row {row} must be between 0 and " +
                    $"{columns - 1}.");
            }
        }

        float[] probabilities = new float[Numel];
        float[] rowLosses = new float[rows];
        void ForwardRow(int row)
        {
            int offset = row * columns;
            float maximum = _data[offset];
            for (int column = 1; column < columns; column++)
            {
                float value = _data[offset + column];
                if (value > maximum)
                    maximum = value;
            }

            float sum = 0f;
            for (int column = 0; column < columns; column++)
            {
                float probability =
                    MathF.Exp(_data[offset + column] - maximum);
                probabilities[offset + column] = probability;
                sum += probability;
            }

            MultiplyValues(
                probabilities,
                offset,
                1f / sum,
                probabilities,
                offset,
                columns);
            rowLosses[row] = maximum
                + MathF.Log(sum)
                - _data[offset + retainedLabels[row]];
        }

        RunBatches(rows, columns, ForwardRow);
        float meanLoss = SumValues(rowLosses, 0, rows) / rows;
        var result = new Tensor([meanLoss], [1], [this]);
        result.Node.BackwardAction = () =>
        {
            float scale = result._grad[0] / rows;
            void BackwardRow(int row)
            {
                int offset = row * columns;
                AddScaledValues(
                    _grad,
                    offset,
                    probabilities,
                    offset,
                    scale,
                    columns);
                _grad[offset + retainedLabels[row]] -= scale;
            }

            RunBatches(rows, columns, BackwardRow);
        };

        return result;
    }
}
