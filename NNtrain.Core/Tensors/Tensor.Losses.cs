namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Computes mean cross entropy directly from rank-1 or rank-2 logits
    /// and integer class labels.
    /// </summary>
    public Tensor CrossEntropyWithLogits(
        int[] labels,
        float labelSmoothing = 0f)
    {
        ArgumentNullException.ThrowIfNull(labels);
        if (!float.IsFinite(labelSmoothing)
            || labelSmoothing < 0f
            || labelSmoothing >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(labelSmoothing),
                labelSmoothing,
                "Label smoothing must be finite and in the range [0, 1).");
        }

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
            float logitSum = 0f;
            for (int column = 0; column < columns; column++)
            {
                logitSum += _data[offset + column];
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
            float logNormalizer = maximum + MathF.Log(sum);
            float negativeLogLikelihood = logNormalizer
                - _data[offset + retainedLabels[row]];
            float uniformLoss = logNormalizer
                - logitSum / columns;
            rowLosses[row] =
                (1f - labelSmoothing) * negativeLogLikelihood
                + labelSmoothing * uniformLoss;
        }

        RunBatches(rows, columns, ForwardRow);
        float meanLoss = SumValues(rowLosses, 0, rows) / rows;
        var result = new Tensor([meanLoss], [1], [this]);
        result.Node.BackwardAction = () =>
        {
            float scale = result._grad[0] / rows;
            float uniformTarget = scale * labelSmoothing / columns;
            float trueTarget = scale * (1f - labelSmoothing);
            void BackwardRow(int row)
            {
                int offset = row * columns;
                int column = 0;
                if (CanUseSimd(columns))
                {
                    int vectorWidth = Vector256<float>.Count;
                    int vectorizedLength =
                        columns - columns % vectorWidth;
                    Vector256<float> scaleVector =
                        Vector256.Create(scale);
                    Vector256<float> uniformTargetVector =
                        Vector256.Create(uniformTarget);
                    for (; column < vectorizedLength; column += vectorWidth)
                    {
                        StoreVector256(
                            LoadVector256(_grad, offset + column)
                                + LoadVector256(
                                    probabilities,
                                    offset + column)
                                    * scaleVector
                                - uniformTargetVector,
                            _grad,
                            offset + column);
                    }
                }

                for (; column < columns; column++)
                {
                    _grad[offset + column] +=
                        probabilities[offset + column] * scale
                        - uniformTarget;
                }
                _grad[offset + retainedLabels[row]] -= trueTarget;
            }

            RunBatches(rows, columns, BackwardRow);
        };

        return result;
    }
}
