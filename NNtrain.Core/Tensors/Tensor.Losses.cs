namespace NNtrain;

partial class Tensor
{
    public const int DefaultCrossEntropyIgnoreIndex = -1;

    private const int MaximumCachedCrossEntropyProbabilities = 1 << 20;

    /// <summary>
    /// Computes mean cross entropy directly from rank-1 or rank-2 logits
    /// and integer class labels.
    /// </summary>
    public Tensor CrossEntropyWithLogits(
        int[] labels,
        float labelSmoothing = 0f,
        int ignoreIndex = DefaultCrossEntropyIgnoreIndex)
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
        int validRows = 0;
        for (int row = 0; row < rows; row++)
        {
            if (retainedLabels[row] == ignoreIndex)
                continue;
            if ((uint)retainedLabels[row] >= (uint)columns)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(labels),
                    retainedLabels[row],
                    $"Label at row {row} must be between 0 and " +
                    $"{columns - 1}.");
            }
            validRows++;
        }
        if (validRows == 0)
        {
            throw new ArgumentException(
                "At least one label must not equal ignoreIndex.",
                nameof(labels));
        }

        float[] rowLosses = new float[rows];
        float[] rowMaximums = new float[rows];
        float[] rowInverseSums = new float[rows];
        float[]? cachedProbabilities = Numel
            <= MaximumCachedCrossEntropyProbabilities
                ? new float[Numel]
                : null;
        void ForwardRow(int row)
        {
            if (retainedLabels[row] == ignoreIndex)
                return;

            int offset = row * columns;
            float maximum = MaxValues(_data, offset, columns);
            rowMaximums[row] = maximum;

            float logitSum = SumValues(_data, offset, columns);
            float sum;
            if (cachedProbabilities is null)
            {
                sum = SumExpShiftedValues(
                    _data,
                    offset,
                    maximum,
                    columns);
            }
            else
            {
                sum = ExpShiftedValues(
                    _data,
                    offset,
                    maximum,
                    cachedProbabilities,
                    offset,
                    columns);
                MultiplyValues(
                    cachedProbabilities,
                    offset,
                    1f / sum,
                    cachedProbabilities,
                    offset,
                    columns);
            }
            rowInverseSums[row] = 1f / sum;
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
        float meanLoss = SumValues(rowLosses, 0, rows) / validRows;
        // Loss values and their reduction are always retained in Float32,
        // even when the logits use a lower-precision physical storage.
        var result = new Tensor(
            [meanLoss],
            [1],
            [this],
            dtype: TensorDType.Float32);
        result.Node.BackwardAction = () =>
        {
            float scale = result._grad[0] / validRows;
            float uniformTarget = scale * labelSmoothing / columns;
            float trueTarget = scale * (1f - labelSmoothing);
            void BackwardRow(int row)
            {
                if (retainedLabels[row] == ignoreIndex)
                    return;

                int offset = row * columns;
                if (cachedProbabilities is null)
                {
                    AccumulateNormalizedExpGradient(
                        _grad,
                        offset,
                        _data,
                        offset,
                        rowMaximums[row],
                        scale * rowInverseSums[row],
                        uniformTarget,
                        columns);
                }
                else
                {
                    AddScaledValues(
                        _grad,
                        offset,
                        cachedProbabilities,
                        offset,
                        scale,
                        columns);
                    AddConstantValuesInPlace(
                        _grad,
                        offset,
                        -uniformTarget,
                        columns);
                }
                _grad[offset + retainedLabels[row]] -= trueTarget;
            }

            RunBatches(rows, columns, BackwardRow);
        };

        return result;
    }
}
