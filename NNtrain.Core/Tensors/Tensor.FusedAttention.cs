namespace NNtrain;

partial class Tensor
{
    /// <summary>
    /// Applies scaled dot-product multi-head attention to a fused QKV
    /// projection shaped [sequence, 3 * model] or
    /// [batch, sequence, 3 * model].
    /// </summary>
    public Tensor FusedMultiHeadAttention(
        int numHeads,
        bool causal = false)
    {
        if (Rank is not 2 and not 3)
        {
            throw new InvalidOperationException(
                "FusedMultiHeadAttention requires rank 2 or rank 3.");
        }

        if (numHeads <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(numHeads),
                numHeads,
                "Head count must be positive.");
        }

        int batch = Rank == 3 ? _shape[0] : 1;
        int sequence = Rank == 3 ? _shape[1] : _shape[0];
        int projectedWidth = _shape[^1];
        if (projectedWidth % 3 != 0)
        {
            throw new ArgumentException(
                "The final QKV dimension must be divisible by three.");
        }

        int modelWidth = projectedWidth / 3;
        if (modelWidth % numHeads != 0)
        {
            throw new ArgumentException(
                "The model width must be divisible by the head count.",
                nameof(numHeads));
        }

        int headWidth = modelWidth / numHeads;
        float scale = 1f / MathF.Sqrt(headWidth);
        int workItemCount = checked(batch * numHeads);
        int probabilityMatrixLength = checked(sequence * sequence);
        float[] probabilities = new float[
            checked(workItemCount * probabilityMatrixLength)];
        float[] output = new float[checked(batch * sequence * modelWidth)];

        void ForwardHead(int workItem)
        {
            int batchIndex = workItem / numHeads;
            int head = workItem % numHeads;
            int headOffset = head * headWidth;
            int projectedBatchOffset =
                batchIndex * sequence * projectedWidth;
            int outputBatchOffset = batchIndex * sequence * modelWidth;
            int probabilityOffset = workItem * probabilityMatrixLength;

            for (int query = 0; query < sequence; query++)
            {
                int queryOffset = projectedBatchOffset
                    + query * projectedWidth
                    + headOffset;
                int probabilityRow =
                    probabilityOffset + query * sequence;
                int lastKey = causal ? query : sequence - 1;
                float maximum = float.NegativeInfinity;

                for (int key = 0; key <= lastKey; key++)
                {
                    int keyOffset = projectedBatchOffset
                        + key * projectedWidth
                        + modelWidth
                        + headOffset;
                    float score = scale * DotProduct(
                        _data,
                        queryOffset,
                        _data,
                        keyOffset,
                        headWidth);
                    probabilities[probabilityRow + key] = score;
                    if (score > maximum)
                        maximum = score;
                }

                int activeKeyCount = lastKey + 1;
                float sum = ExpShiftedValues(
                    probabilities,
                    probabilityRow,
                    maximum,
                    probabilities,
                    probabilityRow,
                    activeKeyCount);
                MultiplyValues(
                    probabilities,
                    probabilityRow,
                    1f / sum,
                    probabilities,
                    probabilityRow,
                    activeKeyCount);
                int outputOffset = outputBatchOffset
                    + query * modelWidth
                    + headOffset;
                for (int key = 0; key <= lastKey; key++)
                {
                    float probability = probabilities[probabilityRow + key];
                    int valueOffset = projectedBatchOffset
                        + key * projectedWidth
                        + 2 * modelWidth
                        + headOffset;
                    AddScaledValues(
                        output,
                        outputOffset,
                        _data,
                        valueOffset,
                        probability,
                        headWidth);
                }
            }
        }

        RunBatches(
            workItemCount,
            (long)sequence * sequence * headWidth,
            ForwardHead);

        int[] outputShape = Rank == 3
            ? [batch, sequence, modelWidth]
            : [sequence, modelWidth];
        var result = new Tensor(output, outputShape, [this]);
        result.Node.BackwardAction = () =>
        {
            void BackwardHead(int workItem)
            {
                int batchIndex = workItem / numHeads;
                int head = workItem % numHeads;
                int headOffset = head * headWidth;
                int projectedBatchOffset =
                    batchIndex * sequence * projectedWidth;
                int outputBatchOffset =
                    batchIndex * sequence * modelWidth;
                int probabilityOffset =
                    workItem * probabilityMatrixLength;
                Span<float> probabilityGradients = sequence <= 256
                    ? stackalloc float[sequence]
                    : new float[sequence];

                for (int query = 0; query < sequence; query++)
                {
                    int queryOffset = projectedBatchOffset
                        + query * projectedWidth
                        + headOffset;
                    int outputOffset = outputBatchOffset
                        + query * modelWidth
                        + headOffset;
                    int probabilityRow =
                        probabilityOffset + query * sequence;
                    int lastKey = causal ? query : sequence - 1;
                    float softmaxDot = 0f;

                    for (int key = 0; key <= lastKey; key++)
                    {
                        int valueOffset = projectedBatchOffset
                            + key * projectedWidth
                            + 2 * modelWidth
                            + headOffset;
                        float probability =
                            probabilities[probabilityRow + key];
                        float probabilityGradient = DotProduct(
                            result._grad,
                            outputOffset,
                            _data,
                            valueOffset,
                            headWidth);
                        probabilityGradients[key] = probabilityGradient;
                        softmaxDot += probabilityGradient * probability;
                        AddScaledValues(
                            _grad,
                            valueOffset,
                            result._grad,
                            outputOffset,
                            probability,
                            headWidth);
                    }

                    for (int key = 0; key <= lastKey; key++)
                    {
                        int keyOffset = projectedBatchOffset
                            + key * projectedWidth
                            + modelWidth
                            + headOffset;
                        float scoreGradient = scale
                            * probabilities[probabilityRow + key]
                            * (probabilityGradients[key] - softmaxDot);
                        AddScaledValues(
                            _grad,
                            queryOffset,
                            _data,
                            keyOffset,
                            scoreGradient,
                            headWidth);
                        AddScaledValues(
                            _grad,
                            keyOffset,
                            _data,
                            queryOffset,
                            scoreGradient,
                            headWidth);
                    }
                }
            }

            RunBatches(
                workItemCount,
                (long)sequence * sequence * headWidth,
                BackwardHead);
        };

        return result;
    }
}
