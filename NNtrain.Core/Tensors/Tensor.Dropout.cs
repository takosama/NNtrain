namespace NNtrain;

partial class Tensor
{
    public Tensor Dropout(float probability, Random? random = null)
    {
        if (!float.IsFinite(probability)
            || probability < 0f
            || probability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(probability),
                probability,
                "Dropout probability must be finite and in [0, 1).");
        }

        if (probability == 0f)
            return this;

        random ??= Random.Shared;
        float scale = 1f / (1f - probability);
        var mask = new float[Numel];
        var output = new float[Numel];
        for (int index = 0; index < Numel; index++)
        {
            float multiplier = random.NextSingle() < probability
                ? 0f
                : scale;
            mask[index] = multiplier;
            output[index] = _data[index] * multiplier;
        }

        var result = new Tensor(output, _shape, new[] { this });
        result.Node.BackwardAction = () =>
        {
            int index = 0;
            if (CanUseSimd(Numel))
            {
                int width = Vector256<float>.Count;
                int vectorizedLength = Numel - Numel % width;
                for (; index < vectorizedLength; index += width)
                {
                    StoreVector256(
                        LoadVector256(_grad, index)
                            + LoadVector256(result._grad, index)
                            * LoadVector256(mask, index),
                        _grad,
                        index);
                }
            }

            for (; index < Numel; index++)
                _grad[index] += result._grad[index] * mask[index];
        };

        return result;
    }
}
