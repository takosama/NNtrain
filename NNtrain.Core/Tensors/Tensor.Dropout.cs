using System.Runtime.InteropServices;

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
        Span<uint> randomBits = MemoryMarshal.Cast<float, uint>(mask.AsSpan());
        random.NextBytes(MemoryMarshal.AsBytes(randomBits));
        uint dropThreshold = (uint)(probability * (uint.MaxValue + 1d));
        for (int index = 0; index < Numel; index++)
        {
            float multiplier = randomBits[index] < dropThreshold
                ? 0f
                : scale;
            mask[index] = multiplier;
        }
        MultiplyElementwiseValues(
            _data,
            0,
            mask,
            0,
            output,
            0,
            Numel);

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
