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

        var output = new float[Numel];
        random ??= Random.Shared;
        uint seed = NextDropoutSeed(random);
        float scale = 1f / (1f - probability);
        uint dropThreshold = (uint)(probability * (uint.MaxValue + 1d));
        int columns = _shape[^1];
        int rows = Numel / columns;

        void ForwardRow(int row)
        {
            int offset = row * columns;
            ApplyDropoutValues(
                _data,
                output,
                offset,
                columns,
                seed,
                dropThreshold,
                scale);
        }

        RunBatches(rows, columns * 8L, ForwardRow);

        var result = new Tensor(output, _shape, new[] { this });
        result.Node.BackwardAction = () =>
        {
            void BackwardRow(int row)
            {
                int offset = row * columns;
                AccumulateDropoutGradient(
                    _grad,
                    result._grad,
                    offset,
                    columns,
                    seed,
                    dropThreshold,
                    scale);
            }

            RunBatches(rows, columns * 3L, BackwardRow);
        };

        return result;
    }

    /// <summary>
    /// Computes <c>residual + dropout(branch)</c> as one allocation and one
    /// autograd node. A counter-based mask is generated per row so large
    /// activations use all configured workers instead of serial random-number
    /// generation. Backward regenerates the same mask from its seed rather
    /// than retaining another activation-sized array.
    /// </summary>
    public Tensor AddDropout(
        Tensor branch,
        float probability,
        Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(branch);
        if (!float.IsFinite(probability)
            || probability < 0f
            || probability >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(probability),
                probability,
                "Dropout probability must be finite and in [0, 1).");
        }
        if (!_shape.AsSpan().SequenceEqual(branch._shape))
        {
            throw new ArgumentException(
                "Residual and dropout branch must have identical shapes.",
                nameof(branch));
        }
        if (probability == 0f)
            return this + branch;

        var output = new float[Numel];
        random ??= Random.Shared;
        uint seed = NextDropoutSeed(random);
        float scale = 1f / (1f - probability);
        uint dropThreshold = (uint)(probability * (uint.MaxValue + 1d));
        int columns = _shape[^1];
        int rows = Numel / columns;

        void ForwardRow(int row)
        {
            int offset = row * columns;
            AddDropoutValues(
                _data,
                branch._data,
                output,
                offset,
                columns,
                seed,
                dropThreshold,
                scale);
        }

        RunBatches(rows, columns * 10L, ForwardRow);

        var result = new Tensor(output, _shape, new[] { this, branch });
        result.Node.BackwardAction = () =>
        {
            bool sameParent = ReferenceEquals(this, branch);

            void BackwardRow(int row)
            {
                int offset = row * columns;
                AccumulateResidualDropoutGradient(
                    _grad,
                    sameParent ? _grad : branch._grad,
                    result._grad,
                    offset,
                    columns,
                    sameParent,
                    seed,
                    dropThreshold,
                    scale);
            }

            RunBatches(rows, columns * 5L, BackwardRow);
        };

        return result;
    }

    private static uint NextDropoutSeed(Random random)
    {
        long value = random.NextInt64();
        return unchecked((uint)value ^ (uint)((ulong)value >> 32));
    }

    private static void ApplyDropoutValues(
        float[] input,
        float[] output,
        int offset,
        int length,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    LoadVector256(input, offset + index)
                        * CreateDropoutMask256(
                            seed,
                            offset + index,
                            dropThreshold,
                            scale),
                    output,
                    offset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                LoadVector128(input, offset + index)
                    * CreateDropoutMask128(
                        seed,
                        offset + index,
                        dropThreshold,
                        scale),
                output,
                offset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            output[valueIndex] = input[valueIndex]
                * DropoutMultiplier(
                    seed,
                    valueIndex,
                    dropThreshold,
                    scale);
        }
    }

    private static void AddDropoutValues(
        float[] residual,
        float[] branch,
        float[] output,
        int offset,
        int length,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    Vector256.FusedMultiplyAdd(
                        LoadVector256(branch, offset + index),
                        CreateDropoutMask256(
                            seed,
                            offset + index,
                            dropThreshold,
                            scale),
                        LoadVector256(residual, offset + index)),
                    output,
                    offset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                Vector128.FusedMultiplyAdd(
                    LoadVector128(branch, offset + index),
                    CreateDropoutMask128(
                        seed,
                        offset + index,
                        dropThreshold,
                        scale),
                    LoadVector128(residual, offset + index)),
                output,
                offset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            output[valueIndex] = residual[valueIndex]
                + branch[valueIndex]
                    * DropoutMultiplier(
                        seed,
                        valueIndex,
                        dropThreshold,
                        scale);
        }
    }

    private static void AccumulateDropoutGradient(
        float[] destination,
        float[] gradient,
        int offset,
        int length,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            for (; index < vectorizedLength; index += width)
            {
                StoreVector256(
                    Vector256.FusedMultiplyAdd(
                        LoadVector256(gradient, offset + index),
                        CreateDropoutMask256(
                            seed,
                            offset + index,
                            dropThreshold,
                            scale),
                        LoadVector256(destination, offset + index)),
                    destination,
                    offset + index);
            }
        }
        if (CanUseVector128(length - index))
        {
            StoreVector128(
                Vector128.FusedMultiplyAdd(
                    LoadVector128(gradient, offset + index),
                    CreateDropoutMask128(
                        seed,
                        offset + index,
                        dropThreshold,
                        scale),
                    LoadVector128(destination, offset + index)),
                destination,
                offset + index);
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            destination[valueIndex] +=
                gradient[valueIndex]
                    * DropoutMultiplier(
                        seed,
                        valueIndex,
                        dropThreshold,
                        scale);
        }
    }

    private static void AccumulateResidualDropoutGradient(
        float[] residualDestination,
        float[] branchDestination,
        float[] gradient,
        int offset,
        int length,
        bool sameParent,
        uint seed,
        uint dropThreshold,
        float scale)
    {
        int index = 0;
        if (CanUseSimd(length))
        {
            int width = Vector256<float>.Count;
            int vectorizedLength = length - length % width;
            Vector256<float> one = Vector256.Create(1f);
            for (; index < vectorizedLength; index += width)
            {
                Vector256<float> gradientVector =
                    LoadVector256(gradient, offset + index);
                Vector256<float> maskVector = CreateDropoutMask256(
                    seed,
                    offset + index,
                    dropThreshold,
                    scale);
                StoreVector256(
                    LoadVector256(residualDestination, offset + index)
                        + gradientVector
                            * (sameParent ? one + maskVector : one),
                    residualDestination,
                    offset + index);
                if (!sameParent)
                {
                    StoreVector256(
                        Vector256.FusedMultiplyAdd(
                            gradientVector,
                            maskVector,
                            LoadVector256(
                                branchDestination,
                                offset + index)),
                        branchDestination,
                        offset + index);
                }
            }
        }
        if (CanUseVector128(length - index))
        {
            Vector128<float> gradientVector =
                LoadVector128(gradient, offset + index);
            Vector128<float> maskVector = CreateDropoutMask128(
                seed,
                offset + index,
                dropThreshold,
                scale);
            StoreVector128(
                LoadVector128(residualDestination, offset + index)
                    + gradientVector
                        * (sameParent
                            ? Vector128.Create(1f) + maskVector
                            : Vector128.Create(1f)),
                residualDestination,
                offset + index);
            if (!sameParent)
            {
                StoreVector128(
                    Vector128.FusedMultiplyAdd(
                        gradientVector,
                        maskVector,
                        LoadVector128(branchDestination, offset + index)),
                    branchDestination,
                    offset + index);
            }
            index += Vector128<float>.Count;
        }
        for (; index < length; index++)
        {
            int valueIndex = offset + index;
            float currentGradient = gradient[valueIndex];
            float multiplier = DropoutMultiplier(
                seed,
                valueIndex,
                dropThreshold,
                scale);
            residualDestination[valueIndex] += currentGradient
                * (sameParent ? 1f + multiplier : 1f);
            if (!sameParent)
            {
                branchDestination[valueIndex] +=
                    currentGradient * multiplier;
            }
        }
    }

    private static Vector256<float> CreateDropoutMask256(
        uint seed,
        int offset,
        uint dropThreshold,
        float scale)
    {
        uint first = unchecked((uint)(offset + 1));
        Vector256<uint> counters = Vector256.Create(
            first,
            first + 1u,
            first + 2u,
            first + 3u,
            first + 4u,
            first + 5u,
            first + 6u,
            first + 7u);
        Vector256<uint> bits = Vector256.Create(seed)
            + counters * Vector256.Create(0x9E3779B9u);
        bits ^= Vector256.ShiftRightLogical(bits, 16);
        bits *= Vector256.Create(0x7FEB352Du);
        bits ^= Vector256.ShiftRightLogical(bits, 15);
        bits *= Vector256.Create(0x846CA68Bu);
        bits ^= Vector256.ShiftRightLogical(bits, 16);

        Vector256<int> dropped = Vector256.LessThan(
            (bits ^ Vector256.Create(0x80000000u)).AsInt32(),
            Vector256.Create(
                unchecked((int)(dropThreshold ^ 0x80000000u))));
        return Vector256.ConditionalSelect(
            dropped.AsSingle(),
            Vector256<float>.Zero,
            Vector256.Create(scale));
    }

    private static Vector128<float> CreateDropoutMask128(
        uint seed,
        int offset,
        uint dropThreshold,
        float scale)
    {
        uint first = unchecked((uint)(offset + 1));
        Vector128<uint> counters = Vector128.Create(
            first,
            first + 1u,
            first + 2u,
            first + 3u);
        Vector128<uint> bits = Vector128.Create(seed)
            + counters * Vector128.Create(0x9E3779B9u);
        bits ^= Vector128.ShiftRightLogical(bits, 16);
        bits *= Vector128.Create(0x7FEB352Du);
        bits ^= Vector128.ShiftRightLogical(bits, 15);
        bits *= Vector128.Create(0x846CA68Bu);
        bits ^= Vector128.ShiftRightLogical(bits, 16);

        Vector128<int> dropped = Vector128.LessThan(
            (bits ^ Vector128.Create(0x80000000u)).AsInt32(),
            Vector128.Create(
                unchecked((int)(dropThreshold ^ 0x80000000u))));
        return Vector128.ConditionalSelect(
            dropped.AsSingle(),
            Vector128<float>.Zero,
            Vector128.Create(scale));
    }

    private static float DropoutMultiplier(
        uint seed,
        int index,
        uint dropThreshold,
        float scale)
    {
        uint counter = unchecked((uint)(index + 1));
        uint bits = unchecked(seed + 0x9E3779B9u * counter);
        bits ^= bits >> 16;
        bits *= 0x7FEB352Du;
        bits ^= bits >> 15;
        bits *= 0x846CA68Bu;
        bits ^= bits >> 16;
        return bits < dropThreshold ? 0f : scale;
    }
}
