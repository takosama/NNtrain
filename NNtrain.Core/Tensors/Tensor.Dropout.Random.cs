namespace NNtrain;

partial class Tensor
{
    private static uint NextDropoutSeed(Random random)
    {
        long value;
        lock (random)
            value = random.NextInt64();
        return unchecked((uint)value ^ (uint)((ulong)value >> 32));
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
