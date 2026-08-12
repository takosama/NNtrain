namespace NNtrain;

public partial class AdamW
{
    private static Vector256<float> LoadBFloat16(
        short[] source,
        int offset)
    {
        Vector128<short> packed = Vector128.LoadUnsafe(ref source[offset]);
        Vector256<int> widened = System.Runtime.Intrinsics.X86.Avx2
            .ConvertToVector256Int32(packed);
        return System.Runtime.Intrinsics.X86.Avx2
            .ShiftLeftLogical(widened.AsUInt32(), 16)
            .AsSingle();
    }

    private static void StoreBFloat16(
        Vector256<float> values,
        short[] destination,
        int offset)
    {
        Vector256<int> bits = values.AsInt32();
        Vector256<int> leastSignificantBit = System.Runtime.Intrinsics.X86.Avx2
            .ShiftRightLogical(bits.AsUInt32(), 16)
            .AsInt32()
            & Vector256.Create(1);
        Vector256<int> rounded = bits
            + Vector256.Create(0x7FFF)
            + leastSignificantBit;
        Vector256<int> upper = System.Runtime.Intrinsics.X86.Avx2
            .ShiftRightArithmetic(rounded, 16);
        Vector256<short> duplicated = System.Runtime.Intrinsics.X86.Avx2
            .PackSignedSaturate(upper, upper);
        Vector256<short> ordered = System.Runtime.Intrinsics.X86.Avx2
            .Permute4x64(duplicated.AsInt64(), 0xD8)
            .AsInt16();
        ordered.GetLower().StoreUnsafe(ref destination[offset]);
    }

    private static short SingleToBFloat16(float value)
    {
        uint bits = unchecked((uint)BitConverter.SingleToInt32Bits(value));
        bits += 0x7FFFu + ((bits >> 16) & 1u);
        return unchecked((short)(bits >> 16));
    }

    private static float BFloat16ToSingle(short value)
        => BitConverter.Int32BitsToSingle(value << 16);

    private static short[] EncodeBFloat16(float[] source)
    {
        var result = new short[source.Length];
        for (int index = 0; index < source.Length; index++)
            result[index] = SingleToBFloat16(source[index]);
        return result;
    }

    private static float[] DecodeBFloat16(short[] source)
    {
        var result = new float[source.Length];
        int index = 0;
        if (Vector256.IsHardwareAccelerated
            && System.Runtime.Intrinsics.X86.Avx2.IsSupported)
        {
            int end = source.Length
                - source.Length % Vector256<float>.Count;
            for (; index < end; index += Vector256<float>.Count)
                LoadBFloat16(source, index).StoreUnsafe(ref result[index]);
        }
        for (; index < source.Length; index++)
            result[index] = BFloat16ToSingle(source[index]);
        return result;
    }
}
