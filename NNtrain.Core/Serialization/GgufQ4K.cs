namespace NNtrain;

/// <summary>llama.cpp Q4_K block decoder (256 weights per block, 144 bytes).</summary>
public static class GgufQ4K
{
    public const int BlockElements = 256;
    public const int BlockBytes = 144;

    public static float[] Dequantize(ReadOnlySpan<byte> source, int elementCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(elementCount);
        int blocks = checked((elementCount + BlockElements - 1) / BlockElements);
        if (source.Length < checked(blocks * BlockBytes))
            throw new ArgumentException("Q4_K payload is shorter than required.", nameof(source));

        var output = new float[elementCount];
        int dst = 0;
        for (int block = 0; block < blocks; block++)
        {
            ReadOnlySpan<byte> b = source.Slice(block * BlockBytes, BlockBytes);
            float d = (float)BitConverter.UInt16BitsToHalf((ushort)(b[0] | b[1] << 8));
            float dmin = (float)BitConverter.UInt16BitsToHalf((ushort)(b[2] | b[3] << 8));
            ReadOnlySpan<byte> scales = b.Slice(4, 12);
            ReadOnlySpan<byte> qs = b.Slice(16, 128);

            Span<byte> sc = stackalloc byte[8];
            Span<byte> m = stackalloc byte[8];
            DecodeScales(scales, sc, m);

            for (int group = 0; group < 8 && dst < elementCount; group++)
            {
                float scale = d * sc[group];
                float min = dmin * m[group];
                int qBase = (group / 2) * 32;
                bool high = (group & 1) != 0;
                for (int i = 0; i < 32 && dst < elementCount; i++)
                {
                    int q = high ? qs[qBase + i] >> 4 : qs[qBase + i] & 0x0f;
                    output[dst++] = scale * q - min;
                }
            }
        }
        return output;
    }

    // ggml get_scale_min_k4 packing: first four 6-bit values live directly
    // in bytes 0..7; values 4..7 take their low nibble from bytes 8..11 and
    // high two bits from bytes 0..7.
    private static void DecodeScales(ReadOnlySpan<byte> packed, Span<byte> scales, Span<byte> mins)
    {
        for (int j = 0; j < 4; j++)
        {
            scales[j] = (byte)(packed[j] & 0x3f);
            mins[j] = (byte)(packed[j + 4] & 0x3f);
            scales[j + 4] = (byte)((packed[j + 8] & 0x0f) | ((packed[j] >> 6) << 4));
            mins[j + 4] = (byte)((packed[j + 8] >> 4) | ((packed[j + 4] >> 6) << 4));
        }
    }
}
