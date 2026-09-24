namespace NNtrain;

/// <summary>llama.cpp Q6_K block decoder (256 weights per block, 210 bytes).</summary>
public static class GgufQ6K
{
    public const int BlockElements = 256;
    public const int BlockBytes = 210;

    public static float[] Dequantize(ReadOnlySpan<byte> source, int elementCount)
    {
        if (elementCount < 0 || elementCount % BlockElements != 0)
            throw new ArgumentOutOfRangeException(nameof(elementCount),
                "Q6_K element count must be a non-negative multiple of 256.");
        int blocks = elementCount / BlockElements;
        if (source.Length < checked(blocks * BlockBytes))
            throw new ArgumentException("Q6_K payload is shorter than required.", nameof(source));

        var output = new float[elementCount];
        int dst = 0;
        for (int block = 0; block < blocks; ++block)
        {
            ReadOnlySpan<byte> b = source.Slice(block * BlockBytes, BlockBytes);
            ReadOnlySpan<byte> ql = b[..128];
            ReadOnlySpan<byte> qh = b.Slice(128, 64);
            ReadOnlySpan<byte> scales = b.Slice(192, 16);
            float d = (float)BitConverter.UInt16BitsToHalf(
                (ushort)(b[208] | (b[209] << 8)));

            for (int n = 0; n < 256; n += 128)
            {
                int qlBase = n / 2;
                int qhBase = n / 4;
                int scBase = n / 16;
                for (int l = 0; l < 32; ++l)
                {
                    int iscale = l / 16;
                    int q1 = ((ql[qlBase + l] & 0x0f)
                        | (((qh[qhBase + l] >> 0) & 3) << 4)) - 32;
                    int q2 = ((ql[qlBase + l + 32] & 0x0f)
                        | (((qh[qhBase + l] >> 2) & 3) << 4)) - 32;
                    int q3 = ((ql[qlBase + l] >> 4)
                        | (((qh[qhBase + l] >> 4) & 3) << 4)) - 32;
                    int q4 = ((ql[qlBase + l + 32] >> 4)
                        | (((qh[qhBase + l] >> 6) & 3) << 4)) - 32;
                    output[dst + l] = d * (sbyte)scales[scBase + iscale] * q1;
                    output[dst + l + 32] = d * (sbyte)scales[scBase + iscale + 2] * q2;
                    output[dst + l + 64] = d * (sbyte)scales[scBase + iscale + 4] * q3;
                    output[dst + l + 96] = d * (sbyte)scales[scBase + iscale + 6] * q4;
                }
                dst += 128;
            }
        }
        return output;
    }
}
