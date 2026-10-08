using System.Buffers.Binary;
using System.Text.RegularExpressions;

namespace Iq2sProductionBench;

internal static class ReferenceData
{
    private static readonly ulong[] Grid = ReadGrid();
    private static ulong[] ReadGrid()
    {
        string source = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "qwen35_iq.cl"));
        string initializer = source.Split("q35l_iq2s_grid[1024]", 2)[1].Split("};", 2)[0];
        ulong[] values = Regex.Matches(initializer, @"0x([0-9a-fA-F]{16})[uU][lL]")
            .Select(m => Convert.ToUInt64(m.Groups[1].Value, 16)).ToArray();
        return values.Length == 1024 ? values : throw new InvalidDataException("IQ2_S table has unexpected length.");
    }

    internal static byte[] Synthetic(int input, int output, int seed, bool stress)
    {
        byte[] packed = new byte[checked(input / 256 * output * 82)];
        var random = new Random(seed);
        random.NextBytes(packed);
        for (int b = 0; b < packed.Length / 82; b++)
        {
            float scale = stress ? MathF.ScaleB(1.013f + random.NextSingle(), b % 16 - 14)
                : .0005f + random.NextSingle() * .003f;
            if (stress && b % 17 == 0) scale = 0;
            BinaryPrimitives.WriteUInt16LittleEndian(packed.AsSpan(b * 82, 2), BitConverter.HalfToUInt16Bits((Half)scale));
        }
        return packed;
    }

    private static float[] DecodeRow(byte[] packed, int column, int inputWidth)
    {
        var values = new float[inputWidth];
        for (int b = 0; b < inputWidth / 256; b++)
        {
            int offset = (column * (inputWidth / 256) + b) * 82;
            float d = (float)BitConverter.UInt16BitsToHalf(BinaryPrimitives.ReadUInt16LittleEndian(packed.AsSpan(offset, 2)));
            for (int i = 0; i < 256; i++)
            {
                int group = i / 32, octet = i / 8 % 4, within = i % 8;
                int code = packed[offset + 2 + group * 4 + octet] | (((packed[offset + 66 + group] >> (octet * 2)) & 3) << 8);
                int q = (int)((Grid[code] >> (within * 8)) & 255);
                int scale = (packed[offset + 74 + group] >> (octet / 2 * 4)) & 15;
                bool negative = ((packed[offset + 34 + group * 4 + octet] >> within) & 1) != 0;
                values[b * 256 + i] = ((d * (0.5f + scale)) * 0.25f) * q * (negative ? -1f : 1f);
            }
        }
        return values;
    }

    internal sealed record Point(int Index, double Value, double AbsoluteSum);
    internal static Point[] Reference(float[] x, byte[] weight, float[] bias, int rows, int input, int output, int count)
    {
        int total = checked(rows * output);
        int requested = Math.Min(total, Math.Max(4, count));
        var indices = new HashSet<int> { 0, output - 1, (rows - 1) * output, total - 1 };
        var random = new Random(831_008);
        if (requested == total) for (int i = 0; i < total; i++) indices.Add(i);
        else while (indices.Count < requested) indices.Add(random.Next(total));
        var result = new List<Point>(indices.Count);
        // Decode only the sampled weight columns, and release each before the next.
        foreach (var columnPoints in indices.GroupBy(index => index % output))
        {
            float[] decoded = DecodeRow(weight, columnPoints.Key, input);
            foreach (int index in columnPoints)
            {
                int row = index / output;
                double value = bias[columnPoints.Key], absolute = Math.Abs(value);
                for (int k = 0; k < input; k++)
                {
                    double term = (double)x[row * input + k] * decoded[k];
                    value += term; absolute += Math.Abs(term);
                }
                result.Add(new(index, value, absolute));
            }
        }
        return result.OrderBy(point => point.Index).ToArray();
    }

    internal sealed record Validation(bool Finite, int CpuPoints, double MaxAbsoluteErrorVsDouble,
        double RelativeL2VsDouble, double WorstToleranceRatio, bool FullOutputCompared,
        double? MaxAbsoluteErrorVsOriginal, double? RelativeL2VsOriginal, bool? BitwiseEqualToOriginal, bool GuardsIntact);

    internal static Validation Validate(float[] values, int count, Point[] reference, float[]? original,
        bool requireBitwise = false)
    {
        if (values.Length != count + 2 || !float.IsNaN(values[count]) || !float.IsNaN(values[count + 1]))
            throw new ArithmeticException("Output guard was overwritten.");
        for (int i = 0; i < count; i++)
            if (!float.IsFinite(values[i])) throw new ArithmeticException($"Non-finite/unwritten output at {i}.");
        double max = 0, error2 = 0, reference2 = 0, worst = 0;
        foreach (Point point in reference)
        {
            double error = Math.Abs(values[point.Index] - point.Value);
            double bound = Math.Max(1e-7, 2e-6 * point.AbsoluteSum);
            if (error > bound) throw new ArithmeticException($"CPU mismatch at {point.Index}: {values[point.Index]:G9} vs {point.Value:G17}, bound={bound:G9}.");
            max = Math.Max(max, error); error2 += error * error; reference2 += point.Value * point.Value;
            worst = Math.Max(worst, error / bound);
        }
        double baselineMax = 0, baselineError2 = 0, baselineReference2 = 0;
        bool sameBits = true;
        if (original is not null)
        {
            for (int i = 0; i < count; i++)
            {
                double error = (double)values[i] - original[i];
                baselineMax = Math.Max(baselineMax, Math.Abs(error));
                baselineError2 += error * error; baselineReference2 += (double)original[i] * original[i];
                sameBits &= BitConverter.SingleToInt32Bits(values[i]) == BitConverter.SingleToInt32Bits(original[i]);
            }
            if (Math.Sqrt(baselineError2 / Math.Max(baselineReference2, 1e-100)) > 5e-5)
                throw new ArithmeticException("Full output disagreement exceeds relative L2 5e-5.");
            if (requireBitwise && !sameBits) throw new ArithmeticException("The candidate did not preserve the required original output bits.");
        }
        return new(true, reference.Length, max, Math.Sqrt(error2 / Math.Max(reference2, 1e-100)), worst,
            original is not null, original is null ? null : baselineMax,
            original is null ? null : Math.Sqrt(baselineError2 / Math.Max(baselineReference2, 1e-100)),
            original is null ? null : sameBits, true);
    }
}
