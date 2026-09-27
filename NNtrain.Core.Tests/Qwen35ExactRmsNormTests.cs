using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35ExactRmsNormTests
{
    [Fact]
    public void SubgroupTreePreservesRmsNormBitsForModelWidthsHeadsAndOddTails()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        Assert.SkipWhen(!lane.Options.XmxMatrices || !lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups"), "Intel SG16 is required.");
        foreach (int width in new[] { 1, 17, 127, 128, 129, 255, 256, 257, 513, 4096, 5120, 5121, 6144, 17408 })
        foreach (int padding in new[] { 0, 17 })
        foreach (int rows in new[] { 1, 5 })
            Compare(lane, width, width + padding, rows, width % 2 == 0 ? 1e-6f : 1e-5f);
    }

    private static void Compare(ArcExecutionLane lane, int width, int inputStride, int rows, float epsilon)
    {
        const int guard = 13;
        float[] input = Enumerable.Repeat(float.NaN, checked(rows * inputStride + guard)).ToArray();
        var random = new Random(width * 31 + rows * 17 + inputStride);
        for (int row = 0; row < rows; row++)
        for (int column = 0; column < width; column++)
        {
            // Zero rows cover both zero signs. Other rows combine mixed signs,
            // magnitudes and values near underflow while keeping squared sums finite.
            float value = row == 1 ? ((column & 1) == 0 ? 0f : -0f)
                : MathF.ScaleB((float)(random.NextDouble() * 2 - 1), (column % 9 - 4) * 12);
            if (row != 1 && column % 31 == 0) value = (column & 1) == 0 ? float.Epsilon : -float.Epsilon;
            input[row * inputStride + column] = value;
        }
        float[] weights = Enumerable.Range(0, width).Select(i => i % 17 == 0 ? -0f
            : (float)(random.NextDouble() * 4 - 2)).ToArray();
        int count = checked(rows * width);
        using ArcBuffer x = lane.Upload(input), weight = lane.Upload(weights);
        using ArcBuffer reference = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
        using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        lane.Run("q35a_rms_norm", (long)rows * 128, 128, x, weight, reference, width, inputStride, epsilon);
        lane.Run("q35a_rms_norm_sg16_exact", (long)rows * 128, 128, x, weight, candidate, width, inputStride, epsilon);
        Assert.Equal(uploads, lane.H2DBytes);
        Assert.Equal(downloads, lane.D2HBytes);
        float[] expected = Read(lane, reference, count + guard), actual = Read(lane, candidate, count + guard);
        for (int i = 0; i < count; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"RMS mismatch width={width}, stride={inputStride}, rows={rows}, index={i}: reference={expected[i]:R}, subgroup={actual[i]:R}.");
        }
        for (int i = count; i < actual.Length; i++)
            Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
        Assert.Equal(input.Select(BitConverter.SingleToInt32Bits), Read(lane, x, input.Length).Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(weights.Select(BitConverter.SingleToInt32Bits), Read(lane, weight, weights.Length).Select(BitConverter.SingleToInt32Bits));
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var values = new float[count];
        lane.Read(buffer, values);
        return values;
    }
}
