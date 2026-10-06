using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingTransposeRollingTests
{
    [Theory]
    [InlineData(1, 512, 37, 16, 1, false)]
    [InlineData(7, 512, 37, 16, 4, true)]
    [InlineData(8, 512, 37, 16, 8, false)]
    [InlineData(9, 512, 37, 16, 8, true)]
    [InlineData(17, 512, 37, 16, 8, false)]
    [InlineData(17, 5120, 1031, 1024, 8, true)]
    public void RollingIq2TransposeMatchesSplitReductionBits(
        int rows, int input, int output, int tile, int chunkRows, bool nativeHalf)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            Qwen35TrainingKernels = true,
            Qwen35NativeHalfScale = nativeHalf,
            BufferPoolBytes = 4 * 1024 * 1024
        });
        Assert.SkipWhen(!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16
            || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups"),
            "The vector transpose candidate requires Intel SG16 support.");
        const int guard = 11;
        int splits = (output + tile - 1) / tile;
        byte[] encoded = EncodedWeights(checked(input * output / 256));
        float[] upstream = Values(rows * output, rows + output);
        float[] initial = Values(rows * input, rows + input);
        using ArcBuffer weight = lane.UploadRaw(encoded);
        using ArcBuffer dy = lane.Upload(upstream);
        using ArcBuffer reference = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, guard)).ToArray());
        using ArcBuffer candidate = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, guard)).ToArray());
        for (int first = 0; first < rows; first += chunkRows)
        {
            int count = Math.Min(chunkRows, rows - first);
            int elements = checked(count * input);
            int dyElements = checked(count * output);
            using ArcBuffer dyTile = lane.Allocate(dyElements);
            lane.CopyBytes(dy, dyTile, checked(first * output * sizeof(float)), 0,
                checked(dyElements * sizeof(float)));
            using ArcBuffer partial = lane.Allocate(checked(count * splits * input));
            using ArcBuffer dxTile = lane.Allocate(elements);
            lane.Run("q35a_zero", elements, 0, dxTile, elements);
            lane.Run("q35t_xpose_iq2_s_vec8_rows8",
                (long)((count + 7) / 8) * splits * (input / 8), 32,
                dyTile, weight, partial, count, input, output, splits, tile);
            lane.Run("q35t_xpose_reduce", elements, 0,
                partial, dxTile, count, input, splits);
            lane.Run("q35t_add_offset", elements, 0, dxTile, reference,
                elements, checked(first * input));

            using ArcBuffer rolling = lane.Allocate(elements);
            lane.Run("q35a_zero", elements, 0, rolling, elements);
            for (int split = 0; split < splits; split++)
                lane.Run("q35t_xpose_iq2_s_vec8_rows8_rolling",
                    (long)((count + 7) / 8) * (input / 8), 32,
                    dy, weight, rolling, count, first, input, output, split, tile);
            lane.Run("q35t_add_offset", elements, 0, rolling, candidate,
                elements, checked(first * input));
        }
        float[] expected = new float[initial.Length + guard];
        float[] actual = new float[expected.Length];
        lane.Read(reference, expected);
        lane.Read(candidate, actual);
        for (int i = 0; i < initial.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"FP32 mismatch at {i}: reference={expected[i]:R}, rolling={actual[i]:R}");
        }
        for (int i = initial.Length; i < actual.Length; i++)
            Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
    }

    private static byte[] EncodedWeights(int blocks)
    {
        const int blockBytes = 82;
        using Stream stream = typeof(Qwen35TrainingTransposeRollingTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        byte[][] cases = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == Qwen35Gguf.IQ2SType)
            .Select(item => Convert.FromBase64String(item.GetProperty("encoded").GetString()!)).ToArray();
        Assert.NotEmpty(cases);
        byte[] encoded = new byte[checked(blocks * blockBytes)];
        for (int b = 0; b < blocks; b++) cases[b % cases.Length].CopyTo(encoded, b * blockBytes);
        return encoded;
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * .4 - .2)).ToArray();
    }
}
