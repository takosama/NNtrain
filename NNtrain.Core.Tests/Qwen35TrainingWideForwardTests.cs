using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingWideForwardTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType, "q4_k", 144)]
    [InlineData(Qwen35Gguf.IQ2SType, "iq2_s", 82)]
    [InlineData(Qwen35Gguf.IQ3SType, "iq3_s", 110)]
    public void WideRowReusePreservesEveryOutputBitAndTail(uint type, string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true
        });
        Assert.SkipWhen(!(lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")), "Intel SG16 is required.");
        const int guard = 7;
        foreach (var shape in new[]
        {
            (Rows: 1, Input: 256, Output: 5),
            (Rows: 2, Input: 512, Output: 7),
            (Rows: 7, Input: 512, Output: 5),
            (Rows: 8, Input: 512, Output: 7),
            (Rows: 9, Input: 512, Output: 9),
            (Rows: 15, Input: 512, Output: 7),
            (Rows: 16, Input: 512, Output: 9),
            (Rows: 17, Input: 512, Output: 5),
            (Rows: 25, Input: 5120, Output: 7),
            (Rows: 31, Input: 17408, Output: 5),
            (Rows: 33, Input: 256, Output: 3)
        })
        {
            int count = shape.Rows * shape.Output;
            byte[] encoded = Encoded(type, shape.Input * shape.Output / 256, blockBytes);
            float[] input = Values(shape.Rows * shape.Input, 131 + shape.Rows);
            float[] bias = Values(shape.Output, 139 + shape.Input);
            using ArcBuffer x = lane.Upload(input), weight = lane.UploadRaw(encoded), b = lane.Upload(bias);
            using ArcBuffer reference = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
            long referenceItems = ((long)(shape.Rows + 3) / 4 * shape.Output + 1) / 2 * 32;
            lane.Run($"q35t_linear_{quant}_rows4", referenceItems, 32,
                x, weight, b, reference, shape.Rows, shape.Input, shape.Output);
            var expected = new float[count + guard];
            lane.Read(reference, expected);
            foreach (int rowTile in new[] { 8, 16 })
            {
                using ArcBuffer result = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
                long global = ((long)(shape.Rows + rowTile - 1) / rowTile * shape.Output + 1) / 2 * 32;
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
                lane.Run($"q35t_linear_{quant}_rows{rowTile}", global, 32,
                    x, weight, b, result, shape.Rows, shape.Input, shape.Output);
                Assert.Equal(uploads, lane.H2DBytes);
                Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(allocated, lane.AllocatedBytes);
                Assert.Equal(encoded.Length, weight.ByteLength);
                var actual = new float[count + guard];
                lane.Read(result, actual);
                for (int i = 0; i < count; i++)
                {
                    Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"{quant}, tile {rowTile}, shape {shape}, output {i}: {expected[i]:R} vs {actual[i]:R}");
                }
                for (int i = count; i < count + guard; i++)
                    Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
            }
        }
    }

    private static byte[] Encoded(uint type, int blocks, int blockBytes)
    {
        var result = new byte[blocks * blockBytes];
        if (type == Qwen2Gguf.Q4KType)
        {
            new Random(211).NextBytes(result);
            for (int block = 0; block < blocks; block++)
            {
                ushort scale = block % 9 == 0 ? (ushort)0x8001
                    : BitConverter.HalfToUInt16Bits((Half)((block % 7 + 1) / 8192f));
                WriteHalf(result, block * blockBytes, scale);
                WriteHalf(result, block * blockBytes + 2,
                    BitConverter.HalfToUInt16Bits((Half)((block % 3 + 1) / 16384f)));
            }
            return result;
        }
        using Stream stream = typeof(Qwen35TrainingWideForwardTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        byte[][] examples = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => Convert.FromBase64String(item.GetProperty("encoded").GetString()!)).ToArray();
        Assert.NotEmpty(examples);
        for (int block = 0; block < blocks; block++) examples[block % examples.Length].CopyTo(result, block * blockBytes);
        return result;
    }

    private static void WriteHalf(byte[] data, int offset, ushort bits)
    {
        data[offset] = (byte)bits;
        data[offset + 1] = (byte)(bits >> 8);
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * .4 - .2)).ToArray();
    }
}
