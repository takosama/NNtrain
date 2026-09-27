using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingLinearTests
{
    [Theory]
    [InlineData("iq2_s", 82, 2)]
    [InlineData("iq2_s", 82, 4)]
    [InlineData("iq3_s", 110, 2)]
    [InlineData("iq3_s", 110, 4)]
    [InlineData("q4_k", 144, 2)]
    [InlineData("q4_k", 144, 4)]
    public void MultirowForwardRetainsSg16BitsAndEncodedStorage(string quant, int blockBytes, int tileRows)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true });
        Assert.SkipWhen(!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16,
            "SG16 compilation requires the Arc SG16 capability.");
        foreach (var shape in new[]
        {
            (Rows: 1, Input: 256, Output: 5, Subnormal: false),
            (Rows: 3, Input: 512, Output: 7, Subnormal: false),
            (Rows: 5, Input: 5120, Output: 17, Subnormal: false),
            (Rows: 8, Input: 17408, Output: 5, Subnormal: false),
            (Rows: 25, Input: 5120, Output: 17, Subnormal: false),
            (Rows: 3, Input: 512, Output: 7, Subnormal: true)
        })
        {
            int blocks = shape.Input * shape.Output / 256, count = shape.Rows * shape.Output;
            var random = new Random(blockBytes * 19 + shape.Input);
            byte[] payload = new byte[blocks * blockBytes]; random.NextBytes(payload);
            for (int block = 0; block < blocks; ++block)
            {
                ushort d = shape.Subnormal ? (ushort)(block % 2 == 0 ? 0x0001 : 0x83ff)
                    : BitConverter.HalfToUInt16Bits((Half)((block % 7 + 1) / 8192f));
                WriteHalf(payload, block * blockBytes, d);
                if (quant == "q4_k") WriteHalf(payload, block * blockBytes + 2, shape.Subnormal ? (ushort)2
                    : BitConverter.HalfToUInt16Bits((Half)((block % 3 + 1) / 16384f)));
            }
            float[] input = Enumerable.Range(0, shape.Rows * shape.Input)
                .Select(_ => ((float)random.NextDouble() * 2 - 1) * (shape.Subnormal ? 10000f : 1f)).ToArray();
            float[] bias = Enumerable.Range(0, shape.Output).Select(i => (i % 9 - 4) * 0.03f).ToArray();
            using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(payload), b = lane.Upload(bias);
            using ArcBuffer expectedBuffer = lane.Allocate(count), actualBuffer = lane.Upload(Enumerable.Repeat(float.NaN, count + 2).ToArray());
            long allocated = lane.AllocatedBytes, uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
            lane.Run($"q35l_{quant}_sg16", ((long)count + 1) / 2 * 32, 32,
                x, w, b, expectedBuffer, shape.Rows, shape.Input, shape.Output);
            long tileOutputs = (long)((shape.Rows + tileRows - 1) / tileRows) * shape.Output;
            lane.Run($"q35t_linear_{quant}_rows{tileRows}", (tileOutputs + 1) / 2 * 32, 32,
                x, w, b, actualBuffer, shape.Rows, shape.Input, shape.Output);
            Assert.Equal(payload.Length, w.ByteLength);
            Assert.Equal(allocated, lane.AllocatedBytes);
            Assert.Equal(uploaded, lane.H2DBytes);
            Assert.Equal(downloaded, lane.D2HBytes);
            var expected = new float[count]; var actual = new float[count + 2];
            lane.Read(expectedBuffer, expected); lane.Read(actualBuffer, actual);
            Assert.True(float.IsNaN(actual[count]) && float.IsNaN(actual[count + 1]));
            for (int i = 0; i < count; ++i)
            {
                Assert.True(float.IsFinite(actual[i]));
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                    $"{quant}, rows{tileRows}, shape={shape}, index={i}: expected {expected[i]:R}, actual {actual[i]:R}");
            }
        }
    }

    private static void WriteHalf(byte[] payload, int offset, ushort bits)
    {
        payload[offset] = (byte)bits; payload[offset + 1] = (byte)(bits >> 8);
    }
}
