using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferencePairProjectionTests
{
    [Theory]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, false)]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, true)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, false)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, true)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, false)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, true)]
    public void PairedOutputsRetainBaseAndFusedLoraBitsWithoutChangingStorage(
        string quant, uint type, int blockBytes, bool nativeHalf)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            Qwen35ProjectionWorkgroupSize = 32,
            Qwen35NativeHalfScale = nativeHalf
        });
        Assert.SkipWhen(!(lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups")), "Intel SG16 is required.");
        var random = new Random(382 + blockBytes);
        float[] Values(int length) => Enumerable.Range(0, length)
            .Select(_ => (float)(random.NextDouble() * .8 - .4)).ToArray();
        foreach (var shape in new[]
        {
            (Rows: 0, Input: 256, Output: 5, Rank: 3),
            (Rows: 1, Input: 256, Output: 0, Rank: 3),
            (Rows: 1, Input: 256, Output: 1, Rank: 3),
            (Rows: 2, Input: 512, Output: 6, Rank: 8),
            (Rows: 3, Input: 512, Output: 7, Rank: 3),
            (Rows: 1, Input: 5120, Output: 33, Rank: 8),
            (Rows: 2, Input: 17408, Output: 9, Rank: 3)
        })
        {
            int count = shape.Rows * shape.Output;
            byte[] encoded = EncodedWeights(type, blockBytes, shape.Input / 256 * shape.Output);
            float[] sentinel = Enumerable.Repeat(float.NaN, count + 3).ToArray();
            using ArcBuffer x = lane.Upload(Values(shape.Rows * shape.Input));
            using ArcBuffer w = lane.UploadRaw(encoded), bias = lane.Upload(Values(shape.Output));
            using ArcBuffer z = lane.Upload(Values(shape.Rows * shape.Rank));
            using ArcBuffer b = lane.Upload(Values(shape.Output * shape.Rank));
            using ArcBuffer reference = lane.Upload(sentinel), candidate = lane.Upload(sentinel);
            long scalarGlobal = Math.Max(32L, ((long)count + 1) / 2 * 32);
            long pairs = (long)shape.Rows * ((shape.Output + 1) / 2);
            long pairGlobal = Math.Max(32L, (pairs + 1) / 2 * 32);
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
            lane.Run($"q35l_{quant}_sg16", scalarGlobal, 32,
                x, w, bias, reference, shape.Rows, shape.Input, shape.Output);
            lane.Run($"q35l_{quant}_sg16_pair", pairGlobal, 32,
                x, w, bias, candidate, shape.Rows, shape.Input, shape.Output);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            Assert.Equal(allocated, lane.AllocatedBytes);
            Assert.Equal(Math.Max(4, encoded.Length), w.ByteLength);
            EqualBits(lane, reference, candidate, count, quant, "base");
            foreach (float scale in new[] { 2f, -.75f, 0f })
            {
                lane.Run($"q35l_{quant}_sg16_lora", scalarGlobal, 32,
                    x, w, bias, reference, shape.Rows, shape.Input, shape.Output,
                    z, b, shape.Rank, scale);
                lane.Run($"q35l_{quant}_sg16_pair_lora", pairGlobal, 32,
                    x, w, bias, candidate, shape.Rows, shape.Input, shape.Output,
                    z, b, shape.Rank, scale);
                EqualBits(lane, reference, candidate, count, quant, $"LoRA scale {scale}");
            }
        }
    }

    private static byte[] EncodedWeights(uint type, int blockBytes, int blocks)
    {
        byte[] encoded = new byte[blocks * blockBytes];
        if (type == Qwen2Gguf.Q4KType)
        {
            new Random(601).NextBytes(encoded);
            ushort[] scales = [0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff,
                0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
            for (int block = 0; block < blocks; block++)
            {
                WriteHalf(encoded, block * blockBytes, scales[block % scales.Length]);
                WriteHalf(encoded, block * blockBytes + 2, scales[(block * 7 + 5) % scales.Length]);
            }
            return encoded;
        }
        using Stream stream = typeof(Qwen35InferencePairProjectionTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        byte[][] cases = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => Convert.FromBase64String(item.GetProperty("encoded").GetString()!)).ToArray();
        Assert.NotEmpty(cases);
        for (int block = 0; block < blocks; block++)
        {
            Assert.Equal(blockBytes, cases[block % cases.Length].Length);
            cases[block % cases.Length].CopyTo(encoded, block * blockBytes);
        }
        return encoded;
    }

    private static void WriteHalf(byte[] data, int offset, ushort bits)
    {
        data[offset] = (byte)bits;
        data[offset + 1] = (byte)(bits >> 8);
    }

    private static void EqualBits(ArcExecutionLane lane, ArcBuffer reference, ArcBuffer candidate,
        int count, string quant, string mode)
    {
        var expected = new float[count + 3];
        var actual = new float[count + 3];
        lane.Read(reference, expected);
        lane.Read(candidate, actual);
        for (int i = 0; i < count; i++)
        {
            Assert.True(float.IsFinite(actual[i]));
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{quant}, {mode}, index {i}: reference={expected[i]:R}, paired={actual[i]:R}");
        }
        for (int i = count; i < count + 3; i++)
            Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
    }
}
