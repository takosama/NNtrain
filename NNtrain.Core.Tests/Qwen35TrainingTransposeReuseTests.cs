using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingTransposeReuseTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType, "q4_k", 144)]
    [InlineData(Qwen2Gguf.Q6KType, "q6_k", 210)]
    [InlineData(Qwen35Gguf.Q5KType, "q5_k", 176)]
    [InlineData(Qwen35Gguf.IQ2SType, "iq2_s", 82)]
    [InlineData(Qwen35Gguf.IQ3SType, "iq3_s", 110)]
    public void RowDecodeReusePreservesEverySplitFmaAndTail(uint type, string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            Qwen35TrainingKernels = true,
            BufferPoolBytes = 4 * 1024 * 1024
        });
        const int input = 512, tile = 1024, output = tile + 7, splits = 2, guard = 13;
        byte[] encoded = EncodedWeights(type, input * output / 256, blockBytes);
        using ArcBuffer weight = lane.UploadRaw(encoded);
        foreach (int rowTile in new[] { 4, 8, 16, 32 })
        foreach (int rows in new[] { 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 25, 31, 32, 33 })
        {
            int count = rows * splits * input;
            float[] upstream = Values(rows * output, 31 + rows);
            float[] initial = Values(rows * input, 53 + rows);
            using ArcBuffer dy = lane.Upload(upstream);
            using ArcBuffer reference = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
            using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
            using ArcBuffer dxReference = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, guard)).ToArray());
            using ArcBuffer dxCandidate = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, guard)).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
            lane.Run("q35t_xpose_" + quant, count, 128,
                dy, weight, reference, rows, input, output, splits, tile);
            lane.Run("q35t_xpose_" + quant + "_rows" + rowTile, ((rows + rowTile - 1) / rowTile) * splits * input, 128,
                dy, weight, candidate, rows, input, output, splits, tile);
            lane.Run("q35t_xpose_reduce", rows * input, 128,
                reference, dxReference, rows, input, splits);
            lane.Run("q35t_xpose_reduce", rows * input, 128,
                candidate, dxCandidate, rows, input, splits);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            Assert.Equal(allocated, lane.AllocatedBytes);
            Assert.Equal(encoded.Length, weight.ByteLength);
            EqualBits(Read(lane, reference, count + guard), Read(lane, candidate, count + guard), count, guard);
            EqualBits(Read(lane, dxReference, initial.Length + guard), Read(lane, dxCandidate, initial.Length + guard), initial.Length, guard);
            Assert.Equal(upstream, Read(lane, dy, upstream.Length));
        }
    }

    private static byte[] EncodedWeights(uint type, int blocks, int blockBytes)
    {
        var result = new byte[blocks * blockBytes];
        if (type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType)
        {
            new Random(211 + (int)type).NextBytes(result);
            for (int block = 0; block < blocks; block++)
            {
                // Include signed subnormal scales as well as ordinary values.
                ushort scale = block % 9 == 0 ? (ushort)0x8001
                    : BitConverter.HalfToUInt16Bits((Half)((block % 7 + 1) / 8192f));
                int offset = block * blockBytes;
                WriteHalf(result, offset + (type == Qwen2Gguf.Q6KType ? 208 : 0), scale);
                if (type == Qwen2Gguf.Q4KType)
                    WriteHalf(result, offset + 2, BitConverter.HalfToUInt16Bits((Half)((block % 3 + 1) / 16384f)));
            }
            return result;
        }
        using Stream stream = typeof(Qwen35TrainingTransposeReuseTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        byte[][] examples = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => Convert.FromBase64String(item.GetProperty("encoded").GetString()!)).ToArray();
        Assert.NotEmpty(examples);
        for (int block = 0; block < blocks; block++)
        {
            byte[] example = examples[block % examples.Length];
            Assert.Equal(blockBytes, example.Length);
            example.CopyTo(result, block * blockBytes);
        }
        return result;
    }

    private static void WriteHalf(byte[] data, int offset, ushort bits)
    {
        data[offset] = (byte)bits;
        data[offset + 1] = (byte)(bits >> 8);
    }

    private static float[] Values(int count, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() * .4 - .2)).ToArray();
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var result = new float[count];
        lane.Read(buffer, result);
        return result;
    }

    private static void EqualBits(float[] expected, float[] actual, int count, int guard)
    {
        for (int i = 0; i < count; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            Assert.Equal(BitConverter.SingleToInt32Bits(expected[i]), BitConverter.SingleToInt32Bits(actual[i]));
        }
        for (int i = count; i < count + guard; i++)
            Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
    }
}
