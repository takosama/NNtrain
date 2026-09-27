using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingTransposeVectorTests
{
    [Theory]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, 4, false)]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, 8, false)]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, 16, false)]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, 4, true)]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, 8, true)]
    [InlineData("iq2_s", Qwen35Gguf.IQ2SType, 82, 16, true)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, 4, false)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, 8, false)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, 16, false)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, 4, true)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, 8, true)]
    [InlineData("iq3_s", Qwen35Gguf.IQ3SType, 110, 16, true)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, 4, false)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, 8, false)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, 16, false)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, 4, true)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, 8, true)]
    [InlineData("q4_k", Qwen2Gguf.Q4KType, 144, 16, true)]
    public void EightComponentDecodeRetainsScalarTransposeBitsAndRowTails(
        string quant, uint type, int blockBytes, int tileRows, bool nativeHalf)
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
        foreach (int rows in new[] { 1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 25, 33 })
            Compare(lane, quant, type, blockBytes, tileRows, rows, input: 512, output: 37, tile: 16);
        // Exercise real hidden/MLP widths, the production split width, and a
        // non-divisible output tail. Native fixtures cover every IQ grid index.
        Compare(lane, quant, type, blockBytes, tileRows, rows: 17, input: 5120, output: 1031, tile: 1024);
        Compare(lane, quant, type, blockBytes, tileRows, rows: 5, input: 17408, output: 19, tile: 16);
    }

    private static void Compare(ArcExecutionLane lane, string quant, uint type, int blockBytes,
        int tileRows, int rows, int input, int output, int tile)
    {
        const int guard = 11;
        int splits = (output + tile - 1) / tile, count = rows * splits * input;
        byte[] encoded = EncodedWeights(type, blockBytes, input * output / 256);
        float[] upstream = Values(rows * output, rows + output);
        float[] initial = Values(rows * input, rows + input);
        using ArcBuffer weight = lane.UploadRaw(encoded), dy = lane.Upload(upstream);
        using ArcBuffer reference = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
        using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + guard).ToArray());
        using ArcBuffer dxReference = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, guard)).ToArray());
        using ArcBuffer dxCandidate = lane.Upload(initial.Concat(Enumerable.Repeat(float.NaN, guard)).ToArray());
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
        lane.Run($"q35t_xpose_{quant}", count, 128,
            dy, weight, reference, rows, input, output, splits, tile);
        long workItems = (long)((rows + tileRows - 1) / tileRows) * splits * (input / 8);
        lane.Run($"q35t_xpose_{quant}_vec8_rows{tileRows}", workItems, 32,
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

    private static byte[] EncodedWeights(uint type, int blockBytes, int blocks)
    {
        byte[] encoded = new byte[blocks * blockBytes];
        if (type == Qwen2Gguf.Q4KType)
        {
            new Random(431).NextBytes(encoded);
            ushort[] scales = [0x0000, 0x8000, 0x0001, 0x8001, 0x03ff, 0x83ff,
                0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
            for (int b = 0; b < blocks; b++)
            {
                WriteHalf(encoded, b * blockBytes, scales[b % scales.Length]);
                WriteHalf(encoded, b * blockBytes + 2, scales[(b * 7 + 5) % scales.Length]);
            }
            return encoded;
        }
        using Stream stream = typeof(Qwen35TrainingTransposeVectorTests).Assembly.GetManifestResourceStream(
            "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
        using JsonDocument fixture = JsonDocument.Parse(stream);
        byte[][] cases = fixture.RootElement.GetProperty("cases").EnumerateArray()
            .Where(item => item.GetProperty("type").GetUInt32() == type)
            .Select(item => Convert.FromBase64String(item.GetProperty("encoded").GetString()!)).ToArray();
        Assert.NotEmpty(cases);
        for (int b = 0; b < blocks; b++)
        {
            Assert.Equal(blockBytes, cases[b % cases.Length].Length);
            cases[b % cases.Length].CopyTo(encoded, b * blockBytes);
        }
        return encoded;
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
        var values = new float[count];
        lane.Read(buffer, values);
        return values;
    }

    private static void EqualBits(float[] expected, float[] actual, int count, int guard)
    {
        for (int i = 0; i < count; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]));
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"FP32 mismatch at {i}: reference={expected[i]:R}, vector={actual[i]:R}");
        }
        for (int i = count; i < count + guard; i++)
            Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
    }
}
