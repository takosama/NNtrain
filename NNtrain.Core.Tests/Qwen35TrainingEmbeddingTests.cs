using System.Text.Json;
using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingEmbeddingTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType, "q4_k", 144)]
    [InlineData(Qwen2Gguf.Q6KType, "q6_k", 210)]
    [InlineData(Qwen35Gguf.Q5KType, "q5_k", 176)]
    [InlineData(Qwen35Gguf.IQ2SType, "iq2_s", 82)]
    [InlineData(Qwen35Gguf.IQ3SType, "iq3_s", 110)]
    public void BatchEmbeddingKeepsTokenOrderRepeatedIdsAndGuards(uint type, string quant, int blockBytes)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true,
            Qwen35NativeHalfScale = true
        });
        const int width = 512, vocab = 7, guard = 9;
        byte[] encoded = new byte[vocab * width / 256 * blockBytes];
        if (type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType)
        {
            new Random(11).NextBytes(encoded);
            for (int offset = 0; offset < encoded.Length; offset += blockBytes)
            {
                int scale = offset + (type == Qwen2Gguf.Q6KType ? 208 : 0);
                encoded[scale] = 0; encoded[scale + 1] = 0x20;
                if (type == Qwen2Gguf.Q4KType) { encoded[offset + 2] = 0; encoded[offset + 3] = 0x10; }
            }
        }
        else
        {
            using Stream stream = typeof(Qwen35TrainingEmbeddingTests).Assembly.GetManifestResourceStream(
                "NNtrain.Core.Tests.Fixtures.IqQuantReference.iq-quant-blocks.json")!;
            using var fixture = JsonDocument.Parse(stream);
            byte[][] blocks = fixture.RootElement.GetProperty("cases").EnumerateArray()
                .Where(x => x.GetProperty("type").GetUInt32() == type)
                .Select(x => Convert.FromBase64String(x.GetProperty("encoded").GetString()!)).ToArray();
            for (int b = 0; b < encoded.Length / blockBytes; b++)
                blocks[b % blocks.Length].CopyTo(encoded, b * blockBytes);
        }
        using var weight = lane.UploadRaw(encoded);
        foreach (int[] ids in new[] { new[] { 6 }, new[] { 6, 0, 3, 3, 1, 6, 2 } })
        {
            var expected = new List<float>();
            foreach (int id in ids)
            {
                using var token = lane.UploadRaw(new[] { id });
                using var one = lane.Allocate(width);
                string prefix = type is Qwen2Gguf.Q4KType or Qwen2Gguf.Q6KType ? "qwen_embedding_" : "q35l_";
                string name = prefix + quant + (prefix == "q35l_" ? "_embedding" : "");
                lane.Run(name, width, 128, weight, token, one, width);
                float[] values = new float[width]; lane.Read(one, values); expected.AddRange(values);
            }
            using var tokens = lane.UploadRaw(ids);
            using var output = lane.Upload(Enumerable.Repeat(float.NaN, ids.Length * width + guard).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            lane.Run("q35t_embedding_" + quant, ids.Length * width, 128, weight, tokens, output, ids.Length, width);
            Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
            float[] actual = new float[ids.Length * width + guard]; lane.Read(output, actual);
            Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.Take(expected.Count).Select(BitConverter.SingleToInt32Bits));
            Assert.All(actual.Skip(expected.Count), x => Assert.True(float.IsNaN(x)));
            Assert.Equal(encoded.Length, weight.ByteLength);
        }
    }
}
