using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35ResidentModelTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void HybridGenerationResetsStateAndKeepsEncodedWeightsResident(bool tiedOutput)
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);
        long resident = Assert.Single(model.ResidentWeightBytes);
        Assert.True(resident > 0);
        // Loading uploads exactly the encoded matrices. Dense auxiliary
        // vectors stay on the host, and no Float32 weight expansion is sent.
        Assert.Equal(resident, Assert.Single(model.UploadedBytes));

        float[] first = model.ForwardToken(1);
        AssertMeaningfulLogits(first);
        long afterFirst = Assert.Single(model.UploadedBytes);
        Assert.InRange(afterFirst - resident, 1L, resident - 1);
        float[] second = model.ForwardToken(2);
        AssertMeaningfulLogits(second);
        long afterSecond = Assert.Single(model.UploadedBytes);
        Assert.InRange(afterSecond - afterFirst, 1L, resident - 1);
        Assert.Equal(resident, Assert.Single(model.ResidentWeightBytes));

        model.Reset();
        Assert.Equal(first, model.ForwardToken(1));
        Assert.Equal(second, model.ForwardToken(2));

        // A fresh token at position zero has different logits from the same
        // token after a prefix, making the reset assertion state-sensitive.
        model.Reset();
        float[] withoutPrefix = model.ForwardToken(2);
        Assert.Contains(second.Zip(withoutPrefix, (left, right) => MathF.Abs(left - right)),
            difference => difference > 1e-7f);

        int[] prompt = [1, 2, 0];
        int[] generated = model.GenerateTokenIds(prompt, 3);
        Assert.Equal(6, generated.Length);
        Assert.Equal(prompt, generated.Take(prompt.Length));
        Assert.All(generated, token => Assert.InRange(token, 0, 3));
        Assert.Equal(generated, model.GenerateTokenIds(prompt, 3));
        Assert.Equal(generated.Take(prompt.Length + 1),
            model.GenerateTokenIds(prompt, 3, eosTokenId: generated[prompt.Length]));
        Assert.Equal(resident, Assert.Single(model.ResidentWeightBytes));
    }

    [Fact]
    public void TwoDevicesMatchSingleDeviceAcrossBothHybridLayers()
    {
        RequireArcDevices(2);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false);
        using Qwen35QuantizedModel single = Qwen35QuantizedModel.Load(file.Path, [0]);
        using Qwen35QuantizedModel split = Qwen35QuantizedModel.Load(file.Path, [0, 1]);

        Assert.Equal(single.ResidentWeightBytes.Sum(), split.ResidentWeightBytes.Sum());
        Assert.Equal(2, split.ResidentWeightBytes.Count);
        Assert.All(split.ResidentWeightBytes, bytes => Assert.True(bytes > 0));
        Assert.Equal(split.ResidentWeightBytes, split.UploadedBytes);
        foreach (int token in new[] { 1, 2, 0, 3 })
        {
            long[] uploadedBefore = split.UploadedBytes.ToArray();
            float[] expected = single.ForwardToken(token);
            float[] actual = split.ForwardToken(token);
            AssertMeaningfulLogits(actual);
            for (int i = 0; i < expected.Length; i++)
                Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, 1e-5f);
            for (int device = 0; device < 2; device++)
                Assert.InRange(split.UploadedBytes[device] - uploadedBefore[device],
                    1L, split.ResidentWeightBytes[device] - 1);
        }
        Assert.Equal(single.GenerateTokenIds([1, 2], 4), split.GenerateTokenIds([1, 2], 4));
    }

    [Fact]
    public void ContextAndDisposalGuardsLeaveResetUsableBeforeDisposal()
    {
        RequireArcDevices(1);
        using TemporaryQwenGguf file = CreateFixture(tiedOutput: false);
        using Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(file.Path, [0]);

        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardToken(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => model.ForwardToken(4));
        float[] first = model.ForwardToken(1);
        for (int i = 1; i < model.Descriptor.ContextLength; i++)
            Assert.Empty(model.ForwardToken(i % 4, returnLogits: false));
        Assert.Contains("context length", Assert.Throws<InvalidOperationException>(
            () => model.ForwardToken(0)).Message);
        model.Reset();
        Assert.Equal(first, model.ForwardToken(1));

        model.Dispose();
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => model.Reset());
        Assert.Throws<ObjectDisposedException>(() => model.ForwardToken(0));
        Assert.Throws<ObjectDisposedException>(() => model.GenerateTokenIds([0], 1));
    }

    private static void RequireArcDevices(int count)
        => Assert.SkipWhen(ArcDevices.Enumerate().Count < count,
            $"{count} Intel Arc GPU(s) required.");

    private static void AssertMeaningfulLogits(float[] logits)
    {
        Assert.Equal(4, logits.Length);
        Assert.All(logits, value => Assert.True(float.IsFinite(value)));
        Assert.True(logits.Max() - logits.Min() > 1e-5f,
            "The fixture must produce nonconstant logits to exercise real projections.");
    }

    private static TemporaryQwenGguf CreateFixture(bool tiedOutput)
    {
        var metadata = new Dictionary<string, object>
        {
            ["general.architecture"] = "qwen35",
            ["tokenizer.ggml.tokens"] = new[] { "a", "b", "c", "d" },
            ["qwen35.block_count"] = 2u,
            ["qwen35.embedding_length"] = 256u,
            ["qwen35.attention.head_count"] = 2u,
            ["qwen35.attention.head_count_kv"] = 1u,
            ["qwen35.attention.key_length"] = 128u,
            ["qwen35.attention.value_length"] = 128u,
            ["qwen35.context_length"] = 8u,
            ["qwen35.feed_forward_length"] = 512u,
            ["qwen35.rope.dimension_count"] = 64u,
            ["qwen35.ssm.group_count"] = 1u,
            ["qwen35.ssm.time_step_rank"] = 2u,
            ["qwen35.ssm.state_size"] = 128u,
            ["qwen35.ssm.inner_size"] = 256u,
            ["qwen35.ssm.conv_kernel"] = 4u,
            ["qwen35.full_attention_interval"] = 2u
        };
        GgufTensorInfo[] directory =
        [
            new("token_embd.weight", [256, 4], Qwen2Gguf.Q4KType, 0),
            new("output_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("output.weight", [256, 4], Qwen2Gguf.Q6KType, 0),
            new("blk.0.attn_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.0.post_attention_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.0.ffn_gate.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ffn_up.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ffn_down.weight", [512, 256], Qwen2Gguf.Q6KType, 0),
            new("blk.0.attn_qkv.weight", [256, 512], Qwen2Gguf.Q6KType, 0),
            new("blk.0.attn_gate.weight", [256, 256], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ssm_alpha.weight", [256, 2], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ssm_beta.weight", [256, 2], Qwen2Gguf.Q4KType, 0),
            new("blk.0.ssm_conv1d.weight", [4, 512], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_a", [2], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_dt.bias", [2], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_norm.weight", [128], Qwen2Gguf.F32Type, 0),
            new("blk.0.ssm_out.weight", [256, 256], Qwen2Gguf.Q4KType, 0),
            new("blk.1.attn_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.1.post_attention_norm.weight", [256], Qwen2Gguf.F32Type, 0),
            new("blk.1.ffn_gate.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.1.ffn_up.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.1.ffn_down.weight", [512, 256], Qwen2Gguf.Q6KType, 0),
            new("blk.1.attn_q.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
            new("blk.1.attn_k.weight", [256, 128], Qwen2Gguf.Q4KType, 0),
            new("blk.1.attn_v.weight", [256, 128], Qwen2Gguf.Q6KType, 0),
            new("blk.1.attn_q_norm.weight", [128], Qwen2Gguf.F32Type, 0),
            new("blk.1.attn_k_norm.weight", [128], Qwen2Gguf.F32Type, 0),
            new("blk.1.attn_output.weight", [256, 256], Qwen2Gguf.Q4KType, 0)
        ];
        using var payload = new MemoryStream();
        var tensors = new List<GgufTensorInfo>();
        foreach (GgufTensorInfo tensor in directory)
        {
            if (tiedOutput && tensor.Name == "output.weight") continue;
            while (payload.Position % 32 != 0) payload.WriteByte(0);
            tensors.Add(tensor with { Offset = (ulong)payload.Position });
            int elements = checked((int)tensor.Shape.Aggregate(1UL, (total, dimension) => total * dimension));
            byte[] bytes = tensor.Type == Qwen2Gguf.F32Type
                ? DensePayload(tensor.Name, elements)
                : QuantizedPayload(tensor.Name, tensor.Type, elements);
            payload.Write(bytes);
        }
        return new TemporaryQwenGguf(metadata, tensors, payload.ToArray());
    }

    private static byte[] DensePayload(string name, int elements)
    {
        var bytes = new byte[elements * sizeof(float)];
        for (int i = 0; i < elements; i++)
        {
            float value = name.EndsWith("ssm_a", StringComparison.Ordinal) ? -0.4f * (i + 1)
                : name.EndsWith("ssm_dt.bias", StringComparison.Ordinal) ? 0f
                : name.EndsWith("ssm_conv1d.weight", StringComparison.Ordinal)
                    ? (i % 4) switch { 0 => 0.03f, 1 => 0.07f, 2 => 0.15f, _ => 0.75f }
                : 1f;
            BitConverter.TryWriteBytes(bytes.AsSpan(i * sizeof(float)), value);
        }
        return bytes;
    }

    private static byte[] QuantizedPayload(string name, uint type, int elements)
    {
        // Fixed seed independent of randomized string hashes. Centered Q4
        // blocks and signed Q6 blocks keep projections small but nonzero.
        var random = new Random(name.Aggregate(17, (seed, ch) => unchecked(seed * 31 + ch)));
        int blockBytes = type == Qwen2Gguf.Q4KType ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        var payload = new byte[elements / 256 * blockBytes];
        for (int offset = 0; offset < payload.Length; offset += blockBytes)
        {
            if (type == Qwen2Gguf.Q4KType)
            {
                WriteHalf(payload, offset, 1f / 1024f);
                WriteHalf(payload, offset + 2, 8f / 1024f);
                payload.AsSpan(offset + 4, 8).Fill(1);
                payload.AsSpan(offset + 12, 4).Fill(0x11);
                random.NextBytes(payload.AsSpan(offset + 16, 128));
            }
            else
            {
                random.NextBytes(payload.AsSpan(offset, 192));
                for (int i = 0; i < 16; i++)
                    payload[offset + 192 + i] = unchecked((byte)(sbyte)(i % 2 == 0 ? 1 : -1));
                WriteHalf(payload, offset + 208, 1f / 4096f);
            }
        }
        return payload;
    }

    private static void WriteHalf(byte[] bytes, int offset, float value)
        => BitConverter.TryWriteBytes(bytes.AsSpan(offset), BitConverter.HalfToUInt16Bits((Half)value));
}
