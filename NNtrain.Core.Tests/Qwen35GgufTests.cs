using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GgufTests
{
    [Fact]
    public void InspectValidatesHybridDirectoryWithoutReadingPayloads()
    {
        using var file = new TemporaryQwenGguf(Metadata(), Directory());
        Qwen35GgufDescriptor descriptor = Qwen35Gguf.Inspect(file.Path);

        Assert.Equal(256, descriptor.EmbeddingLength);
        Assert.Equal(3, descriptor.HeadCount);
        Assert.Equal(256, descriptor.HeadWidth);
        // The query width is independent of the embedding width in Qwen3.5.
        Assert.NotEqual(0, descriptor.EmbeddingLength % descriptor.HeadCount);
        Assert.Equal(1, descriptor.LinearKeyHeads);
        Assert.Equal(2, descriptor.LinearValueHeads);
        Assert.Equal(128, descriptor.LinearHeadWidth);
        Assert.True(descriptor.IsRecurrent(0));
        Assert.False(descriptor.IsRecurrent(1));
        Assert.Equal(Directory().Length, descriptor.Tensors.Count);
    }

    [Theory]
    [InlineData(Qwen2Gguf.Q4KType)]
    [InlineData(Qwen2Gguf.Q6KType)]
    public void InspectAcceptsQuantizedTiedOutput(uint type)
    {
        GgufTensorInfo[] tensors = Directory().Where(t => t.Name != "output.weight")
            .Select(t => t.Name == "token_embd.weight" ? t with { Type = type } : t).ToArray();
        using var file = new TemporaryQwenGguf(Metadata(), tensors);

        Assert.Equal(4, Qwen35Gguf.Inspect(file.Path).VocabularySize);
    }

    public static TheoryData<string, ulong[]> WrongShapes => new()
    {
        { "token_embd.weight", [4, 256] },
        { "blk.0.attn_qkv.weight", [256, 256] },
        { "blk.0.attn_gate.weight", [128, 512] },
        { "blk.0.ssm_alpha.weight", [256, 1] },
        { "blk.0.ssm_conv1d.weight", [512, 4] },
        { "blk.0.ssm_norm.weight", [256] },
        { "blk.0.post_attention_norm.weight", [1, 256] },
        { "blk.1.attn_q.weight", [256, 768] }, // Q projection also contains the output gate.
        { "blk.1.attn_k_norm.weight", [768] },
        { "blk.1.attn_output.weight", [256, 768] },
        { "blk.1.ffn_down.weight", [256, 512] }
    };

    [Theory]
    [MemberData(nameof(WrongShapes))]
    public void InspectRejectsWrongHybridShapesBeforePayloadReads(string name, ulong[] shape)
    {
        using var file = new TemporaryQwenGguf(Metadata(),
            Directory().Select(t => t.Name == name ? t with { Shape = shape } : t).ToArray());

        InvalidDataException error = Assert.Throws<InvalidDataException>(() => Qwen35Gguf.Inspect(file.Path));

        Assert.Contains(name, error.Message);
        Assert.Contains("shape", error.Message);
    }

    [Theory]
    [InlineData("token_embd.weight", Qwen2Gguf.F32Type)]
    [InlineData("blk.0.attn_qkv.weight", Qwen2Gguf.F16Type)]
    [InlineData("blk.0.ssm_conv1d.weight", Qwen2Gguf.Q4KType)]
    [InlineData("blk.1.attn_q_norm.weight", Qwen2Gguf.Q6KType)]
    public void InspectRejectsUnsupportedStorageBeforePayloadReads(string name, uint type)
    {
        using var file = new TemporaryQwenGguf(Metadata(),
            Directory().Select(t => t.Name == name ? t with { Type = type } : t).ToArray());

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Qwen35Gguf.Inspect(file.Path));

        Assert.Contains(name, error.Message);
    }

    [Theory]
    [InlineData(Qwen2Gguf.F16Type)]
    [InlineData(Qwen2Gguf.BF16Type)]
    public void InspectAcceptsSupportedDenseAuxiliaryStorage(uint type)
    {
        using var file = new TemporaryQwenGguf(Metadata(),
            Directory().Select(t => t.Type == Qwen2Gguf.F32Type ? t with { Type = type } : t).ToArray());

        Assert.Equal(2, Qwen35Gguf.Inspect(file.Path).LayerCount);
    }

    [Theory]
    [InlineData("attention.head_count_kv", 2u)]
    [InlineData("attention.value_length", 128u)]
    [InlineData("ssm.inner_size", 512u)]
    [InlineData("ssm.group_count", 3u)]
    [InlineData("rope.dimension_count", 65u)]
    [InlineData("rope.dimension_count", 512u)]
    [InlineData("full_attention_interval", 0u)]
    [InlineData("ssm.conv_kernel", 0u)]
    public void InspectRejectsInconsistentMetadata(string suffix, uint value)
    {
        Dictionary<string, object> metadata = Metadata();
        metadata["qwen35." + suffix] = value;
        using var file = new TemporaryQwenGguf(metadata, Directory());

        Assert.Throws<InvalidDataException>(() => Qwen35Gguf.Inspect(file.Path));
    }

    [Theory]
    [InlineData("qwen35.recurrent_layers")]
    [InlineData("qwen35.attention.recurrent_layers")]
    public void InspectValidatesExplicitRecurrentPattern(string key)
    {
        Dictionary<string, object> metadata = Metadata();
        metadata[key] = new[] { 1, 0 };
        using var valid = new TemporaryQwenGguf(metadata, Directory());
        Assert.True(Qwen35Gguf.Inspect(valid.Path).IsRecurrent(0));

        metadata[key] = new[] { 0, 1 };
        using var invalid = new TemporaryQwenGguf(metadata, Directory());
        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Qwen35Gguf.Inspect(invalid.Path));
        Assert.Contains("full_attention_interval", error.Message);
    }

    [Theory]
    [InlineData(new[] { 1 })]
    [InlineData(new[] { 1, 2 })]
    [InlineData(new[] { -1, 0 })]
    public void InspectRejectsMalformedRecurrentPattern(int[] pattern)
    {
        Dictionary<string, object> metadata = Metadata();
        metadata["qwen35.recurrent_layers"] = pattern;
        using var file = new TemporaryQwenGguf(metadata, Directory());

        Assert.Throws<InvalidDataException>(() => Qwen35Gguf.Inspect(file.Path));
    }

    [Fact]
    public void InspectRejectsMissingOrDuplicateRequiredTensor()
    {
        using var missing = new TemporaryQwenGguf(Metadata(),
            Directory().Where(t => t.Name != "blk.0.ssm_a").ToArray());
        Assert.Contains("blk.0.ssm_a", Assert.Throws<InvalidDataException>(
            () => Qwen35Gguf.Inspect(missing.Path)).Message);
        using var duplicate = new TemporaryQwenGguf(Metadata(), [.. Directory(), Directory()[0]]);
        Assert.Contains("Duplicate", Assert.Throws<InvalidDataException>(
            () => Qwen35Gguf.Inspect(duplicate.Path)).Message);
    }

    [Fact]
    public void InspectRejectsUnknownArchitectureAndUnexpectedTensors()
    {
        Dictionary<string, object> metadata = Metadata();
        metadata["general.architecture"] = "qwen35moe";
        using var moe = new TemporaryQwenGguf(metadata, Directory());
        Assert.Throws<InvalidDataException>(() => Qwen35Gguf.Inspect(moe.Path));
        using var extra = new TemporaryQwenGguf(Metadata(),
            [.. Directory(), new GgufTensorInfo("v.patch_embd.weight", [256], Qwen2Gguf.F32Type, 0)]);
        Assert.Contains("v.patch_embd.weight", Assert.Throws<NotSupportedException>(
            () => Qwen35Gguf.Inspect(extra.Path)).Message);
    }

    private static Dictionary<string, object> Metadata() => new()
    {
        ["general.architecture"] = "qwen35",
        ["tokenizer.ggml.tokens"] = new[] { "a", "b", "c", "d" },
        ["qwen35.block_count"] = 2u,
        ["qwen35.embedding_length"] = 256u,
        ["qwen35.attention.head_count"] = 3u,
        ["qwen35.attention.head_count_kv"] = 1u,
        ["qwen35.attention.key_length"] = 256u,
        ["qwen35.attention.value_length"] = 256u,
        ["qwen35.context_length"] = 16u,
        ["qwen35.feed_forward_length"] = 512u,
        ["qwen35.rope.dimension_count"] = 64u,
        ["qwen35.rope.dimension_sections"] = new[] { 11, 11, 10, 0 },
        ["qwen35.ssm.group_count"] = 1u,
        ["qwen35.ssm.time_step_rank"] = 2u,
        ["qwen35.ssm.state_size"] = 128u,
        ["qwen35.ssm.inner_size"] = 256u,
        ["qwen35.ssm.conv_kernel"] = 4u,
        ["qwen35.full_attention_interval"] = 2u
    };

    private static GgufTensorInfo[] Directory() =>
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
        new("blk.1.attn_q.weight", [256, 1536], Qwen2Gguf.Q4KType, 0),
        new("blk.1.attn_k.weight", [256, 256], Qwen2Gguf.Q4KType, 0),
        new("blk.1.attn_v.weight", [256, 256], Qwen2Gguf.Q6KType, 0),
        new("blk.1.attn_q_norm.weight", [256], Qwen2Gguf.F32Type, 0),
        new("blk.1.attn_k_norm.weight", [256], Qwen2Gguf.F32Type, 0),
        new("blk.1.attn_output.weight", [768, 256], Qwen2Gguf.Q4KType, 0)
    ];
}
