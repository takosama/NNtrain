using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenGgufValidationTests
{
    public static TheoryData<string, ulong[]> InvalidShapes => new()
    {
        { "token_embd.weight", [4, 256] },
        { "blk.0.attn_q.weight", [128, 512] }, // Same element count, wrong orientation/width.
        { "blk.0.attn_q.weight", [65536] },
        { "blk.0.attn_q.weight", [256, 256, 1] },
        { "blk.0.attn_q.weight", [256, 0] },
        { "blk.0.attn_k.weight", [128, 256] },
        { "blk.0.attn_v.weight", [256, 256] },
        { "blk.0.attn_q.bias", [1, 256] },
        { "blk.0.ffn_down.weight", [256, 512] },
        { "output_norm.weight", [1, 256] },
        { "output.weight", [4, 256] }
    };

    [Theory]
    [MemberData(nameof(InvalidShapes))]
    public void QuantizedModelRejectsWrongShapesBeforeReadingPayload(string name, ulong[] shape)
    {
        GgufTensorInfo[] tensors = ValidTensorDirectory()
            .Select(tensor => tensor.Name == name ? tensor with { Shape = shape } : tensor).ToArray();
        // Deliberately omit all payloads. If validation reads even the first
        // embedding before checking the directory, it throws EndOfStream.
        using var file = new TemporaryQwenGguf(ValidMetadata(), tensors);

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => Qwen2Gguf.LoadQuantizedModel(file.Path));

        Assert.Contains(name, error.Message);
        Assert.Contains("shape", error.Message);
    }

    [Fact]
    public void QuantizedModelAcceptsNonBlockAlignedOutputWidthsBeforeReadingPayload()
    {
        // K/V output width 128 and vocabulary width 4 need not be divisible
        // by 256. The contiguous input dimension is the quantization row.
        using var file = new TemporaryQwenGguf(ValidMetadata(), ValidTensorDirectory());

        Assert.Throws<EndOfStreamException>(() => Qwen2Gguf.LoadQuantizedModel(file.Path));
    }

    [Theory]
    [InlineData(Qwen2Gguf.Q4KType)]
    [InlineData(Qwen2Gguf.Q6KType)]
    public void QuantizedTensorRejectsBlocksCrossingRowsBeforeReadingPayload(uint type)
    {
        using var file = new TemporaryQwenGguf(
            new Dictionary<string, object>(), [new GgufTensorInfo("weight", [128, 2], type, 0)]);

        InvalidDataException error = Assert.Throws<InvalidDataException>(
            () => Qwen2Gguf.ReadTensor(file.Path, "weight"));

        Assert.Contains("row width divisible by 256", error.Message);
    }

    [Fact]
    public void TensorRejectsEmptyDimensionBeforeReadingPayload()
    {
        using var file = new TemporaryQwenGguf(
            new Dictionary<string, object>(), [new GgufTensorInfo("weight", [256, 0], Qwen2Gguf.F32Type, 0)]);

        Assert.Throws<InvalidDataException>(() => Qwen2Gguf.ReadTensor(file.Path, "weight"));
    }

    [Theory]
    [InlineData("qwen2.attention.head_count", 0u)]
    [InlineData("qwen2.attention.head_count", 3u)]
    [InlineData("qwen2.attention.head_count_kv", 0u)]
    [InlineData("qwen2.attention.head_count_kv", 3u)]
    [InlineData("qwen2.embedding_length", 0u)]
    public void InspectRejectsInvalidDimensions(string key, uint value)
    {
        Dictionary<string, object> metadata = ValidMetadata();
        metadata[key] = value;
        using var file = new TemporaryQwenGguf(metadata);

        Assert.Throws<InvalidDataException>(() => Qwen2Gguf.Inspect(file.Path));
    }

    [Theory]
    [InlineData(Qwen2Gguf.Q4KType)]
    [InlineData(Qwen2Gguf.Q6KType)]
    public void QuantizedEmbeddingAcceptsTiedHeadBeforeReadingPayload(uint type)
    {
        GgufTensorInfo[] tensors = ValidTensorDirectory()
            .Where(tensor => tensor.Name != "output.weight")
            .Select(tensor => tensor.Name == "token_embd.weight"
                ? tensor with { Type = type } : tensor).ToArray();
        using var file = new TemporaryQwenGguf(ValidMetadata(), tensors);
        Assert.Throws<EndOfStreamException>(() => Qwen2Gguf.LoadQuantizedModel(file.Path));
    }

    [Fact]
    public void DenseEmbeddingRejectsTiedHeadBeforeReadingPayload()
    {
        GgufTensorInfo[] tensors = ValidTensorDirectory()
            .Where(tensor => tensor.Name != "output.weight").ToArray();
        using var file = new TemporaryQwenGguf(ValidMetadata(), tensors);
        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => Qwen2Gguf.LoadQuantizedModel(file.Path));
        Assert.Contains("token_embd.weight", error.Message);
    }

    private static Dictionary<string, object> ValidMetadata() => new()
    {
        ["general.architecture"] = "qwen2",
        ["tokenizer.ggml.tokens"] = new[] { "a", "b", "c", "d" },
        ["qwen2.block_count"] = 1u,
        ["qwen2.embedding_length"] = 256u,
        ["qwen2.attention.head_count"] = 2u,
        ["qwen2.attention.head_count_kv"] = 1u,
        ["qwen2.context_length"] = 16u,
        ["qwen2.feed_forward_length"] = 512u
    };

    private static GgufTensorInfo[] ValidTensorDirectory() =>
    [
        new("token_embd.weight", [256, 4], Qwen2Gguf.F32Type, 0),
        new("blk.0.attn_norm.weight", [256], Qwen2Gguf.F32Type, 0),
        new("blk.0.ffn_norm.weight", [256], Qwen2Gguf.F32Type, 0),
        new("blk.0.attn_q.weight", [256, 256], Qwen2Gguf.Q4KType, 0),
        new("blk.0.attn_q.bias", [256], Qwen2Gguf.F32Type, 0),
        new("blk.0.attn_k.weight", [256, 128], Qwen2Gguf.Q6KType, 0),
        new("blk.0.attn_v.weight", [256, 128], Qwen2Gguf.Q4KType, 0),
        new("blk.0.attn_output.weight", [256, 256], Qwen2Gguf.Q6KType, 0),
        new("blk.0.ffn_gate.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
        new("blk.0.ffn_up.weight", [256, 512], Qwen2Gguf.Q4KType, 0),
        new("blk.0.ffn_down.weight", [512, 256], Qwen2Gguf.Q6KType, 0),
        new("output_norm.weight", [256], Qwen2Gguf.F32Type, 0),
        new("output.weight", [256, 4], Qwen2Gguf.Q6KType, 0)
    ];
}

/// <summary>Minimal on-disk GGUF fixture; no model downloads or GPU are required.</summary>
internal sealed class TemporaryQwenGguf : IDisposable
{
    internal TemporaryQwenGguf(
        IReadOnlyDictionary<string, object> metadata,
        IReadOnlyList<GgufTensorInfo>? tensors = null,
        byte[]? payload = null)
    {
        Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"qwen-test-{Guid.NewGuid():N}.gguf");
        using var writer = new BinaryWriter(File.Create(Path), Encoding.UTF8, leaveOpen: false);
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write((ulong)(tensors?.Count ?? 0));
        writer.Write((ulong)metadata.Count);
        foreach ((string key, object value) in metadata)
        {
            WriteString(writer, key);
            switch (value)
            {
                case string text:
                    writer.Write((uint)GgufValueType.String);
                    WriteString(writer, text);
                    break;
                case uint number:
                    writer.Write((uint)GgufValueType.UInt32);
                    writer.Write(number);
                    break;
                case string[] strings:
                    writer.Write((uint)GgufValueType.Array);
                    writer.Write((uint)GgufValueType.String);
                    writer.Write((ulong)strings.Length);
                    foreach (string item in strings) WriteString(writer, item);
                    break;
                case int[] numbers:
                    writer.Write((uint)GgufValueType.Array);
                    writer.Write((uint)GgufValueType.Int32);
                    writer.Write((ulong)numbers.Length);
                    foreach (int item in numbers) writer.Write(item);
                    break;
                default:
                    throw new ArgumentException($"Unsupported GGUF fixture metadata type for '{key}'.");
            }
        }
        foreach (GgufTensorInfo tensor in tensors ?? [])
        {
            WriteString(writer, tensor.Name);
            writer.Write((uint)tensor.Shape.Count);
            foreach (ulong dimension in tensor.Shape) writer.Write(dimension);
            writer.Write(tensor.Type);
            writer.Write(tensor.Offset);
        }
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        if (payload is not null) writer.Write(payload);
    }

    internal string Path { get; }

    private static void WriteString(BinaryWriter writer, string value)
    {
        byte[] encoded = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)encoded.Length);
        writer.Write(encoded);
    }

    public void Dispose() => File.Delete(Path);
}
