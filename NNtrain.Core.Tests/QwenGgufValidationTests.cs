using System.Buffers.Binary;
using System.Reflection;
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
    [InlineData("inspect", true)]
    [InlineData("quantized", true)]
    [InlineData("model", true)]
    [InlineData("weights", true)]
    [InlineData("tensor", true)]
    [InlineData("inspect", false)]
    public void SplitGgufFailsClearlyBeforeMissingMetadataOrTensor(string entryPoint, bool useUInt16)
    {
        // Real llama.cpp shards use UInt16 split metadata. There is no
        // architecture metadata or tensor directory to accidentally rely on.
        var metadata = new Dictionary<string, object>
        {
            ["split.count"] = useUInt16 ? (object)(ushort)2 : 2u,
            ["split.no"] = (ushort)1
        };
        using var file = new TemporaryQwenGguf(metadata);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() =>
        {
            switch (entryPoint)
            {
                case "inspect": Qwen2Gguf.Inspect(file.Path); break;
                case "quantized": Qwen2Gguf.LoadQuantizedModel(file.Path); break;
                case "model": Qwen2Gguf.LoadModel(file.Path); break;
                case "weights":
                    Qwen2Gguf.LoadWeights(file.Path, new Qwen2ForCausalLM(4, 8, 4, 1, 1, 8, 1));
                    break;
                case "tensor": Qwen2Gguf.ReadTensor(file.Path, "absent"); break;
                default: throw new ArgumentException(nameof(entryPoint));
            }
        });

        Assert.Contains("split.count=2", error.Message);
        Assert.Contains("llama-gguf-split --merge", error.Message);
        Assert.Contains("first-shard", error.Message);
    }

    [Fact]
    public void SingleFileSplitMetadataDoesNotPreventInspection()
    {
        Dictionary<string, object> metadata = ValidMetadata();
        metadata["split.count"] = (ushort)1;
        metadata["split.no"] = (ushort)0;
        using var file = new TemporaryQwenGguf(metadata);

        Assert.Equal(256, Qwen2Gguf.Inspect(file.Path).EmbeddingLength);
    }

    [Theory]
    [InlineData(Qwen2Gguf.Q4KType)]
    [InlineData(Qwen2Gguf.Q6KType)]
    public void TiedQuantizedHeadPreservesPayloadAndMatchesExplicitHead(uint type)
    {
        using var tiedFile = CompleteQuantizedFixture(type, explicitHead: false, out byte[] expected);
        using var explicitFile = CompleteQuantizedFixture(type, explicitHead: true, out _);
        using Qwen2QuantizedForCausalLM tied = Qwen2Gguf.LoadQuantizedModel(tiedFile.Path);
        using Qwen2QuantizedForCausalLM explicitModel = Qwen2Gguf.LoadQuantizedModel(explicitFile.Path);

        // Inspect the immutable encoded matrix without requiring an Arc GPU
        // or introducing a public API just to expose model storage for tests.
        ArcQuantizedMatrix tiedMatrix = HeadMatrix(tied);
        ArcQuantizedMatrix explicitMatrix = HeadMatrix(explicitModel);
        Assert.Equal(type, tiedMatrix.GgmlType);
        Assert.Equal(256, tiedMatrix.InputWidth);
        Assert.Equal(4, tiedMatrix.OutputWidth);
        Assert.Equal(expected, PrivateField<byte[]>(tiedMatrix, "_payload"));
        Assert.Equal(PrivateField<byte[]>(explicitMatrix, "_payload"),
            PrivateField<byte[]>(tiedMatrix, "_payload"));
        Assert.Equal(explicitMatrix.GgmlType, tiedMatrix.GgmlType);
        Assert.Equal(expected, PrivateField<byte[]>(
            PrivateField<ArcQuantizedMatrix>(tied, "_quantizedEmbedding"), "_payload"));
        Assert.Equal(PrivateField<byte[]>(
                PrivateField<ArcQuantizedMatrix>(explicitModel, "_quantizedEmbedding"), "_payload"),
            PrivateField<byte[]>(PrivateField<ArcQuantizedMatrix>(tied, "_quantizedEmbedding"), "_payload"));
    }

    [Fact]
    public void TiedHeadRejectsUnsupportedEmbeddingFormatBeforeReadingPayload()
    {
        using var file = new TemporaryQwenGguf(ValidMetadata(),
            ValidTensorDirectory().Where(t => t.Name != "output.weight").ToArray());

        NotSupportedException error = Assert.Throws<NotSupportedException>(
            () => Qwen2Gguf.LoadQuantizedModel(file.Path));

        Assert.Contains("token_embd.weight", error.Message);
        Assert.Contains("Q4_K/Q6_K", error.Message);
    }

    private static ArcQuantizedMatrix HeadMatrix(Qwen2QuantizedForCausalLM model)
        => PrivateField<ArcQuantizedMatrix>(PrivateField<QwenQuantizedLinear>(model, "_head"), "_weight");

    private static T PrivateField<T>(object instance, string name)
        => Assert.IsType<T>(instance.GetType().GetField(name,
            BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(instance));

    private static TemporaryQwenGguf CompleteQuantizedFixture(
        uint embeddingType, bool explicitHead, out byte[] embeddingPayload)
    {
        var random = new Random(731);
        var directory = new List<GgufTensorInfo>();
        using var payload = new MemoryStream();
        embeddingPayload = [];
        foreach (GgufTensorInfo original in ValidTensorDirectory())
        {
            if (original.Name == "output.weight" && !explicitHead) continue;
            GgufTensorInfo tensor = original.Name is "token_embd.weight" or "output.weight"
                ? original with { Type = embeddingType } : original;
            int elements = checked((int)tensor.Shape.Aggregate(1UL, (a, b) => a * b));
            byte[] encoded;
            if (tensor.Name == "output.weight") encoded = embeddingPayload;
            else if (tensor.Type == Qwen2Gguf.F32Type)
            {
                encoded = new byte[elements * 4];
                for (int i = 0; i < elements; ++i)
                    BinaryPrimitives.WriteSingleLittleEndian(encoded.AsSpan(i * 4, 4), 1f);
            }
            else
            {
                int blockBytes = tensor.Type == Qwen2Gguf.Q4KType ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
                encoded = new byte[elements / 256 * blockBytes];
                random.NextBytes(encoded);
                for (int offset = 0; offset < encoded.Length; offset += blockBytes)
                {
                    if (tensor.Type == Qwen2Gguf.Q4KType)
                    {
                        BinaryPrimitives.WriteUInt16LittleEndian(encoded.AsSpan(offset, 2), 0x3800);
                        BinaryPrimitives.WriteUInt16LittleEndian(encoded.AsSpan(offset + 2, 2), 0x3400);
                    }
                    else BinaryPrimitives.WriteUInt16LittleEndian(encoded.AsSpan(offset + 208, 2), 0x3800);
                }
            }
            if (tensor.Name == "token_embd.weight") embeddingPayload = encoded;
            while (payload.Position % 32 != 0) payload.WriteByte(0);
            directory.Add(tensor with { Offset = (ulong)payload.Position });
            payload.Write(encoded);
        }
        return new TemporaryQwenGguf(ValidMetadata(), directory, payload.ToArray());
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
                case ushort number:
                    writer.Write((uint)GgufValueType.UInt16);
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
