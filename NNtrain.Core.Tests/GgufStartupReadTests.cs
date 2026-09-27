using System.Text.Json;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class GgufStartupReadTests
{
    [Fact]
    public void SharedReaderInspectionMatchesPathInspectionAndRetainsOwnership()
    {
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false);
        using var reader = new GgufReader(Path.GetRelativePath(Environment.CurrentDirectory, file.Path));
        Assert.Equal(Path.GetFullPath(file.Path), reader.FilePath);
        using Stream tensor = reader.OpenTensorData(reader.Tensors[0]);
        long position = tensor.Position;
        Qwen35GgufDescriptor actual = Qwen35Gguf.Inspect(reader);
        Assert.Equal(JsonSerializer.Serialize(Qwen35Gguf.Inspect(file.Path)), JsonSerializer.Serialize(actual));
        Assert.Equal(position, tensor.Position);
        Assert.Equal(reader.ReadTensorBytes(reader.Tensors[0], 8), ReadExactly(tensor, 8));
    }

    [Fact]
    public void SharedReaderTokenizerMatchesPathAndSurvivesReaderDisposal()
    {
        using var file = new TemporaryQwenGguf(TokenizerMetadata());
        Qwen2GgufTokenizer tokenizer;
        using (var reader = new GgufReader(file.Path))
        {
            tokenizer = Qwen2GgufTokenizer.Load(reader);
            Assert.Equal(Qwen2GgufTokenizer.Load(file.Path).Encode("ab"), tokenizer.Encode("ab"));
            Assert.Equal("gpt2", reader.Metadata["tokenizer.ggml.model"]);
            // Reading remains possible: the overload does not dispose its caller's reader.
            Assert.Empty(reader.ReadTensorBytes(new("empty", [1], Qwen2Gguf.F32Type, 0), 0));
        }
        Assert.Equal(new[] { 2 }, tokenizer.Encode("ab"));
        Assert.Equal("ab", tokenizer.Decode([2]));
    }

    [Fact]
    public void SharedReaderOverloadsKeepArchitectureAndTokenizerValidation()
    {
        var metadata = TokenizerMetadata();
        metadata["general.architecture"] = "qwen35moe";
        metadata["tokenizer.ggml.pre"] = "unsupported";
        using var file = new TemporaryQwenGguf(metadata);
        using var reader = new GgufReader(file.Path);
        Assert.Equal(Assert.Throws<InvalidDataException>(() => Qwen35Gguf.Inspect(file.Path)).Message,
            Assert.Throws<InvalidDataException>(() => Qwen35Gguf.Inspect(reader)).Message);
        Assert.Equal(Assert.Throws<NotSupportedException>(() => Qwen2GgufTokenizer.Load(file.Path)).Message,
            Assert.Throws<NotSupportedException>(() => Qwen2GgufTokenizer.Load(reader)).Message);
        Assert.Throws<ArgumentNullException>(() => Qwen35Gguf.Inspect((GgufReader)null!));
        Assert.Throws<ArgumentNullException>(() => Qwen2GgufTokenizer.Load((GgufReader)null!));
    }

    [Fact]
    public void ReplacingAClosedFileDoesNotReusePreviousMetadata()
    {
        using TemporaryQwenGguf first = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false, contextLength: 8);
        using TemporaryQwenGguf second = Qwen35ResidentModelTests.CreateFixture(tiedOutput: false, contextLength: 32);
        Qwen35GgufDescriptor old;
        using (var reader = new GgufReader(first.Path)) old = Qwen35Gguf.Inspect(reader);
        File.Copy(second.Path, first.Path, overwrite: true);
        using var replacement = new GgufReader(first.Path);
        Assert.Equal(8, old.ContextLength);
        Assert.Equal(32, Qwen35Gguf.Inspect(replacement).ContextLength);
        Assert.Equal(32, Qwen35Gguf.Inspect(first.Path).ContextLength);
    }

    [Fact]
    public void PositionalReadsKeepOffsetsAndDoNotChangeBufferedStreamPosition()
    {
        byte[] payload = Enumerable.Range(0, 257).Select(i => (byte)(i * 17)).ToArray();
        GgufTensorInfo[] tensors = [new("first", [7], Qwen2Gguf.F32Type, 0), new("second", [7], Qwen2Gguf.F32Type, 37)];
        using var file = new TemporaryQwenGguf(new Dictionary<string, object>(), tensors, payload);
        using var reader = new GgufReader(file.Path);
        using Stream stream = reader.OpenTensorData(tensors[0]);
        Assert.Equal(payload[0], stream.ReadByte());
        long position = stream.Position;
        Assert.Equal(payload.AsSpan(37, 73).ToArray(), reader.ReadTensorBytes(tensors[1], 73));
        Assert.Equal(position, stream.Position);
        Assert.Equal(payload[1], stream.ReadByte());
        Assert.Equal(payload, reader.ReadTensorBytes(tensors[0], payload.Length));
    }

    [Fact]
    public void ConcurrentTensorReadsDoNotInterleaveFileOffsets()
    {
        const int chunk = 65_537, blocks = 24;
        byte[] payload = new byte[chunk * blocks];
        new Random(712).NextBytes(payload);
        GgufTensorInfo[] tensors = Enumerable.Range(0, blocks)
            .Select(i => new GgufTensorInfo($"block.{i}", [(ulong)chunk], Qwen2Gguf.F32Type, (ulong)(i * chunk))).ToArray();
        using var file = new TemporaryQwenGguf(new Dictionary<string, object>(), tensors, payload);
        using var reader = new GgufReader(file.Path);
        using Stream stream = reader.OpenTensorData(tensors[0]);
        long position = stream.Position;
        Parallel.For(0, blocks * 8, iteration =>
        {
            int index = iteration * 7 % blocks;
            Assert.Equal(payload.AsSpan(index * chunk, chunk).ToArray(), reader.ReadTensorBytes(tensors[index], chunk));
        });
        Assert.Equal(position, stream.Position);
        Assert.Equal(payload[0], stream.ReadByte());
    }

    [Fact]
    public void PositionalReadsRejectTruncationAndOverflowIncludingEmptyRanges()
    {
        GgufTensorInfo tensor = new("data", [1], Qwen2Gguf.F32Type, 0);
        using var file = new TemporaryQwenGguf(new Dictionary<string, object>(), [tensor], [1, 2, 3]);
        using var reader = new GgufReader(file.Path);
        Assert.Equal(new byte[] { 1, 2, 3 }, reader.ReadTensorBytes(tensor, 3));
        Assert.Throws<EndOfStreamException>(() => reader.ReadTensorBytes(tensor, 4));
        Assert.Empty(reader.ReadTensorBytes(tensor with { Offset = 3 }, 0));
        Assert.Throws<EndOfStreamException>(() => reader.ReadTensorBytes(tensor with { Offset = 4 }, 0));
        Assert.Throws<EndOfStreamException>(() => reader.ReadTensorBytes(tensor with { Offset = 3 }, 1));
        Assert.Throws<OverflowException>(() => reader.ReadTensorBytes(tensor with { Offset = ulong.MaxValue }, 0));
        Assert.Throws<OverflowException>(() => reader.ReadTensorBytes(tensor with { Offset = long.MaxValue }, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => reader.ReadTensorBytes(tensor, -1));
        Assert.Throws<ArgumentNullException>(() => reader.ReadTensorBytes(null!, 0));
    }

    private static Dictionary<string, object> TokenizerMetadata() => new()
    {
        ["tokenizer.ggml.model"] = "gpt2", ["tokenizer.ggml.pre"] = "qwen35",
        ["tokenizer.ggml.tokens"] = new[] { "a", "b", "ab" },
        ["tokenizer.ggml.merges"] = new[] { "a b" }
    };

    private static byte[] ReadExactly(Stream stream, int length)
    {
        var bytes = new byte[length];
        stream.ReadExactly(bytes);
        return bytes;
    }
}
