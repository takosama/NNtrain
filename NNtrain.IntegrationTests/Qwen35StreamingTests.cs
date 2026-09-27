using System.Text;
using NNtrain;
using Xunit;

public sealed class Qwen35StreamingTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void GenerationOutputPreservesUnicodeAndStreamsBeforeGenerationReturns(bool stream)
    {
        using var fixture = new ByteTokenizerFixture();
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(fixture.Path);
        const string prompt = "prompt: ";
        const string continuation = "日😀a";
        byte[] bytes = Encoding.UTF8.GetBytes(continuation);
        bool generating = false;
        bool completed = false;
        int calls = 0;
        using var output = new ObservingWriter(() => generating, () => completed);

        QwenGgufCommand.GenerationTiming timing = QwenGgufCommand.WriteGeneration(
            prompt, tokenizer, stream, callback =>
            {
                ++calls;
                generating = true;
                if (stream) Assert.Contains(prompt, output.ToString());
                else Assert.Equal(string.Empty, output.ToString());
                for (int i = 0; i < bytes.Length; ++i)
                {
                    callback(bytes[i]);
                    Assert.DoesNotContain("\ufffd", output.ToString());
                    if (i == 1 && stream) Assert.DoesNotContain("日", output.ToString());
                    if (i == 2 && stream)
                    {
                        Assert.EndsWith("日", output.ToString());
                        Assert.Equal(1, output.ChunksDuringGeneration);
                        Assert.True(output.FlushesDuringGeneration > 0);
                    }
                    if (!stream) Assert.Equal(string.Empty, output.ToString());
                }
                generating = false;
                completed = true;
                return prompt + continuation;
            }, output);

        Assert.Equal(1, calls);
        Assert.Equal("generated:" + Environment.NewLine + prompt + continuation + Environment.NewLine,
            output.ToString());
        Assert.Equal(stream ? 3 : 0, output.ChunksDuringGeneration);
        Assert.Equal(bytes.Length, timing.GeneratedTokens);
        Assert.NotNull(timing.FirstTokenMilliseconds);
        Assert.True(timing.TotalMilliseconds >= timing.FirstTokenMilliseconds.Value);
    }

    [Fact]
    public void ZeroTokenGenerationHasNoFirstTokenOrDecodeRate()
    {
        using var fixture = new ByteTokenizerFixture();
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(fixture.Path);
        using var output = new StringWriter();

        QwenGgufCommand.GenerationTiming timing = QwenGgufCommand.WriteGeneration(
            "a", tokenizer, true, _ => "a", output);

        Assert.Equal(0, timing.GeneratedTokens);
        Assert.Null(timing.FirstTokenMilliseconds);
        Assert.Null(timing.DecodeTokensPerSecond);
        Assert.Equal("generated:" + Environment.NewLine + "a" + Environment.NewLine, output.ToString());
    }

    private sealed class ObservingWriter(Func<bool> isGenerating, Func<bool> isCompleted) : StringWriter
    {
        internal int ChunksDuringGeneration { get; private set; }
        internal int FlushesDuringGeneration { get; private set; }

        public override void Write(string? value)
        {
            if (isGenerating() && !string.IsNullOrEmpty(value))
            {
                Assert.False(isCompleted());
                ChunksDuringGeneration++;
            }
            base.Write(value);
        }

        public override void Flush()
        {
            if (isGenerating()) FlushesDuringGeneration++;
            base.Flush();
        }
    }

    private sealed class ByteTokenizerFixture : IDisposable
    {
        internal ByteTokenizerFixture()
        {
            Path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"qwen-stream-{Guid.NewGuid():N}.gguf");
            using var writer = new BinaryWriter(File.Create(Path), Encoding.UTF8);
            writer.Write(0x46554747u);
            writer.Write(3u);
            writer.Write(0ul);
            writer.Write(4ul);
            WriteString(writer, "tokenizer.ggml.model");
            writer.Write((uint)GgufValueType.String);
            WriteString(writer, "gpt2");
            WriteString(writer, "tokenizer.ggml.pre");
            writer.Write((uint)GgufValueType.String);
            WriteString(writer, "qwen35");
            WriteString(writer, "tokenizer.ggml.tokens");
            writer.Write((uint)GgufValueType.Array);
            writer.Write((uint)GgufValueType.String);
            writer.Write(256ul);
            int escaped = 256;
            for (int value = 0; value < 256; ++value)
            {
                bool literal = value is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255;
                WriteString(writer, ((char)(literal ? value : escaped++)).ToString());
            }
            WriteString(writer, "tokenizer.ggml.merges");
            writer.Write((uint)GgufValueType.Array);
            writer.Write((uint)GgufValueType.String);
            writer.Write(0ul);
        }

        internal string Path { get; }

        private static void WriteString(BinaryWriter writer, string text)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }

        public void Dispose() => File.Delete(Path);
    }
}
