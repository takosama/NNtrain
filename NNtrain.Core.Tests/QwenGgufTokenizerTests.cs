using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenGgufTokenizerTests
{
    [Fact]
    public void EncodeUsesBpeWithinQwenUnicodeAndDigitBoundaries()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out Dictionary<string, int> ids);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        Assert.Equal(
            new[] { ids["hello"], ids["Ġhello"], ids["1"], ids["2"], ids["Ġ"], ids["I"], ids["'M"], ids["!?"] },
            tokenizer.Encode("hello hello12 I'M!?"));
        // These merges exist in the vocabulary, but must not cross the
        // pre-tokenizer's single-digit and line boundaries.
        Assert.Equal(new[] { ids["1"], ids["2"] }, tokenizer.Encode("12"));
        Assert.Equal(
            new[] { ids["hello"], ids["Ċ"], ids["hello"] },
            tokenizer.Encode("hello\nhello"));
    }

    [Theory]
    [InlineData("")]
    [InlineData("hello 日本語")]
    [InlineData("pLN sS 0123456789 I'M we're")]
    [InlineData("  \r\n\t日本語 abc123\u00a0!?😀\n")]
    public void EncodeDecodePreservesEveryUtf8Byte(string text)
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out _);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        int[] encoded = tokenizer.Encode(text);

        if (text.Length > 0) Assert.NotEmpty(encoded);
        Assert.Equal(text, tokenizer.Decode(encoded));
    }

    [Fact]
    public void EncodePreservesSpecialTokenIdsFromGgufMetadata()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out Dictionary<string, int> ids);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        int[] encoded = tokenizer.Encode("hello<|endoftext|>hello");

        Assert.Equal(new[] { ids["hello"], ids["<|endoftext|>"], ids["hello"] }, encoded);
        Assert.Equal(ids["<|endoftext|>"], tokenizer.EosTokenId);
        Assert.Equal("hello<|endoftext|>hello", tokenizer.Decode(encoded));
    }

    private static TemporaryQwenGguf CreateTokenizerFixture(out Dictionary<string, int> ids)
    {
        // A complete GPT-2 byte alphabet in byte-id order, independent of the
        // implementation's encoder. Merges make Encode assertions sensitive
        // to the regex boundaries instead of merely testing UTF-8 itself.
        int escaped = 256;
        var tokens = new List<string>();
        for (int value = 0; value < 256; ++value)
        {
            bool literal = value is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255;
            tokens.Add(((char)(literal ? value : escaped++)).ToString());
        }
        string[] merges = [
            "h e", "he l", "hel l", "hell o", "Ġ hello", "1 2", "' M", "! ?",
            "hello 1", "hello Ċ", "Ċ hello"
        ];
        foreach (string merge in merges) tokens.Add(merge.Replace(" ", "", StringComparison.Ordinal));
        tokens.Add("<|endoftext|>");
        ids = tokens.Select((token, id) => (token, id)).ToDictionary(pair => pair.token, pair => pair.id);
        int[] types = Enumerable.Repeat(1, tokens.Count).ToArray();
        types[^1] = 3;
        return new TemporaryQwenGguf(new Dictionary<string, object>
        {
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.tokens"] = tokens.ToArray(),
            ["tokenizer.ggml.merges"] = merges,
            ["tokenizer.ggml.token_type"] = types,
            ["tokenizer.ggml.eos_token_id"] = (uint)(tokens.Count - 1)
        });
    }
}
