using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TokenizerTests
{
    [Theory]
    [InlineData("a\u0301")]
    [InlineData("a\u0301b")]
    [InlineData("a\u0903")]
    [InlineData("a\u20dd")]
    public void Qwen35MergesLettersAndEveryCombiningMarkCategory(string text)
    {
        using TemporaryQwenGguf file = CreateFixture("qwen35", [text], out Dictionary<string, int> ids);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        int[] encoded = tokenizer.Encode(text);

        Assert.Equal(new[] { ids[text] }, encoded);
        Assert.Equal(text, tokenizer.Decode(encoded));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("qwen2")]
    public void ExistingQwen2BoundaryDoesNotMergeTrailingCombiningMark(string? preTokenizer)
    {
        const string text = "a\u0301";
        using TemporaryQwenGguf file = CreateFixture(preTokenizer, [text], out _);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        // No standalone mark merge is in this fixture: Qwen2 splits after a,
        // so the combining mark remains its two UTF-8 byte tokens.
        Assert.Equal(new[] { (int)'a', 0xcc, 0x81 }, tokenizer.Encode(text));
        Assert.Equal(text, tokenizer.Decode(tokenizer.Encode(text)));
    }

    [Fact]
    public void Qwen35StillSeparatesPunctuationDigitsAndLineBoundaries()
    {
        const string accent = "a\u0301";
        using TemporaryQwenGguf file = CreateFixture("qwen35",
            [accent, accent + "!", "12", accent + "\n", "!?"], out Dictionary<string, int> ids);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        Assert.Equal(new[] { ids[accent], ids["!?"] }, tokenizer.Encode(accent + "!?"));
        Assert.Equal(new[] { (int)'1', (int)'2' }, tokenizer.Encode("12"));
        Assert.Equal(new[] { ids[accent], (int)'\n', ids[accent] }, tokenizer.Encode(accent + "\n" + accent));
    }

    [Fact]
    public void UnsupportedPreTokenizerFailsInsteadOfUsingQwen2Boundaries()
    {
        using TemporaryQwenGguf file = CreateFixture("not-qwen", [], out _);

        NotSupportedException error = Assert.Throws<NotSupportedException>(() => Qwen2GgufTokenizer.Load(file.Path));

        Assert.Contains("not-qwen", error.Message);
    }

    private static TemporaryQwenGguf CreateFixture(
        string? preTokenizer, string[] mergedTexts, out Dictionary<string, int> textIds)
    {
        int escaped = 256;
        var tokens = new List<string>();
        for (int value = 0; value < 256; ++value)
        {
            bool literal = value is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255;
            tokens.Add(((char)(literal ? value : escaped++)).ToString());
        }

        var merges = new List<string>();
        textIds = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string text in mergedTexts)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(text);
            string prefix = tokens[bytes[0]];
            for (int i = 1; i < bytes.Length; ++i)
            {
                string next = tokens[bytes[i]];
                string merged = prefix + next;
                if (!tokens.Contains(merged))
                {
                    merges.Add(prefix + " " + next);
                    tokens.Add(merged);
                }
                prefix = merged;
            }
            textIds.Add(text, tokens.IndexOf(prefix));
        }

        var metadata = new Dictionary<string, object>
        {
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.tokens"] = tokens.ToArray(),
            ["tokenizer.ggml.merges"] = merges.ToArray()
        };
        if (preTokenizer is not null) metadata["tokenizer.ggml.pre"] = preTokenizer;
        return new TemporaryQwenGguf(metadata);
    }
}
