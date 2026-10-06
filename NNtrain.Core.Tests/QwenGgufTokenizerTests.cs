using System.Text;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenGgufTokenizerTests
{
    [Fact]
    public void EncodingRejectsExcessiveInputAndHonorsTokenBudgetAndCancellation()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out _);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);
        Assert.Throws<ArgumentException>(() => tokenizer.Encode(new string('a', 1_048_577)));
        Assert.Throws<ArgumentException>(() => tokenizer.Encode(new string('a', 65_537)));
        Assert.Throws<ArgumentException>(() => tokenizer.Encode("123", 2, TestContext.Current.CancellationToken));
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => tokenizer.Encode("hello", 100, cancellation.Token));
        Assert.Equal("hello", tokenizer.Decode(tokenizer.Encode("hello", 1, TestContext.Current.CancellationToken)));
    }

    [Theory]
    [InlineData(3)]
    [InlineData(4)]
    public void EmptySpecialTokenIsRejectedBeforeEncoding(int type)
    {
        using var file = new TemporaryQwenGguf(new Dictionary<string, object>
        {
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.tokens"] = new[] { "" },
            ["tokenizer.ggml.merges"] = Array.Empty<string>(),
            ["tokenizer.ggml.token_type"] = new[] { type },
            ["tokenizer.ggml.eos_token_id"] = 0u
        });
        Assert.Throws<InvalidDataException>(() => Qwen2GgufTokenizer.Load(file.Path));
    }
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

    [Fact]
    public void StreamingDecoderWaitsForAllBytesOfJapaneseAndEmojiCharacters()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out _);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);
        Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();
        // The fixture assigns each ordinary byte its own token id.
        byte[] bytes = Encoding.UTF8.GetBytes("日😀");

        Assert.Equal(string.Empty, decoder.Append(bytes[0]));
        Assert.Equal(string.Empty, decoder.Append(bytes[1]));
        Assert.Equal("日", decoder.Append(bytes[2]));
        Assert.Equal(string.Empty, decoder.Append(bytes[3]));
        Assert.Equal(string.Empty, decoder.Append(bytes[4]));
        Assert.Equal(string.Empty, decoder.Append(bytes[5]));
        Assert.Equal("😀", decoder.Append(bytes[6]));
        Assert.Equal(string.Empty, decoder.Complete());
        Assert.Equal(string.Empty, decoder.Complete());
        Assert.Throws<InvalidOperationException>(() => decoder.Append((int)'a'));
    }

    [Fact]
    public void StreamingDecoderMatchesFullDecodeAcrossSpecialTokensAndTruncatedFinalBytes()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out Dictionary<string, int> ids);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);
        Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();
        int[] tokens = [0xe6, 0x97, 0xa5, ids["<|endoftext|>"], (int)'a', 0xe6];
        var text = new StringBuilder();
        foreach (int token in tokens) text.Append(decoder.Append(token));

        Assert.Equal("日<|endoftext|>a", text.ToString());
        text.Append(decoder.Complete());

        // An actually truncated sequence is replaced only when finalized,
        // just like the existing non-streaming Decode API.
        Assert.Equal(tokenizer.Decode(tokens), text.ToString());
    }

    [Fact]
    public void StreamingDecoderRejectsInvalidTokenIds()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out _);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);
        Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();

        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Append(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => decoder.Append(tokenizer.VocabularySize));
    }

    [Theory]
    [InlineData("𠮷野家で食べた")]
    [InlineData("a𠮷b")]
    [InlineData("𠀀𠮷")]
    public void EncodeMergesSupplementaryLettersWithAdjacentLetters(string text)
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out Dictionary<string, int> ids, text);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        int[] encoded = tokenizer.Encode(text);

        Assert.Equal(new[] { ids[EncodeBytes(text)] }, encoded);
        Assert.Equal(text, tokenizer.Decode(encoded));
    }

    [Theory]
    [InlineData("e\u0301", "é")]
    [InlineData("か\u3099", "が")]
    [InlineData("A\u030a", "Å")]
    public void EncodeNormalizesOrdinaryTextToNfc(string decomposed, string composed)
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out Dictionary<string, int> ids, composed);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);
        int[] expected = [ids[EncodeBytes(composed)]];

        Assert.Equal(expected, tokenizer.Encode(composed));
        Assert.Equal(expected, tokenizer.Encode(decomposed));
        Assert.Equal(composed, tokenizer.Decode(tokenizer.Encode(decomposed)));
    }

    [Fact]
    public void RuneAwareSplittingPreservesContractionsDigitsPunctuationAndWhitespace()
    {
        const string text = "𠮷's 𝟙𝟚!? \r\n😀a";
        string[] pieces = ["𠮷", "'s", " ", "𝟙", "𝟚", "!?", " \r\n", "😀a"];
        // Supply tempting cross-boundary merges as well. They must not join
        // separate numeric scalars, or the letter and contraction pieces.
        using TemporaryQwenGguf file = CreateTokenizerFixture(
            out Dictionary<string, int> ids, ["𝟙𝟚", "𠮷's", .. pieces]);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);

        int[] encoded = tokenizer.Encode(text);

        Assert.Equal(pieces.Select(piece => ids[EncodeBytes(piece)]).ToArray(), encoded);
        Assert.Equal(text, tokenizer.Decode(encoded));
    }

    [Fact]
    public void UserDefinedToolTokensKeepExactIdsAndLiteralSpelling()
    {
        using TemporaryQwenGguf file = CreateTokenizerFixture(out Dictionary<string, int> ids);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(file.Path);
        const string text = "hello<tool_call>hello</tool_call><tool_e\u0301>";
        int[] expected = [ids["hello"], ids["<tool_call>"], ids["hello"], ids["</tool_call>"], ids["<tool_e\u0301>"]];

        Assert.Equal(expected, tokenizer.Encode(text));
        Assert.Equal(text, tokenizer.Decode(expected));
        Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();
        Assert.Equal(text, string.Concat(expected.Select(decoder.Append)) + decoder.Complete());
    }

    private static TemporaryQwenGguf CreateTokenizerFixture(
        out Dictionary<string, int> ids, params string[] mergedTexts)
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
        List<string> merges = [
            "h e", "he l", "hel l", "hell o", "Ġ hello", "1 2", "' M", "! ?",
            "hello 1", "hello Ċ", "Ċ hello"
        ];
        foreach (string merge in merges) tokens.Add(merge.Replace(" ", "", StringComparison.Ordinal));
        foreach (string text in mergedTexts)
        {
            string encoded = EncodeBytes(text);
            var symbols = encoded.Select(character => character.ToString()).ToList();
            while (symbols.Count > 1)
            {
                // Earlier merges can combine a shared UTF-8 prefix at more
                // than one position. Extend the fixture's actual symbol
                // partition, instead of assuming all later bytes stay single.
                int bestRank = int.MaxValue, best = -1;
                for (int i = 0; i + 1 < symbols.Count; ++i)
                {
                    int rank = merges.IndexOf(symbols[i] + " " + symbols[i + 1]);
                    if (rank >= 0 && rank < bestRank) { bestRank = rank; best = i; }
                }
                if (best < 0)
                {
                    best = 0;
                    merges.Add(symbols[0] + " " + symbols[1]);
                    string joined = symbols[0] + symbols[1];
                    if (!tokens.Contains(joined)) tokens.Add(joined);
                }
                string left = symbols[best], right = symbols[best + 1];
                for (int i = 0; i + 1 < symbols.Count;)
                {
                    if (symbols[i] == left && symbols[i + 1] == right)
                    {
                        symbols[i] += symbols[i + 1];
                        symbols.RemoveAt(i + 1);
                    }
                    else ++i;
                }
            }
        }
        tokens.Add("<|endoftext|>");
        tokens.AddRange(["<tool_call>", "</tool_call>", "<tool_e\u0301>"]);
        ids = tokens.Select((token, id) => (token, id)).ToDictionary(pair => pair.token, pair => pair.id);
        int[] types = Enumerable.Repeat(1, tokens.Count).ToArray();
        types[ids["<|endoftext|>"]] = 3;
        foreach (string token in new[] { "<tool_call>", "</tool_call>", "<tool_e\u0301>" })
            types[ids[token]] = 4;
        return new TemporaryQwenGguf(new Dictionary<string, object>
        {
            ["tokenizer.ggml.model"] = "gpt2",
            ["tokenizer.ggml.tokens"] = tokens.ToArray(),
            ["tokenizer.ggml.merges"] = merges.ToArray(),
            ["tokenizer.ggml.token_type"] = types,
            ["tokenizer.ggml.eos_token_id"] = (uint)ids["<|endoftext|>"]
        });
    }

    private static string EncodeBytes(string text)
    {
        var alphabet = new char[256];
        int escaped = 256;
        for (int value = 0; value < alphabet.Length; ++value)
        {
            bool literal = value is >= 33 and <= 126 or >= 161 and <= 172 or >= 174 and <= 255;
            alphabet[value] = (char)(literal ? value : escaped++);
        }
        return string.Concat(Encoding.UTF8.GetBytes(text).Select(value => alphabet[value]));
    }
}
