using NNtrain;
using Xunit;

public sealed class BpeTokenizerTests
{
    [Fact]
    public void SpecialTokensHaveStableReservedIds()
    {
        Assert.Equal("<pad>", BpeTokenizer.PadToken);
        Assert.Equal("<bos>", BpeTokenizer.BosToken);
        Assert.Equal("<eos>", BpeTokenizer.EosToken);
        Assert.Equal("<unk>", BpeTokenizer.UnknownToken);
        Assert.Equal(0, BpeTokenizer.PadTokenId);
        Assert.Equal(1, BpeTokenizer.BosTokenId);
        Assert.Equal(2, BpeTokenizer.EosTokenId);
        Assert.Equal(3, BpeTokenizer.UnknownTokenId);
    }

    [Fact]
    public void JapaneseTextRoundTripsThroughTrainedTokenizer()
    {
        const string text = "東京は日本の首都です。東京には多くの人が住んでいます。";
        BpeTokenizer tokenizer = BpeTokenizer.Train(
            Enumerable.Repeat(text, 8),
            vocabularySize: 300);

        int[] encoded = tokenizer.Encode(text, addBos: true, addEos: true);

        Assert.Equal(BpeTokenizer.BosTokenId, encoded[0]);
        Assert.Equal(BpeTokenizer.EosTokenId, encoded[^1]);
        Assert.Equal(text, tokenizer.Decode(encoded));
        Assert.True(encoded.Length < System.Text.Encoding.UTF8.GetByteCount(text) + 2);
    }

    [Fact]
    public void SavedTokenizerPreservesEncoding()
    {
        string path = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-bpe-{Guid.NewGuid():N}.json");
        try
        {
            BpeTokenizer original = BpeTokenizer.Train(
                ["りんご りんご みかん", "みかん りんご"],
                vocabularySize: 280);
            original.Save(path);

            BpeTokenizer loaded = BpeTokenizer.Load(path);

            Assert.Equal(original.VocabularySize, loaded.VocabularySize);
            Assert.Equal(
                original.Encode("りんごとみかん"),
                loaded.Encode("りんごとみかん"));
            Assert.Equal(
                "りんごとみかん",
                loaded.Decode(loaded.Encode("りんごとみかん")));
        }
        finally
        {
            if (File.Exists(path))
                File.Delete(path);
        }
    }

    [Fact]
    public void TrainingLimitIsAppliedWithoutBreakingRoundTrip()
    {
        BpeTokenizer tokenizer = BpeTokenizer.Train(
            ["あいうえお", "かきくけこ"],
            vocabularySize: 280,
            maxTrainingBytes: 9);

        const string unseenText = "未知の文字列🙂";
        Assert.Equal(
            unseenText,
            tokenizer.Decode(tokenizer.Encode(unseenText)));
    }
}
