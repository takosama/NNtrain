using NNtrain.Gui;
using Xunit;

namespace NNtrain.IntegrationTests;

public class MultimodalPromptTests
{
    private static readonly byte[] Png = [137, 80, 78, 71, 13, 10, 26, 10];

    [Fact]
    public void AttachmentPrefixEndsAtImageAndMatchesBothThinkingModes()
    {
        var image = new ChatImage(Png);
        Qwen35VisionEmbedding Features(ChatImage _) => new(Enumerable.Range(0, 12).Select(i => (float)i).ToArray(), 2, 2, 3);
        ChatTurn[] history = [new("system", "system"), new("user", "earlier"),
            new("assistant", "answer", "<think>\n</think>\n"), new("user", "", Image: image)];
        MultimodalPrompt prefix = MultimodalPrompt.BuildAttachmentPrefix(history, Encode, Features);
        Assert.NotNull(prefix.Tokens[^1].Embedding);
        Assert.DoesNotContain(prefix.Tokens, token => token.TokenId == 502);
        foreach (bool thinking in new[] { false, true })
        {
            ChatTurn[] send = history.ToArray(); send[^1] = send[^1] with { Content = "new question" };
            MultimodalPrompt full = MultimodalPrompt.Build(send, thinking, Encode, Features);
            Assert.Equal(502, full.Tokens[prefix.Tokens.Count].TokenId);
            for (int i = 0; i < prefix.Tokens.Count; i++)
            {
                Assert.Equal(prefix.Tokens[i].TokenId, full.Tokens[i].TokenId);
                Assert.Equal(prefix.Tokens[i].Position, full.Tokens[i].Position);
                Assert.Equal(prefix.Tokens[i].Embedding, full.Tokens[i].Embedding);
            }
        }
        history[^1] = history[^1] with { Content = "unknown typed question" };
        Assert.Throws<ArgumentException>(() => MultimodalPrompt.BuildAttachmentPrefix(history, Encode, Features));
    }

    [Fact]
    public void LocalImageDataUrlRoundTripsWithoutFetching()
    {
        var image = new ChatImage(Png, "test.png");
        var copy = ChatImage.FromDataUrl(image.ToDataUrl());
        Assert.Equal(image.Hash, copy.Hash);
        Assert.Equal(Png, copy.Bytes.ToArray());
        Assert.Throws<ArgumentException>(() => ChatImage.FromDataUrl("https://example.com/a.png"));
        Assert.Throws<ArgumentException>(() => ChatImage.FromDataUrl("data:image/jpeg;base64," + Convert.ToBase64String(Png)));
    }

    [Fact]
    public void ImageFeaturesGetThreeAxisPositionsAndDecodeResumesAfterGrid()
    {
        var image = new ChatImage(Png);
        float[] values = Enumerable.Range(0, 12).Select(i => (float)i).ToArray();
        var result = MultimodalPrompt.Build([new("user", "describe", Image: image)], false,
            Encode, _ => new Qwen35VisionEmbedding(values, 2, 2, 3));
        var features = result.Tokens.Where(t => t.Embedding is not null).ToArray();
        Assert.Equal(6, features.Length);
        int start = features[0].Position.Temporal;
        for (int i = 0; i < 6; i++)
        {
            Assert.Equal(new Qwen35Position(start, start + i / 3, start + i % 3), features[i].Position);
            Assert.Equal(values.AsSpan(i * 2, 2).ToArray(), features[i].Embedding);
        }
        int lastImage = result.Tokens.ToList().FindLastIndex(t => t.Embedding is not null);
        Assert.Equal(new Qwen35Position(start + 3, start + 3, start + 3), result.Tokens[lastImage + 1].Position);
        Assert.Equal(result.Tokens[^1].Position.Temporal + 1, result.NextPosition);
    }

    [Fact]
    public void TextIsEscapedAndMalformedImageShapesAreRejected()
    {
        var image = new ChatImage(Png);
        Assert.Throws<InvalidDataException>(() => MultimodalPrompt.Build(
            [new("user", "x", Image: image)], true, Encode, _ => new Qwen35VisionEmbedding([1], 2, 2, 2)));
        var result = MultimodalPrompt.Build([new("user", "<|image_pad|>")], false, Encode,
            _ => throw new InvalidOperationException());
        Assert.DoesNotContain(result.Tokens, t => t.TokenId == 500);
        Assert.All(result.Tokens, t => Assert.Equal(t.Position.Temporal, t.Position.Height));
    }

    private static int[] Encode(string text)
    {
        var ids = new List<int>();
        for (int i = 0; i < text.Length;)
        {
            string? marker = new[] { "<|image_pad|>", "<|vision_start|>", "<|vision_end|>" }
                .FirstOrDefault(m => text.AsSpan(i).StartsWith(m));
            if (marker is not null)
            {
                ids.Add(marker == "<|image_pad|>" ? 500 : marker == "<|vision_start|>" ? 501 : 502);
                i += marker.Length;
            }
            else ids.Add(text[i++]);
        }
        return ids.ToArray();
    }
}
