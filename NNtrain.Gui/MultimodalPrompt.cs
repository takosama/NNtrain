using System.Text;
using System.IO;

namespace NNtrain.Gui;

internal sealed record MultimodalPrompt(IReadOnlyList<Qwen35PromptToken> Tokens, int NextPosition)
{
    public static MultimodalPrompt Build(IReadOnlyList<ChatTurn> turns, bool thinking,
        Func<string, int[]> encode, Func<ChatImage, Qwen35VisionEmbedding> imageEmbedding)
        => BuildCore(turns, thinking, encode, imageEmbedding, imagePrefixOnly: false);

    public static MultimodalPrompt BuildAttachmentPrefix(IReadOnlyList<ChatTurn> turns,
        Func<string, int[]> encode, Func<ChatImage, Qwen35VisionEmbedding> imageEmbedding)
    {
        if (turns.Count == 0 || turns[^1].Role != "user" || turns[^1].Image is null || turns[^1].Content.Length != 0)
            throw new ArgumentException("画像の事前準備は本文が空の画像付きユーザー入力で終えてください。");
        return BuildCore(turns, false, encode, imageEmbedding, imagePrefixOnly: true);
    }

    private static MultimodalPrompt BuildCore(IReadOnlyList<ChatTurn> turns, bool thinking,
        Func<string, int[]> encode, Func<ChatImage, Qwen35VisionEmbedding> imageEmbedding, bool imagePrefixOnly)
    {
        if (turns.Count == 0 || turns[^1].Role != "user")
            throw new ArgumentException("会話はユーザーの入力で終わる必要があります。");
        var tokens = new List<Qwen35PromptToken>();
        var text = new StringBuilder();
        int position = 0;
        for (int turnIndex = 0; turnIndex < turns.Count; turnIndex++)
        {
            ChatTurn turn = turns[turnIndex];
            if (turn.Role is not ("system" or "user" or "assistant"))
                throw new ArgumentException("未対応の会話ロールです。");
            text.Append("<|im_start|>").Append(turn.Role).Append('\n')
                .Append(turn.Role == "assistant" ? turn.AssistantPrefix : null);
            if (turn.Image is { } image)
            {
                if (turn.Role != "user") throw new ArgumentException("画像はユーザー入力に添付してください。");
                RequireSpecial("<|vision_start|>");
                RequireSpecial("<|vision_end|>");
                int imageToken = RequireSpecial("<|image_pad|>");
                text.Append("<|vision_start|>");
                FlushText();
                Qwen35VisionEmbedding embedding = imageEmbedding(image);
                int count = checked(embedding.GridHeight * embedding.GridWidth);
                if (count <= 0 || embedding.EmbeddingLength <= 0 || embedding.Values.Length != checked(count * embedding.EmbeddingLength))
                    throw new InvalidDataException("画像埋め込みの形状が一致しません。");
                for (int row = 0; row < embedding.GridHeight; row++)
                    for (int col = 0; col < embedding.GridWidth; col++)
                    {
                        int offset = (row * embedding.GridWidth + col) * embedding.EmbeddingLength;
                        tokens.Add(new Qwen35PromptToken(imageToken,
                            embedding.Values.AsSpan(offset, embedding.EmbeddingLength).ToArray(),
                            new Qwen35Position(position, position + row, position + col)));
                    }
                position = checked(position + Math.Max(embedding.GridHeight, embedding.GridWidth));
                // Stop at the last projected image row. vision_end, the
                // unknown typed question and assistant marker remain for Send.
                if (imagePrefixOnly && turnIndex == turns.Count - 1)
                    return new MultimodalPrompt(tokens, position);
                text.Append("<|vision_end|>");
            }
            text.Append(Escape(turn.Content)).Append("<|im_end|>\n");
        }
        text.Append("<|im_start|>assistant\n<think>\n");
        if (!thinking) text.Append("\n</think>\n\n");
        FlushText();
        return new MultimodalPrompt(tokens, position);

        void FlushText()
        {
            foreach (int id in encode(text.ToString()))
            {
                tokens.Add(new Qwen35PromptToken(id, null, new Qwen35Position(position, position, position)));
                position++;
            }
            text.Clear();
        }
        int RequireSpecial(string marker)
        {
            int[] ids = encode(marker);
            if (ids.Length != 1) throw new NotSupportedException($"モデルの tokenizer に {marker} がありません。");
            return ids[0];
        }
    }

    private static string Escape(string text) => text.Replace("<|", "<\u200b|", StringComparison.Ordinal)
        .Replace("<think>", "<\u200bthink>", StringComparison.Ordinal)
        .Replace("</think>", "<\u200b/think>", StringComparison.Ordinal);
}
