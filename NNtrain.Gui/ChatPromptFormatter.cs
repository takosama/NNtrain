using System.Text;

namespace NNtrain.Gui;

/// <summary>Shared Qwen chat framing for text and multimodal inference.</summary>
internal static class ChatPromptFormatter
{
    internal static string FormatPrompt(IReadOnlyList<ChatTurn> conversation, bool thinking)
    {
        if (conversation.Count == 0 || conversation[^1] is not { Role: "user" })
            throw new ArgumentException("The conversation must end with a user turn.", nameof(conversation));
        var prompt = new StringBuilder();
        AppendTurns(prompt, conversation);
        AppendGenerationPrefix(prompt, thinking);
        return prompt.ToString();
    }

    internal static string FormatHistory(IReadOnlyList<ChatTurn> conversation)
    {
        var prompt = new StringBuilder();
        AppendTurns(prompt, conversation);
        return prompt.ToString();
    }

    private static void AppendTurns(StringBuilder prompt, IReadOnlyList<ChatTurn> conversation)
    {
        foreach (ChatTurn turn in conversation)
        {
            if (turn is null || turn.Role is not ("system" or "user" or "assistant"))
                throw new ArgumentException("Chat roles must be system, user or assistant.", nameof(conversation));
            AppendTurnStart(prompt, turn);
            prompt.Append(EscapeContent(turn.Content ?? string.Empty)).Append("<|im_end|>\n");
        }
    }

    internal static void AppendTurnStart(StringBuilder prompt, ChatTurn turn)
        => prompt.Append("<|im_start|>").Append(turn.Role).Append('\n')
            .Append(turn.Role == "assistant" ? turn.AssistantPrefix : null);

    internal static void AppendGenerationPrefix(StringBuilder prompt, bool thinking)
    {
        prompt.Append("<|im_start|>assistant\n<think>\n");
        if (!thinking) prompt.Append("\n</think>\n\n");
    }

    internal static string EscapeContent(string content) => content
        .Replace("<|", "<\u200b|", StringComparison.Ordinal)
        .Replace("<think>", "<\u200bthink>", StringComparison.Ordinal)
        .Replace("</think>", "<\u200b/think>", StringComparison.Ordinal);
}
