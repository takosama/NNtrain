using System.Text;

namespace NNtrain.Gemma;

internal static class Gemma3ConversationFormatter
{
    public static string Format(IReadOnlyList<Gemma3ChatTurn> turns)
    {
        ArgumentNullException.ThrowIfNull(turns);
        if (turns.Count == 0 || turns[^1].Role != "user")
            throw new ArgumentException("Gemma conversation must end with a user turn.", nameof(turns));

        var prompt = new StringBuilder();
        var system = new StringBuilder();
        bool firstUser = true;
        foreach (Gemma3ChatTurn turn in turns)
        {
            ArgumentNullException.ThrowIfNull(turn);
            if (turn.Role == "system")
            {
                if (!firstUser) throw new ArgumentException("System text must precede user turns.", nameof(turns));
                if (system.Length > 0) system.Append("\n\n");
                system.Append(EscapeMarkers(turn.Content));
                continue;
            }
            string role = turn.Role switch
            {
                "user" => "user",
                "assistant" => "model",
                _ => throw new ArgumentException($"Unsupported Gemma chat role '{turn.Role}'.", nameof(turns))
            };
            prompt.Append("<start_of_turn>").Append(role).Append('\n');
            if (firstUser && role == "user" && system.Length > 0)
                prompt.Append(system).Append("\n\n");
            prompt.Append(EscapeMarkers(turn.Content)).Append("<end_of_turn>\n");
            if (role == "user") firstUser = false;
        }
        prompt.Append("<start_of_turn>model\n");
        return prompt.ToString();
    }

    private static string EscapeMarkers(string text)
        => (text ?? string.Empty)
            .Replace("<start_of_turn>", "<start_of_\u200bturn>", StringComparison.Ordinal)
            .Replace("<end_of_turn>", "<end_of_\u200bturn>", StringComparison.Ordinal);

}
