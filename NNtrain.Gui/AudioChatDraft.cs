namespace NNtrain.Gui;

/// <summary>A voice insertion never replaces selected or already typed chat text.</summary>
internal sealed class AudioChatDraft(string original, int selectionStart, int selectionLength)
{
    internal string Original { get; } = original;
    internal int SelectionStart { get; } = selectionStart;
    internal int SelectionLength { get; } = selectionLength;
    private readonly int _position = Math.Clamp(selectionStart + selectionLength, 0, original.Length);

    internal string Compose(string recognized)
    {
        string voice = recognized.Trim();
        if (voice.Length == 0) return Original;
        string prefix = Original[.._position], suffix = Original[_position..];
        string before = prefix.Length > 0 && !char.IsWhiteSpace(prefix[^1]) ? "\n" : "";
        string after = suffix.Length > 0 && !char.IsWhiteSpace(suffix[0]) ? "\n" : "";
        return prefix + before + voice + after + suffix;
    }
}
