using System.Text;

namespace NNtrain.Gui;

internal readonly record struct ThinkingStreamSnapshot(
    string ThinkingText, string AnswerText, bool HasThinking, bool ThinkingInProgress);

/// <summary>
/// Builds a current view of a generated response. Some Qwen GGUF models emit
/// reasoning and a closing marker even with Thinking disabled, so the view can
/// move provisional answer text into the collapsible reasoning panel later.
/// </summary>
internal sealed class ThinkingStreamParser(bool requestedThinking)
{
    private const string OpenMarker = "<think>";
    private const string EndMarker = "</think>";
    private const string MessageEndMarker = "<|im_end|>";
    private readonly StringBuilder _raw = new();
    private bool _completed;

    public ThinkingStreamSnapshot Append(string chunk)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        if (_completed) throw new InvalidOperationException("The thinking stream is complete.");
        _raw.Append(chunk);
        return Parse(_raw.ToString(), complete: false);
    }

    public ThinkingStreamSnapshot Complete(bool endedByEos = false)
    {
        _completed = true;
        return Parse(_raw.ToString(), complete: true, endedByEos);
    }

    private ThinkingStreamSnapshot Parse(string raw, bool complete, bool endedByEos = false)
    {
        int messageEnd = raw.IndexOf(MessageEndMarker, StringComparison.Ordinal);
        if (messageEnd >= 0) raw = raw[..messageEnd];
        else if (complete)
        {
            // A canceled stream or token limit can end in the middle of a
            // control token. Never save that fragment as assistant content.
            raw = HidePartialMarker(raw, MessageEndMarker, minimumLength: 2);
            raw = HidePartialMarker(raw, EndMarker, minimumLength: 2);
            raw = HidePartialMarker(raw, OpenMarker, minimumLength: 2);
        }
        else raw = HidePartialMarker(raw, MessageEndMarker);
        int closing = raw.IndexOf(EndMarker, StringComparison.Ordinal);
        int opening = raw.IndexOf(OpenMarker, StringComparison.Ordinal);
        bool explicitOpening = opening >= 0 && string.IsNullOrWhiteSpace(raw[..opening]);
        if (closing >= 0)
        {
            int thinkingStart = explicitOpening && opening < closing ? opening + OpenMarker.Length : 0;
            string thought = raw[thinkingStart..closing].Trim('\r', '\n');
            // A LoRA can emit another closing marker after its answer. Remove
            // that marker without discarding answer text generated before it.
            var answer = new StringBuilder();
            int segmentStart = closing + EndMarker.Length;
            while (true)
            {
                int nextClosing = raw.IndexOf(EndMarker, segmentStart, StringComparison.Ordinal);
                string segment = nextClosing >= 0
                    ? raw[segmentStart..nextClosing]
                    : raw[segmentStart..];
                if (nextClosing < 0 && !complete) segment = HidePartialMarker(segment, EndMarker);
                if (segment.Length != 0)
                {
                    // Adjacent text spans need a boundary when the model emits
                    // a marker without whitespace on either side.
                    if (answer.Length != 0 && !char.IsWhiteSpace(answer[^1])
                        && !char.IsWhiteSpace(segment[0])) answer.Append('\n');
                    answer.Append(segment);
                }
                if (nextClosing < 0) break;
                segmentStart = nextClosing + EndMarker.Length;
            }
            bool hasThinking = requestedThinking || thought.Trim().Length != 0;
            return new ThinkingStreamSnapshot(thought, answer.ToString().TrimStart('\r', '\n'),
                hasThinking, false);
        }

        if (requestedThinking || explicitOpening)
        {
            // A few models answer directly despite a <think> assistant prompt.
            // Once the model ends the turn, unmarked text is the complete
            // answer; reaching the token/context limit is still inconclusive.
            if (complete && endedByEos && !explicitOpening && raw.Trim().Length != 0)
                return new ThinkingStreamSnapshot(string.Empty, raw, false, false);
            string thought = explicitOpening ? raw[(opening + OpenMarker.Length)..] : raw;
            if (!complete) thought = HidePartialMarker(thought, EndMarker);
            return new ThinkingStreamSnapshot(thought, string.Empty, true, !complete);
        }

        // Until a closing marker appears, Thinking Off streams a provisional
        // answer. If the marker arrives later, the full view is reclassified.
        string provisionalAnswer = complete ? raw : HidePartialMarker(raw, EndMarker);
        return new ThinkingStreamSnapshot(string.Empty, provisionalAnswer, false, false);
    }

    private static string HidePartialMarker(string text, string marker, int minimumLength = 1)
    {
        for (int length = Math.Min(text.Length, marker.Length - 1); length >= minimumLength; --length)
            if (text.EndsWith(marker[..length], StringComparison.Ordinal))
                return text[..^length];
        return text;
    }
}
