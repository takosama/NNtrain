using System.Text;

namespace NNtrain.Gui;

/// <summary>Settles one answer, including the direct retry and interrupted streaming text.</summary>
internal sealed class ChatGenerationSession(bool thinking)
{
    private int _generation;

    internal ThinkingStreamSnapshot Snapshot { get; private set; }
    internal GenerationStopReason? StopReason { get; private set; }
    internal bool Completed { get; private set; }
    internal bool RetriedWithoutThinking { get; private set; }
    internal string AssistantPrefix => thinking && !RetriedWithoutThinking
        ? "<think>\n</think>\n" : "<think>\n\n</think>\n\n";

    internal async Task RunAsync(
        Func<bool, Action<string>, CancellationToken, Task<LocalGeneration>> generate,
        Action<ThinkingStreamSnapshot> onSnapshot, Action onRetry, CancellationToken ct)
    {
        int firstGeneration = ++_generation;
        var liveParser = new ThinkingStreamParser(thinking);
        var streamedRaw = new StringBuilder();
        var directStreamedRaw = new StringBuilder();
        string? finalRaw = null;
        string firstThought = "";
        ThinkingStreamSnapshot? completedSnapshot = null;
        try
        {
            LocalGeneration first = await generate(thinking, chunk =>
            {
                streamedRaw.Append(chunk);
                if (firstGeneration != _generation) return;
                onSnapshot(liveParser.Append(chunk));
            }, ct);
            finalRaw = first.Text;
            StopReason = first.StopReason;
            var firstFinalParser = new ThinkingStreamParser(thinking);
            firstFinalParser.Append(finalRaw);
            ThinkingStreamSnapshot firstSnapshot = firstFinalParser.Complete(
                StopReason == GenerationStopReason.EndOfMessage);
            completedSnapshot = firstSnapshot;

            // Retry once without placing the failed reasoning into the next prompt.
            if (thinking && string.IsNullOrWhiteSpace(firstSnapshot.AnswerText)
                && StopReason is GenerationStopReason.EndOfMessage or GenerationStopReason.MaximumTokens)
            {
                RetriedWithoutThinking = true;
                int retryGeneration = ++_generation;
                firstThought = firstSnapshot.ThinkingText;
                onSnapshot(firstSnapshot);
                onRetry();
                var directParser = new ThinkingStreamParser(requestedThinking: false);
                StopReason = null;
                LocalGeneration directResult = await generate(false, chunk =>
                {
                    directStreamedRaw.Append(chunk);
                    if (retryGeneration != _generation) return;
                    ThinkingStreamSnapshot direct = directParser.Append(chunk);
                    onSnapshot(new ThinkingStreamSnapshot(firstThought,
                        direct.AnswerText, firstThought.Length != 0, false));
                }, ct);
                StopReason = directResult.StopReason;
                var directFinalParser = new ThinkingStreamParser(requestedThinking: false);
                directFinalParser.Append(directResult.Text);
                completedSnapshot = CombineRetrySnapshot(firstThought, directFinalParser.Complete(
                    StopReason == GenerationStopReason.EndOfMessage));
            }
            Completed = true;
        }
        finally
        {
            ++_generation; // Ignore callbacks arriving after the final text is settled.
            if (RetriedWithoutThinking && !Completed)
            {
                var directFinalParser = new ThinkingStreamParser(requestedThinking: false);
                directFinalParser.Append(directStreamedRaw.ToString());
                completedSnapshot = CombineRetrySnapshot(firstThought, directFinalParser.Complete(false));
            }
            else if (completedSnapshot is null)
            {
                var finalParser = new ThinkingStreamParser(thinking);
                finalParser.Append(finalRaw ?? streamedRaw.ToString());
                completedSnapshot = finalParser.Complete(StopReason == GenerationStopReason.EndOfMessage);
            }
            Snapshot = completedSnapshot.Value;
        }
    }

    private static ThinkingStreamSnapshot CombineRetrySnapshot(string firstThought, ThinkingStreamSnapshot direct)
    {
        string combinedThought = string.IsNullOrWhiteSpace(direct.ThinkingText)
            ? firstThought : string.IsNullOrWhiteSpace(firstThought)
                ? direct.ThinkingText : firstThought + "\n\n" + direct.ThinkingText;
        return new ThinkingStreamSnapshot(combinedThought, direct.AnswerText,
            !string.IsNullOrWhiteSpace(combinedThought), false);
    }
}
