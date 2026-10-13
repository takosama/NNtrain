namespace NNtrain.Gui;

public sealed record GenerationSampling(float Temperature, float TopP, int TopK);
public enum GenerationStopReason { EndOfMessage, MaximumTokens, ContextLimit }
public sealed record GenerationStats(int PromptTokens, int CompletionTokens, int ReusedPromptTokens,
    double? FirstTokenMilliseconds, int? FirstTokenId, GenerationStopReason StopReason);
public sealed record PromptPrimeStats(int PromptTokens, int ReusedPromptTokens, bool Cached);
public sealed record ImagePreparationStats(int GridWidth, int GridHeight, bool Cached,
    int PrimedPromptTokens = 0, int ReusedPromptTokens = 0, bool PrefixCached = false,
    double PrefixPreparationMilliseconds = 0);
