namespace NNtrain;

/// <summary>
/// Common training and generation contract for Wikipedia language models.
/// </summary>
public interface IWikiLanguageModel
{
    int VocabularySize { get; }

    int ContextLength { get; }

    int ModelWidth { get; }

    bool IsTraining { get; }

    IReadOnlyList<Parameter> HiddenWeightParameters { get; }

    IReadOnlyList<Parameter> AuxiliaryParameters { get; }

    /// <summary>
    /// Groups parameters by module depth for GainShareAdamW. Every Wikipedia
    /// model derives from <see cref="Module"/>, whose public method with the
    /// same signature satisfies this member implicitly.
    /// </summary>
    IReadOnlyList<IReadOnlyList<Parameter>> MakeGainShareParameterGroups(
        int blockDepth = 1);

    Tensor Forward(int[] tokenIds, int batchSize, int sequenceLength);

    Tensor forward(int[] input_ids, int batch_size, int sequence_length)
        => Forward(input_ids, batch_size, sequence_length);

    int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random);

    /// <summary>
    /// Generates tokens, invoking <paramref name="onToken"/> as each one is
    /// sampled so a caller can display the text while it is produced.
    /// </summary>
    /// <remarks>
    /// Models that have not implemented streaming fall back to generating
    /// everything first and then replaying the tokens through the callback,
    /// which keeps the output identical but not incremental.
    /// </remarks>
    int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random,
        Action<int>? onToken)
    {
        int[] generated = GenerateTokenIds(
            promptTokenIds,
            maxNewTokens,
            temperature,
            topK,
            stopTokenId,
            random);
        if (onToken is not null)
        {
            int promptLength = promptTokenIds.Count();
            for (int index = promptLength; index < generated.Length; index++)
                onToken(generated[index]);
        }
        return generated;
    }

    string Generate(
        string prompt,
        BpeTokenizer tokenizer,
        int maxNewTokens,
        float temperature,
        int topK,
        Random? random);

    IEnumerable<Parameter> Parameters();

    ModuleState CaptureState();

    void RestoreState(ModuleState state);

    void Train();

    void Eval();

    IWikiLanguageModel train()
    {
        Train();
        return this;
    }

    IWikiLanguageModel eval()
    {
        Eval();
        return this;
    }

    IEnumerable<Parameter> parameters() => Parameters();

    ModuleState state_dict() => CaptureState();

    void load_state_dict(ModuleState state) => RestoreState(state);
}
