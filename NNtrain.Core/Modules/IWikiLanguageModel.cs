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

    Tensor Forward(int[] tokenIds, int batchSize, int sequenceLength);

    int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random);

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
