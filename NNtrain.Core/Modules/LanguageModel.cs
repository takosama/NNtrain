namespace NNtrain;

/// <summary>Base module for token language models.</summary>
public abstract class LanguageModel : Module
{
    protected LanguageModel(TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
    }

    public abstract int VocabularySize { get; }
    public abstract int ContextLength { get; }
    public abstract int ModelWidth { get; }
    public abstract IReadOnlyList<Parameter> HiddenWeightParameters { get; }
    public abstract IReadOnlyList<Parameter> AuxiliaryParameters { get; }

    internal abstract Tensor Forward(
        int[] tokenIds,
        int batchSize,
        int sequenceLength);

    public Tensor forward(
        int[] input_ids,
        int batch_size,
        int sequence_length)
        => Forward(input_ids, batch_size, sequence_length);

    internal abstract int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random);

    internal virtual int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random,
        Action<int>? onToken)
    {
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        int[] prompt = promptTokenIds.ToArray();
        int[] generated = GenerateTokenIds(
            prompt,
            maxNewTokens,
            temperature,
            topK,
            stopTokenId,
            random);
        if (onToken is not null)
        {
            for (int index = prompt.Length; index < generated.Length; index++)
                onToken(generated[index]);
        }
        return generated;
    }

    internal abstract string Generate(
        string prompt,
        BpeTokenizer tokenizer,
        int maxNewTokens,
        float temperature,
        int topK,
        Random? random);

    public int[] generate_token_ids(
        IEnumerable<int> prompt_token_ids,
        int max_new_tokens,
        float temperature,
        int top_k,
        int? stop_token_id,
        Random? random,
        Action<int>? on_token = null)
        => GenerateTokenIds(
            prompt_token_ids,
            max_new_tokens,
            temperature,
            top_k,
            stop_token_id,
            random,
            on_token);

    public string generate(
        string prompt,
        BpeTokenizer tokenizer,
        int max_new_tokens,
        float temperature,
        int top_k,
        Random? random)
        => Generate(
            prompt,
            tokenizer,
            max_new_tokens,
            temperature,
            top_k,
            random);
}
