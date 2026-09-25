namespace NNtrain;

/// <summary>
/// Qwen2-family decoder-only causal LM.  The first implementation targets
/// correctness on Intel Arc; the operation boundaries are intentionally kept
/// independent from the existing GPT MHA path.
/// </summary>
public sealed class Qwen2ForCausalLM : LanguageModel
{
    private readonly Parameter _tokenEmbedding;
    private readonly QwenBlock[] _blocks;
    private readonly QwenRmsNorm _finalNorm;
    private readonly Linear _lmHead;
    private readonly Parameter[] _hiddenWeightParameters;
    private readonly Parameter[] _auxiliaryParameters;
    private LoraAdapterSet? _loraAdapters;

    public Qwen2ForCausalLM(
        int vocabularySize,
        int contextLength,
        int modelWidth,
        int queryHeads,
        int kvHeads,
        int hiddenWidth,
        int numLayers,
        float rmsEpsilon = 1e-6f,
        float ropeTheta = 1_000_000f,
        Random? random = null,
        float initializationScale = 0.02f,
        TensorDType dtype = TensorDType.Float32)
        : base(dtype)
    {
        if (vocabularySize <= 0) throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (contextLength <= 0) throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (modelWidth <= 0) throw new ArgumentOutOfRangeException(nameof(modelWidth));
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
            throw new ArgumentException("Query heads must be a positive multiple of KV heads.");
        if (modelWidth % queryHeads != 0)
            throw new ArgumentException("modelWidth must be divisible by queryHeads.");
        if (hiddenWidth <= 0) throw new ArgumentOutOfRangeException(nameof(hiddenWidth));
        if (numLayers <= 0) throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (!(rmsEpsilon > 0f) || !float.IsFinite(rmsEpsilon))
            throw new ArgumentOutOfRangeException(nameof(rmsEpsilon));
        if (!(ropeTheta > 0f) || !float.IsFinite(ropeTheta))
            throw new ArgumentOutOfRangeException(nameof(ropeTheta));

        VocabularySize = vocabularySize;
        ContextLength = contextLength;
        ModelWidth = modelWidth;
        QueryHeads = queryHeads;
        KvHeads = kvHeads;
        HiddenWidth = hiddenWidth;
        RmsEpsilon = rmsEpsilon;
        RopeTheta = ropeTheta;

        random ??= new Random(1);
        _tokenEmbedding = RegisterParameter(CreateEmbedding(
            vocabularySize, modelWidth, "token_embd.weight",
            random, initializationScale, dtype));
        _blocks = new QwenBlock[numLayers];
        for (int i = 0; i < numLayers; ++i)
        {
            _blocks[i] = RegisterModule(new QwenBlock(
                modelWidth, queryHeads, kvHeads, hiddenWidth,
                rmsEpsilon, ropeTheta, random, initializationScale, dtype));
        }
        _finalNorm = RegisterModule(new QwenRmsNorm(modelWidth, rmsEpsilon, dtype));
        _lmHead = RegisterModule(new Linear(
            modelWidth, vocabularySize, random, initializationScale, dtype));

        _hiddenWeightParameters = _blocks
            .SelectMany(block => block.Parameters())
            .Where(parameter => parameter.T.Rank >= 2)
            .Concat([_lmHead.W])
            .ToArray();
        var hidden = new HashSet<Parameter>(
            _hiddenWeightParameters, ReferenceEqualityComparer.Instance);
        _auxiliaryParameters = Parameters()
            .Where(parameter => !hidden.Contains(parameter))
            .ToArray();
    }

    public override int VocabularySize { get; }
    public override int ContextLength { get; }
    public override int ModelWidth { get; }
    public int QueryHeads { get; }
    public int KvHeads { get; }
    public int HiddenWidth { get; }
    public float RmsEpsilon { get; }
    public float RopeTheta { get; }
    internal Parameter TokenEmbedding => _tokenEmbedding;
    internal IReadOnlyList<QwenBlock> Blocks => _blocks;
    internal QwenRmsNorm FinalNorm => _finalNorm;
    internal Linear LmHead => _lmHead;

    public override IReadOnlyList<Parameter> HiddenWeightParameters
        => Array.AsReadOnly(_hiddenWeightParameters);
    public override IReadOnlyList<Parameter> AuxiliaryParameters
        => Array.AsReadOnly(_auxiliaryParameters);

    public LoraAdapterSet AttachLora(
        int rank = 8,
        float alpha = 16f,
        string[]? targets = null,
        int seed = 1234)
    {
        if (_loraAdapters is not null)
            throw new InvalidOperationException("LoRA is already attached.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(rank);
        if (!(alpha > 0f) || !float.IsFinite(alpha))
            throw new ArgumentOutOfRangeException(nameof(alpha));

        var selected = new HashSet<string>(
            targets ?? ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            StringComparer.Ordinal);
        string[] allowed =
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"];
        if (selected.Count == 0 || selected.Any(t => !allowed.Contains(t, StringComparer.Ordinal)))
            throw new ArgumentException("Unknown or empty Qwen LoRA target list.", nameof(targets));

        var adapters = new LoraAdapterSet(DType);
        var random = new Random(seed);
        for (int i = 0; i < _blocks.Length; ++i)
        {
            _blocks[i].SetBaseFrozenForLora(true);
            _blocks[i].AttachLora(adapters, i, rank, alpha, selected, random);
        }
        _lmHead.FrozenForLora = true;
        adapters.SetPrecisionMode(PrecisionMode);
        _loraAdapters = adapters;
        return adapters;
    }

    internal override Tensor Forward(
        int[] tokenIds,
        int batchSize,
        int sequenceLength)
    {
        Tensor hidden = ForwardHidden(tokenIds, batchSize, sequenceLength);
        return _lmHead.ForwardBatch(
            hidden.Reshape(batchSize * sequenceLength, ModelWidth));
    }

    internal override Tensor ForwardLoss(
        int[] tokenIds,
        int[] targetIds,
        int batchSize,
        int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
    {
        ArgumentNullException.ThrowIfNull(targetIds);
        if (targetIds.Length != checked(batchSize * sequenceLength))
            throw new ArgumentException("Target count must equal batchSize * sequenceLength.", nameof(targetIds));

        Tensor hidden = ForwardHidden(tokenIds, batchSize, sequenceLength);
        Tensor logits = _lmHead.ForwardBatch(hidden);
        return logits.CrossEntropyWithLogits(targetIds, ignoreIndex: ignoreIndex);
    }

    private Tensor ForwardHidden(int[] tokenIds, int batchSize, int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (sequenceLength <= 0 || sequenceLength > ContextLength)
            throw new ArgumentOutOfRangeException(nameof(sequenceLength));
        if (tokenIds.Length != checked(batchSize * sequenceLength))
            throw new ArgumentException("Token count must equal batchSize * sequenceLength.", nameof(tokenIds));
        if (tokenIds.Any(token => (uint)token >= (uint)VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(tokenIds));

        Tensor hidden = _tokenEmbedding.T.EmbeddingLookup(
            tokenIds, batchSize, sequenceLength);
        foreach (QwenBlock block in _blocks)
            hidden = block.Forward(hidden);
        return _finalNorm.Forward(hidden);
    }

    internal override int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random)
    {
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        if (maxNewTokens < 0) throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
        var result = promptTokenIds.ToList();
        if (result.Count == 0) throw new ArgumentException("Prompt cannot be empty.", nameof(promptTokenIds));
        if (result.Count > ContextLength) throw new ArgumentOutOfRangeException(nameof(promptTokenIds));
        random ??= Random.Shared;

        bool wasTraining = IsTraining;
        Eval();
        try
        {
            using (AutogradContext.NoGrad())
            {
                for (int generated = 0;
                     generated < maxNewTokens && result.Count < ContextLength;
                     ++generated)
                {
                    // The logits must be sampled while their Arc buffers are
                    // alive; release every detached activation after this token.
                    using IDisposable? arcInference = Tensor.BeginArcInferenceFrame();
                    int sequence = result.Count;
                    Tensor logits = Forward(result.ToArray(), 1, sequence);
                    int offset = checked((sequence - 1) * VocabularySize);
                    int next = SampleLogits(
                        logits, offset, VocabularySize, temperature, topK, random);
                    result.Add(next);
                    if (stopTokenId.HasValue && next == stopTokenId.Value) break;
                }
            }
        }
        finally
        {
            if (wasTraining) Train();
        }
        return result.ToArray();
    }

    internal override string Generate(
        string prompt,
        BpeTokenizer tokenizer,
        int maxNewTokens,
        float temperature,
        int topK,
        Random? random)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tokenizer.VocabularySize != VocabularySize)
            throw new ArgumentException("Tokenizer vocabulary does not match Qwen model.", nameof(tokenizer));
        int[] ids = tokenizer.Encode(prompt, addBos: false);
        return tokenizer.Decode(GenerateTokenIds(
            ids, maxNewTokens, temperature, topK, BpeTokenizer.EosTokenId, random));
    }

    private static Parameter CreateEmbedding(
        int rows, int width, string name, Random random,
        float scale, TensorDType dtype)
    {
        var values = new float[checked(rows * width)];
        for (int i = 0; i < values.Length; ++i)
            values[i] = ((float)random.NextDouble() * 2f - 1f) * scale;
        return new Parameter(values, [rows, width], name, WeightDecayPolicy.Apply, dtype);
    }
}
