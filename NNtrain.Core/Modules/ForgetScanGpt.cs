namespace NNtrain;

/// <summary>
/// Decoder-only language model using content-dependent forgetful state scans.
/// </summary>
public sealed class ForgetScanGpt : Module, IWikiLanguageModel
{
    private readonly Parameter _tokenEmbedding;
    private readonly Dropout _embeddingDropout;
    private readonly ForgetScanBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _languageModelHead;
    private readonly Parameter[] _hiddenWeightParameters;
    private readonly Parameter[] _auxiliaryParameters;

    public ForgetScanGpt(
        int vocabularySize,
        int contextLength,
        int modelWidth,
        int hiddenWidth,
        int numLayers,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f)
    {
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (contextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (modelWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(modelWidth));
        if (hiddenWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(hiddenWidth));
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (!float.IsFinite(initializationScale)
            || initializationScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(initializationScale));
        }
        if (!float.IsFinite(dropout) || dropout < 0f || dropout >= 1f)
            throw new ArgumentOutOfRangeException(nameof(dropout));

        VocabularySize = vocabularySize;
        ContextLength = contextLength;
        ModelWidth = modelWidth;
        random ??= new Random(1);

        _tokenEmbedding = RegisterParameter(
            CreateEmbedding(
                vocabularySize,
                modelWidth,
                random,
                initializationScale));
        _embeddingDropout = RegisterModule(new Dropout(dropout, random));
        _blocks = new ForgetScanBlock[numLayers];
        for (int layer = 0; layer < numLayers; layer++)
        {
            _blocks[layer] = RegisterModule(
                new ForgetScanBlock(
                    modelWidth,
                    hiddenWidth,
                    random,
                    initializationScale,
                    dropout));
        }
        _finalNorm = RegisterModule(new LayerNorm(modelWidth));
        _languageModelHead = RegisterModule(
            new Linear(
                modelWidth,
                vocabularySize,
                random,
                initializationScale));

        _hiddenWeightParameters = _blocks
            .SelectMany(block => block.Parameters())
            .Where(parameter => parameter.T.Rank >= 2)
            .ToArray();
        var hiddenWeightSet = new HashSet<Parameter>(
            _hiddenWeightParameters,
            ReferenceEqualityComparer.Instance);
        _auxiliaryParameters = Parameters()
            .Where(parameter => !hiddenWeightSet.Contains(parameter))
            .ToArray();
    }

    public int VocabularySize { get; }

    public int ContextLength { get; }

    public int ModelWidth { get; }

    public IReadOnlyList<Parameter> HiddenWeightParameters
        => Array.AsReadOnly(_hiddenWeightParameters);

    public IReadOnlyList<Parameter> AuxiliaryParameters
        => Array.AsReadOnly(_auxiliaryParameters);

    public Tensor Forward(
        int[] tokenIds,
        int batchSize,
        int sequenceLength)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        if (batchSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(batchSize));
        if (sequenceLength <= 0 || sequenceLength > ContextLength)
        {
            throw new ArgumentOutOfRangeException(
                nameof(sequenceLength),
                sequenceLength,
                $"Sequence length must be between 1 and {ContextLength}.");
        }
        if (tokenIds.Length != checked(batchSize * sequenceLength))
        {
            throw new ArgumentException(
                "Token count must equal batchSize * sequenceLength.",
                nameof(tokenIds));
        }

        Tensor hidden = _embeddingDropout.Forward(
            _tokenEmbedding.T.EmbeddingLookup(
                tokenIds,
                batchSize,
                sequenceLength));
        foreach (ForgetScanBlock block in _blocks)
            hidden = block.Forward(hidden);
        hidden = _finalNorm.Forward(hidden);
        return _languageModelHead.ForwardBatch(
            hidden.Reshape(batchSize * sequenceLength, ModelWidth));
    }

    public int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature = 0.8f,
        int topK = 40,
        int? stopTokenId = BpeTokenizer.EosTokenId,
        Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        if (maxNewTokens < 0)
            throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
        if (!float.IsFinite(temperature) || temperature < 0f)
            throw new ArgumentOutOfRangeException(nameof(temperature));
        if (topK < 0)
            throw new ArgumentOutOfRangeException(nameof(topK));
        if (stopTokenId.HasValue
            && (uint)stopTokenId.Value >= (uint)VocabularySize)
        {
            throw new ArgumentOutOfRangeException(nameof(stopTokenId));
        }

        var result = promptTokenIds.ToList();
        if (result.Count == 0)
        {
            throw new ArgumentException(
                "At least one prompt token is required.",
                nameof(promptTokenIds));
        }
        if (result.Any(token => (uint)token >= (uint)VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(promptTokenIds));

        random ??= new Random();
        bool wasTraining = IsTraining;
        Eval();
        try
        {
            using (AutogradContext.NoGrad())
            {
                for (int generated = 0; generated < maxNewTokens; generated++)
                {
                    int sequenceLength = Math.Min(ContextLength, result.Count);
                    int[] context = result
                        .Skip(result.Count - sequenceLength)
                        .ToArray();
                    Tensor logits = Forward(context, 1, sequenceLength);
                    int offset = (sequenceLength - 1) * VocabularySize;
                    int nextToken = Sample(
                        logits.Data,
                        offset,
                        VocabularySize,
                        temperature,
                        topK,
                        random);
                    result.Add(nextToken);
                    if (stopTokenId.HasValue && nextToken == stopTokenId.Value)
                        break;
                }
            }
        }
        finally
        {
            if (wasTraining)
                Train();
        }

        return result.ToArray();
    }

    public string Generate(
        string prompt,
        BpeTokenizer tokenizer,
        int maxNewTokens,
        float temperature = 0.8f,
        int topK = 40,
        Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tokenizer.VocabularySize != VocabularySize)
        {
            throw new ArgumentException(
                "Tokenizer vocabulary size does not match the model.",
                nameof(tokenizer));
        }

        int[] promptIds = tokenizer.Encode(prompt, addBos: true);
        int[] generated = GenerateTokenIds(
            promptIds,
            maxNewTokens,
            temperature,
            topK,
            BpeTokenizer.EosTokenId,
            random);
        return tokenizer.Decode(generated);
    }

    private static Parameter CreateEmbedding(
        int rows,
        int width,
        Random random,
        float scale)
    {
        var values = new float[checked(rows * width)];
        for (int index = 0; index < values.Length; index++)
            values[index] = ((float)random.NextDouble() * 2f - 1f) * scale;
        return new Parameter(
            values,
            [rows, width],
            "TokenEmbedding",
            WeightDecayPolicy.Apply);
    }

    private static int Sample(
        IReadOnlyList<float> logits,
        int offset,
        int count,
        float temperature,
        int topK,
        Random random)
    {
        if (temperature == 0f || topK == 1)
            return ArgMax(logits, offset, count);

        int candidateCount = topK == 0 ? count : Math.Min(topK, count);
        int[] candidates = Enumerable.Range(0, count)
            .OrderByDescending(index => logits[offset + index])
            .Take(candidateCount)
            .ToArray();
        float maximum = candidates.Max(index => logits[offset + index]);
        var weights = new float[candidateCount];
        float sum = 0f;
        for (int index = 0; index < candidateCount; index++)
        {
            float weight = MathF.Exp(
                (logits[offset + candidates[index]] - maximum)
                / temperature);
            weights[index] = weight;
            sum += weight;
        }

        double threshold = random.NextDouble() * sum;
        float cumulative = 0f;
        for (int index = 0; index < candidateCount; index++)
        {
            cumulative += weights[index];
            if (threshold <= cumulative)
                return candidates[index];
        }
        return candidates[^1];
    }

    private static int ArgMax(
        IReadOnlyList<float> values,
        int offset,
        int count)
    {
        int result = 0;
        float maximum = values[offset];
        for (int index = 1; index < count; index++)
        {
            float value = values[offset + index];
            if (value > maximum)
            {
                maximum = value;
                result = index;
            }
        }
        return result;
    }
}
