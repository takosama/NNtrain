namespace NNtrain;

/// <summary>
/// A compact decoder-only Transformer for Japanese Wikipedia text.
/// </summary>
public sealed class GptRinWikiJp : LanguageModel
{
    private readonly Parameter _tokenEmbedding;
    private readonly Parameter _positionEmbedding;
    private readonly Dropout _embeddingDropout;
    private readonly TransformerBlock[] _blocks;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _languageModelHead;
    private readonly Parameter[] _hiddenWeightParameters;
    private readonly Parameter[] _auxiliaryParameters;

    public GptRinWikiJp(
        int vocabularySize,
        int contextLength,
        int dModel,
        int numHeads,
        int dHidden,
        int numLayers,
        Random? rng = null,
        float initializationScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float32,
        bool tieWordEmbeddings = false)
        : base(dtype)
    {
        if (vocabularySize <= 0)
            throw new ArgumentOutOfRangeException(nameof(vocabularySize));
        if (contextLength <= 0)
            throw new ArgumentOutOfRangeException(nameof(contextLength));
        if (dModel <= 0)
            throw new ArgumentOutOfRangeException(nameof(dModel));
        if (numHeads <= 0 || dModel % numHeads != 0)
        {
            throw new ArgumentException(
                "Head count must be positive and evenly divide dModel.",
                nameof(numHeads));
        }
        if (dHidden <= 0)
            throw new ArgumentOutOfRangeException(nameof(dHidden));
        if (numLayers <= 0)
            throw new ArgumentOutOfRangeException(nameof(numLayers));
        if (!float.IsFinite(initializationScale) || initializationScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(initializationScale));
        }
        if (!float.IsFinite(dropout) || dropout < 0f || dropout >= 1f)
            throw new ArgumentOutOfRangeException(nameof(dropout));

        VocabularySize = vocabularySize;
        ContextLength = contextLength;
        ModelWidth = dModel;
        rng ??= new Random(1);

        _tokenEmbedding = RegisterParameter(
            CreateEmbedding(
                vocabularySize,
                dModel,
                "TokenEmbedding",
                rng,
                initializationScale,
                dtype));
        _positionEmbedding = RegisterParameter(
            CreateEmbedding(
                contextLength,
                dModel,
                "PositionEmbedding",
                rng,
                initializationScale,
                dtype));
        _embeddingDropout = RegisterModule(new Dropout(dropout, rng, dtype));
        _blocks = new TransformerBlock[numLayers];
        for (int layer = 0; layer < numLayers; layer++)
        {
            _blocks[layer] = RegisterModule(
                new TransformerBlock(
                    dModel,
                    numHeads,
                    dHidden,
                    causal: true,
                    rng,
                    initializationScale,
                    dropout,
                    dtype));
        }
        _finalNorm = RegisterModule(new LayerNorm(dModel, dtype: dtype));
        _languageModelHead = RegisterModule(
            tieWordEmbeddings
                ? new Linear(_tokenEmbedding, dModel, vocabularySize)
                : new Linear(
                    dModel, vocabularySize, rng, initializationScale, dtype));

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

    public override int VocabularySize { get; }

    public override int ContextLength { get; }

    public override int ModelWidth { get; }

    /// <summary>
    /// Transformer matrix weights updated by NekoMuon.
    /// </summary>
    public override IReadOnlyList<Parameter> HiddenWeightParameters
        => Array.AsReadOnly(_hiddenWeightParameters);

    /// <summary>
    /// Embeddings, normalization parameters, biases, and language-model head
    /// updated by the auxiliary AdamW optimizer.
    /// </summary>
    public override IReadOnlyList<Parameter> AuxiliaryParameters
        => Array.AsReadOnly(_auxiliaryParameters);

    /// <summary>
    /// Returns flattened next-token logits with shape
    /// [batchSize * sequenceLength, vocabularySize].
    /// </summary>
    internal override Tensor Forward(
        int[] tokenIds,
        int batchSize,
        int sequenceLength)
    {
        Tensor hidden = ForwardHidden(tokenIds, batchSize, sequenceLength);
        return _languageModelHead.ForwardBatch(
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
        {
            throw new ArgumentException(
                "Target count must equal batchSize * sequenceLength.",
                nameof(targetIds));
        }

        Tensor hidden = ForwardHidden(tokenIds, batchSize, sequenceLength);
        Tensor logits = Tensor.ExecutionDevice == TensorDevice.Cuda
                && hidden.DType == TensorDType.Bfp8
                && _languageModelHead.W.T.DType == TensorDType.Bfp8
                && _languageModelHead.B.T.DType == TensorDType.Bfp8
            ? hidden.LinearLastDimBFloat16ForLoss(
                _languageModelHead.W.T,
                _languageModelHead.B.T)
            : _languageModelHead.ForwardBatch(hidden);
        return logits.CrossEntropyWithLogits(
            targetIds,
            ignoreIndex: ignoreIndex);
    }

    private Tensor ForwardHidden(
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
            _tokenEmbedding.T.EmbeddingLookupWithPositions(
                _positionEmbedding.T,
                tokenIds,
                batchSize,
                sequenceLength));
        foreach (TransformerBlock block in _blocks)
            hidden = block.Forward(hidden);
        hidden = _finalNorm.Forward(hidden);

        return hidden;
    }

    private Tensor ForwardHiddenIncremental(
        int tokenId,
        int position,
        IReadOnlyList<CudaAttentionKvCache> caches)
    {
        Tensor token = _tokenEmbedding.T
            .EmbeddingLookup([tokenId], 1)
            .Reshape(1, 1, ModelWidth);
        Tensor positional = _positionEmbedding.T
            .EmbeddingLookup([position], 1)
            .Reshape(1, 1, ModelWidth);
        Tensor hidden = _embeddingDropout.Forward(token + positional);
        for (int layer = 0; layer < _blocks.Length; ++layer)
        {
            hidden = _blocks[layer].ForwardIncremental(
                hidden, caches[layer], position);
        }
        return _finalNorm.Forward(hidden);
    }

    private Tensor ForwardHiddenPrefill(
        int[] tokenIds,
        IReadOnlyList<CudaAttentionKvCache> caches)
    {
        int sequence = tokenIds.Length;
        Tensor hidden = _embeddingDropout.Forward(
            _tokenEmbedding.T.EmbeddingLookupWithPositions(
                _positionEmbedding.T,
                tokenIds,
                1,
                sequence));
        for (int layer = 0; layer < _blocks.Length; ++layer)
        {
            hidden = _blocks[layer].ForwardPrefill(
                hidden, caches[layer], sequence);
        }
        return _finalNorm.Forward(hidden);
    }

    /// <summary>
    /// Autoregressively samples token ids and returns the prompt plus generated
    /// continuation.
    /// </summary>
    internal override int[] GenerateTokenIds(
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
            throw new ArgumentException(
                "At least one prompt token is required.",
                nameof(promptTokenIds));
        if (result.Any(token => (uint)token >= (uint)VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(promptTokenIds));

        random ??= new Random();
        bool wasTraining = IsTraining;
        Eval();
        try
        {
            using (AutogradContext.NoGrad())
            using (CudaBfp8InferenceComputeScope.Begin(
                Tensor.ExecutionDevice == TensorDevice.Cuda
                && _tokenEmbedding.T.DType == TensorDType.Bfp8))
            using (CudaInferenceScope cacheSession = CudaInferenceScope.Begin(
                resetPool: true,
                clearPoolOnDispose: true))
            {
                int generated = 0;
                bool stopped = false;
                if (Tensor.ExecutionDevice == TensorDevice.Cuda
                    && DType is TensorDType.BFloat16 or TensorDType.Bfp8
                    && result.Count <= ContextLength
                    && maxNewTokens > 0
                    && !CudaDispatchPolicy.Current.DisableKvCache)
                {
                    CudaAttentionKvCache[] caches = _blocks
                        .Select(block => block.Attn.CreateIncrementalCache(
                            ContextLength))
                        .ToArray();
                    try
                    {
                        // Prefill the prompt with the normal tiled Tensor Core
                        // path, copying each layer's projected K/V into its
                        // persistent cache. Only generated tokens use the
                        // one-token path.
                        using (CudaInferenceScope prefillScope =
                            CudaInferenceScope.Begin())
                        {
                            Tensor hidden = ForwardHiddenPrefill(
                                result.ToArray(), caches);
                            Tensor logits = _languageModelHead.ForwardBatch(
                                hidden.SelectLastSequenceToken());
                            int nextToken = SampleLogits(
                                logits,
                                0,
                                VocabularySize,
                                temperature,
                                topK,
                                random);
                            result.Add(nextToken);
                            ++generated;
                            if (stopTokenId.HasValue
                                && nextToken == stopTokenId.Value)
                            {
                                stopped = true;
                            }
                        }

                        int position = result.Count - 1;
                        while (!stopped && generated < maxNewTokens
                            && position < ContextLength)
                        {
                            using CudaInferenceScope inferenceScope =
                                CudaInferenceScope.Begin();
                            Tensor hidden = ForwardHiddenIncremental(
                                result[^1], position, caches);
                            Tensor logits = _languageModelHead.ForwardBatch(
                                hidden.SelectLastSequenceToken());
                            int nextToken = SampleLogits(
                                logits,
                                0,
                                VocabularySize,
                                temperature,
                                topK,
                                random);
                            result.Add(nextToken);
                            ++generated;
                            ++position;
                            if (stopTokenId.HasValue
                                && nextToken == stopTokenId.Value)
                            {
                                stopped = true;
                                break;
                            }
                        }
                    }
                    finally
                    {
                        foreach (CudaAttentionKvCache cache in caches)
                            cache.Dispose();
                    }
                }

                // Once a sliding window has filled, absolute positional
                // embeddings change for every retained token. Fall back to a
                // correct full-window pass for only that remaining suffix.
                for (; !stopped && generated < maxNewTokens; ++generated)
                {
                    using CudaInferenceScope inferenceScope =
                        CudaInferenceScope.Begin();
                    int sequenceLength = Math.Min(ContextLength, result.Count);
                    int[] context = result
                        .Skip(result.Count - sequenceLength)
                        .ToArray();
                    Tensor hidden = ForwardHidden(context, 1, sequenceLength);
                    Tensor logits = _languageModelHead.ForwardBatch(
                        hidden.SelectLastSequenceToken());
                    const int offset = 0;
                    int nextToken = SampleLogits(
                        logits,
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

    /// <summary>
    /// Encodes a prompt, generates a continuation, and decodes it to text.
    /// </summary>
    internal override string Generate(
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
        string name,
        Random random,
        float scale,
        TensorDType dtype)
    {
        var values = new float[checked(rows * width)];
        for (int index = 0; index < values.Length; index++)
            values[index] = ((float)random.NextDouble() * 2f - 1f) * scale;
        return new Parameter(
            values,
            [rows, width],
            name,
            WeightDecayPolicy.Apply,
            dtype);
    }

}
