namespace NNtrain;

/// <summary>
/// A compact decoder-only Transformer for Japanese Wikipedia text.
/// </summary>
public sealed partial class GptRinWikiJp : LanguageModel
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

    internal ArcTransformerMemoryPlan? LastArcMemoryPlan { get; private set; }

    /// <summary>Use the two Arc lanes in the current inference session.</summary>
    internal bool ArcTensorParallelEnabled
    {
        get => _arcTensorParallelEnabled;
        set
        {
            if (_arcTensorParallelEnabled == value) return;
            if (!value) ReleaseArcTensorParallelShards();
            _arcTensorParallelFullWeightsReleased = false;
            _arcTensorParallelEnabled = value;
        }
    }

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
        using IDisposable? arcPrimaryScope = ArcTensorParallelEnabled
            ? PushArcTensorParallelPrimary()
            : null;
        if (ArcTensorParallelEnabled && batchSize != 1)
            throw new NotSupportedException("Arc tensor-parallel inference requires batch size 1.");
        Tensor hidden = ArcTensorParallelEnabled
            ? ForwardHiddenArcTensorParallel(tokenIds, sequenceLength)
            : ForwardHidden(tokenIds, batchSize, sequenceLength);
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
        if (Tensor.ExecutionDevice == TensorDevice.Arc && Tensor.ArcLane.Options.ChunkedLossHead)
            return hidden.ArcLinearCrossEntropy(_languageModelHead.W.T, _languageModelHead.B.T, targetIds, ignoreIndex);
        Tensor logits = (Tensor.ExecutionDevice is TensorDevice.Cuda or TensorDevice.Arc)
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
        int checkpointPrefix = 0;
        bool checkpointFfn = false;
        LastArcMemoryPlan = null;
        if (IsTraining && AutogradContext.IsRecordingEnabled && Tensor.ArcResident)
        {
            var options = Tensor.ArcLane.Options;
            checkpointPrefix = options.TransformerCheckpointing
                ? Math.Min(_blocks.Length, options.TransformerCheckpointLayers) : 0;
            checkpointFfn = options.TransformerFfnCheckpointing;
            // Explicit A/B policy always wins, including an explicit empty
            // prefix. Auto may change graph retention only, never batch/LR or
            // storage/compute precision.
            if (options.AutomaticTransformerMemoryPlan
                && !options.TransformerCheckpointing && !options.TransformerFfnCheckpointing)
            {
                Parameter[] parameters = Parameters().ToArray();
                var policy = TensorExecutionContext.ActivePrecisionPolicy
                    ?? throw new InvalidOperationException("Arc memory planning requires an active precision policy.");
                bool bf16Activations = options.Mix8_16Bf16Activations
                    && policy.Mode == NNtrain.Runtime.Execution.PrecisionMode.Mix8_16
                    && policy.NonWeightSelection == NNtrain.Runtime.Execution.NonWeightSelectionPolicy.FastestAvailable
                    && (policy.AllowedActivationStorageFormats & NNtrain.Runtime.Execution.NumericFormatSet.BFloat16) != 0;
                int fusedBfp8LinearWidth = 0;
                bool fusedBfp8Ffn = false;
                if (bf16Activations)
                {
                    int rows = checked(batchSize * sequenceLength);
                    var lane = Tensor.ArcLane;
                    bool FusedOutput(Linear linear, bool relu)
                    {
                        Tensor weight = linear.W.T, bias = linear.B.T;
                        int n = weight.Shape[0], k = weight.Shape[1];
                        var descriptor = weight.Bfp8Quantization;
                        return options.FusedBfp8Linear && options.PackedMatrixStorage
                            && rows >= 4096 && n % 32 == 0 && k <= 2048
                            && weight.DType == TensorDType.Bfp8 && bias.DType == TensorDType.Bfp8
                            && descriptor is not null && descriptor == bias.Bfp8Quantization
                            && descriptor == _tokenEmbedding.T.Bfp8Quantization
                            && descriptor.GetEffectiveBlockSize(checked(rows * n)) == 32
                            && ArcXmxStorageOperand.CanRunAny(lane, rows, n, k,
                                tb: true, hasBias: true, relu: relu);
                    }
                    TransformerBlock block = _blocks[0];
                    if (FusedOutput(block.Attn.Qkv, false)) fusedBfp8LinearWidth += 3 * ModelWidth;
                    if (FusedOutput(block.Attn.Wo, false)) fusedBfp8LinearWidth += ModelWidth;
                    fusedBfp8Ffn = FusedOutput(block.Ffn.Fc1, true);
                    if (fusedBfp8Ffn) fusedBfp8LinearWidth += block.Ffn.Fc1.W.T.Shape[0];
                    if (FusedOutput(block.Ffn.Fc2, false)) fusedBfp8LinearWidth += ModelWidth;
                }
                ArcTransformerMemoryPlan plan = ArcTransformerMemoryPlan.Create(
                    batchSize, sequenceLength, ModelWidth, _blocks[0].Attn.NumHeads,
                    _blocks[0].Ffn.Fc1.W.T.Shape[0], _blocks.Length,
                    _tokenEmbedding.T.DType,
                    policy.Mode,
                    _tokenEmbedding.T.Bfp8Quantization?.BlockSize ?? 0,
                    parameters.Sum(parameter => (long)parameter.T.Numel),
                    parameters.Sum(parameter => (long)parameter.T.StorageByteLength),
                    parameters.Max(parameter => (long)parameter.T.Numel),
                    parameters.Select(parameter => (long)parameter.T.Numel)
                        .OrderByDescending(elements => elements).Take(2).Sum(),
                    HiddenWeightParameters.Select(parameter =>
                    {
                        long elements = parameter.T.Numel;
                        long rows = parameter.T.Rank >= 2 ? parameter.T.Shape[0] : 1;
                        long columns = elements / rows;
                        long gram = Math.Min(rows, columns);
                        return checked(16 * elements + 12 * gram * gram);
                    }).DefaultIfEmpty(0).Max(),
                    Tensor.ArcLane.Device.GlobalMemoryBytes,
                    retainAttentionOutputBFloat16: Tensor.ArcUseAttentionRowDelta(
                        sequenceLength, ModelWidth / _blocks[0].Attn.NumHeads, causal: true),
                    publishBFloat16Activations: bf16Activations,
                    fusedBfp8LinearOutputWidthPerLayer: fusedBfp8LinearWidth,
                    fusedBfp8FfnIntermediate: fusedBfp8Ffn);
                LastArcMemoryPlan = plan;
                checkpointPrefix = plan.CheckpointPrefixLayers;
                checkpointFfn = plan.CheckpointFfn;
            }
        }
        for (int layer = 0; layer < _blocks.Length; layer++)
        {
            TransformerBlock block = _blocks[layer];
            hidden = layer < checkpointPrefix
                ? hidden.ArcCheckpoint(block.CaptureArcCheckpointForward(checkpointFfn),
                    block.Parameters().Select(parameter => parameter.T).ToArray())
                : block.ForwardArcPlanned(hidden, checkpointFfn);
        }
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
        => GenerateTokenIds(promptTokenIds, maxNewTokens, temperature, topK,
            stopTokenId, random, null);

    internal override int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature,
        int topK,
        int? stopTokenId,
        Random? random,
        Action<int>? onToken)
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

        using IDisposable? arcPrimaryScope = ArcTensorParallelEnabled
            ? PushArcTensorParallelPrimary()
            : null;
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
                if (CanUseArcKvCache()
                    && result.Count <= ContextLength && maxNewTokens > 0)
                    (generated, stopped) = GenerateArcCached(result, maxNewTokens,
                        temperature, topK, stopTokenId, random, onToken);
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
                            onToken?.Invoke(nextToken);
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
                            onToken?.Invoke(nextToken);
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
                    using IDisposable? arcInference = Tensor.BeginArcInferenceFrame();
                    using CudaInferenceScope inferenceScope =
                        CudaInferenceScope.Begin();
                    int sequenceLength = Math.Min(ContextLength, result.Count);
                    int[] context = result
                        .Skip(result.Count - sequenceLength)
                        .ToArray();
                    Tensor hidden = ArcTensorParallelEnabled
                        ? ForwardHiddenArcTensorParallel(context, sequenceLength)
                        : ForwardHidden(context, 1, sequenceLength);
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
                    onToken?.Invoke(nextToken);
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
