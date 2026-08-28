namespace NNtrain;

/// <summary>
/// Decoder-only language model using stable matrix-valued delta memories.
/// </summary>
public class ForgetMemoryV2Gpt : LanguageModel
{
    private readonly Parameter _tokenEmbedding;
    private readonly Dropout _embeddingDropout;
    private readonly ForgetMemoryV2Layer[] _layers;
    private readonly LayerNorm _finalNorm;
    private readonly Linear _languageModelHead;
    private readonly Parameter[] _hiddenWeightParameters;
    private readonly Parameter[] _auxiliaryParameters;

    public ForgetMemoryV2Gpt(
        int vocabularySize,
        int contextLength,
        int modelWidth,
        int hiddenWidth,
        int numLayers,
        int keyWidth = 16,
        int valueWidth = 16,
        float retentionMinimum = 0.5f,
        float retentionMaximum = 0.99f,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float16)
        : this(
            vocabularySize,
            contextLength,
            modelWidth,
            hiddenWidth,
            numLayers,
            keyWidth,
            valueWidth,
            retentionMinimum,
            retentionMaximum,
            random,
            initializationScale,
            dropout,
            dtype,
            useV3: false,
            useDrn: false)
    {
    }

    protected ForgetMemoryV2Gpt(
        int vocabularySize,
        int contextLength,
        int modelWidth,
        int hiddenWidth,
        int numLayers,
        int keyWidth,
        int valueWidth,
        float retentionMinimum,
        float retentionMaximum,
        Random? random,
        float initializationScale,
        float dropout,
        TensorDType dtype,
        bool useV3,
        bool useDrn)
        : base(dtype)
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
        if (keyWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(keyWidth));
        if (valueWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(valueWidth));
        ValidateRetentionRange(retentionMinimum, retentionMaximum);
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
        KeyWidth = keyWidth;
        ValueWidth = valueWidth;
        RetentionMinimum = retentionMinimum;
        RetentionMaximum = retentionMaximum;
        UseV3 = useV3;
        UseDrn = useDrn;
        random ??= new Random(1);

        _tokenEmbedding = RegisterParameter(
            CreateEmbedding(
                vocabularySize,
                modelWidth,
                random,
                initializationScale,
                dtype));
        _embeddingDropout = RegisterModule(new Dropout(dropout, random, dtype));
        _layers = new ForgetMemoryV2Layer[numLayers];
        for (int layerIndex = 0; layerIndex < numLayers; layerIndex++)
        {
            float retentionFloor = numLayers == 1
                ? retentionMaximum
                : retentionMinimum
                    + (retentionMaximum - retentionMinimum)
                        * layerIndex
                        / (numLayers - 1f);
            _layers[layerIndex] = RegisterModule(
                new ForgetMemoryV2Layer(
                    modelWidth,
                    hiddenWidth,
                    keyWidth,
                    valueWidth,
                    retentionFloor,
                    random,
                    initializationScale,
                    dropout,
                    dtype,
                    useV3,
                    useDrn));
        }
        _finalNorm = RegisterModule(
            new LayerNorm(modelWidth, dtype: dtype));
        _languageModelHead = RegisterModule(
            new Linear(
                modelWidth,
                vocabularySize,
                random,
                initializationScale,
                dtype));

        _hiddenWeightParameters = _layers
            .SelectMany(layer => layer.Parameters())
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

    public int KeyWidth { get; }

    public int ValueWidth { get; }

    public float RetentionMinimum { get; }

    public float RetentionMaximum { get; }

    public bool UseV3 { get; }

    public bool UseDrn { get; }

    public IReadOnlyList<ForgetMemoryV2Layer> Layers
        => Array.AsReadOnly(_layers);

    public override IReadOnlyList<Parameter> HiddenWeightParameters
        => Array.AsReadOnly(_hiddenWeightParameters);

    public override IReadOnlyList<Parameter> AuxiliaryParameters
        => Array.AsReadOnly(_auxiliaryParameters);

    internal override Tensor Forward(
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
        foreach (ForgetMemoryV2Layer layer in _layers)
            hidden = layer.Forward(hidden);
        hidden = _finalNorm.Forward(hidden);
        return _languageModelHead.ForwardBatch(
            hidden.Reshape(batchSize * sequenceLength, ModelWidth));
    }

    /// <summary>
    /// Creates an empty recurrent memory for <see cref="Advance"/>.
    /// </summary>
    public ForgetMemoryV2RecurrentState CreateRecurrentState()
        => new(_layers.Length, checked(KeyWidth * ValueWidth));

    /// <summary>
    /// Advances <paramref name="state"/> by <paramref name="tokenIds"/> and
    /// returns logits of shape [1, tokens, vocabulary].
    /// </summary>
    /// <remarks>
    /// Cost is linear in the number of tokens supplied and independent of how
    /// many tokens the state has already absorbed, because the model carries a
    /// fixed-size matrix memory instead of a growing key/value cache. The
    /// context-length bound that <see cref="Forward"/> enforces does not apply
    /// here: the recurrence has no positional embedding to run past. Feeding
    /// more tokens than the model was trained on is therefore possible but is
    /// extrapolation, not a guarantee.
    /// </remarks>
    public Tensor Advance(
        IReadOnlyList<int> tokenIds,
        ForgetMemoryV2RecurrentState state)
        => Advance(tokenIds, state, allPositions: true);

    /// <summary>
    /// Advances <paramref name="state"/> by <paramref name="tokenIds"/> and
    /// returns logits of shape [1, 1, vocabulary] for the final position only.
    /// </summary>
    /// <remarks>
    /// Sampling needs one distribution per generated token, not one per prompt
    /// token, and the language-model head is the widest matrix in the model.
    /// Skipping it for the positions nobody reads is what makes absorbing a
    /// long prompt cheap.
    /// </remarks>
    public Tensor AdvanceToLastLogits(
        IReadOnlyList<int> tokenIds,
        ForgetMemoryV2RecurrentState state)
        => Advance(tokenIds, state, allPositions: false);

    private Tensor Advance(
        IReadOnlyList<int> tokenIds,
        ForgetMemoryV2RecurrentState state,
        bool allPositions)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        ArgumentNullException.ThrowIfNull(state);
        if (tokenIds.Count == 0)
        {
            throw new ArgumentException(
                "At least one token is required.",
                nameof(tokenIds));
        }
        if (state.LayerCount != _layers.Length
            || state.StateSize != checked(KeyWidth * ValueWidth))
        {
            throw new ArgumentException(
                "The recurrent state was created for a different model.",
                nameof(state));
        }

        int[] ids = tokenIds as int[] ?? tokenIds.ToArray();
        foreach (int token in ids)
        {
            if ((uint)token >= (uint)VocabularySize)
                throw new ArgumentOutOfRangeException(nameof(tokenIds));
        }

        bool wasTraining = IsTraining;
        Eval();
        try
        {
            using (AutogradContext.NoGrad())
            {
                Tensor hidden = _embeddingDropout.Forward(
                    _tokenEmbedding.T.EmbeddingLookup(ids, 1, ids.Length));
                for (int layer = 0; layer < _layers.Length; layer++)
                {
                    hidden = _layers[layer].Continue(
                        hidden,
                        state.Memory(layer));
                }
                if (!allPositions && ids.Length > 1)
                {
                    hidden = hidden.Slice(1, ids.Length - 1, 1);
                }
                hidden = _finalNorm.Forward(hidden);
                Tensor logits = _languageModelHead.ForwardBatch(hidden);
                state.Advanced(ids.Length);
                return logits;
            }
        }
        finally
        {
            if (wasTraining)
                Train();
        }
    }

    internal override int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds,
        int maxNewTokens,
        float temperature = 0.8f,
        int topK = 40,
        int? stopTokenId = BpeTokenizer.EosTokenId,
        Random? random = null)
        => GenerateTokenIds(
            promptTokenIds,
            maxNewTokens,
            temperature,
            topK,
            stopTokenId,
            random,
            onToken: null);

    /// <summary>
    /// Generates tokens, reporting each one to <paramref name="onToken"/> as
    /// soon as it is sampled.
    /// </summary>
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
            using (CudaInferenceScope cacheSession = CudaInferenceScope.Begin(
                resetPool: true,
                clearPoolOnDispose: true))
            {
                // The prompt is absorbed once; every generated token then costs
                // one recurrent step against a fixed-size memory rather than a
                // full forward pass over the whole prefix.
                using ForgetMemoryV2RecurrentState state =
                    CreateRecurrentState();
                CudaInferenceScope inferenceScope =
                    CudaInferenceScope.Begin();
                try
                {
                    Tensor logits = AdvanceToLastLogits(result, state);
                    for (int generated = 0; generated < maxNewTokens; generated++)
                    {
                        int nextToken = SampleLogits(
                            logits,
                            0,
                            VocabularySize,
                            temperature,
                            topK,
                            random);
                        result.Add(nextToken);
                        onToken?.Invoke(nextToken);
                        inferenceScope.Dispose();
                        if (stopTokenId.HasValue && nextToken == stopTokenId.Value)
                            break;
                        if (generated + 1 < maxNewTokens)
                        {
                            inferenceScope = CudaInferenceScope.Begin();
                            logits = AdvanceToLastLogits([nextToken], state);
                        }
                    }
                }
                finally
                {
                    inferenceScope.Dispose();
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

    private static void ValidateRetentionRange(
        float minimum,
        float maximum)
    {
        if (!float.IsFinite(minimum)
            || !float.IsFinite(maximum)
            || minimum < 0f
            || minimum > maximum
            || maximum >= 1f)
        {
            throw new ArgumentException(
                "Retention bounds must be finite and satisfy " +
                "0 <= minimum <= maximum < 1.");
        }
    }

    private static Parameter CreateEmbedding(
        int rows,
        int width,
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
            "TokenEmbedding",
            WeightDecayPolicy.Apply,
            dtype);
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

/// <summary>
/// Decoder-only language model using the V3 matrix-memory recurrence.
/// </summary>
public sealed class ForgetMemoryV3Gpt : ForgetMemoryV2Gpt
{
    public ForgetMemoryV3Gpt(
        int vocabularySize,
        int contextLength,
        int modelWidth,
        int hiddenWidth,
        int numLayers,
        int keyWidth = 16,
        int valueWidth = 16,
        float retentionMinimum = 0.5f,
        float retentionMaximum = 0.99f,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float16)
        : base(
            vocabularySize,
            contextLength,
            modelWidth,
            hiddenWidth,
            numLayers,
            keyWidth,
            valueWidth,
            retentionMinimum,
            retentionMaximum,
            random,
            initializationScale,
            dropout,
            dtype,
            useV3: true,
            useDrn: false)
    {
    }
}

/// <summary>
/// Decoder-only language model using delta writes, read-before-write, and
/// L2-normalized queries and keys.
/// </summary>
public sealed class ForgetMemoryDRNGpt : ForgetMemoryV2Gpt
{
    public ForgetMemoryDRNGpt(
        int vocabularySize,
        int contextLength,
        int modelWidth,
        int hiddenWidth,
        int numLayers,
        int keyWidth = 16,
        int valueWidth = 16,
        float retentionMinimum = 0.5f,
        float retentionMaximum = 0.99f,
        Random? random = null,
        float initializationScale = 0.02f,
        float dropout = 0f,
        TensorDType dtype = TensorDType.Float16)
        : base(
            vocabularySize,
            contextLength,
            modelWidth,
            hiddenWidth,
            numLayers,
            keyWidth,
            valueWidth,
            retentionMinimum,
            retentionMaximum,
            random,
            initializationScale,
            dropout,
            dtype,
            useV3: false,
            useDrn: true)
    {
    }
}

/// <summary>
/// Fixed-size recurrent memory for <see cref="ForgetMemoryV2Gpt.Advance"/>.
/// </summary>
/// <remarks>
/// The whole point of the architecture is that this object does not grow with
/// the number of tokens seen. One layer carries valueWidth * keyWidth floats,
/// so a 16x16 memory is 1 KiB per layer regardless of whether the model has
/// read a hundred tokens or a million.
/// </remarks>
public sealed class ForgetMemoryV2RecurrentState : IDisposable
{
    private readonly ForgetMemoryRecurrentMemory[] _memories;
    private int _disposed;

    internal ForgetMemoryV2RecurrentState(int layerCount, int stateSize)
    {
        if (layerCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(layerCount));
        if (stateSize <= 0)
            throw new ArgumentOutOfRangeException(nameof(stateSize));

        _memories = new ForgetMemoryRecurrentMemory[layerCount];
        for (int layer = 0; layer < layerCount; layer++)
            _memories[layer] = new ForgetMemoryRecurrentMemory(stateSize);
        LayerCount = layerCount;
        StateSize = stateSize;
    }

    public int LayerCount { get; }

    /// <summary>Floats of memory per layer, that is valueWidth * keyWidth.</summary>
    public int StateSize { get; }

    /// <summary>Bytes of recurrent state, constant for the whole run.</summary>
    public long StateBytes => (long)LayerCount * StateSize * sizeof(float);

    /// <summary>Tokens absorbed so far. Does not affect the memory size.</summary>
    public long TokensSeen { get; private set; }

    /// <summary>Largest absolute memory entry, for divergence checks.</summary>
    public float PeakMagnitude()
    {
        ThrowIfDisposed();
        float peak = 0f;
        foreach (ForgetMemoryRecurrentMemory memory in _memories)
        {
            foreach (float value in memory.HostSnapshot())
                peak = MathF.Max(peak, MathF.Abs(value));
        }
        return peak;
    }

    public void Reset()
    {
        ThrowIfDisposed();
        foreach (ForgetMemoryRecurrentMemory memory in _memories)
            memory.Reset();
        TokensSeen = 0;
    }

    internal ForgetMemoryRecurrentMemory Memory(int layer)
    {
        ThrowIfDisposed();
        return _memories[layer];
    }

    internal void Advanced(int tokens)
    {
        ThrowIfDisposed();
        TokensSeen += tokens;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;

        List<Exception>? failures = null;
        foreach (ForgetMemoryRecurrentMemory memory in _memories)
        {
            try
            {
                memory.Dispose();
            }
            catch (Exception failure)
            {
                (failures ??= []).Add(failure);
            }
        }
        GC.SuppressFinalize(this);
        if (failures is not null)
        {
            throw new AggregateException(
                "One or more recurrent CUDA memories failed to release.",
                failures);
        }
    }

    private void ThrowIfDisposed()
        => ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
}
