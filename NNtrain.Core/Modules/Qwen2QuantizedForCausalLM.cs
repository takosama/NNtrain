namespace NNtrain;

/// <summary>Qwen2 attention whose large matrices stay in GGUF K-quant form.</summary>
internal sealed class QwenQuantizedAttention : Module, IDisposable
{
    private readonly QwenQuantizedLinear _q, _k, _v, _o;

    internal QwenQuantizedAttention(
        QwenQuantizedLinear q, QwenQuantizedLinear k,
        QwenQuantizedLinear v, QwenQuantizedLinear o,
        int queryHeads, int kvHeads, float ropeTheta, TensorDType dtype)
        : base(dtype)
    {
        _q = RegisterModule(q); _k = RegisterModule(k);
        _v = RegisterModule(v); _o = RegisterModule(o);
        QueryHeads = queryHeads; KvHeads = kvHeads; RopeTheta = ropeTheta;
    }

    internal int QueryHeads { get; }
    internal int KvHeads { get; }
    internal float RopeTheta { get; }

    internal Tensor Forward(Tensor input, ArcQwenKvCache? cache = null, int position = 0)
    {
        Tensor q = _q.Forward(input);
        Tensor k = _k.Forward(input);
        Tensor v = _v.Forward(input);
        Tensor attention = cache is null
            ? q.QwenGroupedQueryAttention(k, v, QueryHeads, KvHeads, RopeTheta, causal: true)
            : q.ArcQwenCachedAttention(k, v, cache, position, RopeTheta);
        return _o.Forward(attention);
    }

    public void Dispose() { _q.Dispose(); _k.Dispose(); _v.Dispose(); _o.Dispose(); }
}

internal sealed class QwenQuantizedMlp : Module, IDisposable
{
    private readonly QwenQuantizedLinear _gate, _up, _down;

    internal QwenQuantizedMlp(
        QwenQuantizedLinear gate, QwenQuantizedLinear up,
        QwenQuantizedLinear down, TensorDType dtype)
        : base(dtype)
    {
        _gate = RegisterModule(gate);
        _up = RegisterModule(up);
        _down = RegisterModule(down);
    }

    internal Tensor Forward(Tensor input)
        => _down.Forward(_gate.Forward(input).SiluMultiply(_up.Forward(input)));

    public void Dispose() { _gate.Dispose(); _up.Dispose(); _down.Dispose(); }
}

internal sealed class QwenQuantizedBlock : Module, IDisposable
{
    private readonly QwenRmsNorm _inputNorm, _postNorm;
    private readonly QwenQuantizedAttention _attention;
    private readonly QwenQuantizedMlp _mlp;

    internal QwenQuantizedBlock(
        QwenRmsNorm inputNorm, QwenQuantizedAttention attention,
        QwenRmsNorm postNorm, QwenQuantizedMlp mlp, TensorDType dtype)
        : base(dtype)
    {
        _inputNorm = RegisterModule(inputNorm);
        _attention = RegisterModule(attention);
        _postNorm = RegisterModule(postNorm);
        _mlp = RegisterModule(mlp);
    }

    internal Tensor Forward(Tensor input, ArcQwenKvCache? cache = null, int position = 0)
    {
        Tensor h = input + _attention.Forward(_inputNorm.Forward(input), cache, position);
        return h + _mlp.Forward(_postNorm.Forward(h));
    }

    public void Dispose() { _attention.Dispose(); _mlp.Dispose(); }
}

/// <summary>
/// Inference-first Qwen2 model.  Quantized GGUF matrix payloads stay encoded
/// in host memory and, after first use, encoded in Arc VRAM.
/// </summary>
public sealed class Qwen2QuantizedForCausalLM : LanguageModel, IDisposable
{
    private readonly Parameter? _embedding;
    private readonly ArcQuantizedMatrix? _quantizedEmbedding;
    private readonly QwenQuantizedBlock[] _blocks;
    private readonly QwenRmsNorm _finalNorm;
    private readonly QwenQuantizedLinear _head;

    // Preserve callers that construct a model with a dense embedding.
    internal Qwen2QuantizedForCausalLM(
        int vocabulary, int context, int width, int heads, int kvHeads,
        Parameter embedding, QwenQuantizedBlock[] blocks,
        QwenRmsNorm finalNorm, QwenQuantizedLinear head, TensorDType dtype)
        : this(vocabulary, context, width, heads, kvHeads,
            embedding, null, blocks, finalNorm, head, dtype) { }

    internal Qwen2QuantizedForCausalLM(
        int vocabulary, int context, int width, int heads, int kvHeads,
        Parameter? embedding, ArcQuantizedMatrix? quantizedEmbedding, QwenQuantizedBlock[] blocks,
        QwenRmsNorm finalNorm, QwenQuantizedLinear head,
        TensorDType dtype)
        : base(dtype)
    {
        VocabularySize = vocabulary; ContextLength = context; ModelWidth = width;
        QueryHeads = heads; KvHeads = kvHeads;
        if ((embedding is null) == (quantizedEmbedding is null))
            throw new ArgumentException("Exactly one embedding representation is required.");
        _embedding = embedding is null ? null : RegisterParameter(embedding);
        _quantizedEmbedding = quantizedEmbedding;
        _blocks = blocks.Select(RegisterModule).ToArray();
        _finalNorm = RegisterModule(finalNorm);
        _head = RegisterModule(head);
    }

    public override int VocabularySize { get; }
    public override int ContextLength { get; }
    public override int ModelWidth { get; }
    public int QueryHeads { get; }
    public int KvHeads { get; }
    public override IReadOnlyList<Parameter> HiddenWeightParameters => [];
    public override IReadOnlyList<Parameter> AuxiliaryParameters
        => Parameters().ToArray();

    internal override Tensor Forward(int[] tokenIds, int batchSize, int sequenceLength)
    {
        Tensor h = Hidden(tokenIds, batchSize, sequenceLength);
        return _head.Forward(h).Reshape(batchSize * sequenceLength, VocabularySize);
    }

    internal override Tensor ForwardLoss(
        int[] tokenIds, int[] targetIds, int batchSize, int sequenceLength,
        int ignoreIndex = Tensor.DefaultCrossEntropyIgnoreIndex)
        => throw new NotSupportedException(
            "Quantized Qwen model is inference-first; training is intentionally deferred.");

    private Tensor Hidden(int[] tokenIds, int batch, int sequence,
        ArcQwenKvCache[]? caches = null, int position = 0)
    {
        if (sequence <= 0 || sequence > ContextLength)
            throw new ArgumentOutOfRangeException(nameof(sequence));
        if (tokenIds.Length != checked(batch * sequence))
            throw new ArgumentException("Token count does not match batch * sequence.", nameof(tokenIds));
        Tensor h = _quantizedEmbedding is not null
            ? _quantizedEmbedding.LookupEmbedding(tokenIds, batch, sequence)
            : _embedding!.T.EmbeddingLookup(tokenIds, batch, sequence);
        for (int layer = 0; layer < _blocks.Length; ++layer)
            h = _blocks[layer].Forward(h, caches?[layer], position);
        return _finalNorm.Forward(h);
    }

    internal Tensor ForwardLastLogits(int[] tokenIds, ArcQwenKvCache[]? caches = null,
        int position = 0)
        => _head.Forward(Hidden(tokenIds, 1, tokenIds.Length, caches, position)
            .SelectLastSequenceToken());

    internal ArcQwenKvCache[] CreateArcKvCaches(int capacity)
    {
        var owned = new List<ArcQwenKvCache>(_blocks.Length);
        try
        {
            for (int layer = 0; layer < _blocks.Length; ++layer)
                owned.Add(new ArcQwenKvCache(QueryHeads, KvHeads,
                    ModelWidth / QueryHeads, capacity));
            return owned.ToArray();
        }
        catch
        {
            foreach (ArcQwenKvCache cache in owned) cache.Dispose();
            throw;
        }
    }

    internal override int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds, int maxNewTokens,
        float temperature, int topK, int? stopTokenId, Random? random)
        => GenerateTokenIds(promptTokenIds, maxNewTokens, temperature, topK,
            stopTokenId, random, onToken: null);

    internal override int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds, int maxNewTokens,
        float temperature, int topK, int? stopTokenId, Random? random,
        Action<int>? onToken)
    {
        ArgumentNullException.ThrowIfNull(promptTokenIds);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        var result = promptTokenIds.ToList();
        if (result.Count == 0) throw new ArgumentException("Prompt cannot be empty.", nameof(promptTokenIds));
        if (result.Count > ContextLength) throw new ArgumentOutOfRangeException(nameof(promptTokenIds));
        if (result.Any(token => (uint)token >= (uint)VocabularySize))
            throw new ArgumentOutOfRangeException(nameof(promptTokenIds));
        random ??= Random.Shared;
        bool training = IsTraining; Eval();
        ArcQwenKvCache[]? caches = null;
        try
        {
            using (AutogradContext.NoGrad())
            {
                int maximum = Math.Min(maxNewTokens, ContextLength - result.Count);
                if (maximum > 0 && Tensor.ArcResident && Tensor.ArcLane.Options.QwenInferenceKvCache)
                    caches = CreateArcKvCaches(checked(result.Count + maximum - 1));
                int position = 0;
                int[] input = result.ToArray();
                for (int generated = 0;
                     generated < maximum;
                     ++generated)
                {
                    // Quantized weights belong to the model. Only detached
                    // activations are released after the logits have been read.
                    using IDisposable? arcInference = Tensor.BeginArcInferenceFrame();
                    Tensor logits = caches is null
                        ? Forward(input, 1, input.Length)
                        : ForwardLastLogits(input, caches, position);
                    int next = SampleLogits(
                        logits, caches is null ? checked((input.Length - 1) * VocabularySize) : 0,
                        VocabularySize, temperature, topK, random);
                    result.Add(next);
                    onToken?.Invoke(next);
                    if (stopTokenId.HasValue && next == stopTokenId.Value) break;
                    position = caches is null ? 0 : result.Count - 1;
                    input = caches is null ? result.ToArray() : [next];
                }
            }
        }
        finally
        {
            if (caches is not null)
                foreach (ArcQwenKvCache cache in caches) cache.Dispose();
            if (training) Train();
        }
        return result.ToArray();
    }

    public string Generate(
        string prompt, Qwen2GgufTokenizer tokenizer, int maxNewTokens,
        float temperature = 0f, int topK = 1, Random? random = null)
        => GenerateText(prompt, tokenizer, maxNewTokens, onToken: null,
            temperature, topK, random);

    /// <summary>Emits each new token immediately after sampling, including EOS.</summary>
    public string GenerateStreaming(
        string prompt, Qwen2GgufTokenizer tokenizer, int maxNewTokens,
        Action<int> onToken, float temperature = 0f, int topK = 1,
        Random? random = null)
    {
        ArgumentNullException.ThrowIfNull(onToken);
        return GenerateText(prompt, tokenizer, maxNewTokens, onToken,
            temperature, topK, random);
    }

    private string GenerateText(
        string prompt, Qwen2GgufTokenizer tokenizer, int maxNewTokens,
        Action<int>? onToken, float temperature, int topK, Random? random)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tokenizer.VocabularySize != VocabularySize)
            throw new ArgumentException("GGUF tokenizer vocabulary does not match the model.", nameof(tokenizer));
        int[] promptIds = tokenizer.Encode(prompt);
        int[] generated = GenerateTokenIds(
            promptIds, maxNewTokens, temperature, topK,
            tokenizer.EosTokenId, random, onToken);
        return tokenizer.Decode(generated);
    }

    internal override string Generate(
        string prompt, BpeTokenizer tokenizer, int maxNewTokens,
        float temperature, int topK, Random? random)
        => throw new NotSupportedException(
            "Use Generate(prompt, Qwen2GgufTokenizer, ...) for GGUF Qwen models.");

    public void Dispose()
    {
        foreach (QwenQuantizedBlock block in _blocks) block.Dispose();
        _head.Dispose();
        _quantizedEmbedding?.Dispose();
    }
}
