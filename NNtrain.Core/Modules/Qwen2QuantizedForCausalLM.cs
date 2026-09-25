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

    internal Tensor Forward(Tensor input)
    {
        Tensor q = _q.Forward(input);
        Tensor k = _k.Forward(input);
        Tensor v = _v.Forward(input);
        return _o.Forward(q.QwenGroupedQueryAttention(
            k, v, QueryHeads, KvHeads, RopeTheta, causal: true));
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

    internal Tensor Forward(Tensor input)
    {
        Tensor h = input + _attention.Forward(_inputNorm.Forward(input));
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
    private readonly Parameter _embedding;
    private readonly QwenQuantizedBlock[] _blocks;
    private readonly QwenRmsNorm _finalNorm;
    private readonly QwenQuantizedLinear _head;

    internal Qwen2QuantizedForCausalLM(
        int vocabulary, int context, int width, int heads, int kvHeads,
        Parameter embedding, QwenQuantizedBlock[] blocks,
        QwenRmsNorm finalNorm, QwenQuantizedLinear head,
        TensorDType dtype)
        : base(dtype)
    {
        VocabularySize = vocabulary; ContextLength = context; ModelWidth = width;
        QueryHeads = heads; KvHeads = kvHeads;
        _embedding = RegisterParameter(embedding);
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

    private Tensor Hidden(int[] tokenIds, int batch, int sequence)
    {
        if (sequence <= 0 || sequence > ContextLength)
            throw new ArgumentOutOfRangeException(nameof(sequence));
        if (tokenIds.Length != checked(batch * sequence))
            throw new ArgumentException("Token count does not match batch * sequence.", nameof(tokenIds));
        Tensor h = _embedding.T.EmbeddingLookup(tokenIds, batch, sequence);
        foreach (QwenQuantizedBlock block in _blocks) h = block.Forward(h);
        return _finalNorm.Forward(h);
    }

    internal override int[] GenerateTokenIds(
        IEnumerable<int> promptTokenIds, int maxNewTokens,
        float temperature, int topK, int? stopTokenId, Random? random)
    {
        var result = promptTokenIds.ToList();
        if (result.Count == 0) throw new ArgumentException("Prompt cannot be empty.", nameof(promptTokenIds));
        random ??= Random.Shared;
        bool training = IsTraining; Eval();
        try
        {
            using (AutogradContext.NoGrad())
            {
                for (int generated = 0;
                     generated < maxNewTokens && result.Count < ContextLength;
                     ++generated)
                {
                    Tensor logits = Forward(result.ToArray(), 1, result.Count);
                    int next = SampleLogits(
                        logits, (result.Count - 1) * VocabularySize,
                        VocabularySize, temperature, topK, random);
                    result.Add(next);
                    if (stopTokenId.HasValue && next == stopTokenId.Value) break;
                }
            }
        }
        finally { if (training) Train(); }
        return result.ToArray();
    }

    internal override string Generate(
        string prompt, BpeTokenizer tokenizer, int maxNewTokens,
        float temperature, int topK, Random? random)
        => throw new NotSupportedException(
            "Qwen GGUF tokenizer integration is the next inference milestone.");

    public void Dispose()
    {
        foreach (QwenQuantizedBlock block in _blocks) block.Dispose();
        _head.Dispose();
    }
}
