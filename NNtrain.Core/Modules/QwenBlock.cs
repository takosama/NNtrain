namespace NNtrain;

internal sealed class QwenRmsNorm : Module
{
    public QwenRmsNorm(int width, float epsilon, TensorDType dtype)
        : base(dtype)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (!(epsilon > 0f) || !float.IsFinite(epsilon))
            throw new ArgumentOutOfRangeException(nameof(epsilon));
        Epsilon = epsilon;
        Weight = RegisterParameter(new Parameter(
            Enumerable.Repeat(1f, width).ToArray(),
            [width],
            "weight",
            WeightDecayPolicy.Exclude,
            dtype));
    }

    public Parameter Weight { get; }
    public float Epsilon { get; }

    public Tensor Forward(Tensor input)
        => input.RmsNormLastDim(Weight.T, Epsilon);
}

internal sealed class QwenAttention : Module
{
    public QwenAttention(
        int modelWidth,
        int queryHeads,
        int kvHeads,
        float ropeTheta,
        Random random,
        float initScale,
        TensorDType dtype)
        : base(dtype)
    {
        if (queryHeads <= 0 || kvHeads <= 0 || queryHeads % kvHeads != 0)
            throw new ArgumentException("Qwen query heads must be a multiple of KV heads.");
        if (modelWidth % queryHeads != 0)
            throw new ArgumentException("Qwen model width must be divisible by query heads.");
        ModelWidth = modelWidth;
        QueryHeads = queryHeads;
        KvHeads = kvHeads;
        HeadWidth = modelWidth / queryHeads;
        RopeTheta = ropeTheta;
        int kvWidth = checked(kvHeads * HeadWidth);
        QProj = RegisterModule(new Linear(modelWidth, modelWidth, random, initScale, dtype));
        KProj = RegisterModule(new Linear(modelWidth, kvWidth, random, initScale, dtype));
        VProj = RegisterModule(new Linear(modelWidth, kvWidth, random, initScale, dtype));
        OProj = RegisterModule(new Linear(modelWidth, modelWidth, random, initScale, dtype));
    }

    public int ModelWidth { get; }
    public int QueryHeads { get; }
    public int KvHeads { get; }
    public int HeadWidth { get; }
    public float RopeTheta { get; }
    public Linear QProj { get; }
    public Linear KProj { get; }
    public Linear VProj { get; }
    public Linear OProj { get; }

    public Tensor Forward(Tensor input)
    {
        Tensor q = QProj.ForwardBatch(input);
        Tensor k = KProj.ForwardBatch(input);
        Tensor v = VProj.ForwardBatch(input);
        Tensor attended = q.QwenGroupedQueryAttention(
            k, v, QueryHeads, KvHeads, RopeTheta, causal: true);
        return OProj.ForwardBatch(attended);
    }

    internal void SetBaseFrozenForLora(bool frozen)
    {
        QProj.FrozenForLora = frozen;
        KProj.FrozenForLora = frozen;
        VProj.FrozenForLora = frozen;
        OProj.FrozenForLora = frozen;
    }

    internal void AttachLora(
        LoraAdapterSet set, int layer, int rank, float alpha,
        IReadOnlySet<string> targets, Random random)
    {
        foreach (var (name, linear) in new[]
        {
            ("q_proj", QProj), ("k_proj", KProj),
            ("v_proj", VProj), ("o_proj", OProj),
        })
        {
            if (!targets.Contains(name)) continue;
            set.Add($"layers.{layer}.self_attn.{name}",
                linear.AttachLora(rank, alpha, random));
        }
    }
}

internal sealed class QwenMlp : Module
{
    public QwenMlp(
        int modelWidth,
        int hiddenWidth,
        Random random,
        float initScale,
        TensorDType dtype)
        : base(dtype)
    {
        GateProj = RegisterModule(new Linear(modelWidth, hiddenWidth, random, initScale, dtype));
        UpProj = RegisterModule(new Linear(modelWidth, hiddenWidth, random, initScale, dtype));
        DownProj = RegisterModule(new Linear(hiddenWidth, modelWidth, random, initScale, dtype));
    }

    public Linear GateProj { get; }
    public Linear UpProj { get; }
    public Linear DownProj { get; }

    public Tensor Forward(Tensor input)
        => DownProj.ForwardBatch(
            GateProj.ForwardBatch(input).SiluMultiply(
                UpProj.ForwardBatch(input)));

    internal void SetBaseFrozenForLora(bool frozen)
    {
        GateProj.FrozenForLora = frozen;
        UpProj.FrozenForLora = frozen;
        DownProj.FrozenForLora = frozen;
    }

    internal void AttachLora(
        LoraAdapterSet set, int layer, int rank, float alpha,
        IReadOnlySet<string> targets, Random random)
    {
        foreach (var (name, linear) in new[]
        {
            ("gate_proj", GateProj), ("up_proj", UpProj), ("down_proj", DownProj),
        })
        {
            if (!targets.Contains(name)) continue;
            set.Add($"layers.{layer}.mlp.{name}",
                linear.AttachLora(rank, alpha, random));
        }
    }
}

internal sealed class QwenBlock : Module
{
    public QwenBlock(
        int modelWidth,
        int queryHeads,
        int kvHeads,
        int hiddenWidth,
        float rmsEpsilon,
        float ropeTheta,
        Random random,
        float initScale,
        TensorDType dtype)
        : base(dtype)
    {
        InputNorm = RegisterModule(new QwenRmsNorm(modelWidth, rmsEpsilon, dtype));
        Attention = RegisterModule(new QwenAttention(
            modelWidth, queryHeads, kvHeads, ropeTheta, random, initScale, dtype));
        PostAttentionNorm = RegisterModule(new QwenRmsNorm(modelWidth, rmsEpsilon, dtype));
        Mlp = RegisterModule(new QwenMlp(modelWidth, hiddenWidth, random, initScale, dtype));
    }

    public QwenRmsNorm InputNorm { get; }
    public QwenAttention Attention { get; }
    public QwenRmsNorm PostAttentionNorm { get; }
    public QwenMlp Mlp { get; }

    public Tensor Forward(Tensor input)
    {
        Tensor afterAttention = input + Attention.Forward(InputNorm.Forward(input));
        return afterAttention + Mlp.Forward(PostAttentionNorm.Forward(afterAttention));
    }

    internal void SetBaseFrozenForLora(bool frozen)
    {
        Attention.SetBaseFrozenForLora(frozen);
        Mlp.SetBaseFrozenForLora(frozen);
    }

    internal void AttachLora(
        LoraAdapterSet set, int layer, int rank, float alpha,
        IReadOnlySet<string> targets, Random random)
    {
        Attention.AttachLora(set, layer, rank, alpha, targets, random);
        Mlp.AttachLora(set, layer, rank, alpha, targets, random);
    }
}
