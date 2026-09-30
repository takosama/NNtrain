using System.Text.Json;

namespace NNtrain;

internal sealed record QwenLoraTrainingConfiguration
{
    public int SchemaVersion { get; init; } = 1;
    public string DataPath { get; init; } = "qwen-lora.example.jsonl";
    public string AdapterPath { get; init; } = "checkpoints/qwen35.adapter.bin";
    public int[] Devices { get; init; } = [0, 1];
    public int ContextLength { get; init; } = 64;
    public int MaxSteps { get; init; } = 10;
    public int SaveEverySteps { get; init; } = 1;
    // The default preserves the exact training trajectory. FP16 uses Arc XMX
    // for the frozen IQ2_S forward projection and changes roundoff slightly.
    public string Iq2ForwardPrecision { get; init; } = "exact";
    // Optional exact FP32 host cache for long-sequence IQ2_S recomputation.
    public int Iq2ProjectionCacheMiB { get; init; }
    public bool Iq2ProjectionCachePrioritize { get; init; }
    // Bounded per-Arc alternative to the host cache; outputs never leave VRAM.
    public int Iq2GpuProjectionCacheMiB { get; init; }
    // Maximum reusable Arc training buffers retained per device.
    public int TrainingBufferPoolMiB { get; init; } = 2048;
    public string? LossGraphPath { get; init; }
    public bool ShowLossGraph { get; init; } = true;
    public bool OpenLossGraph { get; init; } = true;
    public int LossGraphEverySteps { get; init; } = 1;
    public int Rank { get; init; } = 8;
    public float Alpha { get; init; } = 16f;
    public float LearningRate { get; init; } = 0.0001f;
    public float WeightDecay { get; init; }
    public float GradientClip { get; init; } = 1f;
    public int Seed { get; init; } = 1;
    // Omitted: shuffle new runs, but preserve legacy sequential order on resume.
    // Set false to request the original dataset order explicitly.
    public bool? ShuffleExamples { get; init; }
    public int[]? Layers { get; init; }
    public string[] Targets { get; init; } =
    [
        "attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate",
        "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"
    ];
    public bool IncludeOutput { get; init; }
    public string PromptPrefix { get; init; } = "<|im_start|>user\n";
    public string ResponsePrefix { get; init; } = "<|im_end|>\n<|im_start|>assistant\n";

    internal static QwenLoraTrainingConfiguration Load(string path)
        => JsonSerializer.Deserialize<QwenLoraTrainingConfiguration>(File.ReadAllText(path), LoraConfiguration.Json)
            ?? throw new InvalidDataException("Empty Qwen3.5 LoRA configuration.");

    internal void Validate()
    {
        if (SchemaVersion != 1) throw new NotSupportedException("Qwen3.5 LoRA configuration schema must be 1.");
        if (string.IsNullOrWhiteSpace(DataPath) || string.IsNullOrWhiteSpace(AdapterPath))
            throw new ArgumentException("dataPath and adapterPath must not be blank.");
        if (Devices is null || Devices.Length == 0 || Devices.Any(x => x < 0)
            || Devices.Distinct().Count() != Devices.Length)
            throw new ArgumentException("devices must contain distinct nonnegative Arc indices.");
        if (ContextLength < 2 || MaxSteps <= 0 || SaveEverySteps <= 0)
            throw new ArgumentException("contextLength must be at least 2; maxSteps and saveEverySteps must be positive.");
        if (Iq2ForwardPrecision is not ("exact" or "fp16"))
            throw new ArgumentException("iq2ForwardPrecision must be exact or fp16.");
        if (Iq2ProjectionCacheMiB is < 0 or > 16384)
            throw new ArgumentException("iq2ProjectionCacheMiB must be between 0 and 16384.");
        if (Iq2GpuProjectionCacheMiB is < 0 or > 1024)
            throw new ArgumentException("iq2GpuProjectionCacheMiB must be between 0 and 1024 per Arc.");
        if (TrainingBufferPoolMiB is < 0 or > 2048)
            throw new ArgumentException("trainingBufferPoolMiB must be between 0 and 2048 per Arc.");
        if (Iq2ProjectionCacheMiB > 0 && Iq2GpuProjectionCacheMiB > 0)
            throw new ArgumentException("Choose either the host or GPU IQ2 projection cache.");
        if (LossGraphEverySteps <= 0)
            throw new ArgumentException("lossGraphEverySteps must be positive.");
        if (LossGraphPath is not null && string.IsNullOrWhiteSpace(LossGraphPath))
            throw new ArgumentException("lossGraphPath must be null for the configuration-derived path or a non-empty HTML path.");
        if (Rank is < 1 or > 256 || !float.IsFinite(Alpha) || Alpha <= 0)
            throw new ArgumentException("rank must be 1..256 and alpha must be finite and positive.");
        if (!float.IsFinite(LearningRate) || LearningRate <= 0
            || !float.IsFinite(WeightDecay) || WeightDecay < 0
            || !float.IsFinite(GradientClip) || GradientClip <= 0)
            throw new ArgumentException("Invalid learningRate, weightDecay or gradientClip.");
        if (Layers is not null && (Layers.Length == 0 || Layers.Any(x => x < 0)
            || Layers.Distinct().Count() != Layers.Length))
            throw new ArgumentException("layers must be null for all layers, or distinct nonnegative indices.");
        if (Targets is null || Targets.Length == 0 || Targets.Distinct(StringComparer.Ordinal).Count() != Targets.Length
            || Targets.Any(x => x is not ("attn_q" or "attn_k" or "attn_v" or "attn_output"
                or "attn_qkv" or "attn_gate" or "ssm_alpha" or "ssm_beta" or "ssm_out"
                or "ffn_gate" or "ffn_up" or "ffn_down")))
            throw new ArgumentException("targets must contain distinct supported Attention/MLP projection names.");
        if (PromptPrefix is null || ResponsePrefix is null)
            throw new ArgumentException("promptPrefix and responsePrefix must be strings.");
        AdapterOptions().Validate();
    }

    internal Qwen35LoraOptions AdapterOptions() => new()
    {
        Rank = Rank, Alpha = Alpha, LearningRate = LearningRate, WeightDecay = WeightDecay,
        GradientClip = GradientClip, Seed = Seed, Layers = Layers, Targets = Targets,
        IncludeOutput = IncludeOutput
    };
}
