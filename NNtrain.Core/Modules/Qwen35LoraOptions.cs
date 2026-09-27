namespace NNtrain;

/// <summary>FP32 LoRA on a frozen, encoded Qwen3.5 base. One update per sequence.</summary>
public sealed record Qwen35LoraOptions
{
    public int Rank { get; init; } = 8;
    public float Alpha { get; init; } = 16;
    public float LearningRate { get; init; } = 0.0001f;
    public float WeightDecay { get; init; }
    public float GradientClip { get; init; } = 1;
    public int Seed { get; init; } = 1;
    public int[]? Layers { get; init; }
    public string[] Targets { get; init; } = ["attn_q", "attn_k", "attn_v", "attn_output",
        "attn_qkv", "attn_gate", "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"];
    public bool IncludeOutput { get; init; }

    public void Validate()
    {
        if (Rank is < 1 or > 256 || !float.IsFinite(Alpha) || Alpha is <= 0 or > 1024
            || !float.IsFinite(LearningRate) || LearningRate is <= 0 or > 1
            || !float.IsFinite(WeightDecay) || WeightDecay is < 0 or > 1
            || !float.IsFinite(GradientClip) || GradientClip is <= 0 or > 1e6f)
            throw new ArgumentException("Invalid Qwen3.5 LoRA rank, alpha or optimizer settings.");
        string[] supported = ["attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate",
            "ssm_alpha", "ssm_beta", "ssm_out", "ffn_gate", "ffn_up", "ffn_down"];
        if (Targets is null || Targets.Distinct().Count() != Targets.Length
            || Targets.Any(target => !supported.Contains(target)) || (Targets.Length == 0 && !IncludeOutput))
            throw new ArgumentException("Select distinct supported LoRA target names.");
        if (Layers is not null && (Layers.Length == 0 || Layers.Any(layer => layer < 0)
            || Layers.Distinct().Count() != Layers.Length))
            throw new ArgumentException("Select distinct nonnegative LoRA layer indices.");
    }
}

public sealed record Qwen35LoraStepResult(int Step, double Loss, double GradientNorm, int SupervisedTokens);
