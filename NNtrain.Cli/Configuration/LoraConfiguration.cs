using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

internal sealed record LoraConfiguration
{
    internal static readonly JsonSerializerOptions Json = new() {
        PropertyNameCaseInsensitive = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        ReadCommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true };
    public int SchemaVersion { get; init; } = 1;
    public string Architecture { get; init; } = "forgetmemorydrn";
    public int Rank { get; init; } = 8;
    public float Alpha { get; init; } = 16;
    public string[] Targets { get; init; } = ["memoryOutput", "ffnInput", "ffnOutput"];
    public string PrecisionMode { get; init; } = "mix16_32";
    public int Bfp8BlockSize { get; init; } = 32;
    public int Seed { get; init; } = 1234;
    internal void Validate()
    {
        if (SchemaVersion != 1 || Architecture != "forgetmemorydrn")
            throw new NotSupportedException("LoRA schema 1 / forgetmemorydrn only; other models are unsupported.");
        if (Rank <= 0 || Rank > 256 || !float.IsFinite(Alpha) || Alpha <= 0)
            throw new ArgumentException("LoRA rank must be 1..256 and alpha finite and positive.");
        if (PrecisionMode is not ("float32" or "mix16_32" or "mix8_32"))
            throw new NotSupportedException("LoRA precisionMode supports float32, mix16_32 and mix8_32.");
        if (Bfp8BlockSize <= 0) throw new ArgumentException("bfp8BlockSize must be positive.");
        if (Targets is null || Targets.Length == 0 || Targets.Distinct().Count() != Targets.Length ||
            Targets.Any(t => t is not ("memoryProjection" or "memoryOutput" or "ffnInput" or "ffnOutput")))
            throw new ArgumentException("Unknown, duplicate or empty LoRA targets.");
    }
    internal static T Load<T>(string path) => JsonSerializer.Deserialize<T>(File.ReadAllText(path), Json)
        ?? throw new InvalidDataException($"Empty configuration: {path}");

    internal LoraConfiguration WithTrainingOverrides(LoraTrainingConfiguration training)
    {
        var effective = this with { PrecisionMode = training.PrecisionMode ?? PrecisionMode,
            Bfp8BlockSize = training.Bfp8BlockSize ?? Bfp8BlockSize };
        effective.Validate();
        return effective;
    }
}

internal sealed record LoraTrainingConfiguration
{
    public int SchemaVersion { get; init; } = 1;
    public string Objective { get; init; } = "sft";
    public string LoraConfig { get; init; } = "lora.json";
    public string? PrecisionMode { get; init; }
    public int? Bfp8BlockSize { get; init; }
    public bool AllowPrecisionConversionOnResume { get; init; }
    public string TokenizerPath { get; init; } = "fineweb-bpe.json";
    public string DataPath { get; init; } = "data/lora/train.jsonl";
    public string Dataset { get; init; } = "jsonl";
    public string TextColumn { get; init; } = "text";
    public string AdapterPath { get; init; } = "checkpoints/lora/drn.adapter.json";
    public string Device { get; init; } = "cuda";
    public int DeviceIndex { get; init; }
    public int ContextLength { get; init; } = 512;
    public int BatchSize { get; init; } = 2;
    public int GradientAccumulationSteps { get; init; } = 8;
    public int Epochs { get; init; } = 1;
    public int MaxSteps { get; init; } = 100;
    public float LearningRate { get; init; } = 5e-5f;
    public float WeightDecay { get; init; }
    public float GradientClip { get; init; } = 1;
    public int SaveEverySteps { get; init; } = 50;
    public bool Resume { get; init; }
    public bool AutoResume { get; init; }
    public string? LossGraphPath { get; init; }
    public int LossGraphEverySteps { get; init; } = 10;
    public int SampleEverySteps { get; init; } = 100;
    public string PromptPrefix { get; init; } = "### 指示:\n";
    public string ResponsePrefix { get; init; } = "\n\n### 回答:\n";
    public int MaxNewTokens { get; init; } = 128;
    public float Temperature { get; init; } = .7f;
    public int TopK { get; init; } = 40;
    public float DpoBeta { get; init; } = .1f;
    public int GenerationSlots { get; init; } = 4;
    public int[]? GenerationDeviceIndices { get; init; }
    public int CompletedQueueCapacity { get; init; } = 16;
    public float CutMinimum { get; init; } = .01f;
    public float CutMaximum { get; init; } = .99f;
    internal int GenerationDeviceForWorker(int worker)
    {
        if (worker < 0 || worker >= GenerationSlots) throw new ArgumentOutOfRangeException(nameof(worker));
        return GenerationDeviceIndices is { Length: > 0 } devices
            ? devices[worker % devices.Length] : DeviceIndex;
    }
    internal void Validate()
    {
        if (SchemaVersion != 1 || Objective is not ("sft" or "dpo")) throw new NotSupportedException("LoRA supports SFT or DPO.");
        if (PrecisionMode is not (null or "float32" or "mix16_32" or "mix8_32"))
            throw new ArgumentException("precisionMode must be float32, mix16_32 or mix8_32.");
        if (Bfp8BlockSize is <= 0) throw new ArgumentException("bfp8BlockSize must be positive.");
        if (SampleEverySteps < 0) throw new ArgumentException("sampleEverySteps must be nonnegative (0 disables DPO samples).");
        if (LossGraphEverySteps <= 0 || (LossGraphPath is not null && string.IsNullOrWhiteSpace(LossGraphPath)))
            throw new ArgumentException("lossGraphEverySteps must be positive and lossGraphPath must not be blank.");
        if (GenerationDeviceIndices is not null && (Objective != "dpo" || Device != "cuda" ||
            GenerationDeviceIndices.Length == 0 || GenerationDeviceIndices.Length > GenerationSlots ||
            GenerationDeviceIndices.Any(d => d < 0) ||
            GenerationDeviceIndices.Distinct().Count() != GenerationDeviceIndices.Length))
            throw new ArgumentException("generationDeviceIndices requires CUDA DPO, unique nonnegative devices, and at least one generation slot per device.");
        if (Dataset is not ("jsonl" or "fineweb") || string.IsNullOrWhiteSpace(TextColumn))
            throw new ArgumentException("dataset must be jsonl or fineweb, with a non-empty textColumn.");
        if (Objective == "sft" && Dataset != "jsonl") throw new NotSupportedException("SFT requires prompt/response JSONL.");
        if (Objective == "dpo" && (PromptPrefix != "" || ResponsePrefix != ""))
            throw new ArgumentException("DPO uses raw document continuations; promptPrefix and responsePrefix must be empty.");
        if (Objective == "dpo" && (!float.IsFinite(DpoBeta) || DpoBeta <= 0 || GenerationSlots <= 0 ||
            CompletedQueueCapacity < GenerationSlots || CompletedQueueCapacity < BatchSize ||
            !float.IsFinite(CutMinimum) || !float.IsFinite(CutMaximum) || CutMinimum <= 0 ||
            CutMaximum >= 1 || CutMinimum > CutMaximum)) throw new ArgumentException("Invalid DPO beta, queue, generation slots or cut range.");
        if (Device is not ("cpu" or "cuda") || DeviceIndex < 0) throw new ArgumentException("Invalid LoRA device.");
        if (ContextLength < 2 || BatchSize <= 0 || GradientAccumulationSteps <= 0 || Epochs <= 0 ||
            MaxSteps <= 0 || SaveEverySteps <= 0 || MaxNewTokens <= 0 || TopK <= 0)
            throw new ArgumentException("LoRA lengths, batch, accumulation, epochs, steps and save interval must be positive.");
        if ((long)BatchSize * GradientAccumulationSteps > int.MaxValue)
            throw new ArgumentException("Effective LoRA batch exceeds Int32 capacity.");
        if (!float.IsFinite(LearningRate) || LearningRate <= 0 || !float.IsFinite(WeightDecay) || WeightDecay < 0 ||
            !float.IsFinite(GradientClip) || GradientClip <= 0 || !float.IsFinite(Temperature) || Temperature < 0)
            throw new ArgumentException("Invalid LoRA optimizer or sampling settings.");
    }
}
