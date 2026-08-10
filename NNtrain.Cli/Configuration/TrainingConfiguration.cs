using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

sealed record TrainingConfiguration
{
    internal const string GainShareAdamWOptimizer = "gainshareadamw";
    internal const string LionOptimizer = "lion";
    internal const string NekoMuonOptimizer = "nekomuon";
    internal const string AdamWOptimizer = "adamw";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        AllowTrailingCommas = true,
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Skip,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    public DatasetConfiguration TrainingData { get; init; } = new();

    public DatasetConfiguration EvaluationData { get; init; } = new();

    public int Epochs { get; init; } = 200;

    public int BatchSize { get; init; } = 32;

    public int? MicroBatchSize { get; init; }

    public int MicroBatchCount { get; init; } = 1;

    public string Optimizer { get; init; } = GainShareAdamWOptimizer;

    public float LearningRate { get; init; } = 3e-4f;

    public float AuxiliaryLearningRate { get; init; } = 3e-4f;

    public float WeightDecay { get; init; } = 5e-4f;

    public int GainShareBlockDepth { get; init; } = 1;

    public float GainShareBeta1 { get; init; } = 0.9f;

    public float GainShareBeta2 { get; init; } = 0.999f;

    public float GainShareEpsilon { get; init; } = 1e-8f;

    public float GainShareRho { get; init; } = 0.95f;

    public float GainShareGamma { get; init; } = 1f;

    public float GainShareMinScale { get; init; } = 0.5f;

    public float GainShareMaxScale { get; init; } = 2f;

    public float LabelSmoothing { get; init; } = 0.1f;

    public int WarmupEpochs { get; init; }

    public float MinimumLearningRateRatio { get; init; } = 0.01f;

    public int EarlyStoppingPatience { get; init; }

    public float EarlyStoppingMinimumDelta { get; init; } = 1e-4f;

    public bool UseSimd { get; init; } = true;

    public bool ShowLossGraph { get; init; } = true;

    public int Seed { get; init; } = 1234;

    public ModelConfiguration Model { get; init; } = new();

    public static TrainingConfiguration Load(string configurationPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configurationPath);

        string fullConfigurationPath = Path.GetFullPath(configurationPath);
        string json = File.ReadAllText(fullConfigurationPath);
        TrainingConfiguration configuration =
            JsonSerializer.Deserialize<TrainingConfiguration>(
                json,
                JsonOptions)
            ?? throw new InvalidDataException(
                "Training configuration cannot be JSON null.");

        configuration.Validate();
        string configurationDirectory =
            Path.GetDirectoryName(fullConfigurationPath)
            ?? Environment.CurrentDirectory;

        return configuration with
        {
            TrainingData = configuration.TrainingData.ResolvePaths(
                configurationDirectory),
            EvaluationData = configuration.EvaluationData.ResolvePaths(
                configurationDirectory),
        };
    }

    internal void Validate()
    {
        if (TrainingData is null)
        {
            throw new ArgumentException(
                "Training dataset configuration cannot be null.",
                nameof(TrainingData));
        }

        if (EvaluationData is null)
        {
            throw new ArgumentException(
                "Evaluation dataset configuration cannot be null.",
                nameof(EvaluationData));
        }

        TrainingData.Validate("Training");
        EvaluationData.Validate("Evaluation");

        if (Epochs <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Epochs),
                Epochs,
                "Epoch count must be positive.");
        }

        if (BatchSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(BatchSize),
                BatchSize,
                "Batch size must be positive.");
        }

        if (MicroBatchSize.HasValue && MicroBatchSize.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MicroBatchSize),
                MicroBatchSize,
                "Micro-batch size must be positive when specified.");
        }

        if (MicroBatchCount <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MicroBatchCount),
                MicroBatchCount,
                "Micro-batch count must be positive.");
        }

        if ((long)ResolvedMicroBatchSize * MicroBatchCount > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MicroBatchCount),
                MicroBatchCount,
                "Effective batch size must not exceed Int32.MaxValue.");
        }

        if (!IsOptimizer(GainShareAdamWOptimizer)
            && !IsOptimizer(LionOptimizer)
            && !IsOptimizer(NekoMuonOptimizer)
            && !IsOptimizer(AdamWOptimizer))
        {
            throw new ArgumentException(
                $"Unsupported optimizer '{Optimizer}'. Supported " +
                $"optimizers are '{GainShareAdamWOptimizer}', " +
                $"'{LionOptimizer}', " +
                $"'{NekoMuonOptimizer}', and " +
                $"'{AdamWOptimizer}'.",
                nameof(Optimizer));
        }

        if (!float.IsFinite(LearningRate) || LearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(LearningRate),
                LearningRate,
                "Learning rate must be finite and positive.");
        }

        if (!float.IsFinite(AuxiliaryLearningRate)
            || AuxiliaryLearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(AuxiliaryLearningRate),
                AuxiliaryLearningRate,
                "Auxiliary AdamW learning rate must be finite and positive.");
        }

        if (!float.IsFinite(WeightDecay) || WeightDecay < 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(WeightDecay),
                WeightDecay,
                "Weight decay must be finite and non-negative.");
        }

        if (GainShareBlockDepth < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareBlockDepth),
                GainShareBlockDepth,
                "GainShare block depth must be non-negative.");
        }

        ValidateGainShareUnitInterval(
            GainShareBeta1,
            nameof(GainShareBeta1),
            "beta1");
        ValidateGainShareUnitInterval(
            GainShareBeta2,
            nameof(GainShareBeta2),
            "beta2");

        if (!float.IsFinite(GainShareEpsilon)
            || GainShareEpsilon <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareEpsilon),
                GainShareEpsilon,
                "GainShare epsilon must be finite and positive.");
        }

        if (!float.IsFinite(GainShareRho)
            || GainShareRho < 0f
            || GainShareRho >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareRho),
                GainShareRho,
                "GainShare rho must be finite and in [0, 1).");
        }

        if (!float.IsFinite(GainShareGamma) || GainShareGamma < 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareGamma),
                GainShareGamma,
                "GainShare gamma must be finite and non-negative.");
        }

        if (!float.IsFinite(GainShareMinScale)
            || GainShareMinScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareMinScale),
                GainShareMinScale,
                "GainShare minimum scale must be finite and positive.");
        }

        if (!float.IsFinite(GainShareMaxScale)
            || GainShareMaxScale < GainShareMinScale)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareMaxScale),
                GainShareMaxScale,
                "GainShare maximum scale must be finite and not less " +
                "than the minimum scale.");
        }

        if (!float.IsFinite(LabelSmoothing)
            || LabelSmoothing < 0f
            || LabelSmoothing >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(LabelSmoothing),
                LabelSmoothing,
                "Label smoothing must be finite and in the range [0, 1).");
        }

        if (WarmupEpochs < 0 || WarmupEpochs >= Epochs)
        {
            throw new ArgumentOutOfRangeException(
                nameof(WarmupEpochs),
                WarmupEpochs,
                "Warmup epochs must be non-negative and less than the " +
                "total epoch count.");
        }

        if (!float.IsFinite(MinimumLearningRateRatio)
            || MinimumLearningRateRatio <= 0f
            || MinimumLearningRateRatio > 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MinimumLearningRateRatio),
                MinimumLearningRateRatio,
                "Minimum learning-rate ratio must be finite and in " +
                "the range (0, 1].");
        }

        if (EarlyStoppingPatience < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(EarlyStoppingPatience),
                EarlyStoppingPatience,
                "Early-stopping patience must be non-negative.");
        }

        if (!float.IsFinite(EarlyStoppingMinimumDelta)
            || EarlyStoppingMinimumDelta < 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(EarlyStoppingMinimumDelta),
                EarlyStoppingMinimumDelta,
                "Early-stopping minimum delta must be finite and " +
                "non-negative.");
        }

        if (Model is null)
        {
            throw new ArgumentException(
                "Model configuration cannot be null.",
                nameof(Model));
        }

        Model.Validate();
    }

    private static void ValidateGainShareUnitInterval(
        float value,
        string parameterName,
        string settingName)
    {
        if (!float.IsFinite(value) || value < 0f || value >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                value,
                $"GainShare {settingName} must be finite and in [0, 1).");
        }
    }

    internal bool IsOptimizer(string expectedOptimizer)
        => string.Equals(
            Optimizer,
            expectedOptimizer,
            StringComparison.OrdinalIgnoreCase);

    internal int ResolvedMicroBatchSize => MicroBatchSize ?? BatchSize;

    internal int EffectiveBatchSize
        => checked(ResolvedMicroBatchSize * MicroBatchCount);
}

sealed record DatasetConfiguration
{
    internal const string MnistType = "mnist";
    internal const string Cifar100Type = "cifar100";

    public string Type { get; init; } = MnistType;

    public string ImagePath { get; init; } = string.Empty;

    public string LabelPath { get; init; } = string.Empty;

    public string DataPath { get; init; } = string.Empty;

    public int PatchSize { get; init; } = 4;

    public bool Normalize { get; init; }

    public Cifar100AugmentationConfiguration Augmentation { get; init; } =
        new();

    internal void Validate(string role)
    {
        if (string.IsNullOrWhiteSpace(Type))
        {
            throw new ArgumentException(
                $"{role} dataset type cannot be null, empty, or whitespace.",
                nameof(Type));
        }

        if (IsType(MnistType))
        {
            if (string.IsNullOrWhiteSpace(ImagePath))
            {
                throw new ArgumentException(
                    $"{role} image path cannot be null, empty, or " +
                    "whitespace.",
                    nameof(ImagePath));
            }

            if (string.IsNullOrWhiteSpace(LabelPath))
            {
                throw new ArgumentException(
                    $"{role} label path cannot be null, empty, or " +
                    "whitespace.",
                    nameof(LabelPath));
            }

            return;
        }

        if (IsType(Cifar100Type))
        {
            if (string.IsNullOrWhiteSpace(DataPath))
            {
                throw new ArgumentException(
                    $"{role} CIFAR-100 data path cannot be null, empty, " +
                    "or whitespace.",
                    nameof(DataPath));
            }

            if (Augmentation is null)
            {
                throw new ArgumentException(
                    $"{role} CIFAR-100 augmentation configuration cannot " +
                    "be null.",
                    nameof(Augmentation));
            }

            if (PatchSize <= 0 || 32 % PatchSize != 0)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(PatchSize),
                    PatchSize,
                    $"{role} CIFAR-100 patch size must be a positive " +
                    "divisor of 32.");
            }

            Augmentation.Validate(role);

            return;
        }

        throw new ArgumentException(
            $"Unsupported {role.ToLowerInvariant()} dataset type '{Type}'. " +
            $"Supported types are '{MnistType}' and '{Cifar100Type}'.",
            nameof(Type));
    }

    internal DatasetConfiguration ResolvePaths(string configurationDirectory)
    {
        if (IsType(Cifar100Type))
        {
            return this with
            {
                DataPath = Path.GetFullPath(
                    DataPath,
                    configurationDirectory),
            };
        }

        return this with
        {
            ImagePath = Path.GetFullPath(ImagePath, configurationDirectory),
            LabelPath = Path.GetFullPath(LabelPath, configurationDirectory),
        };
    }

    internal bool IsType(string expectedType)
        => string.Equals(Type, expectedType, StringComparison.OrdinalIgnoreCase);
}

sealed record Cifar100AugmentationConfiguration
{
    public int RandomCropPadding { get; init; } = 4;

    public bool HorizontalFlip { get; init; } = true;

    public bool VerticalFlip { get; init; }

    internal void Validate(string role)
    {
        if (RandomCropPadding < 0 || RandomCropPadding > 32)
        {
            throw new ArgumentOutOfRangeException(
                nameof(RandomCropPadding),
                RandomCropPadding,
                $"{role} CIFAR-100 random crop padding must be between " +
                "0 and 32.");
        }
    }
}

sealed record ModelConfiguration
{
    public int Heads { get; init; } = 1;

    public int HiddenSize { get; init; } = 128;

    public int Layers { get; init; } = 32;

    public int Seed { get; init; }

    public float InitializationScale { get; init; } = 0.02f;

    public float Dropout { get; init; }

    internal void Validate()
    {
        if (Heads <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Heads),
                Heads,
                "Attention head count must be positive.");
        }

        if (HiddenSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(HiddenSize),
                HiddenSize,
                "Hidden size must be positive.");
        }

        if (Layers <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Layers),
                Layers,
                "Layer count must be positive.");
        }

        if (!float.IsFinite(InitializationScale)
            || InitializationScale <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(InitializationScale),
                InitializationScale,
                "Initialization scale must be finite and positive.");
        }

        if (!float.IsFinite(Dropout) || Dropout < 0f || Dropout >= 1f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Dropout),
                Dropout,
                "Dropout probability must be finite and in [0, 1).");
        }
    }

    internal void ValidateForModelWidth(int modelWidth)
    {
        if (modelWidth % Heads != 0)
        {
            throw new ArgumentException(
                $"Attention head count '{Heads}' must evenly divide model " +
                $"width '{modelWidth}'.",
                nameof(Heads));
        }
    }
}
