using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

sealed record TrainingConfiguration
{
    internal const string GainShareAdamWOptimizer = "gainshareadamw";
    internal const string LionOptimizer = "lion";
    internal const string NekoMuonOptimizer = "nekomuon";
    internal const string AdamWOptimizer = "adamw";
    internal const string LinearWarmupCosineAnnealingScheduler =
        "linearWarmupCosineAnnealing";

    private static readonly string[] LegacyOptimizationPropertyNames =
    [
        "optimizer",
        "learningRate",
        "auxiliaryLearningRate",
        "weightDecay",
        "gainShareBlockDepth",
        "gainShareBeta1",
        "gainShareBeta2",
        "gainShareEpsilon",
        "gainShareRho",
        "gainShareGamma",
        "gainShareMinScale",
        "gainShareMaxScale",
        "warmupEpochs",
        "minimumLearningRateRatio",
    ];

    private static readonly string[] LegacyCheckpointPropertyNames =
    [
        "resumeFromCheckpoint",
        "autoResume",
        "checkpointPath",
    ];

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

    public bool ResumeFromCheckpoint { get; init; }

    public bool AutoResume { get; init; }

    public string CheckpointPath { get; init; } = string.Empty;

    public ClassificationCheckpointConfiguration? Checkpoint { get; init; }

    public ClassificationOptimizationConfiguration? Optimization
    {
        get;
        init;
    }

    public int Seed { get; init; } = 1234;

    public ModelConfiguration Model { get; init; } = new();

    public static TrainingConfiguration Load(string configurationPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configurationPath);

        string fullConfigurationPath = Path.GetFullPath(configurationPath);
        string json = File.ReadAllText(fullConfigurationPath);
        TrainingConfigurationV2.NormalizedConfiguration normalized =
            TrainingConfigurationV2.Normalize(json);
        if (normalized.IsV2
            && !string.Equals(
                normalized.TaskType,
                TrainingConfigurationV2.ClassificationTask,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Configuration task '{normalized.TaskType}' is not an " +
                "image-classification task.");
        }
        json = normalized.Json;
        ValidateJsonSectionExclusivity(json);
        TrainingConfiguration configuration =
            JsonSerializer.Deserialize<TrainingConfiguration>(
                json,
                JsonOptions)
            ?? throw new InvalidDataException(
                "Training configuration cannot be JSON null.");

        string configurationDirectory =
            Path.GetDirectoryName(fullConfigurationPath)
            ?? Environment.CurrentDirectory;
        configuration = configuration.NormalizeStructuredSettings(
            fullConfigurationPath,
            configurationDirectory);
        configuration.Validate();

        return configuration with
        {
            TrainingData = configuration.TrainingData.ResolvePaths(
                configurationDirectory),
            EvaluationData = configuration.EvaluationData.ResolvePaths(
                configurationDirectory),
        };
    }

    private static void ValidateJsonSectionExclusivity(string json)
    {
        using JsonDocument document = JsonDocument.Parse(
            json,
            new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling = JsonCommentHandling.Skip,
            });

        if (document.RootElement.ValueKind != JsonValueKind.Object)
        {
            return;
        }

        RejectMixedJsonSettings(
            document.RootElement,
            "optimization",
            LegacyOptimizationPropertyNames);
        RejectMixedJsonSettings(
            document.RootElement,
            "checkpoint",
            LegacyCheckpointPropertyNames);
    }

    private static void RejectMixedJsonSettings(
        JsonElement root,
        string sectionName,
        IReadOnlyCollection<string> legacyNames)
    {
        bool hasSection = root.EnumerateObject().Any(
            property => string.Equals(
                property.Name,
                sectionName,
                StringComparison.OrdinalIgnoreCase));
        if (!hasSection)
        {
            return;
        }

        string[] specifiedLegacyNames = root.EnumerateObject()
            .Where(property => legacyNames.Any(
                legacyName => string.Equals(
                    property.Name,
                    legacyName,
                    StringComparison.OrdinalIgnoreCase)))
            .Select(property => property.Name)
            .ToArray();
        if (specifiedLegacyNames.Length == 0)
        {
            return;
        }

        throw new InvalidDataException(
            $"The '{sectionName}' section cannot be combined with legacy " +
            $"root settings: {string.Join(", ", specifiedLegacyNames)}.");
    }

    private TrainingConfiguration NormalizeStructuredSettings(
        string fullConfigurationPath,
        string configurationDirectory)
    {
        ClassificationOptimizerConfiguration? optimizerSettings =
            Optimization?.Optimizer;
        ClassificationSchedulerConfiguration? schedulerSettings =
            Optimization?.Scheduler;

        if (Optimization is not null && optimizerSettings is null)
        {
            throw new InvalidDataException(
                "The 'optimization.optimizer' setting cannot be JSON null.");
        }

        if (Optimization is not null && schedulerSettings is null)
        {
            throw new InvalidDataException(
                "The 'optimization.scheduler' setting cannot be JSON null.");
        }

        string schedulerType = schedulerSettings?.Type
            ?? LinearWarmupCosineAnnealingScheduler;
        if (!string.Equals(
            schedulerType,
            LinearWarmupCosineAnnealingScheduler,
            StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"Unsupported scheduler '{schedulerType}'. Supported " +
                $"scheduler is '{LinearWarmupCosineAnnealingScheduler}'.",
                nameof(ClassificationSchedulerConfiguration.Type));
        }

        string checkpointPath = ResolveCheckpointPath(
            fullConfigurationPath,
            configurationDirectory);

        return this with
        {
            Optimizer = optimizerSettings?.Type ?? Optimizer,
            LearningRate = optimizerSettings?.LearningRate ?? LearningRate,
            AuxiliaryLearningRate =
                optimizerSettings?.AuxiliaryLearningRate
                ?? AuxiliaryLearningRate,
            WeightDecay = optimizerSettings?.WeightDecay ?? WeightDecay,
            GainShareBlockDepth =
                optimizerSettings?.GainShareBlockDepth
                ?? GainShareBlockDepth,
            GainShareBeta1 =
                optimizerSettings?.GainShareBeta1 ?? GainShareBeta1,
            GainShareBeta2 =
                optimizerSettings?.GainShareBeta2 ?? GainShareBeta2,
            GainShareEpsilon =
                optimizerSettings?.GainShareEpsilon ?? GainShareEpsilon,
            GainShareRho =
                optimizerSettings?.GainShareRho ?? GainShareRho,
            GainShareGamma =
                optimizerSettings?.GainShareGamma ?? GainShareGamma,
            GainShareMinScale =
                optimizerSettings?.GainShareMinScale ?? GainShareMinScale,
            GainShareMaxScale =
                optimizerSettings?.GainShareMaxScale ?? GainShareMaxScale,
            WarmupEpochs =
                schedulerSettings?.WarmupEpochs ?? WarmupEpochs,
            MinimumLearningRateRatio =
                schedulerSettings?.MinimumLearningRateRatio
                ?? MinimumLearningRateRatio,
            ResumeFromCheckpoint =
                Checkpoint?.Resume ?? ResumeFromCheckpoint,
            AutoResume = Checkpoint?.AutoResume ?? AutoResume,
            CheckpointPath = checkpointPath,
        };
    }

    private string ResolveCheckpointPath(
        string fullConfigurationPath,
        string configurationDirectory)
    {
        if (Checkpoint is null)
        {
            return string.IsNullOrWhiteSpace(CheckpointPath)
                ? Path.ChangeExtension(
                    fullConfigurationPath,
                    ".checkpoint.json")
                : Path.GetFullPath(
                    CheckpointPath,
                    configurationDirectory);
        }

        string directory = Checkpoint.Directory
            ?? configurationDirectory;
        if (string.IsNullOrWhiteSpace(directory))
        {
            throw new InvalidDataException(
                "The 'checkpoint.directory' setting cannot be empty or " +
                "whitespace.");
        }

        string fileName = Checkpoint.FileName
            ?? Path.GetFileName(
                Path.ChangeExtension(
                    fullConfigurationPath,
                    ".checkpoint.json"));
        if (string.IsNullOrWhiteSpace(fileName)
            || Path.IsPathRooted(fileName)
            || !string.Equals(
                Path.GetFileName(fileName),
                fileName,
                StringComparison.Ordinal)
            || fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new InvalidDataException(
                "The 'checkpoint.fileName' setting must be a valid file " +
                "name without a directory component.");
        }

        try
        {
            string fullDirectory = Path.GetFullPath(
                directory,
                configurationDirectory);
            return Path.Combine(fullDirectory, fileName);
        }
        catch (Exception exception) when (
            exception is ArgumentException
            or NotSupportedException
            or PathTooLongException)
        {
            throw new InvalidDataException(
                "The checkpoint directory could not be resolved.",
                exception);
        }
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

sealed record ClassificationCheckpointConfiguration
{
    public string Directory { get; init; } = string.Empty;

    public string? FileName { get; init; }

    public bool? Resume { get; init; }

    public bool? AutoResume { get; init; }
}

sealed record ClassificationOptimizationConfiguration
{
    public ClassificationOptimizerConfiguration Optimizer { get; init; } =
        new();

    public ClassificationSchedulerConfiguration Scheduler { get; init; } =
        new();
}

sealed record ClassificationOptimizerConfiguration
{
    public string? Type { get; init; }

    public float? LearningRate { get; init; }

    public float? AuxiliaryLearningRate { get; init; }

    public float? WeightDecay { get; init; }

    public int? GainShareBlockDepth { get; init; }

    public float? GainShareBeta1 { get; init; }

    public float? GainShareBeta2 { get; init; }

    public float? GainShareEpsilon { get; init; }

    public float? GainShareRho { get; init; }

    public float? GainShareGamma { get; init; }

    public float? GainShareMinScale { get; init; }

    public float? GainShareMaxScale { get; init; }
}

sealed record ClassificationSchedulerConfiguration
{
    public string? Type { get; init; }

    public int? WarmupEpochs { get; init; }

    public float? MinimumLearningRateRatio { get; init; }
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
