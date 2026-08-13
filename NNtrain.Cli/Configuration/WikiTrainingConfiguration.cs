using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

sealed record WikiTrainingConfiguration
{
    internal const string TaskName = "gpt_rin_wiki_jp";
    internal const string NekoMuonOptimizer = "nekomuon";
    internal const string AdamWOptimizer = "adamw";
    internal const string WarmupCosineProgressScheduler =
        "warmupCosineProgress";
    internal const string TransformerArchitecture = "transformer";
    internal const string HyenaArchitecture = "hyena";
    internal const string ForgetScanArchitecture = "forgetscan";
    internal const string ForgetMemoryV2Architecture = "forgetmemoryv2";
    internal const string FrogetMemoryV2ArchitectureAlias = "frogetmemoryv2";
    internal const string Float16ModelDType = "float16";
    internal const string Float32ModelDType = "float32";
    internal const string AutoHyenaConvolution = "auto";
    internal const string DirectHyenaConvolution = "direct";
    internal const string FftHyenaConvolution = "fft";

    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        AllowTrailingCommas = true,
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Skip,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    public string Task { get; init; } = TaskName;

    public string DataPath { get; init; } = "data/wiki";

    public string TextColumn { get; init; } = "text";

    public string TokenizerPath { get; init; } = string.Empty;

    public string CheckpointPath { get; init; } = string.Empty;

    public bool ResumeFromCheckpoint { get; init; }

    public bool AutoResume { get; init; }

    public WikiCheckpointConfiguration? Checkpoint { get; init; }

    public int VocabularySize { get; init; } = 2048;

    public int TokenizerTrainingDocuments { get; init; } = 2000;

    public int TokenizerTrainingBytes { get; init; } = 2_000_000;

    public int MaxTrainingDocuments { get; init; } = 10_000;

    public int MaxTrainingTokens { get; init; } = 1_000_000;

    public int MaxDocumentTokens { get; init; } = 4096;

    public float ValidationFraction { get; init; } = 0.05f;

    public int Epochs { get; init; } = 5;

    public int BatchSize { get; init; } = 2;

    public int ContextLength { get; init; } = 64;

    public int ModelWidth { get; init; } = 128;

    public int Heads { get; init; } = 4;

    public int HiddenSize { get; init; } = 512;

    public int Layers { get; init; } = 4;

    public string ModelArchitecture { get; init; } =
        ForgetMemoryV2Architecture;

    public string? ModelDType { get; init; }

    public int ForgetMemoryKeyWidth { get; init; } = 16;

    public int ForgetMemoryValueWidth { get; init; } = 16;

    public float ForgetMemoryRetentionMinimum { get; init; } = 0.5f;

    public float ForgetMemoryRetentionMaximum { get; init; } = 0.99f;

    public int HyenaFilterWidth { get; init; } = 64;

    public string HyenaConvolutionAlgorithm { get; init; } =
        AutoHyenaConvolution;

    public float Dropout { get; init; } = 0.1f;

    public float InitializationScale { get; init; } = 0.02f;

    public string Optimizer { get; init; } = NekoMuonOptimizer;

    public float LearningRate { get; init; } = 3e-4f;

    public float AuxiliaryLearningRate { get; init; } = 3e-4f;

    public int NekoMuonNewtonSchulzInterval { get; init; } = 5;

    public float WarmupPercent { get; init; } = 20f;

    public float WeightDecay { get; init; } = 0.01f;

    public bool AdamWUseBFloat16FirstMoment { get; init; }

    public bool AdamWUseBFloat16SecondMoment { get; init; }

    public WikiOptimizationConfiguration? Optimization { get; init; }

    public int Seed { get; init; } = 1234;

    public int LogEveryBatches { get; init; } = 10;

    public bool ShowLossGraph { get; init; } = true;

    public int GraphUpdateSteps { get; init; } = 100;

    public int DatasetSampleEverySteps { get; init; } = 1000;

    public int DatasetSamplePoolSize { get; init; } = 32;

    public int MaxNewTokens { get; init; } = 80;

    public float Temperature { get; init; } = 0.8f;

    public int TopK { get; init; } = 40;

    public bool UseSimd { get; init; } = true;

    public int MaxDegreeOfParallelism { get; init; }

    internal static bool IsWikiConfiguration(string path)
    {
        using JsonDocument document = JsonDocument.Parse(
            File.ReadAllText(Path.GetFullPath(path)),
            new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling = JsonCommentHandling.Skip,
            });
        if (document.RootElement.ValueKind != JsonValueKind.Object)
            return false;

        foreach (JsonProperty property in document.RootElement.EnumerateObject())
        {
            if (string.Equals(
                    property.Name,
                    "task",
                    StringComparison.OrdinalIgnoreCase)
                && property.Value.ValueKind == JsonValueKind.String)
            {
                return string.Equals(
                    property.Value.GetString(),
                    TaskName,
                    StringComparison.OrdinalIgnoreCase);
            }
        }
        return false;
    }

    internal static WikiTrainingConfiguration Load(string path)
    {
        string fullPath = Path.GetFullPath(path);
        string json = File.ReadAllText(fullPath);
        using JsonDocument document = JsonDocument.Parse(
            json,
            new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling = JsonCommentHandling.Skip,
            });
        WikiTrainingConfiguration configuration =
            JsonSerializer.Deserialize<WikiTrainingConfiguration>(
                json,
                JsonOptions)
            ?? throw new InvalidDataException(
                "Wiki training configuration cannot be JSON null.");
        configuration = ApplyGroupedSettings(
            configuration,
            document.RootElement);
        configuration.Validate();

        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        return configuration with
        {
            DataPath = Path.GetFullPath(configuration.DataPath, directory),
            TokenizerPath = string.IsNullOrWhiteSpace(
                configuration.TokenizerPath)
                ? Path.Combine(directory, "wiki-jp-bpe.json")
                : Path.GetFullPath(configuration.TokenizerPath, directory),
            CheckpointPath = ResolveCheckpointPath(
                configuration,
                fullPath,
                directory),
        };
    }

    private static WikiTrainingConfiguration ApplyGroupedSettings(
        WikiTrainingConfiguration configuration,
        JsonElement root)
    {
        bool hasCheckpoint = HasProperty(root, "checkpoint");
        if (hasCheckpoint)
        {
            RejectLegacyProperties(
                root,
                "checkpoint",
                "checkpointPath",
                "resumeFromCheckpoint",
                "autoResume");
            WikiCheckpointConfiguration checkpoint =
                configuration.Checkpoint
                ?? throw new InvalidDataException(
                    "The 'checkpoint' section cannot be null.");
            if (string.IsNullOrWhiteSpace(checkpoint.Directory))
            {
                throw new InvalidDataException(
                    "checkpoint.directory is required.");
            }
            ValidateCheckpointFileName(checkpoint.FileName);
            configuration = configuration with
            {
                ResumeFromCheckpoint = checkpoint.Resume,
                AutoResume = checkpoint.AutoResume,
            };
        }

        bool hasOptimization = HasProperty(root, "optimization");
        if (!hasOptimization)
            return configuration;

        RejectLegacyProperties(
            root,
            "optimization",
            "optimizer",
            "learningRate",
            "auxiliaryLearningRate",
            "weightDecay",
            "nekoMuonNewtonSchulzInterval",
            "warmupPercent",
            "adamWUseBFloat16FirstMoment",
            "adamWUseBFloat16SecondMoment");
        WikiOptimizationConfiguration optimization =
            configuration.Optimization
            ?? throw new InvalidDataException(
                "The 'optimization' section cannot be null.");
        WikiOptimizerConfiguration optimizer = optimization.Optimizer
            ?? throw new InvalidDataException(
                "optimization.optimizer cannot be null.");
        WikiSchedulerConfiguration scheduler = optimization.Scheduler
            ?? throw new InvalidDataException(
                "optimization.scheduler cannot be null.");
        if (!string.Equals(
                scheduler.Type,
                WarmupCosineProgressScheduler,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"Unsupported scheduler '{scheduler.Type}'. The supported " +
                $"scheduler is '{WarmupCosineProgressScheduler}'.",
                nameof(Optimization));
        }

        return configuration with
        {
            Optimizer = optimizer.Type,
            LearningRate = optimizer.LearningRate,
            AuxiliaryLearningRate = optimizer.AuxiliaryLearningRate,
            WeightDecay = optimizer.WeightDecay,
            NekoMuonNewtonSchulzInterval =
                optimizer.NekoMuonNewtonSchulzInterval,
            AdamWUseBFloat16FirstMoment =
                optimizer.AdamWUseBFloat16FirstMoment,
            AdamWUseBFloat16SecondMoment =
                optimizer.AdamWUseBFloat16SecondMoment,
            WarmupPercent = scheduler.WarmupPercent,
        };
    }

    private static string ResolveCheckpointPath(
        WikiTrainingConfiguration configuration,
        string fullConfigurationPath,
        string configurationDirectory)
    {
        string defaultPath = Path.ChangeExtension(
            fullConfigurationPath,
            ".wiki-model.json");
        if (configuration.Checkpoint is not WikiCheckpointConfiguration grouped)
        {
            return string.IsNullOrWhiteSpace(configuration.CheckpointPath)
                ? defaultPath
                : Path.GetFullPath(
                    configuration.CheckpointPath,
                    configurationDirectory);
        }

        string checkpointDirectory = Path.GetFullPath(
            grouped.Directory,
            configurationDirectory);
        string fileName = string.IsNullOrWhiteSpace(grouped.FileName)
            ? Path.GetFileName(defaultPath)
            : grouped.FileName;
        return Path.Combine(checkpointDirectory, fileName);
    }

    private static void ValidateCheckpointFileName(string? fileName)
    {
        if (string.IsNullOrWhiteSpace(fileName))
            return;
        if (Path.IsPathRooted(fileName)
            || !string.Equals(
                Path.GetFileName(fileName),
                fileName,
                StringComparison.Ordinal)
            || fileName.IndexOfAny(Path.GetInvalidFileNameChars()) >= 0)
        {
            throw new InvalidDataException(
                "checkpoint.fileName must be a file name without a " +
                "directory component.");
        }
    }

    private static void RejectLegacyProperties(
        JsonElement root,
        string sectionName,
        params string[] legacyNames)
    {
        foreach (string legacyName in legacyNames)
        {
            if (!HasProperty(root, legacyName))
                continue;
            throw new InvalidDataException(
                $"'{legacyName}' cannot be used together with the " +
                $"'{sectionName}' section. Move the setting into " +
                $"'{sectionName}'.");
        }
    }

    private static bool HasProperty(JsonElement root, string name)
    {
        if (root.ValueKind != JsonValueKind.Object)
            return false;
        foreach (JsonProperty property in root.EnumerateObject())
        {
            if (string.Equals(
                    property.Name,
                    name,
                    StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }
        return false;
    }

    internal void Validate()
    {
        if (!string.Equals(Task, TaskName, StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                $"Wiki task must be '{TaskName}'.",
                nameof(Task));
        }
        if (string.IsNullOrWhiteSpace(DataPath))
            throw new ArgumentException("Wiki data path is required.", nameof(DataPath));
        if (string.IsNullOrWhiteSpace(TextColumn))
            throw new ArgumentException("Wiki text column is required.", nameof(TextColumn));
        if (VocabularySize < BpeTokenizer.BaseVocabularySize)
        {
            throw new ArgumentOutOfRangeException(
                nameof(VocabularySize),
                VocabularySize,
                $"Vocabulary size must be at least " +
                $"{BpeTokenizer.BaseVocabularySize}.");
        }
        ValidatePositive(TokenizerTrainingDocuments, nameof(TokenizerTrainingDocuments));
        ValidatePositive(TokenizerTrainingBytes, nameof(TokenizerTrainingBytes));
        ValidateNonNegative(MaxTrainingDocuments, nameof(MaxTrainingDocuments));
        ValidateNonNegative(MaxTrainingTokens, nameof(MaxTrainingTokens));
        ValidatePositive(MaxDocumentTokens, nameof(MaxDocumentTokens));
        ValidatePositive(Epochs, nameof(Epochs));
        ValidatePositive(BatchSize, nameof(BatchSize));
        ValidatePositive(ContextLength, nameof(ContextLength));
        ValidatePositive(ModelWidth, nameof(ModelWidth));
        ValidatePositive(Heads, nameof(Heads));
        ValidatePositive(HiddenSize, nameof(HiddenSize));
        ValidatePositive(Layers, nameof(Layers));
        ValidatePositive(
            ForgetMemoryKeyWidth,
            nameof(ForgetMemoryKeyWidth));
        ValidatePositive(
            ForgetMemoryValueWidth,
            nameof(ForgetMemoryValueWidth));
        ValidatePositive(HyenaFilterWidth, nameof(HyenaFilterWidth));
        ValidatePositive(LogEveryBatches, nameof(LogEveryBatches));
        ValidatePositive(GraphUpdateSteps, nameof(GraphUpdateSteps));
        ValidatePositive(
            DatasetSampleEverySteps,
            nameof(DatasetSampleEverySteps));
        ValidatePositive(DatasetSamplePoolSize, nameof(DatasetSamplePoolSize));
        ValidatePositive(MaxNewTokens, nameof(MaxNewTokens));

        if (MaxTrainingTokens > 0
            && MaxTrainingTokens < ContextLength * 2L + 2L)
        {
            throw new ArgumentOutOfRangeException(
                nameof(MaxTrainingTokens),
                MaxTrainingTokens,
                "Maximum training tokens must contain at least two contexts.");
        }
        if (ModelWidth % Heads != 0)
        {
            throw new ArgumentException(
                "Head count must evenly divide model width.",
                nameof(Heads));
        }
        if (!IsArchitecture(TransformerArchitecture)
            && !IsArchitecture(HyenaArchitecture)
            && !IsArchitecture(ForgetScanArchitecture)
            && !IsForgetMemoryV2Architecture())
        {
            throw new ArgumentException(
                $"Unsupported model architecture '{ModelArchitecture}'. " +
                $"Supported architectures are '{TransformerArchitecture}' " +
                $"'{HyenaArchitecture}', '{ForgetScanArchitecture}', and " +
                $"'{ForgetMemoryV2Architecture}'.",
                nameof(ModelArchitecture));
        }
        TensorDType? explicitModelDType = GetExplicitModelDType();
        if (explicitModelDType == TensorDType.Float16
            && !IsForgetMemoryV2Architecture())
        {
            throw new ArgumentException(
                "modelDType 'float16' is currently supported only for the " +
                $"'{ForgetMemoryV2Architecture}' architecture.",
                nameof(ModelDType));
        }
        if (!float.IsFinite(ForgetMemoryRetentionMinimum)
            || !float.IsFinite(ForgetMemoryRetentionMaximum)
            || ForgetMemoryRetentionMinimum < 0f
            || ForgetMemoryRetentionMinimum
                > ForgetMemoryRetentionMaximum
            || ForgetMemoryRetentionMaximum >= 1f)
        {
            throw new ArgumentException(
                "ForgetMemory retention bounds must satisfy " +
                "0 <= minimum <= maximum < 1.",
                nameof(ForgetMemoryRetentionMinimum));
        }
        if (!IsHyenaConvolution(AutoHyenaConvolution)
            && !IsHyenaConvolution(DirectHyenaConvolution)
            && !IsHyenaConvolution(FftHyenaConvolution))
        {
            throw new ArgumentException(
                $"Unsupported Hyena convolution algorithm " +
                $"'{HyenaConvolutionAlgorithm}'. Supported algorithms are " +
                $"'{AutoHyenaConvolution}', '{DirectHyenaConvolution}', and " +
                $"'{FftHyenaConvolution}'.",
                nameof(HyenaConvolutionAlgorithm));
        }
        if (!float.IsFinite(ValidationFraction)
            || ValidationFraction < 0f
            || ValidationFraction >= 0.5f)
        {
            throw new ArgumentOutOfRangeException(nameof(ValidationFraction));
        }
        if (MaxTrainingTokens == 0 && ValidationFraction != 0f)
        {
            throw new ArgumentException(
                "Streaming all-data training requires validationFraction 0.",
                nameof(ValidationFraction));
        }
        if (!float.IsFinite(Dropout) || Dropout < 0f || Dropout >= 1f)
            throw new ArgumentOutOfRangeException(nameof(Dropout));
        if (!float.IsFinite(InitializationScale) || InitializationScale <= 0f)
            throw new ArgumentOutOfRangeException(nameof(InitializationScale));
        if (!IsOptimizer(NekoMuonOptimizer) && !IsOptimizer(AdamWOptimizer))
        {
            throw new ArgumentException(
                $"Unsupported optimizer '{Optimizer}'. Supported optimizers " +
                $"are '{NekoMuonOptimizer}' and '{AdamWOptimizer}'.",
                nameof(Optimizer));
        }
        if (!float.IsFinite(LearningRate) || LearningRate <= 0f)
            throw new ArgumentOutOfRangeException(nameof(LearningRate));
        if (!float.IsFinite(AuxiliaryLearningRate)
            || AuxiliaryLearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(AuxiliaryLearningRate));
        }
        ValidatePositive(
            NekoMuonNewtonSchulzInterval,
            nameof(NekoMuonNewtonSchulzInterval));
        if (!float.IsFinite(WarmupPercent)
            || WarmupPercent < 0f
            || WarmupPercent >= 100f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(WarmupPercent),
                WarmupPercent,
                "Warmup percent must be finite and in [0, 100).");
        }
        if (!float.IsFinite(WeightDecay) || WeightDecay < 0f)
            throw new ArgumentOutOfRangeException(nameof(WeightDecay));
        if (!float.IsFinite(Temperature) || Temperature < 0f)
            throw new ArgumentOutOfRangeException(nameof(Temperature));
        if (TopK < 0)
            throw new ArgumentOutOfRangeException(nameof(TopK));
        ValidateNonNegative(
            MaxDegreeOfParallelism,
            nameof(MaxDegreeOfParallelism));
    }

    private static void ValidatePositive(int value, string name)
    {
        if (value <= 0)
            throw new ArgumentOutOfRangeException(name, value, "Value must be positive.");
    }

    private static void ValidateNonNegative(int value, string name)
    {
        if (value < 0)
        {
            throw new ArgumentOutOfRangeException(
                name,
                value,
                "Value must be non-negative; zero means unlimited.");
        }
    }

    internal bool IsOptimizer(string expectedOptimizer)
        => string.Equals(
            Optimizer,
            expectedOptimizer,
            StringComparison.OrdinalIgnoreCase);

    internal bool IsArchitecture(string expectedArchitecture)
        => string.Equals(
            ModelArchitecture,
            expectedArchitecture,
            StringComparison.OrdinalIgnoreCase);

    internal bool IsForgetMemoryV2Architecture()
        => IsArchitecture(ForgetMemoryV2Architecture)
            || IsArchitecture(FrogetMemoryV2ArchitectureAlias);

    internal TensorDType? GetExplicitModelDType()
    {
        if (ModelDType is null)
            return null;

        if (string.Equals(
            ModelDType,
            Float16ModelDType,
            StringComparison.OrdinalIgnoreCase))
        {
            return TensorDType.Float16;
        }
        if (string.Equals(
            ModelDType,
            Float32ModelDType,
            StringComparison.OrdinalIgnoreCase))
        {
            return TensorDType.Float32;
        }

        throw new ArgumentException(
            $"Unsupported modelDType '{ModelDType}'. Supported values are " +
            $"'{Float16ModelDType}' and '{Float32ModelDType}'.",
            nameof(ModelDType));
    }

    internal TensorDType GetModelDType()
        => GetExplicitModelDType()
            ?? (IsForgetMemoryV2Architecture()
                ? TensorDType.Float16
                : TensorDType.Float32);

    internal bool IsHyenaConvolution(string expectedAlgorithm)
        => string.Equals(
            HyenaConvolutionAlgorithm,
            expectedAlgorithm,
            StringComparison.OrdinalIgnoreCase);

    internal HyenaConvolutionAlgorithm GetHyenaConvolutionAlgorithm()
        => HyenaConvolutionAlgorithm.ToLowerInvariant() switch
        {
            DirectHyenaConvolution => NNtrain.HyenaConvolutionAlgorithm.Direct,
            FftHyenaConvolution => NNtrain.HyenaConvolutionAlgorithm.Fft,
            _ => NNtrain.HyenaConvolutionAlgorithm.Auto,
        };
}
