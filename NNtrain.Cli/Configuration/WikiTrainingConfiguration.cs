using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

sealed record WikiTrainingConfiguration
{
    internal const string TaskName = "gpt_rin_wiki_jp";
    internal const string NekoMuonOptimizer = "nekomuon";
    internal const string AdamWOptimizer = "adamw";
    internal const string GainShareAdamWOptimizer = "gainshareadamw";
    internal const string LionOptimizer = "lion";
    internal const string WarmupCosineProgressScheduler =
        "warmupCosineProgress";
    internal const string TransformerArchitecture = "transformer";
    internal const string HyenaArchitecture = "hyena";
    internal const string ForgetScanArchitecture = "forgetscan";
    internal const string ForgetMemoryV2Architecture = "forgetmemoryv2";
    internal const string ForgetMemoryV2ArchitectureAlias = "frogetmemoryv2";
    internal const string ForgetMemoryV3Architecture = "forgetmemoryv3";
    internal const string ForgetMemoryDrnArchitecture = "forgetmemorydrn";
    internal const string Float32PrecisionMode = TensorPrecisionModeNames.Float32;
    internal const string BFloat16PrecisionMode = TensorPrecisionModeNames.BFloat16;
    internal const string Mix16_32PrecisionMode = TensorPrecisionModeNames.Mix16_32;
    internal const string Bfp8PrecisionMode = TensorPrecisionModeNames.Bfp8;
    internal const string Mix8_32PrecisionMode = TensorPrecisionModeNames.Mix8_32;
    internal const string LegacyFloat16ModelDType = "float16";
    internal const string CpuDevice = "cpu";
    internal const string CudaDevice = "cuda";
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

    /// <summary>
    /// Documents held back to randomize reading order. Zero reads the globally
    /// shuffled Parquet row groups without additional document mixing. A
    /// larger buffer mixes individual documents further but costs memory,
    /// because the held documents are raw text: roughly 2.7 KB per Japanese
    /// Wikipedia article, so the default holds on the order of 700 MB and
    /// delays the first training step until the buffer is full.
    /// </summary>
    public int ShuffleBufferSize { get; init; } = 262_144;

    public float ValidationFraction { get; init; } = 0.05f;

    public int Epochs { get; init; } = 5;

    public int BatchSize { get; init; } = 2;

    /// <summary>
    /// Number of microbatches whose gradients form one optimizer update.
    /// BatchSize remains the CUDA microbatch size; the effective batch size is
    /// BatchSize multiplied by this value.
    /// </summary>
    public int GradientAccumulationSteps { get; init; } = 1;

    public int ContextLength { get; init; } = 64;

    public int ModelWidth { get; init; } = 128;

    public int Heads { get; init; } = 4;

    public int HiddenSize { get; init; } = 512;

    public int Layers { get; init; } = 4;

    public string ModelArchitecture { get; init; } =
        ForgetMemoryV3Architecture;

    /// <summary>
    /// Numeric execution contract: float32, bfloat16, mix16_32 (fp16_32
    /// alias), bfp8, or mix8_32.
    /// </summary>
    public string? PrecisionMode { get; init; }

    /// <summary>Canonical numeric execution setting.</summary>
    public string? Precision { get; init; }

    private int _bfp8BlockSize =
        Bfp8QuantizationDescriptor.DefaultBlockSize;
    private bool _bfp8BlockSizeWasSet;

    [JsonPropertyName("bfp8_block_size")]
    public int Bfp8BlockSize
    {
        get => _bfp8BlockSize;
        init
        {
            _bfp8BlockSize = value;
            _bfp8BlockSizeWasSet = true;
        }
    }

    internal bool HasExplicitBfp8BlockSize => _bfp8BlockSizeWasSet;

    /// <summary>Legacy physical-storage setting. Use precisionMode.</summary>
    public string? ModelDType { get; init; }

    public bool TieWordEmbeddings { get; init; }

    public string Device { get; init; } = CpuDevice;

    public int DeviceIndex { get; init; }

    public int[]? DeviceIndices { get; init; }

    public bool AdaptiveCudaSharding { get; init; } = true;

    public double CudaShardEmaAlpha { get; init; } = 0.2d;

    public double CudaMinimumRelativeShardSize { get; init; } = 0.5d;

    public int CudaMaximumBatchAdjustmentPerStep { get; init; } = 1;

    /// <summary>
    /// Upper bound for cached compiled CUDA Graph allocations. The active
    /// shape is retained; older completed shapes are retired first.
    /// </summary>
    public int CudaGraphCacheBudgetMiB { get; init; } = 512;

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

    public string? NekoMuonNewtonSchulzDepthMode { get; init; }

    public float? NekoMuonNewtonSchulzDepth { get; init; }

    public int GainShareBlockDepth { get; init; } = 1;

    public float GainShareBeta1 { get; init; } = 0.9f;

    public float GainShareBeta2 { get; init; } = 0.999f;

    public float GainShareEpsilon { get; init; } = 1e-8f;

    public float GainShareRho { get; init; } = 0.95f;

    public float GainShareGamma { get; init; } = 1f;

    public float GainShareMinScale { get; init; } = 0.5f;

    public float GainShareMaxScale { get; init; } = 2f;

    public float WarmupPercent { get; init; } = 20f;

    public float WeightDecay { get; init; } = 0.01f;

    public WikiOptimizationConfiguration? Optimization { get; init; }

    public int Seed { get; init; } = 1234;

    public int LogEveryBatches { get; init; } = 10;

    public bool ShowLossGraph { get; init; } = true;

    public int GraphUpdateSteps { get; init; } = 100;

    public int DatasetSampleEverySteps { get; init; } = 2000;

    public int DatasetSamplePoolSize { get; init; } = 32;

    public int MaxNewTokens { get; init; } = 80;

    public float Temperature { get; init; } = 0.8f;

    public int TopK { get; init; } = 40;

    public bool UseSimd { get; init; } = true;

    public int MaxDegreeOfParallelism { get; init; }

    internal static bool IsWikiConfiguration(string path)
    {
        string json = File.ReadAllText(Path.GetFullPath(path));
        TrainingConfigurationV2.NormalizedConfiguration normalized =
            TrainingConfigurationV2.Normalize(json);
        if (normalized.IsV2)
        {
            return string.Equals(
                normalized.TaskType,
                TrainingConfigurationV2.LanguageModelTask,
                StringComparison.OrdinalIgnoreCase);
        }

        using JsonDocument document = JsonDocument.Parse(
            json,
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
        TrainingConfigurationV2.NormalizedConfiguration normalized =
            TrainingConfigurationV2.Normalize(json);
        return LoadNormalized(fullPath, normalized);
    }

    internal static WikiTrainingConfiguration LoadNormalized(
        string path,
        TrainingConfigurationV2.NormalizedConfiguration normalized)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(normalized);

        string fullPath = Path.GetFullPath(path);
        if (normalized.IsV2
            && !string.Equals(
                normalized.TaskType,
                TrainingConfigurationV2.LanguageModelTask,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Configuration task '{normalized.TaskType}' is not a " +
                "wiki-language-model task.");
        }
        string json = normalized.Json;
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
            "nekoMuonNewtonSchulzDepthMode",
            "nekoMuonNewtonSchulzDepth",
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
            NekoMuonNewtonSchulzDepthMode =
                optimizer.NekoMuonNewtonSchulzDepthMode,
            NekoMuonNewtonSchulzDepth =
                optimizer.NekoMuonNewtonSchulzDepth,
            GainShareBlockDepth = optimizer.GainShareBlockDepth,
            GainShareBeta1 = optimizer.GainShareBeta1,
            GainShareBeta2 = optimizer.GainShareBeta2,
            GainShareEpsilon = optimizer.GainShareEpsilon,
            GainShareRho = optimizer.GainShareRho,
            GainShareGamma = optimizer.GainShareGamma,
            GainShareMinScale = optimizer.GainShareMinScale,
            GainShareMaxScale = optimizer.GainShareMaxScale,
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
        ValidateNonNegative(MaxDocumentTokens, nameof(MaxDocumentTokens));
        ValidateNonNegative(ShuffleBufferSize, nameof(ShuffleBufferSize));
        ValidatePositive(Epochs, nameof(Epochs));
        ValidatePositive(BatchSize, nameof(BatchSize));
        ValidatePositive(
            GradientAccumulationSteps,
            nameof(GradientAccumulationSteps));
        if ((long)BatchSize * GradientAccumulationSteps > int.MaxValue)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GradientAccumulationSteps),
                GradientAccumulationSteps,
                "Effective batch size must not exceed Int32.MaxValue.");
        }
        ValidatePositive(ContextLength, nameof(ContextLength));
        ValidatePositive(ModelWidth, nameof(ModelWidth));
        ValidatePositive(Heads, nameof(Heads));
        ValidatePositive(HiddenSize, nameof(HiddenSize));
        ValidatePositive(Layers, nameof(Layers));
        ValidateNonNegative(DeviceIndex, nameof(DeviceIndex));
        if (DeviceIndices is { Length: 0 })
            throw new ArgumentException("deviceIndices cannot be empty.", nameof(DeviceIndices));
        if (DeviceIndices is not null
            && (DeviceIndices.Any(index => index < 0)
                || DeviceIndices.Distinct().Count() != DeviceIndices.Length))
        {
            throw new ArgumentException(
                "deviceIndices must contain unique, non-negative indices.",
                nameof(DeviceIndices));
        }
        if (!double.IsFinite(CudaShardEmaAlpha)
            || CudaShardEmaAlpha <= 0d
            || CudaShardEmaAlpha > 1d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(CudaShardEmaAlpha),
                "cudaShardEmaAlpha must be in (0, 1].");
        }
        if (!double.IsFinite(CudaMinimumRelativeShardSize)
            || CudaMinimumRelativeShardSize <= 0d
            || CudaMinimumRelativeShardSize > 1d)
        {
            throw new ArgumentOutOfRangeException(
                nameof(CudaMinimumRelativeShardSize),
                "cudaMinimumRelativeShardSize must be in (0, 1].");
        }
        ValidatePositive(
            CudaMaximumBatchAdjustmentPerStep,
            nameof(CudaMaximumBatchAdjustmentPerStep));
        ValidatePositive(
            CudaGraphCacheBudgetMiB,
            nameof(CudaGraphCacheBudgetMiB));
        ValidatePositive(
            ForgetMemoryKeyWidth,
            nameof(ForgetMemoryKeyWidth));
        ValidatePositive(
            ForgetMemoryValueWidth,
            nameof(ForgetMemoryValueWidth));
        ValidatePositive(HyenaFilterWidth, nameof(HyenaFilterWidth));
        ValidatePositive(LogEveryBatches, nameof(LogEveryBatches));
        ValidatePositive(GraphUpdateSteps, nameof(GraphUpdateSteps));
        if (DatasetSampleEverySteps < 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(DatasetSampleEverySteps),
                DatasetSampleEverySteps,
                "Dataset sample interval must be zero (disabled) or positive.");
        }
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
            && !IsForgetMemoryArchitecture())
        {
            throw new ArgumentException(
                $"Unsupported model architecture '{ModelArchitecture}'. " +
                $"Supported architectures are '{TransformerArchitecture}' " +
                $"'{HyenaArchitecture}', '{ForgetScanArchitecture}', and " +
                $"'{ForgetMemoryV2Architecture}', and " +
                $"'{ForgetMemoryV3Architecture}', and " +
                $"'{ForgetMemoryDrnArchitecture}'.",
                nameof(ModelArchitecture));
        }
        if (Bfp8BlockSize <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(Bfp8BlockSize),
                Bfp8BlockSize,
                "BFP8 block size must be positive.");
        }
        TensorDType? explicitModelDType = GetExplicitModelDType();
        if (explicitModelDType is TensorDType.Float16
                or TensorDType.BFloat16
                or TensorDType.Bfp8
            && !IsForgetMemoryArchitecture()
            && !IsArchitecture(TransformerArchitecture))
        {
            throw new ArgumentException(
                "Reduced-precision modes are currently supported only for the " +
                "Transformer and ForgetMemory architectures.",
                nameof(PrecisionMode));
        }
        _ = GetExecutionDevice();
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
        if (!IsOptimizer(NekoMuonOptimizer)
            && !IsOptimizer(AdamWOptimizer)
            && !IsOptimizer(GainShareAdamWOptimizer)
            && !IsOptimizer(LionOptimizer))
        {
            throw new ArgumentException(
                $"Unsupported optimizer '{Optimizer}'. Supported optimizers " +
                $"are '{NekoMuonOptimizer}', '{AdamWOptimizer}', " +
                $"'{GainShareAdamWOptimizer}', and '{LionOptimizer}'.",
                nameof(Optimizer));
        }
        ValidateGainShareSettings();
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
        ValidateNekoMuonNewtonSchulzDepthPolicy();
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

    private void ValidateGainShareSettings()
    {
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
        ValidateGainShareUnitInterval(
            GainShareRho,
            nameof(GainShareRho),
            "rho");

        if (!float.IsFinite(GainShareEpsilon) || GainShareEpsilon <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareEpsilon),
                GainShareEpsilon,
                "GainShare epsilon must be finite and positive.");
        }

        if (!float.IsFinite(GainShareGamma) || GainShareGamma < 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(GainShareGamma),
                GainShareGamma,
                "GainShare gamma must be finite and non-negative.");
        }

        if (!float.IsFinite(GainShareMinScale) || GainShareMinScale <= 0f)
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
                "GainShare maximum scale must be finite and not less than " +
                "the minimum scale.");
        }
    }

    private void ValidateNekoMuonNewtonSchulzDepthPolicy()
    {
        if (NekoMuonNewtonSchulzDepthMode is null)
        {
            if (NekoMuonNewtonSchulzDepth.HasValue)
            {
                throw new ArgumentException(
                    "nekoMuonNewtonSchulzDepth requires " +
                    "nekoMuonNewtonSchulzDepthMode.",
                    nameof(NekoMuonNewtonSchulzDepth));
            }
            return;
        }

        if (!IsOptimizer(NekoMuonOptimizer))
        {
            throw new ArgumentException(
                "NekoMuon Newton-Schulz depth policy can only be used with " +
                "the NekoMuon optimizer.",
                nameof(NekoMuonNewtonSchulzDepthMode));
        }

        if (!Enum.TryParse(
                NekoMuonNewtonSchulzDepthMode,
                ignoreCase: true,
                out global::NNtrain.NekoMuonNewtonSchulzDepthMode mode)
            || !Enum.IsDefined(mode))
        {
            throw new ArgumentException(
                $"Unsupported NekoMuon Newton-Schulz depth mode " +
                $"'{NekoMuonNewtonSchulzDepthMode}'. Expected adaptive, " +
                "minimum, or fixed.",
                nameof(NekoMuonNewtonSchulzDepthMode));
        }

        if (mode
            == global::NNtrain.NekoMuonNewtonSchulzDepthMode.Adaptive)
        {
            if (NekoMuonNewtonSchulzDepth.HasValue)
            {
                throw new ArgumentException(
                    "Adaptive NekoMuon Newton-Schulz depth must not specify " +
                    "nekoMuonNewtonSchulzDepth.",
                    nameof(NekoMuonNewtonSchulzDepth));
            }
            return;
        }

        if (!NekoMuonNewtonSchulzDepth.HasValue)
        {
            throw new ArgumentException(
                $"NekoMuon Newton-Schulz depth mode '{mode}' requires " +
                "nekoMuonNewtonSchulzDepth.",
                nameof(NekoMuonNewtonSchulzDepth));
        }

        float depth = NekoMuonNewtonSchulzDepth.Value;
        int maximumDepth = new NekoMuonOptions().MaxNewtonSchulzSteps;
        if (!float.IsFinite(depth) || depth < 0f || depth > maximumDepth)
        {
            throw new ArgumentOutOfRangeException(
                nameof(NekoMuonNewtonSchulzDepth),
                depth,
                $"NekoMuon Newton-Schulz depth must be finite and in " +
                $"[0, {maximumDepth}].");
        }
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

    internal bool HasNekoMuonNewtonSchulzDepthPolicyOverride
        => NekoMuonNewtonSchulzDepthMode is not null;

    internal global::NNtrain.NekoMuonNewtonSchulzDepthMode
        GetNekoMuonNewtonSchulzDepthMode()
        => NekoMuonNewtonSchulzDepthMode is null
            ? global::NNtrain.NekoMuonNewtonSchulzDepthMode.Adaptive
            : Enum.Parse<global::NNtrain.NekoMuonNewtonSchulzDepthMode>(
                NekoMuonNewtonSchulzDepthMode,
                ignoreCase: true);

    internal float GetNekoMuonNewtonSchulzDepth()
        => NekoMuonNewtonSchulzDepth ?? 0f;

    internal bool IsArchitecture(string expectedArchitecture)
        => string.Equals(
            ModelArchitecture,
            expectedArchitecture,
            StringComparison.OrdinalIgnoreCase);

    internal bool IsForgetMemoryV2Architecture()
        => IsArchitecture(ForgetMemoryV2Architecture)
            || IsArchitecture(ForgetMemoryV2ArchitectureAlias);

    internal bool IsForgetMemoryV3Architecture()
        => IsArchitecture(ForgetMemoryV3Architecture);

    internal bool IsForgetMemoryDrnArchitecture()
        => IsArchitecture(ForgetMemoryDrnArchitecture);

    internal bool IsForgetMemoryArchitecture()
        => IsForgetMemoryV2Architecture()
            || IsForgetMemoryV3Architecture()
            || IsForgetMemoryDrnArchitecture();

    internal TensorPrecisionMode? GetExplicitPrecisionMode()
    {
        int configuredNames = (Precision is null ? 0 : 1)
            + (PrecisionMode is null ? 0 : 1)
            + (ModelDType is null ? 0 : 1);
        if (configuredNames > 1)
        {
            throw new ArgumentException(
                "precision, precisionMode, and the legacy modelDType " +
                "setting cannot be combined.",
                nameof(Precision));
        }

        string? configured = Precision ?? PrecisionMode ?? ModelDType;
        if (configured is null)
            return null;

        if (ModelDType is not null
            && string.Equals(
                configured,
                LegacyFloat16ModelDType,
                StringComparison.OrdinalIgnoreCase))
        {
            return TensorPrecisionMode.Mix16_32;
        }

        try
        {
            return TensorPrecisionModeNames.Parse(configured);
        }
        catch (ArgumentException exception)
        {
            string parameterName = Precision is not null
                ? nameof(Precision)
                : PrecisionMode is not null
                    ? nameof(PrecisionMode)
                    : nameof(ModelDType);
            throw new ArgumentException(
                $"Unsupported precision mode '{configured}'. Supported values " +
                $"are {TensorPrecisionModeNames.SupportedValuesDescription}.",
                parameterName,
                exception);
        }
    }

    internal TensorDType? GetExplicitModelDType()
        => GetExplicitPrecisionMode()?.ToStorageDType();

    internal TensorPrecisionMode GetPrecisionMode()
        => GetExplicitPrecisionMode()
            ?? (IsForgetMemoryArchitecture()
                ? TensorPrecisionMode.Mix16_32
                : TensorPrecisionMode.Float32);

    internal TensorDType GetModelDType()
        => GetPrecisionMode().ToStorageDType();

    internal TensorDevice GetExecutionDevice()
    {
        if (string.Equals(Device, CpuDevice, StringComparison.OrdinalIgnoreCase))
            return TensorDevice.Cpu;
        if (string.Equals(Device, CudaDevice, StringComparison.OrdinalIgnoreCase))
            return TensorDevice.Cuda;

        throw new ArgumentException(
            $"Unsupported device '{Device}'. Supported values are " +
            $"'{CpuDevice}' and '{CudaDevice}'.",
            nameof(Device));
    }

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
