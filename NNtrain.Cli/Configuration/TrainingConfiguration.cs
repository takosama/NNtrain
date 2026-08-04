using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

sealed record TrainingConfiguration
{
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

    public float LearningRate { get; init; } = 1e-4f;

    public bool UseSimd { get; init; } = true;

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

        if (!float.IsFinite(LearningRate) || LearningRate <= 0f)
        {
            throw new ArgumentOutOfRangeException(
                nameof(LearningRate),
                LearningRate,
                "Learning rate must be finite and positive.");
        }

        if (Model is null)
        {
            throw new ArgumentException(
                "Model configuration cannot be null.",
                nameof(Model));
        }

        Model.Validate();
    }
}

sealed record DatasetConfiguration
{
    internal const string MnistType = "mnist";
    internal const string Cifar100Type = "cifar100";

    public string Type { get; init; } = MnistType;

    public string ImagePath { get; init; } = string.Empty;

    public string LabelPath { get; init; } = string.Empty;

    public string DataPath { get; init; } = string.Empty;

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

sealed record ModelConfiguration
{
    public int Heads { get; init; } = 1;

    public int HiddenSize { get; init; } = 128;

    public int Layers { get; init; } = 32;

    public int Seed { get; init; }

    public float InitializationScale { get; init; } = 0.02f;

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
