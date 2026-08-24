using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain;

internal sealed record GenerationConfiguration
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        AllowTrailingCommas = true,
        PropertyNameCaseInsensitive = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        ReadCommentHandling = JsonCommentHandling.Skip,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
    };

    public string TrainingConfigPath { get; init; } = string.Empty;
    public string SafeTensorsPath { get; init; } = string.Empty;
    public string? TokenizerPath { get; init; }
    public string Prompt { get; init; } = string.Empty;
    public string Sampling { get; init; } = "topK";
    public int MaxNewTokens { get; init; } = 80;
    public float? Temperature { get; init; }

    // Keep the spelling requested by existing generate.json files.
    public float? Templator { get; init; }
    public int TopK { get; init; } = 40;
    public int? Seed { get; init; }

    public float EffectiveTemperature => Temperature ?? Templator ?? 0.8f;
    public bool IsGreedy => string.Equals(
        Sampling,
        "greedy",
        StringComparison.OrdinalIgnoreCase);

    internal static GenerationConfiguration Load(string path)
    {
        string fullPath = Path.GetFullPath(path);
        GenerationConfiguration configuration = JsonSerializer.Deserialize<GenerationConfiguration>(
            File.ReadAllText(fullPath),
            JsonOptions)
            ?? throw new InvalidDataException(
                "Generation configuration cannot be JSON null.");
        configuration.Validate();

        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        return configuration with
        {
            TrainingConfigPath = Path.GetFullPath(
                configuration.TrainingConfigPath,
                directory),
            SafeTensorsPath = Path.GetFullPath(
                configuration.SafeTensorsPath,
                directory),
            TokenizerPath = string.IsNullOrWhiteSpace(configuration.TokenizerPath)
                ? null
                : Path.GetFullPath(configuration.TokenizerPath, directory),
        };
    }

    internal void Validate()
    {
        if (string.IsNullOrWhiteSpace(TrainingConfigPath))
            throw new ArgumentException("trainingConfigPath is required.");
        if (string.IsNullOrWhiteSpace(SafeTensorsPath))
            throw new ArgumentException("safeTensorsPath is required.");
        if (string.IsNullOrEmpty(Prompt))
            throw new ArgumentException("prompt is required.");
        if (!string.Equals(Sampling, "topK", StringComparison.OrdinalIgnoreCase)
            && !IsGreedy)
        {
            throw new ArgumentException(
                "sampling must be either 'topK' or 'greedy'.");
        }
        if (MaxNewTokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(MaxNewTokens));
        if (Temperature is not null && Templator is not null)
        {
            throw new ArgumentException(
                "Specify either temperature or templator, not both.");
        }
        if (!float.IsFinite(EffectiveTemperature)
            || EffectiveTemperature < 0f)
        {
            throw new ArgumentOutOfRangeException(nameof(Temperature));
        }
        if (TopK <= 0)
            throw new ArgumentOutOfRangeException(nameof(TopK));
    }
}
