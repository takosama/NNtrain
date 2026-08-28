using System.Text.Json;

namespace NNtrain;

/// <summary>
/// Reads and normalizes one training document exactly once, then dispatches
/// its already-normalized representation to the task-specific compatibility
/// loader.
/// </summary>
internal static class ConfigLoader
{
    internal static CanonicalTrainingSpec Load(string configurationPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configurationPath);

        string fullPath = Path.GetFullPath(configurationPath);
        string json = File.ReadAllText(fullPath);
        TrainingConfigurationV2.NormalizedConfiguration normalized =
            TrainingConfigurationV2.Normalize(json);
        int? sourceSchemaVersion = normalized.IsV2
            ? TrainingConfigurationV2.CurrentSchemaVersion
            : null;

        return ResolveTaskKind(normalized) switch
        {
            CanonicalTrainingTaskKind.WikiLanguageModel =>
                new CanonicalWikiTrainingSpec(
                    fullPath,
                    sourceSchemaVersion,
                    WikiTrainingConfiguration.LoadNormalized(
                        fullPath,
                        normalized)),
            _ => new CanonicalClassificationTrainingSpec(
                fullPath,
                sourceSchemaVersion,
                TrainingConfiguration.LoadNormalized(fullPath, normalized)),
        };
    }

    internal static CanonicalTrainingTaskKind ResolveTaskKind(
        TrainingConfigurationV2.NormalizedConfiguration normalized)
    {
        ArgumentNullException.ThrowIfNull(normalized);
        if (normalized.IsV2)
        {
            return string.Equals(
                normalized.TaskType,
                TrainingConfigurationV2.LanguageModelTask,
                StringComparison.OrdinalIgnoreCase)
                    ? CanonicalTrainingTaskKind.WikiLanguageModel
                    : CanonicalTrainingTaskKind.ImageClassification;
        }

        using JsonDocument document = JsonDocument.Parse(
            normalized.Json,
            new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling = JsonCommentHandling.Skip,
            });
        if (document.RootElement.ValueKind != JsonValueKind.Object)
            return CanonicalTrainingTaskKind.ImageClassification;

        foreach (JsonProperty property in
            document.RootElement.EnumerateObject())
        {
            if (string.Equals(
                    property.Name,
                    "task",
                    StringComparison.OrdinalIgnoreCase)
                && property.Value.ValueKind == JsonValueKind.String
                && string.Equals(
                    property.Value.GetString(),
                    WikiTrainingConfiguration.TaskName,
                    StringComparison.OrdinalIgnoreCase))
            {
                return CanonicalTrainingTaskKind.WikiLanguageModel;
            }
        }
        return CanonicalTrainingTaskKind.ImageClassification;
    }
}
