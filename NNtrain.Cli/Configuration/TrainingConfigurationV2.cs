using System.Text.Json;
using System.Text.Json.Nodes;

namespace NNtrain;

internal static class TrainingConfigurationV2
{
    internal const int CurrentSchemaVersion = 2;
    internal const string ClassificationTask = "image-classification";
    internal const string LanguageModelTask = "wiki-language-model";

    private static readonly HashSet<string> SectionNames = new(
        [
            "schemaVersion",
            "task",
            "runtime",
            "data",
            "model",
            "training",
            "optimization",
            "checkpoint",
            "reporting",
        ],
        StringComparer.OrdinalIgnoreCase);

    internal static NormalizedConfiguration Normalize(string json)
    {
        ArgumentNullException.ThrowIfNull(json);
        JsonNode? parsed = JsonNode.Parse(
            json,
            new JsonNodeOptions { PropertyNameCaseInsensitive = true },
            new JsonDocumentOptions
            {
                AllowTrailingCommas = true,
                CommentHandling = JsonCommentHandling.Skip,
            });
        if (parsed is not JsonObject root)
            return new NormalizedConfiguration(null, json, IsV2: false);
        if (!TryGet(root, "schemaVersion", out JsonNode? versionNode))
            return new NormalizedConfiguration(null, json, IsV2: false);

        int? version = versionNode?.GetValue<int?>();
        if (version != CurrentSchemaVersion)
        {
            throw new InvalidDataException(
                $"Unsupported training configuration schemaVersion " +
                $"'{version}'. Expected {CurrentSchemaVersion}.");
        }

        string[] unknown = root
            .Select(pair => pair.Key)
            .Where(name => !SectionNames.Contains(name))
            .ToArray();
        if (unknown.Length != 0)
        {
            throw new InvalidDataException(
                $"Unknown v2 configuration section '{unknown[0]}'.");
        }

        JsonObject task = RequireObject(root, "task");
        string taskType = RequireString(task, "type");
        if (!string.Equals(
                taskType,
                ClassificationTask,
                StringComparison.OrdinalIgnoreCase)
            && !string.Equals(
                taskType,
                LanguageModelTask,
                StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Unsupported task.type '{taskType}'. Supported values are " +
                $"'{ClassificationTask}' and '{LanguageModelTask}'.");
        }

        var flattened = new JsonObject(
            new JsonNodeOptions { PropertyNameCaseInsensitive = true });
        if (string.Equals(
            taskType,
            LanguageModelTask,
            StringComparison.OrdinalIgnoreCase))
        {
            flattened["task"] = WikiTrainingConfiguration.TaskName;
        }

        MergeProperties(task, flattened, excludedName: "type");
        MergeSection(root, "runtime", flattened);
        MergeSection(root, "data", flattened);
        MergeSection(root, "training", flattened);
        MergeSection(root, "reporting", flattened);

        if (TryGet(root, "model", out JsonNode? modelNode))
        {
            JsonObject model = RequireObject(modelNode, "model");
            if (string.Equals(
                taskType,
                ClassificationTask,
                StringComparison.OrdinalIgnoreCase))
            {
                AddUnique(flattened, "model", model.DeepClone());
            }
            else
            {
                MergeProperties(model, flattened);
            }
        }

        CopySection(root, "optimization", flattened);
        CopySection(root, "checkpoint", flattened);

        return new NormalizedConfiguration(
            taskType,
            flattened.ToJsonString(new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            }),
            IsV2: true);
    }

    private static void MergeSection(
        JsonObject root,
        string sectionName,
        JsonObject target)
    {
        if (!TryGet(root, sectionName, out JsonNode? node))
            return;
        MergeProperties(RequireObject(node, sectionName), target);
    }

    private static void CopySection(
        JsonObject root,
        string sectionName,
        JsonObject target)
    {
        if (!TryGet(root, sectionName, out JsonNode? node))
            return;
        AddUnique(
            target,
            sectionName,
            RequireObject(node, sectionName).DeepClone());
    }

    private static void MergeProperties(
        JsonObject source,
        JsonObject target,
        string? excludedName = null)
    {
        foreach ((string name, JsonNode? value) in source)
        {
            if (excludedName is not null
                && string.Equals(
                    name,
                    excludedName,
                    StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }
            AddUnique(target, name, value?.DeepClone());
        }
    }

    private static void AddUnique(
        JsonObject target,
        string name,
        JsonNode? value)
    {
        if (target.Any(pair => string.Equals(
            pair.Key,
            name,
            StringComparison.OrdinalIgnoreCase)))
        {
            throw new InvalidDataException(
                $"Configuration property '{name}' is specified in more " +
                "than one v2 section.");
        }
        target[name] = value;
    }

    private static JsonObject RequireObject(
        JsonObject parent,
        string name)
    {
        if (!TryGet(parent, name, out JsonNode? node))
            throw new InvalidDataException($"'{name}' section is required.");
        return RequireObject(node, name);
    }

    private static JsonObject RequireObject(JsonNode? node, string name)
        => node as JsonObject
            ?? throw new InvalidDataException(
                $"'{name}' must be a JSON object.");

    private static string RequireString(JsonObject parent, string name)
    {
        if (!TryGet(parent, name, out JsonNode? node)
            || node is null)
        {
            throw new InvalidDataException($"'{name}' is required.");
        }
        string? value = node.GetValue<string?>();
        return string.IsNullOrWhiteSpace(value)
            ? throw new InvalidDataException($"'{name}' cannot be empty.")
            : value;
    }

    private static bool TryGet(
        JsonObject value,
        string name,
        out JsonNode? result)
    {
        foreach ((string propertyName, JsonNode? propertyValue) in value)
        {
            if (string.Equals(
                propertyName,
                name,
                StringComparison.OrdinalIgnoreCase))
            {
                result = propertyValue;
                return true;
            }
        }
        result = null;
        return false;
    }

    internal sealed record NormalizedConfiguration(
        string? TaskType,
        string Json,
        bool IsV2);
}
