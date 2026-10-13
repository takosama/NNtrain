using System.IO;
using System.Text.Json;
using Microsoft.AspNetCore.Http;

namespace NNtrain.Gui;

/// <summary>Parses and validates the local API contract before acquiring the model gate.</summary>
internal static class LocalApiRequests
{
    internal static LocalChatRequest ParseChat(JsonElement json, string? initialLora)
    {
        string model = ValidateFilePath(RequiredString(json, "model"), ".gguf", "model");
        string? lora = ResolveLora(json, initialLora);
        string? mmproj = ResolveMmproj(json);
        int[]? devices = OptionalDevices(json);
        string? gemmaBackend = OptionalGemmaBackend(json);
        ChatTurn[] messages = RequiredMessages(json);
        int maxTokens = OptionalInt(json, "max_completion_tokens")
            ?? OptionalInt(json, "max_tokens") ?? 512;
        if (maxTokens is < 1 or > 8192)
            throw new LocalApiException(400, "max_tokens must be between 1 and 8192.", "max_tokens");
        float temperature = OptionalFloat(json, "temperature") ?? 0.6f;
        float topP = OptionalFloat(json, "top_p") ?? 0.95f;
        int topK = OptionalInt(json, "top_k") ?? 20;
        if (!float.IsFinite(temperature) || temperature is < 0f or > 2f)
            throw new LocalApiException(400, "temperature must be between 0 and 2.", "temperature");
        if (!float.IsFinite(topP) || topP is <= 0f or > 1f)
            throw new LocalApiException(400, "top_p must be greater than 0 and at most 1.", "top_p");
        if (topK is < 1 or > 256)
            throw new LocalApiException(400, "top_k must be between 1 and 256.", "top_k");
        bool stream = OptionalBoolean(json, "stream") ?? false;
        bool think = OptionalBoolean(json, "think") ?? true;
        bool primeHistory = OptionalBoolean(json, "prime_history") ?? false;
        var sampling = new GenerationSampling(temperature, topP, topK);
        return new LocalChatRequest(model, lora, mmproj, devices, gemmaBackend,
            messages, maxTokens, stream, think, primeHistory, sampling);
    }

    internal static string? ResolveLora(JsonElement json, string? initialLora)
    {
        if (!json.TryGetProperty("lora", out JsonElement value)) return initialLora;
        if (value.ValueKind == JsonValueKind.Null) return null;
        if (value.ValueKind != JsonValueKind.String)
            throw new LocalApiException(400, "lora must be a path or null.", "lora");
        string? path = value.GetString();
        return string.IsNullOrWhiteSpace(path) ? null : ValidateFilePath(path, ".bin", "lora");
    }

    internal static string ValidateFilePath(string path, string extension, string parameter)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new LocalApiException(400, $"{parameter} path is required.", parameter);
        string normalized = path.Replace('/', '\\');
        if (normalized.StartsWith("\\\\", StringComparison.Ordinal)
            || normalized.StartsWith("\\", StringComparison.Ordinal))
            throw new LocalApiException(400, "Only local drive paths are accepted.", parameter);
        if (path.Split(['/', '\\']).Any(segment => segment == ".."))
            throw new LocalApiException(400, $"{parameter} path must not contain '..' segments.", parameter);
        string fullPath;
        try { fullPath = Path.GetFullPath(path); }
        catch (Exception error) when (error is ArgumentException or NotSupportedException or PathTooLongException)
        {
            throw new LocalApiException(400, $"Invalid {parameter} path.", parameter);
        }
        string? root = Path.GetPathRoot(fullPath);
        if (root is null || root.StartsWith("\\\\", StringComparison.Ordinal)
            || new DriveInfo(root).DriveType == DriveType.Network)
            throw new LocalApiException(400, "Network model paths are not accepted.", parameter);
        string actualExtension = Path.GetExtension(fullPath);
        bool accepted = actualExtension.Equals(extension, StringComparison.OrdinalIgnoreCase) ||
            parameter == "lora" && actualExtension.Equals(".gguf", StringComparison.OrdinalIgnoreCase);
        if (!accepted)
            throw new LocalApiException(400, parameter == "lora"
                ? "lora must be an NNtrain adapter.bin or GGUF LoRA adapter."
                : $"{parameter} must be a {extension} file.", parameter);
        if (!File.Exists(fullPath))
            throw new LocalApiException(400, $"{parameter} file does not exist: {fullPath}", parameter);
        return fullPath;
    }

    internal static async Task<JsonElement> ReadObjectAsync(HttpContext context, CancellationToken ct)
    {
        if (!context.Request.HasJsonContentType())
            throw new LocalApiException(415, "Content-Type must be application/json.");
        JsonElement json;
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(15));
        try { json = await context.Request.ReadFromJsonAsync<JsonElement>(cancellationToken: timeout.Token).ConfigureAwait(false); }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        { throw new LocalApiException(408, "Request body read timed out."); }
        catch (JsonException error) { throw new LocalApiException(400, $"Invalid JSON: {error.Message}"); }
        if (json.ValueKind != JsonValueKind.Object)
            throw new LocalApiException(400, "Request body must be a JSON object.");
        return json;
    }

    internal static string RequiredString(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(value.GetString()))
            throw new LocalApiException(400, $"{name} is required and must be a nonempty string.", name);
        return value.GetString()!;
    }

    internal static string? OptionalString(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.String)
            throw new LocalApiException(400, $"{name} must be a string.", name);
        return value.GetString();
    }

    internal static int? OptionalInt(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int result))
            throw new LocalApiException(400, $"{name} must be an integer.", name);
        return result;
    }

    internal static float? OptionalFloat(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float result))
            throw new LocalApiException(400, $"{name} must be a number.", name);
        return result;
    }

    internal static bool? OptionalBoolean(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new LocalApiException(400, $"{name} must be a boolean.", name);
        return value.GetBoolean();
    }

    internal static int[]? OptionalDevices(JsonElement json)
    {
        if (!json.TryGetProperty("devices", out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Array)
            throw new LocalApiException(400, "devices must be an array of GPU indices.", "devices");
        int[] indices = value.EnumerateArray().Select(item =>
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int index) || index < 0)
                throw new LocalApiException(400, "devices must contain nonnegative integer GPU indices.", "devices");
            return index;
        }).ToArray();
        if (indices.Length > 2 || indices.Distinct().Count() != indices.Length)
            throw new LocalApiException(400, "devices must contain at most two distinct GPU indices; an empty array selects CPU for Gemma 3.", "devices");
        return indices;
    }

    internal static string? OptionalGemmaBackend(JsonElement json)
    {
        string? backend = OptionalString(json, "gemma_backend");
        if (backend is not null and not ("cpu" or "opencl"))
            throw new LocalApiException(400, "gemma_backend must be cpu or opencl.", "gemma_backend");
        return backend;
    }

    internal static ChatTurn[] RequiredMessages(JsonElement json)
    {
        if (!json.TryGetProperty("messages", out JsonElement value) || value.ValueKind != JsonValueKind.Array)
            throw new LocalApiException(400, "messages must be an array.", "messages");
        if (value.GetArrayLength() > 4096)
            throw new LocalApiException(400, "Too many messages.", "messages");
        long textCharacters = 0;
        ChatTurn[] turns = value.EnumerateArray().Select(item =>
        {
            if (item.ValueKind != JsonValueKind.Object)
                throw new LocalApiException(400, "Each message must be an object.", "messages");
            string role = RequiredString(item, "role");
            if (role == "developer") role = "system";
            if (role is not ("system" or "user" or "assistant"))
                throw new LocalApiException(400, "Only system, developer, user and assistant roles are supported.", "messages");
            if (!item.TryGetProperty("content", out JsonElement content))
                throw new LocalApiException(400, "Message content is required.", "messages");
            string text;
            ChatImage? image = null;
            if (content.ValueKind == JsonValueKind.String) text = content.GetString() ?? "";
            else if (content.ValueKind == JsonValueKind.Array)
            {
                var parts = new List<string>();
                foreach (JsonElement part in content.EnumerateArray())
                {
                    string type = RequiredString(part, "type");
                    if (type == "text") parts.Add(RequiredString(part, "text"));
                    else if (type == "image_url" && role == "user" && image is null)
                    {
                        if (!part.TryGetProperty("image_url", out var source) || source.ValueKind != JsonValueKind.Object)
                            throw new LocalApiException(400, "image_url must contain a data URL.", "messages");
                        try { image = ChatImage.FromDataUrl(RequiredString(source, "url")); }
                        catch (Exception ex) when (ex is ArgumentException or FormatException)
                        { throw new LocalApiException(400, ex.Message, "messages"); }
                    }
                    else throw new LocalApiException(400, "Use text and at most one local PNG/JPEG image per user turn.", "messages");
                }
                text = string.Join("\n", parts);
            }
            else throw new LocalApiException(400, "content must be text or text/image parts.", "messages");
            string? prefix = OptionalString(item, "assistant_prefix");
            textCharacters += text.Length + (long)(prefix?.Length ?? 0);
            if (textCharacters > 1024 * 1024)
                throw new LocalApiException(400, "Conversation exceeds the text character budget.", "messages");
            return new ChatTurn(role, text, prefix, image);
        }).ToArray();
        if (turns.Length == 0 || turns[^1].Role != "user")
            throw new LocalApiException(400, "messages must end with a user message.", "messages");
        return turns;
    }

    internal static string? ResolveMmproj(JsonElement json)
    {
        string? path = OptionalString(json, "mmproj");
        return string.IsNullOrWhiteSpace(path) ? null : ValidateFilePath(path, ".gguf", "mmproj");
    }

}

internal sealed class LocalApiException(int statusCode, string message, string? parameter = null)
    : Exception(message)
{
    public int StatusCode { get; } = statusCode;
    public string? Parameter { get; } = parameter;
}

internal sealed record LocalChatRequest(string Model, string? Lora, string? Mmproj,
    int[]? Devices, string? GemmaBackend, ChatTurn[] Messages, int MaxTokens,
    bool Stream, bool Thinking, bool PrimeHistory, GenerationSampling Sampling);
