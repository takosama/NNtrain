using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace NNtrain.Gui;

internal sealed record LocalInferenceState(bool IsLoaded, string? ModelPath, string? AdapterPath,
    string? MmprojPath, int[]? Devices, string? GemmaBackend, double? IdleSecondsRemaining);

internal sealed record LocalGeneration(string Text, GenerationStopReason? StopReason);

internal sealed record LocalGenerationRequest(IReadOnlyList<ChatTurn> Conversation, string ModelPath,
    string? AdapterPath, string? MmprojPath, IReadOnlyList<int> Devices, string? GemmaBackend,
    bool Thinking, bool Stream, int MaxNewTokens, GenerationSampling Sampling);

/// <summary>Authenticated local API transport. It has no dependency on WPF or GPU lifetimes.</summary>
internal sealed class LocalInferenceClient : IDisposable
{
    private readonly HttpClient _client;

    internal LocalInferenceClient(string authenticationToken, HttpMessageHandler? handler = null)
    {
        _client = handler is null ? new HttpClient() : new HttpClient(handler);
        _client.Timeout = Timeout.InfiniteTimeSpan;
        _client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", authenticationToken);
    }

    internal void SetAddress(Uri address) => _client.BaseAddress = address;

    internal async Task<bool> ProbeAsync(CancellationToken ct)
    {
        using HttpResponseMessage response = await _client.GetAsync("health", ct);
        return response.IsSuccessStatusCode;
    }

    internal async Task PostAsync(string path, object body, CancellationToken ct)
    {
        using HttpRequestMessage request = CreateRequest(path, body);
        using HttpResponseMessage response = await _client.SendAsync(request, ct);
        await EnsureSuccessAsync(response, ct);
    }

    internal Task LoadAsync(string modelPath, string? adapterPath, string? mmprojPath,
        IReadOnlyList<int> devices, string? gemmaBackend, CancellationToken ct) =>
        PostAsync("internal/load", new { model = modelPath, lora = adapterPath,
            mmproj = mmprojPath, devices, gemma_backend = gemmaBackend }, ct);

    internal Task PrepareImageAsync(string? modelPath, string? adapterPath, string? mmprojPath,
        IReadOnlyList<int>? devices, IReadOnlyList<ChatTurn> conversation, CancellationToken ct) =>
        PostAsync("internal/prepare-image", new { model = modelPath, lora = adapterPath,
            mmproj = mmprojPath, devices, messages = Messages(conversation) }, ct);

    internal async Task<LocalInferenceState> ReadStateAsync(CancellationToken ct)
    {
        using HttpResponseMessage response = await _client.GetAsync("internal/state", ct);
        await EnsureSuccessAsync(response, ct);
        using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
        JsonElement state = document.RootElement;
        return new LocalInferenceState(ReadBoolean(state, "is_loaded"), ReadString(state, "model"),
            ReadString(state, "lora"), ReadString(state, "mmproj"),
            state.TryGetProperty("devices", out JsonElement devices) && devices.ValueKind == JsonValueKind.Array
                ? devices.EnumerateArray().Select(value => value.GetInt32()).ToArray() : null,
            ReadString(state, "gemma_backend"),
            state.TryGetProperty("idle_seconds_remaining", out JsonElement idle) && idle.ValueKind == JsonValueKind.Number
                ? idle.GetDouble() : null);
    }

    internal async Task<LocalGeneration> GenerateAsync(LocalGenerationRequest generation,
        Action<string> onText, CancellationToken ct)
    {
        using HttpRequestMessage request = CreateRequest("v1/chat/completions", new
        {
            model = generation.ModelPath,
            messages = Messages(generation.Conversation),
            max_tokens = generation.MaxNewTokens,
            temperature = generation.Sampling.Temperature,
            top_p = generation.Sampling.TopP,
            top_k = generation.Sampling.TopK,
            stream = generation.Stream,
            prime_history = generation.Stream,
            think = generation.Thinking,
            lora = generation.AdapterPath,
            mmproj = generation.MmprojPath,
            devices = generation.Devices,
            gemma_backend = generation.GemmaBackend
        });
        using HttpResponseMessage response = await _client.SendAsync(request,
            HttpCompletionOption.ResponseHeadersRead, ct);
        await EnsureSuccessAsync(response, ct);
        if (!generation.Stream)
        {
            using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
            JsonElement root = document.RootElement;
            ThrowIfApiError(root);
            JsonElement choice = root.GetProperty("choices")[0];
            string raw = ReadString(choice.GetProperty("message"), "content") ?? string.Empty;
            GenerationStopReason? stopReason = ParseStopReason(
                ReadString(choice, "nntrain_stop_reason") ?? ReadString(choice, "finish_reason"));
            if (stopReason is null)
                throw new InvalidDataException("サーバー応答に生成の終了理由がありません。");
            if (raw.Length != 0) onText(raw);
            return new LocalGeneration(raw, stopReason);
        }

        await using Stream stream = await response.Content.ReadAsStreamAsync(ct);
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return await ReadEventsAsync(reader, onText, ct);
    }

    private static async Task<LocalGeneration> ReadEventsAsync(StreamReader reader,
        Action<string> onText, CancellationToken ct)
    {
        var rawText = new StringBuilder();
        var eventData = new StringBuilder();
        GenerationStopReason? stopReason = null;
        bool done = false;
        while (await reader.ReadLineAsync(ct) is { } line)
        {
            if (line.Length == 0)
            {
                ConsumeEvent();
                if (done) break;
                continue;
            }
            if (!line.StartsWith("data:", StringComparison.Ordinal)) continue;
            if (eventData.Length != 0) eventData.Append('\n');
            eventData.Append(line.AsSpan(5).TrimStart());
        }
        ConsumeEvent();
        ct.ThrowIfCancellationRequested();
        if (!done || stopReason is null)
            throw new IOException("ストリームが正常に終了する前に接続が切れました。");
        return new LocalGeneration(rawText.ToString(), stopReason);

        void ConsumeEvent()
        {
            if (eventData.Length == 0) return;
            string data = eventData.ToString();
            eventData.Clear();
            if (data == "[DONE]") { done = true; return; }
            using JsonDocument eventJson = JsonDocument.Parse(data);
            JsonElement root = eventJson.RootElement;
            ThrowIfApiError(root);
            if (!root.TryGetProperty("choices", out JsonElement choices) || choices.GetArrayLength() == 0)
                return;
            JsonElement choice = choices[0];
            if (choice.TryGetProperty("delta", out JsonElement delta))
            {
                string? chunk = ReadString(delta, "content");
                if (!string.IsNullOrEmpty(chunk))
                {
                    rawText.Append(chunk);
                    onText(chunk);
                }
            }
            stopReason = ParseStopReason(
                ReadString(choice, "nntrain_stop_reason") ?? ReadString(choice, "finish_reason")) ?? stopReason;
        }
    }

    private static HttpRequestMessage CreateRequest(string path, object body) => new(HttpMethod.Post, path)
    {
        Content = new StringContent(JsonSerializer.Serialize(body), Encoding.UTF8, "application/json")
    };

    private static object[] Messages(IReadOnlyList<ChatTurn> conversation) => conversation.Select(turn => (object)new
    {
        role = turn.Role,
        content = MessageContent(turn),
        assistant_prefix = turn.AssistantPrefix
    }).ToArray();

    private static object MessageContent(ChatTurn turn) => turn.Image is null ? turn.Content
        : new object[]
        {
            new { type = "image_url", image_url = new { url = turn.Image.ToDataUrl() } },
            new { type = "text", text = turn.Content }
        };

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, CancellationToken ct)
    {
        if (response.IsSuccessStatusCode) return;
        string detail = await response.Content.ReadAsStringAsync(ct);
        try
        {
            using JsonDocument body = JsonDocument.Parse(detail);
            if (body.RootElement.TryGetProperty("error", out JsonElement error))
                detail = error.ValueKind == JsonValueKind.Object ? ReadString(error, "message") ?? detail : error.ToString();
        }
        catch (JsonException) { }
        throw new HttpRequestException($"サーバー応答 {(int)response.StatusCode}: {detail}");
    }

    private static void ThrowIfApiError(JsonElement root)
    {
        if (!root.TryGetProperty("error", out JsonElement error)) return;
        string detail = error.ValueKind == JsonValueKind.Object
            ? ReadString(error, "message") ?? error.ToString() : error.ToString();
        throw new InvalidOperationException(detail);
    }

    private static GenerationStopReason? ParseStopReason(string? reason) => reason switch
    {
        "stop" or "end_of_message" => GenerationStopReason.EndOfMessage,
        "length" or "maximum_tokens" => GenerationStopReason.MaximumTokens,
        "context_limit" => GenerationStopReason.ContextLimit,
        _ => null
    };

    private static string? ReadString(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() : null;

    private static bool ReadBoolean(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.True;

    public void Dispose() => _client.Dispose();
}
