using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace NNtrain.Gui;

public sealed partial class LocalOpenAiServer
{
    private void MapEndpoints(WebApplication app)
    {
        app.MapGet("/health", () => Results.Json(new { status = "ok" }));
        app.MapGet("/v1/models", () => Results.Json(new
        {
            @object = "list",
            data = EnumerateModels().Select(path => new
            {
                id = path,
                @object = "model",
                created = 0,
                owned_by = "nntrain"
            }).ToArray()
        }));
        app.MapGet("/internal/state", () => Results.Json(State()));
        app.MapPost("/internal/load", context => ExecuteAsync(context, LoadAsync));
        app.MapPost("/internal/prepare-image", context => ExecuteAsync(context, PrepareImageAsync));
        app.MapPost("/internal/unload", context => ExecuteAsync(context, UnloadAsync));
        app.MapPost("/internal/shutdown", context => ExecuteAsync(context, async (context, ct) =>
        {
            await LocalApiRequests.ReadObjectAsync(context, ct).ConfigureAwait(false);
            Log("Shutdown requested.");
            await context.Response.WriteAsJsonAsync(new { status = "stopping" }).ConfigureAwait(false);
            _app?.Lifetime.StopApplication();
        }));
        app.MapPost("/v1/chat/completions", context => ExecuteAsync(context, ChatAsync));
    }

    private async Task LoadAsync(HttpContext context, CancellationToken ct)
    {
        JsonElement json = await LocalApiRequests.ReadObjectAsync(context, ct).ConfigureAwait(false);
        string model = LocalApiRequests.ValidateFilePath(LocalApiRequests.RequiredString(json, "model"), ".gguf", "model");
        string? lora = LocalApiRequests.ResolveLora(json, _initialLora);
        string? mmproj = LocalApiRequests.ResolveMmproj(json);
        int[]? devices = LocalApiRequests.OptionalDevices(json);
        string? gemmaBackend = LocalApiRequests.OptionalGemmaBackend(json);
        await _requestGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            await LoadCoreAsync(model, lora, devices, ct, mmproj, gemmaBackend).ConfigureAwait(false);
            await context.Response.WriteAsJsonAsync(State(), ct).ConfigureAwait(false);
        }
        finally { _requestGate.Release(); }
    }

    private async Task UnloadAsync(HttpContext context, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_session.IsLoaded)
            {
                Log("Unloading model.");
                await _session.UnloadAsync().ConfigureAwait(false);
            }
            Interlocked.Exchange(ref _lastCompletedUnixMilliseconds, -1);
            await context.Response.WriteAsJsonAsync(State(), ct).ConfigureAwait(false);
        }
        finally { _requestGate.Release(); }
    }

    private async Task PrepareImageAsync(HttpContext context, CancellationToken ct)
    {
        JsonElement json = await LocalApiRequests.ReadObjectAsync(context, ct).ConfigureAwait(false);
        string model = LocalApiRequests.ValidateFilePath(LocalApiRequests.RequiredString(json, "model"), ".gguf", "model");
        string? lora = LocalApiRequests.ResolveLora(json, _initialLora);
        string? mmproj = LocalApiRequests.ResolveMmproj(json);
        if (mmproj is null) throw new LocalApiException(400, "mmproj is required to prepare an image.", "mmproj");
        int[]? devices = LocalApiRequests.OptionalDevices(json);
        ChatTurn[]? messages = json.TryGetProperty("messages", out _) ? LocalApiRequests.RequiredMessages(json) : null;
        string? imageUrl = LocalApiRequests.OptionalString(json, "image_url");
        // A history request already contains the attachment. Avoid sending its
        // base64 twice, which can exceed the 40 MiB limit for one 16 MiB image.
        ChatImage image = imageUrl is not null ? ChatImage.FromDataUrl(imageUrl)
            : messages is { Length: > 0 } && messages[^1].Image is { } attached ? attached
            : throw new LocalApiException(400, "image_url or a final image message is required.", "image_url");
        if (messages is not null && (messages[^1].Content.Length != 0 || messages[^1].Image?.Hash != image.Hash))
            throw new LocalApiException(400, "Preparation messages must end with the same image and empty user text.", "messages");
        await _requestGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            await LoadCoreAsync(model, lora, devices, ct, mmproj).ConfigureAwait(false);
            ImagePreparationStats result = await _session.PrepareImageAsync(image, messages, ct).ConfigureAwait(false);
            MarkCompleted();
            await context.Response.WriteAsJsonAsync(new
            {
                image_hash = image.Hash,
                grid_width = result.GridWidth,
                grid_height = result.GridHeight,
                cached = result.Cached,
                image_preparation_ms = _session.LastImagePreparationMilliseconds,
                primed_prompt_tokens = result.PrimedPromptTokens,
                reused_prompt_tokens = result.ReusedPromptTokens,
                prefix_cached = result.PrefixCached,
                prefix_preparation_ms = result.PrefixPreparationMilliseconds
            }, ct).ConfigureAwait(false);
        }
        finally { _requestGate.Release(); }
    }

}
