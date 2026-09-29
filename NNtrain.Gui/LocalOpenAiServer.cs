using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace NNtrain.Gui;

/// <summary>
/// Loopback-only OpenAI-compatible endpoint for the GUI's console child process.
/// This class owns the GPU session; all model operations share one request gate.
/// </summary>
public sealed class LocalOpenAiServer : IAsyncDisposable
{
    private static readonly TimeSpan IdleLimit = TimeSpan.FromMinutes(5);
    private readonly InferenceSession _session = new();
    private readonly SemaphoreSlim _requestGate = new(1, 1);
    private readonly CancellationTokenSource _lifetime = new();
    private WebApplication? _app;
    private Task? _idleTask;
    private string? _initialLora;
    private long _lastCompletedUnixMilliseconds = -1;
    private int _disposed;

    public Uri BaseAddress { get; private set; } = null!;

    public async Task<Uri> StartAsync(int port, string? initialLora, CancellationToken ct)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        if (_app is not null) throw new InvalidOperationException("The server is already running.");
        if (port is < 0 or > 65535) throw new ArgumentOutOfRangeException(nameof(port));
        _initialLora = string.IsNullOrWhiteSpace(initialLora)
            ? null : ValidateFilePath(initialLora, ".bin", "lora");

        WebApplicationBuilder builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            Args = [],
            ContentRootPath = AppContext.BaseDirectory
        });
        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Listen(IPAddress.Loopback, port);
            options.Limits.MaxRequestBodySize = 4 * 1024 * 1024;
        });
        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        WebApplication app = builder.Build();
        MapEndpoints(app);
        app.Lifetime.ApplicationStopping.Register(() => _lifetime.Cancel());
        try
        {
            await app.StartAsync(ct).ConfigureAwait(false);
            string address = app.Services.GetRequiredService<IServer>().Features
                .Get<IServerAddressesFeature>()?.Addresses
                .FirstOrDefault(value => value.StartsWith("http://127.0.0.1:", StringComparison.OrdinalIgnoreCase))
                ?? throw new InvalidOperationException("Could not determine the loopback listening address.");
            BaseAddress = new Uri(address.TrimEnd('/') + "/", UriKind.Absolute);
            _app = app;
            _idleTask = IdleLoopAsync(_lifetime.Token);
            Log($"Server listening at {BaseAddress}");
            if (_initialLora is not null) Log($"Default LoRA: {_initialLora}");
            return BaseAddress;
        }
        catch
        {
            await app.DisposeAsync().ConfigureAwait(false);
            throw;
        }
    }

    public Task WaitForShutdownAsync(CancellationToken ct = default)
        => (_app ?? throw new InvalidOperationException("Start the server first."))
            .WaitForShutdownAsync(ct);

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
        app.MapPost("/internal/unload", context => ExecuteAsync(context, UnloadAsync));
        app.MapPost("/internal/shutdown", async context =>
        {
            Log("Shutdown requested.");
            await context.Response.WriteAsJsonAsync(new { status = "stopping" }).ConfigureAwait(false);
            _app?.Lifetime.StopApplication();
        });
        app.MapPost("/v1/chat/completions", context => ExecuteAsync(context, ChatAsync));
    }

    private async Task ExecuteAsync(HttpContext context, Func<HttpContext, CancellationToken, Task> action)
    {
        using var linked = CancellationTokenSource.CreateLinkedTokenSource(
            context.RequestAborted, _lifetime.Token);
        try
        {
            await action(context, linked.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (context.RequestAborted.IsCancellationRequested)
        {
            Log($"Client disconnected: {context.Request.Path}");
        }
        catch (OperationCanceledException) when (_lifetime.IsCancellationRequested)
        {
            Log($"Server is stopping: {context.Request.Path}");
        }
        catch (Exception error)
        {
            Log($"ERROR {context.Request.Path}: {error}");
            if (context.RequestAborted.IsCancellationRequested) return;
            int status = error switch
            {
                ApiException api => api.StatusCode,
                JsonException or BadHttpRequestException or ArgumentException
                    or FileNotFoundException or InvalidDataException or NotSupportedException => 400,
                _ => 500
            };
            object body = new
            {
                error = new
                {
                    message = error.Message,
                    type = status == 500 ? "server_error" : "invalid_request_error",
                    param = error is ApiException requestError ? requestError.Parameter : null,
                    code = status == 500 ? "internal_error" : "invalid_request"
                }
            };
            if (context.Response.HasStarted)
            {
                if (context.Response.ContentType?.StartsWith("text/event-stream", StringComparison.Ordinal) == true)
                {
                    try
                    {
                        await WriteEventAsync(context.Response, body, CancellationToken.None).ConfigureAwait(false);
                    }
                    catch (Exception) { /* The client may have closed the SSE connection. */ }
                }
                return;
            }
            context.Response.StatusCode = status;
            await context.Response.WriteAsJsonAsync(body).ConfigureAwait(false);
        }
    }

    private async Task LoadAsync(HttpContext context, CancellationToken ct)
    {
        JsonElement json = await ReadObjectAsync(context, ct).ConfigureAwait(false);
        string model = ValidateFilePath(RequiredString(json, "model"), ".gguf", "model");
        string? lora = ResolveLora(json);
        int[]? devices = OptionalDevices(json);
        await _requestGate.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            await LoadCoreAsync(model, lora, devices, ct).ConfigureAwait(false);
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

    private async Task ChatAsync(HttpContext context, CancellationToken ct)
    {
        Stopwatch requestTimer = Stopwatch.StartNew();
        JsonElement json = await ReadObjectAsync(context, ct).ConfigureAwait(false);
        string model = ValidateFilePath(RequiredString(json, "model"), ".gguf", "model");
        string? lora = ResolveLora(json);
        int[]? devices = OptionalDevices(json);
        ChatTurn[] messages = RequiredMessages(json);
        int maxTokens = OptionalInt(json, "max_completion_tokens")
            ?? OptionalInt(json, "max_tokens") ?? 512;
        if (maxTokens is < 1 or > 8192)
            throw new ApiException(400, "max_tokens must be between 1 and 8192.", "max_tokens");
        float temperature = OptionalFloat(json, "temperature") ?? 0.6f;
        float topP = OptionalFloat(json, "top_p") ?? 0.95f;
        int topK = OptionalInt(json, "top_k") ?? 20;
        if (!float.IsFinite(temperature) || temperature is < 0f or > 2f)
            throw new ApiException(400, "temperature must be between 0 and 2.", "temperature");
        if (!float.IsFinite(topP) || topP is <= 0f or > 1f)
            throw new ApiException(400, "top_p must be greater than 0 and at most 1.", "top_p");
        if (topK is < 1 or > 256)
            throw new ApiException(400, "top_k must be between 1 and 256.", "top_k");
        bool stream = OptionalBoolean(json, "stream") ?? false;
        bool think = OptionalBoolean(json, "think") ?? true;
        bool primeHistory = OptionalBoolean(json, "prime_history") ?? false;
        var sampling = new GenerationSampling(temperature, topP, topK);
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        await _requestGate.WaitAsync(ct).ConfigureAwait(false);
        double gateWaitMilliseconds = requestTimer.Elapsed.TotalMilliseconds;
        try
        {
            await LoadCoreAsync(model, lora, devices, ct).ConfigureAwait(false);
            Log($"Chat {id}: model={model}, lora={lora ?? "none"}, stream={stream}, think={think}, " +
                $"max_tokens={maxTokens}, temperature={temperature.ToString(CultureInfo.InvariantCulture)}, " +
                $"top_p={topP.ToString(CultureInfo.InvariantCulture)}, top_k={topK}");
            Log($"Chat {id} history: turns={messages.Length}, characters={messages.Sum(message => message.Content.Length)}");
            Log($"Chat {id} input [{messages[^1].Role}]: {messages[^1].Content}");
            if (stream)
            {
                context.Response.StatusCode = 200;
                context.Response.ContentType = "text/event-stream; charset=utf-8";
                context.Response.Headers.CacheControl = "no-cache";
                context.Response.Headers.Append("X-Accel-Buffering", "no");
                await WriteEventAsync(context.Response, new
                {
                    id, @object = "chat.completion.chunk", created, model,
                    choices = new[] { new { index = 0, delta = new { role = "assistant" }, finish_reason = (string?)null } }
                }, ct).ConfigureAwait(false);
            }
            Log($"Chat {id} output: gate_wait_ms={gateWaitMilliseconds:F1}");
            Stopwatch generationTimer = Stopwatch.StartNew();
            bool firstText = true;
            double? firstTextFromRequestMilliseconds = null;
            string raw = await _session.GenerateAsync(messages, think, maxTokens, chunk =>
            {
                if (firstText)
                {
                    firstText = false;
                    firstTextFromRequestMilliseconds = requestTimer.Elapsed.TotalMilliseconds;
                    Log($"Chat {id} first text after {firstTextFromRequestMilliseconds / 1000:F2}s " +
                        $"from request ({generationTimer.Elapsed.TotalSeconds:F2}s generating).");
                }
                Console.Write(chunk);
                Console.Out.Flush();
                if (stream)
                    WriteEventAsync(context.Response, new
                    {
                        id, @object = "chat.completion.chunk", created, model,
                        choices = new[] { new { index = 0, delta = new { content = chunk }, finish_reason = (string?)null } }
                    }, ct).GetAwaiter().GetResult();
            }, ct, sampling).ConfigureAwait(false);
            Console.WriteLine();
            GenerationStats? stats = _session.LastGenerationStats;
            string finishReason = stats?.StopReason == GenerationStopReason.EndOfMessage ? "stop" : "length";
            string stopDetail = stats?.StopReason switch
            {
                GenerationStopReason.EndOfMessage => "end_of_message",
                GenerationStopReason.ContextLimit => "context_limit",
                _ => "maximum_tokens"
            };
            Log($"Chat {id} completed: prompt_tokens={stats?.PromptTokens}, " +
                $"completion_tokens={stats?.CompletionTokens}, reused={stats?.ReusedPromptTokens}, " +
                $"uncached={(stats is null ? null : stats.PromptTokens - stats.ReusedPromptTokens)}, " +
                $"gate_wait_ms={gateWaitMilliseconds:F1}, " +
                $"first_text_ms_from_request={firstTextFromRequestMilliseconds:F1}, " +
                $"first_token_ms_from_generation={stats?.FirstTokenMilliseconds:F1}, " +
                $"finish_reason={finishReason}");
            MarkCompleted();
            if (stream)
            {
                await WriteEventAsync(context.Response, new
                {
                    id, @object = "chat.completion.chunk", created, model,
                    choices = new[] { new { index = 0, delta = new { },
                        finish_reason = finishReason, nntrain_stop_reason = stopDetail } }
                }, ct).ConfigureAwait(false);
                await context.Response.WriteAsync("data: [DONE]\n\n", ct).ConfigureAwait(false);
                await context.Response.Body.FlushAsync(ct).ConfigureAwait(false);
                if (primeHistory)
                {
                    try
                    {
                        await PrimeHistoryAsync(messages, raw, think, stats?.StopReason,
                            _lifetime.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (_lifetime.IsCancellationRequested) { }
                    catch (Exception error) { Log($"Chat {id} history prefill failed: {error}"); }
                }
            }
            else
            {
                int promptTokens = stats?.PromptTokens ?? 0;
                int completionTokens = stats?.CompletionTokens ?? 0;
                await context.Response.WriteAsJsonAsync(new
                {
                    id, @object = "chat.completion", created, model,
                    choices = new[] { new
                    {
                        index = 0,
                        message = new { role = "assistant", content = raw },
                        finish_reason = finishReason,
                        nntrain_stop_reason = stopDetail
                    } },
                    usage = new
                    {
                        prompt_tokens = promptTokens,
                        completion_tokens = completionTokens,
                        total_tokens = promptTokens + completionTokens
                    }
                }, ct).ConfigureAwait(false);
            }
        }
        finally
        {
            if (_session.IsLoaded) MarkCompleted();
            _requestGate.Release();
        }
    }

    private async Task PrimeHistoryAsync(ChatTurn[] messages, string raw, bool thinking,
        GenerationStopReason? stopReason, CancellationToken ct)
    {
        var parser = new ThinkingStreamParser(thinking);
        parser.Append(raw);
        ThinkingStreamSnapshot snapshot = parser.Complete(stopReason == GenerationStopReason.EndOfMessage);
        if (string.IsNullOrWhiteSpace(snapshot.AnswerText)) return;
        string prefix = thinking ? "<think>\n</think>\n" : "<think>\n\n</think>\n\n";
        ChatTurn[] history = [.. messages, new ChatTurn("assistant", snapshot.AnswerText, prefix)];
        Stopwatch timer = Stopwatch.StartNew();
        Log("Priming sanitized conversation history for the next turn.");
        PromptPrimeStats result = await _session.PrimeHistoryAsync(history, ct).ConfigureAwait(false);
        Log($"History prefill: prompt_tokens={result.PromptTokens}, reused={result.ReusedPromptTokens}, " +
            $"uncached={result.PromptTokens - result.ReusedPromptTokens}, cached={result.Cached}, " +
            $"elapsed={timer.Elapsed.TotalSeconds:F2}s.");
        MarkCompleted();
    }

    private async Task LoadCoreAsync(string model, string? lora, int[]? devices, CancellationToken ct)
    {
        if (_session.IsLoaded && string.Equals(_session.LoadedModelPath, model, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_session.LoadedAdapterPath, lora, StringComparison.OrdinalIgnoreCase)
            && (devices is null || _session.LoadedDevices is { } loaded && loaded.SequenceEqual(devices)))
        {
            MarkCompleted();
            return;
        }
        Log($"Loading model: {model}; LoRA: {lora ?? "none"}; devices: " +
            (devices is null ? "automatic" : string.Join(',', devices)));
        Stopwatch timer = Stopwatch.StartNew();
        await _session.LoadAsync(model, lora, new Progress<string>(Log), ct, devices).ConfigureAwait(false);
        timer.Stop();
        Log($"Model loaded in {timer.Elapsed.TotalSeconds:F2}s.");
        MarkCompleted();
    }

    private object State()
    {
        long completed = Interlocked.Read(ref _lastCompletedUnixMilliseconds);
        DateTimeOffset? when = completed < 0 ? null : DateTimeOffset.FromUnixTimeMilliseconds(completed);
        int? idle = _session.IsLoaded && when is not null
            ? Math.Max(0, (int)Math.Ceiling((IdleLimit - (DateTimeOffset.UtcNow - when.Value)).TotalSeconds))
            : null;
        return new
        {
            is_loaded = _session.IsLoaded,
            model = _session.LoadedModelPath,
            lora = _session.LoadedAdapterPath,
            devices = _session.LoadedDevices,
            last_completed_utc = when,
            idle_seconds_remaining = idle
        };
    }

    private void MarkCompleted()
        => Interlocked.Exchange(ref _lastCompletedUnixMilliseconds,
            DateTimeOffset.UtcNow.ToUnixTimeMilliseconds());

    private async Task IdleLoopAsync(CancellationToken ct)
    {
        using var timer = new PeriodicTimer(TimeSpan.FromSeconds(15));
        try
        {
            while (await timer.WaitForNextTickAsync(ct).ConfigureAwait(false))
            {
                long completed = Interlocked.Read(ref _lastCompletedUnixMilliseconds);
                if (completed < 0 || DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - completed < IdleLimit.TotalMilliseconds)
                    continue;
                if (!await _requestGate.WaitAsync(0, ct).ConfigureAwait(false)) continue;
                try
                {
                    completed = Interlocked.Read(ref _lastCompletedUnixMilliseconds);
                    if (!_session.IsLoaded || completed < 0
                        || DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() - completed < IdleLimit.TotalMilliseconds)
                        continue;
                    Log("Five-minute idle timeout: unloading model.");
                    await _session.UnloadAsync().ConfigureAwait(false);
                    Interlocked.Exchange(ref _lastCompletedUnixMilliseconds, -1);
                }
                catch (Exception error) { Log($"Idle unload failed: {error}"); }
                finally { _requestGate.Release(); }
            }
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested) { }
    }

    private string? ResolveLora(JsonElement json)
    {
        if (!json.TryGetProperty("lora", out JsonElement value)) return _initialLora;
        if (value.ValueKind == JsonValueKind.Null) return null;
        if (value.ValueKind != JsonValueKind.String)
            throw new ApiException(400, "lora must be a path or null.", "lora");
        string? path = value.GetString();
        return string.IsNullOrWhiteSpace(path) ? null : ValidateFilePath(path, ".bin", "lora");
    }

    private static string ValidateFilePath(string path, string extension, string parameter)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ApiException(400, $"{parameter} path is required.", parameter);
        if (path.Split(['/', '\\']).Any(segment => segment == ".."))
            throw new ApiException(400, $"{parameter} path must not contain '..' segments.", parameter);
        string fullPath;
        try { fullPath = Path.GetFullPath(path); }
        catch (Exception error) when (error is ArgumentException or NotSupportedException or PathTooLongException)
        {
            throw new ApiException(400, $"Invalid {parameter} path.", parameter);
        }
        if (!Path.GetExtension(fullPath).Equals(extension, StringComparison.OrdinalIgnoreCase))
            throw new ApiException(400, parameter == "lora"
                ? "lora must be an NNtrain adapter.bin file; GGUF LoRA adapters are not supported."
                : $"{parameter} must be a {extension} file.", parameter);
        if (!File.Exists(fullPath))
            throw new ApiException(400, $"{parameter} file does not exist: {fullPath}", parameter);
        return fullPath;
    }

    private static async Task<JsonElement> ReadObjectAsync(HttpContext context, CancellationToken ct)
    {
        if (!context.Request.HasJsonContentType())
            throw new ApiException(415, "Content-Type must be application/json.");
        JsonElement json;
        try { json = await context.Request.ReadFromJsonAsync<JsonElement>(cancellationToken: ct).ConfigureAwait(false); }
        catch (JsonException error) { throw new ApiException(400, $"Invalid JSON: {error.Message}"); }
        if (json.ValueKind != JsonValueKind.Object)
            throw new ApiException(400, "Request body must be a JSON object.");
        return json;
    }

    private static string RequiredString(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind != JsonValueKind.String
            || string.IsNullOrWhiteSpace(value.GetString()))
            throw new ApiException(400, $"{name} is required and must be a nonempty string.", name);
        return value.GetString()!;
    }

    private static string? OptionalString(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.String)
            throw new ApiException(400, $"{name} must be a string.", name);
        return value.GetString();
    }

    private static int? OptionalInt(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetInt32(out int result))
            throw new ApiException(400, $"{name} must be an integer.", name);
        return result;
    }

    private static float? OptionalFloat(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Number || !value.TryGetSingle(out float result))
            throw new ApiException(400, $"{name} must be a number.", name);
        return result;
    }

    private static bool? OptionalBoolean(JsonElement json, string name)
    {
        if (!json.TryGetProperty(name, out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
            throw new ApiException(400, $"{name} must be a boolean.", name);
        return value.GetBoolean();
    }

    private static int[]? OptionalDevices(JsonElement json)
    {
        if (!json.TryGetProperty("devices", out JsonElement value) || value.ValueKind == JsonValueKind.Null)
            return null;
        if (value.ValueKind != JsonValueKind.Array)
            throw new ApiException(400, "devices must be an array of GPU indices.", "devices");
        int[] indices = value.EnumerateArray().Select(item =>
        {
            if (item.ValueKind != JsonValueKind.Number || !item.TryGetInt32(out int index) || index < 0)
                throw new ApiException(400, "devices must contain nonnegative integer GPU indices.", "devices");
            return index;
        }).ToArray();
        if (indices.Length is < 1 or > 2 || indices.Distinct().Count() != indices.Length)
            throw new ApiException(400, "devices must contain one or two distinct GPU indices.", "devices");
        return indices;
    }

    private static ChatTurn[] RequiredMessages(JsonElement json)
    {
        if (!json.TryGetProperty("messages", out JsonElement value) || value.ValueKind != JsonValueKind.Array)
            throw new ApiException(400, "messages must be an array.", "messages");
        ChatTurn[] turns = value.EnumerateArray().Select(item =>
        {
            if (item.ValueKind != JsonValueKind.Object)
                throw new ApiException(400, "Each message must be an object.", "messages");
            string role = RequiredString(item, "role");
            if (role == "developer") role = "system";
            if (role is not ("system" or "user" or "assistant"))
                throw new ApiException(400, "Only system, developer, user and assistant roles are supported.", "messages");
            if (!item.TryGetProperty("content", out JsonElement content) || content.ValueKind != JsonValueKind.String)
                throw new ApiException(400, "Only text message content is supported.", "messages");
            return new ChatTurn(role, content.GetString() ?? "", OptionalString(item, "assistant_prefix"));
        }).ToArray();
        if (turns.Length == 0 || turns[^1].Role != "user")
            throw new ApiException(400, "messages must end with a user message.", "messages");
        return turns;
    }

    private IEnumerable<string> EnumerateModels()
    {
        var paths = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        if (_session.LoadedModelPath is string loaded) paths.Add(loaded);
        foreach (string origin in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            DirectoryInfo? current = new(origin);
            while (current is not null)
            {
                string models = Path.Combine(current.FullName, "models");
                if (Directory.Exists(models))
                {
                    try
                    {
                        foreach (string path in Directory.EnumerateFiles(models, "*.gguf", SearchOption.AllDirectories))
                            if (!Path.GetFileName(path).StartsWith("lora_", StringComparison.OrdinalIgnoreCase))
                                paths.Add(Path.GetFullPath(path));
                    }
                    catch (IOException) { }
                    catch (UnauthorizedAccessException) { }
                    break;
                }
                current = current.Parent;
            }
        }
        return paths.Order(StringComparer.OrdinalIgnoreCase).ToArray();
    }

    private static async Task WriteEventAsync(HttpResponse response, object value, CancellationToken ct)
    {
        await response.WriteAsync("data: " + JsonSerializer.Serialize(value) + "\n\n", ct).ConfigureAwait(false);
        await response.Body.FlushAsync(ct).ConfigureAwait(false);
    }

    private static void Log(string message)
    {
        Console.WriteLine($"[{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss}] {message}");
        Console.Out.Flush();
    }

    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _lifetime.Cancel();
        if (_app is not null)
        {
            try { await _app.StopAsync().ConfigureAwait(false); }
            finally { await _app.DisposeAsync().ConfigureAwait(false); }
        }
        if (_idleTask is not null) await _idleTask.ConfigureAwait(false);
        await _requestGate.WaitAsync().ConfigureAwait(false);
        try { _session.Dispose(); }
        finally
        {
            _requestGate.Release();
            _requestGate.Dispose();
            _lifetime.Dispose();
        }
    }

    private sealed class ApiException(int statusCode, string message, string? parameter = null)
        : Exception(message)
    {
        public int StatusCode { get; } = statusCode;
        public string? Parameter { get; } = parameter;
    }
}
