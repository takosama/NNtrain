using System.Net;
using System.Security.Cryptography;
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
public sealed partial class LocalOpenAiServer : IAsyncDisposable
{
    private static readonly TimeSpan IdleLimit = TimeSpan.FromMinutes(5);
    private readonly InferenceSession _session = new();
    private readonly SemaphoreSlim _requestGate = new(1, 1);
    private readonly LocalApiAccessPolicy _access;
    public const string TokenEnvironmentVariable = "NNTRAIN_GUI_SERVER_TOKEN";
    public string AuthenticationToken { get; }

    public LocalOpenAiServer(string? authenticationToken = null)
    {
        AuthenticationToken = authenticationToken ?? Convert.ToHexString(RandomNumberGenerator.GetBytes(32));
        if (AuthenticationToken.Length < 32 || AuthenticationToken.Any(c => !char.IsAsciiLetterOrDigit(c)))
            throw new ArgumentException("The session token must contain at least 32 ASCII letters or digits.", nameof(authenticationToken));
        _access = new LocalApiAccessPolicy(AuthenticationToken);
    }
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
            ? null : LocalApiRequests.ValidateFilePath(initialLora, ".bin", "lora");

        WebApplicationBuilder builder = WebApplication.CreateBuilder(new WebApplicationOptions
        {
            Args = [],
            ContentRootPath = AppContext.BaseDirectory
        });
        builder.WebHost.ConfigureKestrel(options =>
        {
            options.Listen(IPAddress.Loopback, port);
            options.Limits.MaxRequestBodySize = 40 * 1024 * 1024;
            options.Limits.MaxConcurrentConnections = 16;
            options.Limits.MaxRequestHeadersTotalSize = 16 * 1024;
            options.Limits.RequestHeadersTimeout = TimeSpan.FromSeconds(15);
            options.Limits.KeepAliveTimeout = TimeSpan.FromSeconds(30);
        });
        builder.Logging.ClearProviders();
        builder.Logging.AddConsole();
        WebApplication app = builder.Build();
        app.Use(_access.InvokeAsync);
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
            int status = LocalApiResponses.ErrorStatus(error);
            object body = LocalApiResponses.Error(error, status);
            if (context.Response.HasStarted)
            {
                if (context.Response.ContentType?.StartsWith("text/event-stream", StringComparison.Ordinal) == true)
                {
                    try
                    {
                        await LocalApiResponses.WriteEventAsync(context.Response, body, CancellationToken.None).ConfigureAwait(false);
                    }
                    catch (Exception) { /* The client may have closed the SSE connection. */ }
                }
                return;
            }
            context.Response.StatusCode = status;
            await context.Response.WriteAsJsonAsync(body).ConfigureAwait(false);
        }
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
            _access.Dispose();
        }
    }

}
