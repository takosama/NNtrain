using System.Diagnostics;
using System.IO;

namespace NNtrain.Gui;

public sealed partial class LocalOpenAiServer
{
    private async Task LoadCoreAsync(string model, string? lora, int[]? devices, CancellationToken ct,
        string? mmproj = null, string? gemmaBackend = null)
    {
        if (_session.IsLoaded && string.Equals(_session.LoadedModelPath, model, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_session.LoadedAdapterPath, lora, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_session.LoadedMmprojPath, mmproj, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_session.LoadedGemmaBackend, gemmaBackend ??
                (_session.LoadedGemmaBackend is null ? null : "cpu"), StringComparison.Ordinal)
            && (devices is null || _session.LoadedDevices is { } loaded && loaded.SequenceEqual(devices)))
        {
            MarkCompleted();
            return;
        }
        Log($"Loading model: {model}; LoRA: {lora ?? "none"}; devices: " +
            (devices is null ? "automatic" : string.Join(',', devices)) +
            $"; gemma_backend: {gemmaBackend ?? "default"}");
        Stopwatch timer = Stopwatch.StartNew();
        try
        {
            await _session.LoadAsync(model, lora, new Progress<string>(Log), ct, devices, mmproj,
                gemmaBackend).ConfigureAwait(false);
        }
        finally
        {
            // A rejected vision tower can leave a useful text model resident.
            // Keep its normal idle lifetime even when the load request fails.
            if (_session.IsLoaded) MarkCompleted();
        }
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
            mmproj = _session.LoadedMmprojPath,
            devices = _session.LoadedDevices,
            gemma_backend = _session.LoadedGemmaBackend,
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
                            if (!Path.GetFileName(path).StartsWith("lora_", StringComparison.OrdinalIgnoreCase)
                                && !HuggingFaceModelDownload.IsMmproj(path))
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

}
