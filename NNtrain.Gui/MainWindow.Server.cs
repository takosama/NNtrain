using System.IO;
using System.Windows.Threading;

namespace NNtrain.Gui;

public partial class MainWindow
{
    private readonly LocalInferenceServerProcess _server;
    private readonly LocalInferenceClient _api;
    private readonly DispatcherTimer _idleTimer = new() { Interval = TimeSpan.FromSeconds(15) };
    private bool _serverReady => _server.IsReady;
    private bool _isLoaded;
    private string? _loadedModelPath, _loadedAdapterPath;
    private int[]? _loadedDevices;
    private string? _loadedGemmaBackend;
    private string? _serverGemmaBackend;
    private double? _idleSecondsRemaining;

    private async Task StartServerAsync()
    {
        try
        {
            SetStatus("ローカル推論サーバーを起動しています…");
            await _server.StartAsync(_api);
            await RefreshServerStateAsync();
            _idleTimer.Start();
            SetStatus("推論サーバーの準備ができました。");
            UpdateControls();
        }
        catch (OperationCanceledException) { SetStatus("推論サーバーの起動がタイムアウトしました。"); }
        catch (Exception ex) { SetStatus($"推論サーバーの起動に失敗: {ex.Message}"); }
    }

    private async Task EnsureLoadedAsync(string modelPath, string? adapterPath,
        IReadOnlyList<int> deviceIndices, CancellationToken ct, bool gemmaOpenClAuto = false)
    {
        await RestartServerIfStoppedAsync(ct);
        string? gemmaBackend = _selectedModelIsGemma ? gemmaOpenClAuto ? "opencl" : "cpu" : null;
        if (gemmaBackend is not null && _serverGemmaBackend is { } prior
            && !string.Equals(prior, gemmaBackend, StringComparison.Ordinal))
            await RestartServerForGemmaBackendChangeAsync(ct);
        if (_isLoaded && string.Equals(_loadedModelPath, modelPath, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_loadedAdapterPath, adapterPath, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_loadedMmprojPath, SelectedMmprojPath, StringComparison.OrdinalIgnoreCase)
            && _loadedDevices is { } loadedDevices && loadedDevices.SequenceEqual(deviceIndices)
            && string.Equals(_loadedGemmaBackend, gemmaBackend, StringComparison.Ordinal))
        {
            if (gemmaBackend is not null) _serverGemmaBackend = gemmaBackend;
            EnsureServerReady();
            await _api.LoadAsync(modelPath, adapterPath, SelectedMmprojPath, deviceIndices, gemmaBackend, ct);
            await RefreshServerStateAsync(ct);
            SetStatus("選択中のモデルはプリロード済みです。");
            return;
        }
        ct.ThrowIfCancellationRequested();
        SetStatus($"読み込み中: {Path.GetFileName(modelPath)}");
        // Native backend initialization can succeed even when model loading later fails.
        // Remember the attempted backend so the next backend switch restarts the server.
        if (gemmaBackend is not null) _serverGemmaBackend = gemmaBackend;
        EnsureServerReady();
        await _api.LoadAsync(modelPath, adapterPath, SelectedMmprojPath, deviceIndices, gemmaBackend, ct);
        ct.ThrowIfCancellationRequested();
        await RefreshServerStateAsync(ct);
        string gpu = gemmaBackend == "opencl" ? "OpenCL GPU（自動）"
            : deviceIndices.Count == 0 ? "CPU" : $"Arc {string.Join(",", deviceIndices)}";
        SetStatus(adapterPath is null
            ? $"プリロード完了。モデルを {gpu} に保持しています。"
            : $"プリロード完了。モデルと LoRA を {gpu} に保持しています。");
    }

    private async Task RestartServerForGemmaBackendChangeAsync(CancellationToken ct)
    {
        SetStatus("Gemma 3 の計算方式を切り替えるため、推論サーバーを再起動しています…");
        using var timeout = CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(TimeSpan.FromSeconds(10));
        await PostAsync("internal/shutdown", new { }, timeout.Token);
        await _server.WaitForExitAsync(timeout.Token);
        await RestartServerIfStoppedAsync(ct);
    }

    private async Task RestartServerIfStoppedAsync(CancellationToken ct)
    {
        if (_server.IsRunning)
        {
            if (_serverReady) return;
            throw new InvalidOperationException("推論サーバーの起動が完了していません。");
        }
        ct.ThrowIfCancellationRequested();
        _idleTimer.Stop();
        _isLoaded = false;
        _loadedModelPath = null;
        _loadedAdapterPath = null;
        _loadedMmprojPath = null;
        _loadedDevices = null;
        _loadedGemmaBackend = null;
        _serverGemmaBackend = null;
        _server.ResetExitedProcess();
        SetStatus("推論サーバーが停止したため、再起動しています…");
        await StartServerAsync();
        ct.ThrowIfCancellationRequested();
        EnsureServerReady();
    }

    private async Task UnloadServerAsync(CancellationToken ct = default)
    {
        if (!_serverReady) return;
        await PostAsync("internal/unload", new { }, ct);
        await RefreshServerStateAsync(ct);
    }

    private async Task PostAsync(string path, object body, CancellationToken ct)
    {
        EnsureServerReady();
        await _api.PostAsync(path, body, ct);
    }

    private async Task RefreshServerStateAsync(CancellationToken ct = default)
    {
        if (!_serverReady) return;
        LocalInferenceState state = await _api.ReadStateAsync(ct);
        _isLoaded = state.IsLoaded;
        _loadedModelPath = state.ModelPath;
        _loadedAdapterPath = state.AdapterPath;
        _loadedMmprojPath = state.MmprojPath;
        _loadedDevices = state.Devices;
        _loadedGemmaBackend = state.GemmaBackend;
        if (_loadedGemmaBackend is not null) _serverGemmaBackend = _loadedGemmaBackend;
        _idleSecondsRemaining = state.IdleSecondsRemaining;
        UpdateMemoryStatus();
    }

    private void EnsureServerReady() => _server.EnsureReady();

    private Task<LocalGeneration> GenerateViaServerAsync(
        IReadOnlyList<ChatTurn> conversation, string modelPath, string? adapterPath,
        IReadOnlyList<int> devices, bool gemmaOpenClAuto, bool thinking, bool stream, int maxNewTokens,
        GenerationSampling sampling, Action<string> onText, CancellationToken ct)
    {
        EnsureServerReady();
        var request = new LocalGenerationRequest(conversation, modelPath, adapterPath,
            SelectedMmprojPath, devices,
            _selectedModelIsGemma ? gemmaOpenClAuto ? "opencl" : "cpu" : null,
            thinking, stream, maxNewTokens, sampling);
        return _api.GenerateAsync(request, onText, ct);
    }

    private async void IdleTimer_Tick(object? sender, EventArgs e)
    {
        if (_busy || !_serverReady || _closingFinished) return;
        bool wasLoaded = _isLoaded;
        try
        {
            await RefreshServerStateAsync();
            if (wasLoaded && !_isLoaded)
                SetStatus("5 分間操作がなかったため、サーバーが GPU 資源を解放しました。");
        }
        catch (Exception ex) { SetStatus($"サーバー状態を取得できません: {ex.Message}"); }
    }
}
