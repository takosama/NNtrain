using System.ComponentModel;
using System.Windows;
using System.Windows.Threading;

namespace NNtrain.Gui;

public partial class MainWindow : Window
{
    private readonly GuiLaunchOptions _launchOptions;
    private readonly SemaphoreSlim _requestGate = new(1, 1);
    private CancellationTokenSource? _activeCancellation;
    private bool _busy, _refreshing, _pendingUnload, _closingFinished, _closingStarted;

    internal MainWindow(GuiLaunchOptions launchOptions)
    {
        _launchOptions = launchOptions;
        _server = new LocalInferenceServerProcess(launchOptions.LoraPath, LocalOpenAiServer.TokenEnvironmentVariable);
        _api = new LocalInferenceClient(_server.AuthenticationToken);
        InitializeComponent();
        ModelComboBox.ItemsSource = _models;
        AdapterComboBox.ItemsSource = _adapters;
        GpuComboBox.ItemsSource = _gpus;
        ChatItems.ItemsSource = _messages;
        _idleTimer.Tick += IdleTimer_Tick;
        RefreshGpuChoices();
        RefreshChoices();
        InitializeHuggingFaceTab();
        InitializeAudio();
        ApplyLaunchOptions();
        UpdateControls();
        Loaded += async (_, _) => await StartServerAsync();
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
        _activeCancellation?.Cancel();
        CancelPendingImagePreparation();
        SetStatus("停止しています…");
    }

    private async Task ExecuteAsync(Func<CancellationToken, Task> action)
    {
        if (!_requestGate.Wait(0)) return;
        _busy = true;
        var cancellation = new CancellationTokenSource();
        _activeCancellation = cancellation;
        UpdateControls();
        try { await action(cancellation.Token); }
        catch (OperationCanceledException) { SetStatus("処理を停止しました。"); }
        catch (Exception ex) { SetStatus($"エラー: {ex.Message}"); }
        finally
        {
            if (_pendingUnload)
            {
                _pendingUnload = false;
                try
                {
                    if (_isLoaded) await UnloadServerAsync();
                    SetStatus("選択変更により GPU 資源を解放しました。");
                }
                catch (Exception ex) { SetStatus($"GPU 資源の解放に失敗: {ex.Message}"); }
            }
            _activeCancellation = null;
            cancellation.Dispose();
            _busy = false;
            _requestGate.Release();
            try { await RefreshServerStateAsync(); }
            catch (Exception ex) { SetStatus($"サーバー状態を取得できません: {ex.Message}"); }
            UpdateControls();
        }
    }

    private void SetStatus(string message)
    {
        StatusText.Text = message;
        UpdateMemoryStatus();
    }

    private void UpdateControls()
    {
        UpdateAudioControls();
        ModelComboBox.IsEnabled = !_busy;
        AdapterComboBox.IsEnabled = !_busy && !_selectedModelIsGemma;
        MmprojComboBox.IsEnabled = !_busy && !_selectedModelIsGemma;
        AttachImageButton.IsEnabled = !_busy && !_asrActive && !_selectedModelIsGemma;
        NewConversationButton.IsEnabled = !_busy && !_asrActive;
        UpdateImageAttachment();
        GpuComboBox.IsEnabled = !_busy;
        MessageBox.IsEnabled = !_busy;
        MaxTokensBox.IsEnabled = !_busy;
        TemperatureBox.IsEnabled = !_busy;
        TopPBox.IsEnabled = !_busy;
        TopKBox.IsEnabled = !_busy;
        ThinkingCheckBox.IsEnabled = !_busy && !_selectedModelIsGemma;
        StreamCheckBox.IsEnabled = !_busy;
        PreloadButton.IsEnabled = _serverReady && !_busy && !_asrActive && ModelComboBox.SelectedItem is FileChoice
            && GpuComboBox.SelectedItem is GpuChoice;
        SendButton.IsEnabled = _serverReady && !_busy && !_asrActive && ModelComboBox.SelectedItem is FileChoice
            && GpuComboBox.SelectedItem is GpuChoice;
        StopButton.IsEnabled = _busy;
        UpdateMemoryStatus();
    }

    private void UpdateMemoryStatus()
    {
        if (_isLoaded)
        {
            string gpu = _loadedGemmaBackend == "opencl" ? "OpenCL GPU（自動）"
                : _loadedDevices is { Length: 0 } ? "CPU" : _loadedDevices is { Length: > 0 } devices
                ? $"Arc {string.Join(",", devices)}" : "GPU";
            string remaining = _idleSecondsRemaining is double seconds ?
                $" · 約{Math.Max(0, (int)Math.Ceiling(seconds / 60))}分後に解放" : "";
            MemoryStatusText.Text = $"{gpu}{remaining}";
        }
        else MemoryStatusText.Text = _selectedModelIsGemma && GpuComboBox.SelectedItem is GpuChoice
            { IsGemmaOpenClAuto: true } ? "OpenCL GPU: 未読込・解放済み"
            : _selectedModelIsGemma ? "CPU: 未読込・解放済み" : "GPU: 未読込・解放済み";
    }

    private void ScrollToEnd() => Dispatcher.BeginInvoke(DispatcherPriority.Background,
        new Action(() => ChatScrollViewer.ScrollToEnd()));

    private async void MainWindow_Closing(object? sender, CancelEventArgs e)
    {
        if (_closingFinished) return;
        e.Cancel = true;
        if (_closingStarted) return;
        _closingStarted = true;
        IsEnabled = false;
        _idleTimer.Stop();
        CancelAudio();
        if (_asrTask is { } audioTask)
        {
            try { await audioTask; } catch { }
        }
        _activeCancellation?.Cancel();
        _hfOperationCancellation?.Cancel();
        CancelPendingImagePreparation();
        try { await _server.StopAsync(_api); }
        finally
        {
            _asrModel?.Dispose();
            _api.Dispose();
            _hfClient.Dispose();
            _closingFinished = true;
            _ = Dispatcher.BeginInvoke(new Action(Close));
        }
    }
}
