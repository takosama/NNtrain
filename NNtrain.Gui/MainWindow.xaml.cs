using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Threading;
using Microsoft.Win32;
using NNtrain.Arc;

namespace NNtrain.Gui;

public partial class MainWindow : Window
{
    private static readonly TimeSpan IdleLimit = TimeSpan.FromMinutes(5);
    private readonly InferenceSession _session = new();
    private readonly ObservableCollection<FileChoice> _models = [];
    private readonly ObservableCollection<FileChoice> _adapters = [];
    private readonly ObservableCollection<GpuChoice> _gpus = [];
    private readonly ObservableCollection<ChatBubble> _messages = [];
    private readonly List<ChatTurn> _conversation = [];
    private readonly SemaphoreSlim _sessionGate = new(1, 1);
    private readonly DispatcherTimer _idleTimer = new() { Interval = TimeSpan.FromSeconds(15) };
    private CancellationTokenSource? _activeCancellation;
    private DateTimeOffset? _lastCompletedUtc;
    private string? _modelsDirectory;
    private bool _busy, _refreshing, _pendingUnload, _closingFinished;
    private int _streamingGeneration;

    public MainWindow()
    {
        InitializeComponent();
        ModelComboBox.ItemsSource = _models;
        AdapterComboBox.ItemsSource = _adapters;
        GpuComboBox.ItemsSource = _gpus;
        ChatItems.ItemsSource = _messages;
        _idleTimer.Tick += IdleTimer_Tick;
        RefreshGpuChoices();
        RefreshChoices();
        UpdateControls();
    }

    private void RefreshChoices()
    {
        _refreshing = true;
        try
        {
            string? selectedModel = (ModelComboBox.SelectedItem as FileChoice)?.FilePath;
            string? selectedAdapter = (AdapterComboBox.SelectedItem as FileChoice)?.FilePath;
            FileChoice[] manualModels = _models.Where(item => item.IsManual).ToArray();
            FileChoice[] manualAdapters = _adapters.Where(item => item.IsManual).ToArray();
            _models.Clear();
            _adapters.Clear();
            _adapters.Add(new FileChoice("なし", null));
            _modelsDirectory = FindModelsDirectory();
            if (_modelsDirectory is not null)
            {
                foreach (string path in EnumerateModelFiles(_modelsDirectory)) AddChoice(_models, path, false);
                foreach (string path in EnumerateAdapterFiles(_modelsDirectory)) AddChoice(_adapters, path, false);
                string? repository = Directory.GetParent(_modelsDirectory)?.FullName;
                if (repository is not null)
                {
                    string checkpoints = Path.Combine(repository, "checkpoints");
                    if (Directory.Exists(checkpoints))
                        foreach (string path in EnumerateAdapterFiles(checkpoints)) AddChoice(_adapters, path, false);
                }
            }
            foreach (FileChoice item in manualModels) AddChoice(_models, item.FilePath!, true);
            foreach (FileChoice item in manualAdapters) AddChoice(_adapters, item.FilePath!, true);
            ModelComboBox.SelectedItem = FindChoice(_models, selectedModel) ?? _models.FirstOrDefault();
            AdapterComboBox.SelectedItem = FindChoice(_adapters, selectedAdapter) ?? _adapters[0];
            if (_models.Count == 0)
                SetStatus("models フォルダーに GGUF がありません。参照からモデルを選択してください。");
        }
        finally { _refreshing = false; }
        if (_session.IsLoaded && !SelectionMatchesLoaded())
            _ = ReleaseForSelectionChangeAsync();
        UpdateControls();
    }

    private void RefreshGpuChoices()
    {
        bool wasRefreshing = _refreshing;
        _refreshing = true;
        int[]? selected = (GpuComboBox.SelectedItem as GpuChoice)?.DeviceIndices;
        try
        {
            _gpus.Clear();
            IReadOnlyList<ArcDeviceInfo> available = ArcDevices.Enumerate();
            for (int first = 0; first < available.Count; first++)
                for (int second = first + 1; second < available.Count; second++)
                    _gpus.Add(new GpuChoice($"Arc {first} + Arc {second}（2台）", [first, second]));
            for (int index = 0; index < available.Count; index++)
                _gpus.Add(new GpuChoice($"Arc {index}（1台）", [index]));
            GpuComboBox.SelectedItem = _gpus.FirstOrDefault(choice =>
                selected is not null && choice.DeviceIndices.SequenceEqual(selected)) ?? _gpus.FirstOrDefault();
        }
        finally { _refreshing = wasRefreshing; }
    }

    private static IEnumerable<string> EnumerateModelFiles(string directory)
    {
        IEnumerable<string> files;
        try { files = Directory.GetFiles(directory, "*.gguf", SearchOption.AllDirectories); }
        catch (IOException) { yield break; }
        catch (UnauthorizedAccessException) { yield break; }
        foreach (string path in files.Order(StringComparer.OrdinalIgnoreCase))
            if (!Path.GetFileName(path).StartsWith("lora_", StringComparison.OrdinalIgnoreCase))
                yield return path;
    }

    private static IEnumerable<string> EnumerateAdapterFiles(string directory)
    {
        IEnumerable<string> files;
        try { files = Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories); }
        catch (IOException) { yield break; }
        catch (UnauthorizedAccessException) { yield break; }
        foreach (string path in files.Order(StringComparer.OrdinalIgnoreCase)) yield return path;
    }

    private static string? FindModelsDirectory()
    {
        foreach (string origin in new[] { AppContext.BaseDirectory, Environment.CurrentDirectory })
        {
            DirectoryInfo? current = new(origin);
            while (current is not null)
            {
                string path = Path.Combine(current.FullName, "models");
                if (Directory.Exists(path)) return path;
                current = current.Parent;
            }
        }
        return null;
    }

    private static FileChoice AddChoice(ObservableCollection<FileChoice> choices, string path, bool manual)
    {
        string fullPath = Path.GetFullPath(path);
        FileChoice? existing = FindChoice(choices, fullPath);
        if (existing is not null) return existing;
        string label = Path.GetExtension(fullPath).Equals(".bin", StringComparison.OrdinalIgnoreCase)
            ? $"{Path.GetFileName(Path.GetDirectoryName(fullPath) ?? string.Empty)} / {Path.GetFileName(fullPath)}"
            : Path.GetFileName(fullPath);
        var choice = new FileChoice(label, fullPath, manual);
        choices.Add(choice);
        return choice;
    }

    private static FileChoice? FindChoice(IEnumerable<FileChoice> choices, string? path) =>
        choices.FirstOrDefault(item => string.Equals(item.FilePath, path, StringComparison.OrdinalIgnoreCase));

    private async void ModelOrAdapter_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_refreshing || _closingFinished) return;
        if (_session.IsLoaded && !SelectionMatchesLoaded()) await ReleaseForSelectionChangeAsync();
        UpdateControls();
    }

    private bool SelectionMatchesLoaded() =>
        string.Equals(_session.LoadedModelPath, (ModelComboBox.SelectedItem as FileChoice)?.FilePath,
            StringComparison.OrdinalIgnoreCase) &&
        string.Equals(_session.LoadedAdapterPath, (AdapterComboBox.SelectedItem as FileChoice)?.FilePath,
            StringComparison.OrdinalIgnoreCase) &&
        _session.LoadedDevices is { } loadedDevices &&
        GpuComboBox.SelectedItem is GpuChoice gpu && loadedDevices.SequenceEqual(gpu.DeviceIndices);

    private async Task ReleaseForSelectionChangeAsync()
    {
        if (_busy)
        {
            _pendingUnload = true;
            _activeCancellation?.Cancel();
            SetStatus("選択が変わりました。現在の処理を停止して GPU 資源を解放します…");
            return;
        }
        await ExecuteAsync(async _ =>
        {
            if (!_session.IsLoaded || SelectionMatchesLoaded()) return;
            await _session.UnloadAsync();
            _lastCompletedUtc = null;
            SetStatus("選択が変わったため GPU 資源を解放しました。");
        });
    }

    private void BrowseModel_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        var dialog = new OpenFileDialog
        {
            Title = "Qwen3.5 / Qwen3.8 GGUF モデルを選択",
            Filter = "GGUF モデル (*.gguf)|*.gguf",
            CheckFileExists = true,
            InitialDirectory = _modelsDirectory ?? Environment.CurrentDirectory
        };
        if (dialog.ShowDialog(this) == true)
        {
            if (Path.GetFileName(dialog.FileName).StartsWith("lora_", StringComparison.OrdinalIgnoreCase))
            {
                SetStatus("選択したファイルは LoRA アダプターです。ベースの GGUF モデルを選んでください。");
                return;
            }
            ModelComboBox.SelectedItem = AddChoice(_models, dialog.FileName, true);
        }
    }

    private void BrowseAdapter_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        var dialog = new OpenFileDialog
        {
            Title = "LoRA アダプターを選択",
            Filter = "NNtrain LoRA アダプター (*.bin)|*.bin",
            CheckFileExists = true,
            InitialDirectory = _modelsDirectory ?? Environment.CurrentDirectory
        };
        if (dialog.ShowDialog(this) == true)
            AdapterComboBox.SelectedItem = AddChoice(_adapters, dialog.FileName, true);
    }

    private void RefreshModels_Click(object sender, RoutedEventArgs e)
    {
        if (!_busy)
        {
            RefreshGpuChoices();
            RefreshChoices();
        }
    }

    private void Thinking_Changed(object sender, RoutedEventArgs e)
    {
        if (ThinkingCheckBox is not null)
            ThinkingCheckBox.Content = ThinkingCheckBox.IsChecked == true ? "Thinking: On" : "Thinking: Off";
    }

    private async void Preload_Click(object sender, RoutedEventArgs e)
    {
        if (!TrySelectedFiles(out string? modelPath, out string? adapterPath, out GpuChoice? gpu)) return;
        await ExecuteAsync(ct => EnsureLoadedAsync(modelPath!, adapterPath, gpu!.DeviceIndices, ct));
    }

    private async Task EnsureLoadedAsync(string modelPath, string? adapterPath,
        IReadOnlyList<int> deviceIndices, CancellationToken ct)
    {
        if (_session.IsLoaded && string.Equals(_session.LoadedModelPath, modelPath, StringComparison.OrdinalIgnoreCase)
            && string.Equals(_session.LoadedAdapterPath, adapterPath, StringComparison.OrdinalIgnoreCase)
            && _session.LoadedDevices is { } loadedDevices && loadedDevices.SequenceEqual(deviceIndices))
        {
            SetStatus("選択中のモデルはプリロード済みです。");
            MarkCompleted();
            return;
        }
        if (_session.IsLoaded)
        {
            SetStatus("前のモデルの GPU 資源を解放しています…");
            await _session.UnloadAsync();
            _lastCompletedUtc = null;
        }
        ct.ThrowIfCancellationRequested();
        SetStatus($"読み込み中: {Path.GetFileName(modelPath)}");
        var progress = new Progress<string>(message => SetStatus(message));
        await _session.LoadAsync(modelPath, adapterPath, progress, ct, deviceIndices);
        ct.ThrowIfCancellationRequested();
        string gpu = _session.LoadedDevices is { Count: > 0 } devices
            ? $"Arc {string.Join(",", devices)}" : "Arc GPU";
        SetStatus(adapterPath is null
            ? $"プリロード完了。モデルを {gpu} に保持しています。"
            : $"プリロード完了。モデルと LoRA を {gpu} に保持しています。");
        MarkCompleted();
    }

    private async void Send_Click(object sender, RoutedEventArgs e) => await SendMessageAsync();

    private void MessageBox_PreviewKeyDown(object sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && Keyboard.Modifiers == ModifierKeys.Control)
        {
            e.Handled = true;
            _ = SendMessageAsync();
        }
    }

    private async Task SendMessageAsync()
    {
        string message = MessageBox.Text.Trim();
        if (message.Length == 0 ||
            !TrySelectedFiles(out string? modelPath, out string? adapterPath, out GpuChoice? gpu)) return;
        if (!int.TryParse(MaxTokensBox.Text, out int maxNewTokens) || maxNewTokens is < 1 or > 8192)
        {
            SetStatus("最大生成数は 1 ～ 8192 の整数で指定してください。");
            return;
        }
        if (!float.TryParse(TemperatureBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture,
                out float temperature) || !float.IsFinite(temperature) || temperature is < 0f or > 2f)
        {
            SetStatus("temperature は 0 ～ 2 の数値で指定してください。");
            return;
        }
        if (!float.TryParse(TopPBox.Text, NumberStyles.Float, CultureInfo.InvariantCulture,
                out float topP) || !float.IsFinite(topP) || topP is <= 0f or > 1f)
        {
            SetStatus("top-p は 0 より大きく 1 以下の数値で指定してください。");
            return;
        }
        if (!int.TryParse(TopKBox.Text, out int topK) || topK is < 1 or > 256)
        {
            SetStatus("top-k は 1 ～ 256 の整数で指定してください。");
            return;
        }
        var sampling = new GenerationSampling(temperature, topP, topK);
        await ExecuteAsync(async ct =>
        {
            await EnsureLoadedAsync(modelPath!, adapterPath, gpu!.DeviceIndices, ct);
            ct.ThrowIfCancellationRequested();
            MessageBox.Clear();
            _conversation.Add(new ChatTurn("user", message));
            _messages.Add(new ChatBubble("あなた", message));
            bool thinking = ThinkingCheckBox.IsChecked == true;
            var answer = new ChatBubble("アシスタント", "", thinking);
            _messages.Add(answer);
            ScrollToEnd();
            SetStatus(thinking ? "思考中…" : "生成中…");
            int firstGeneration = ++_streamingGeneration;
            var liveParser = new ThinkingStreamParser(thinking);
            var streamedRaw = new StringBuilder();
            string? finalRaw = null;
            ThinkingStreamSnapshot? completedSnapshot = null;
            GenerationStopReason? stopReason = null;
            bool usedThinkingForAnswer = thinking;
            bool retriedWithoutThinking = false;
            try
            {
                finalRaw = await _session.GenerateAsync(_conversation.ToArray(),
                    thinking, maxNewTokens, chunk =>
                    {
                        streamedRaw.Append(chunk);
                        Dispatcher.BeginInvoke(DispatcherPriority.Background, new Action(() =>
                        {
                            if (firstGeneration != _streamingGeneration) return;
                            ThinkingStreamSnapshot snapshot = liveParser.Append(chunk);
                            answer.Update(snapshot);
                            if (snapshot.ThinkingInProgress) SetStatus("思考中…");
                            else if (snapshot.AnswerText.Length != 0) SetStatus("回答を生成中…");
                            ScrollToEnd();
                        }));
                    }, ct, sampling);
                stopReason = _session.LastGenerationStats?.StopReason;
                var firstFinalParser = new ThinkingStreamParser(thinking);
                firstFinalParser.Append(finalRaw);
                ThinkingStreamSnapshot firstSnapshot = firstFinalParser.Complete();
                completedSnapshot = firstSnapshot;

                // A model can finish its turn or exhaust the token limit before
                // emitting an answer delimiter. Keep its reasoning collapsed and
                // make one direct-answer attempt without feeding that reasoning
                // into the next prompt.
                if (thinking && string.IsNullOrWhiteSpace(firstSnapshot.AnswerText)
                    && stopReason is GenerationStopReason.EndOfMessage or GenerationStopReason.MaximumTokens)
                {
                    retriedWithoutThinking = true;
                    usedThinkingForAnswer = false;
                    int retryGeneration = ++_streamingGeneration;
                    string firstThought = firstSnapshot.ThinkingText;
                    answer.Update(firstSnapshot);
                    SetStatus("思考だけで終了したため、回答を生成しています…");
                    var directParser = new ThinkingStreamParser(requestedThinking: false);
                    string directRaw = await _session.GenerateAsync(_conversation.ToArray(),
                        thinking: false, maxNewTokens, chunk =>
                        {
                            Dispatcher.BeginInvoke(DispatcherPriority.Background, new Action(() =>
                            {
                                if (retryGeneration != _streamingGeneration) return;
                                ThinkingStreamSnapshot direct = directParser.Append(chunk);
                                answer.Update(new ThinkingStreamSnapshot(firstThought,
                                    direct.AnswerText, firstThought.Length != 0, false));
                                if (direct.AnswerText.Length != 0) SetStatus("回答を生成中…");
                                ScrollToEnd();
                            }));
                        }, ct, sampling);
                    stopReason = _session.LastGenerationStats?.StopReason;
                    var directFinalParser = new ThinkingStreamParser(requestedThinking: false);
                    directFinalParser.Append(directRaw);
                    ThinkingStreamSnapshot directFinal = directFinalParser.Complete(
                        stopReason == GenerationStopReason.EndOfMessage);
                    string combinedThought = string.IsNullOrWhiteSpace(directFinal.ThinkingText)
                        ? firstThought : string.IsNullOrWhiteSpace(firstThought)
                            ? directFinal.ThinkingText : firstThought + "\n\n" + directFinal.ThinkingText;
                    completedSnapshot = new ThinkingStreamSnapshot(combinedThought,
                        directFinal.AnswerText, !string.IsNullOrWhiteSpace(combinedThought), false);
                }
                MarkCompleted();
                GenerationStats? stats = _session.LastGenerationStats;
                SetStatus(stats is null ? "生成が完了しました。" :
                    (retriedWithoutThinking ? "回答を再生成しました。" : "生成完了。") +
                    $"入力 {stats.PromptTokens} token、KV再利用 {stats.ReusedPromptTokens} token、" +
                    (stats.FirstTokenMilliseconds is double first
                        ? $"最初の出力まで {first / 1000:F2} 秒。" : "出力トークンなし。"));
            }
            finally
            {
                ++_streamingGeneration; // Ignore queued callbacks after settling the final text.
                if (completedSnapshot is null)
                {
                    var finalParser = new ThinkingStreamParser(thinking);
                    finalParser.Append(finalRaw ?? streamedRaw.ToString());
                    completedSnapshot = finalParser.Complete(
                        _session.LastGenerationStats?.StopReason == GenerationStopReason.EndOfMessage);
                }
                answer.Update(completedSnapshot.Value);
                answer.FinishThinking(stopReason);
                if (answer.Content.Length == 0 && answer.ThinkingContent.Length == 0 && stopReason is null)
                    _messages.Remove(answer);
                else if (answer.Content.Length != 0)
                {
                    // Keep the exact assistant prefix used by the preceding
                    // prompt while retaining only the visible answer in history.
                    // The next prompt can then extend the cached token prefix.
                    string assistantPrefix = usedThinkingForAnswer
                        ? "<think>\n</think>\n" : "<think>\n\n</think>\n\n";
                    _conversation.Add(new ChatTurn("assistant", answer.Content, assistantPrefix));
                }
                ScrollToEnd();
            }
        });
    }

    private void Stop_Click(object sender, RoutedEventArgs e)
    {
        _activeCancellation?.Cancel();
        SetStatus("停止しています…");
    }

    private async Task ExecuteAsync(Func<CancellationToken, Task> action)
    {
        if (!_sessionGate.Wait(0)) return;
        _busy = true;
        _idleTimer.Stop();
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
                    if (_session.IsLoaded) await _session.UnloadAsync();
                    _lastCompletedUtc = null;
                    SetStatus("選択変更により GPU 資源を解放しました。");
                }
                catch (Exception ex) { SetStatus($"GPU 資源の解放に失敗: {ex.Message}"); }
            }
            _activeCancellation = null;
            cancellation.Dispose();
            _busy = false;
            _sessionGate.Release();
            if (_session.IsLoaded)
            {
                // A stopped or failed generation can have occupied the GPU for
                // minutes. Restart the idle window when that operation ends.
                _lastCompletedUtc = DateTimeOffset.UtcNow;
                _idleTimer.Start();
            }
            else _lastCompletedUtc = null;
            UpdateControls();
        }
    }

    private async void IdleTimer_Tick(object? sender, EventArgs e)
    {
        if (_busy || !_session.IsLoaded || _lastCompletedUtc is null) return;
        UpdateMemoryStatus();
        if (DateTimeOffset.UtcNow - _lastCompletedUtc.Value < IdleLimit) return;
        await ExecuteAsync(async _ =>
        {
            if (!_session.IsLoaded) return;
            await _session.UnloadAsync();
            _lastCompletedUtc = null;
            SetStatus("最終プリロード/生成から 5 分経過したため GPU 資源を解放しました。");
        });
    }

    private void MarkCompleted()
    {
        _lastCompletedUtc = DateTimeOffset.UtcNow;
        _idleTimer.Start();
    }

    private bool TrySelectedFiles(out string? modelPath, out string? adapterPath, out GpuChoice? gpu)
    {
        modelPath = (ModelComboBox.SelectedItem as FileChoice)?.FilePath;
        adapterPath = (AdapterComboBox.SelectedItem as FileChoice)?.FilePath;
        gpu = GpuComboBox.SelectedItem as GpuChoice;
        if (modelPath is null || !File.Exists(modelPath))
        {
            SetStatus("GGUF モデルを選択してください。");
            return false;
        }
        if (adapterPath is not null && !File.Exists(adapterPath))
        {
            SetStatus("選択中の LoRA アダプターが見つかりません。");
            return false;
        }
        if (adapterPath is not null && string.Equals(modelPath, adapterPath, StringComparison.OrdinalIgnoreCase))
        {
            SetStatus("モデルと LoRA には別のファイルを選んでください。");
            return false;
        }
        if (gpu is null)
        {
            SetStatus("使用する Arc GPU を選択してください。");
            return false;
        }
        return true;
    }

    private void SetStatus(string message)
    {
        StatusText.Text = message;
        UpdateMemoryStatus();
    }

    private void UpdateControls()
    {
        ModelComboBox.IsEnabled = !_busy;
        AdapterComboBox.IsEnabled = !_busy;
        GpuComboBox.IsEnabled = !_busy;
        MessageBox.IsEnabled = !_busy;
        MaxTokensBox.IsEnabled = !_busy;
        TemperatureBox.IsEnabled = !_busy;
        TopPBox.IsEnabled = !_busy;
        TopKBox.IsEnabled = !_busy;
        ThinkingCheckBox.IsEnabled = !_busy;
        PreloadButton.IsEnabled = !_busy && ModelComboBox.SelectedItem is FileChoice
            && GpuComboBox.SelectedItem is GpuChoice;
        SendButton.IsEnabled = !_busy && ModelComboBox.SelectedItem is FileChoice
            && GpuComboBox.SelectedItem is GpuChoice;
        StopButton.IsEnabled = _busy;
        UpdateMemoryStatus();
    }

    private void UpdateMemoryStatus()
    {
        if (_session.IsLoaded)
        {
            string gpu = _session.LoadedDevices is { Count: > 0 } devices
                ? $"Arc {string.Join(",", devices)}" : "GPU";
            string remaining = _lastCompletedUtc is null ? "" :
                $" · 約{Math.Max(0, (int)Math.Ceiling((IdleLimit - (DateTimeOffset.UtcNow - _lastCompletedUtc.Value)).TotalSeconds / 60))}分後に解放";
            MemoryStatusText.Text = $"{gpu}{remaining}";
        }
        else MemoryStatusText.Text = "GPU: 未読込・解放済み";
    }

    private void ScrollToEnd() => Dispatcher.BeginInvoke(DispatcherPriority.Background,
        new Action(() => ChatScrollViewer.ScrollToEnd()));

    private async void MainWindow_Closing(object? sender, CancelEventArgs e)
    {
        if (_closingFinished) return;
        e.Cancel = true;
        IsEnabled = false;
        _idleTimer.Stop();
        _activeCancellation?.Cancel();
        await _sessionGate.WaitAsync();
        try
        {
            await _session.UnloadAsync();
        }
        catch { /* Closing still disposes the owned session. */ }
        finally
        {
            try { _session.Dispose(); }
            catch { /* A failing driver must not prevent the window from closing. */ }
            _sessionGate.Release();
            _closingFinished = true;
            Close();
        }
    }

    public sealed record FileChoice(string Label, string? FilePath, bool IsManual = false);
    public sealed record GpuChoice(string Label, int[] DeviceIndices);

    public sealed class ChatBubble : INotifyPropertyChanged
    {
        private string _content;
        private string _thinkingContent = "";
        private bool _hasThinking;
        private bool _thinkingInProgress;
        private bool _thinkingExpanded;
        private bool _generationFinished;
        private GenerationStopReason? _stopReason;

        public ChatBubble(string author, string content, bool hasThinking = false)
        {
            Author = author;
            _content = content;
            _hasThinking = hasThinking;
            _thinkingInProgress = hasThinking;
        }

        public string Author { get; }
        public bool HasThinking => _hasThinking;
        public Visibility ThinkingVisibility => _hasThinking && (_thinkingInProgress || _thinkingContent.Length != 0)
            ? Visibility.Visible : Visibility.Collapsed;
        public string ThinkingHeader => _thinkingInProgress ? "思考中…（クリックで表示）" : "思考の内容";
        public string DisplayContent
        {
            get
            {
                if (_content.Length != 0) return _content;
                if (_generationFinished)
                {
                    if (_hasThinking && !string.IsNullOrWhiteSpace(_thinkingContent))
                        return _stopReason switch
                        {
                            GenerationStopReason.MaximumTokens => "思考中に最大生成数へ達しました。最大生成数を増やして再実行してください。",
                            GenerationStopReason.ContextLimit => "会話がモデルの文脈上限に達しました。会話を短くして再実行してください。",
                            GenerationStopReason.EndOfMessage => "思考の終了タグがないままモデルが終了しました。思考の内容を確認してください。",
                            _ => "生成が中断されました。思考の内容を確認してください。"
                        };
                    return _stopReason switch
                    {
                        GenerationStopReason.EndOfMessage => "モデルは回答を出さずに終了しました。",
                        GenerationStopReason.MaximumTokens => "最大生成数に達しましたが、表示できる回答がありません。",
                        GenerationStopReason.ContextLimit => "会話がモデルの文脈上限に達しました。",
                        _ => ""
                    };
                }
                if (_thinkingInProgress) return "";
                return _hasThinking ? "回答を生成中…" : "生成中…";
            }
        }

        public string Content
        {
            get => _content;
            set
            {
                if (_content == value) return;
                _content = value;
                Notify(nameof(Content));
                Notify(nameof(DisplayContent));
            }
        }

        public string ThinkingContent
        {
            get => _thinkingContent;
            set
            {
                if (_thinkingContent == value) return;
                _thinkingContent = value;
                Notify(nameof(ThinkingContent));
                Notify(nameof(ThinkingVisibility));
                Notify(nameof(DisplayContent));
            }
        }

        public bool ThinkingExpanded
        {
            get => _thinkingExpanded;
            set
            {
                if (_thinkingExpanded == value) return;
                _thinkingExpanded = value;
                Notify(nameof(ThinkingExpanded));
            }
        }

        internal void Update(ThinkingStreamSnapshot snapshot)
        {
            if (_hasThinking != snapshot.HasThinking)
            {
                _hasThinking = snapshot.HasThinking;
                Notify(nameof(HasThinking));
                Notify(nameof(ThinkingVisibility));
            }
            ThinkingContent = snapshot.ThinkingText;
            Content = snapshot.AnswerText;
            SetThinkingInProgress(snapshot.ThinkingInProgress);
        }

        internal void FinishThinking(GenerationStopReason? stopReason = null)
        {
            _generationFinished = true;
            _stopReason = stopReason;
            SetThinkingInProgress(false);
            ThinkingExpanded = false;
            Notify(nameof(DisplayContent));
        }

        private void SetThinkingInProgress(bool value)
        {
            if (_thinkingInProgress == value) return;
            _thinkingInProgress = value;
            Notify(nameof(ThinkingHeader));
            Notify(nameof(ThinkingVisibility));
            Notify(nameof(DisplayContent));
        }

        private void Notify(string name) => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
        public event PropertyChangedEventHandler? PropertyChanged;
    }
}
