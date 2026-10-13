using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;
using NNtrain.Arc;

namespace NNtrain.Gui;

public partial class MainWindow
{
    private readonly ObservableCollection<FileChoice> _models = [];
    private readonly ObservableCollection<FileChoice> _adapters = [];
    private readonly ObservableCollection<GpuChoice> _gpus = [];
    private string? _modelsDirectory;
    private bool _selectedModelIsGemma;

    private void ApplyLaunchOptions()
    {
        if (_launchOptions.ModelPath is { } model)
        {
            if (!File.Exists(model)) throw new FileNotFoundException("--model の GGUF が見つかりません。", model);
            ModelComboBox.SelectedItem = AddChoice(_models, model, true);
        }
        if (_launchOptions.LoraPath is { } lora)
        {
            if (!File.Exists(lora)) throw new FileNotFoundException("--lora のアダプターが見つかりません。", lora);
            AdapterComboBox.SelectedItem = AddChoice(_adapters, lora, true);
        }
        if (_launchOptions.TopP is { } p) TopPBox.Text = p;
        if (_launchOptions.TopK is { } k) TopKBox.Text = k;
        if (_launchOptions.Temperature is { } t) TemperatureBox.Text = t;
        if (_launchOptions.MaxTokens is { } max) MaxTokensBox.Text = max;
        ThinkingCheckBox.IsChecked = !_selectedModelIsGemma && _launchOptions.Think;
        StreamCheckBox.IsChecked = _launchOptions.Stream;
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
            RefreshMmprojChoices();
            UpdateSelectedModelBackend();
            if (_models.Count == 0)
                SetStatus("models フォルダーに GGUF がありません。参照からモデルを選択してください。");
        }
        finally { _refreshing = false; }
        if (_isLoaded && !SelectionMatchesLoaded())
            _ = ReleaseForSelectionChangeAsync();
        UpdateControls();
    }

    private void RefreshGpuChoices()
    {
        bool wasRefreshing = _refreshing;
        _refreshing = true;
        GpuChoice? selected = GpuComboBox.SelectedItem as GpuChoice;
        try
        {
            _gpus.Clear();
            IReadOnlyList<ArcDeviceInfo> available = ArcDevices.Enumerate();
            for (int first = 0; first < available.Count; first++)
                for (int second = first + 1; second < available.Count; second++)
                    _gpus.Add(new GpuChoice($"Arc {first} + Arc {second}（2台）", [first, second]));
            for (int index = 0; index < available.Count; index++)
                _gpus.Add(new GpuChoice($"Arc {index}（1台）", [index]));
            if (available.Count != 0)
                _gpus.Add(new GpuChoice("OpenCL GPU（自動 / Gemma 3）", [], IsGemmaOpenClAuto: true));
            _gpus.Add(new GpuChoice("CPU（Gemma 3）", []));
            GpuComboBox.SelectedItem = _selectedModelIsGemma
                ? _gpus.FirstOrDefault(choice => selected is { DeviceIndices.Length: 0 }
                    && choice.IsGemmaOpenClAuto == selected.IsGemmaOpenClAuto
                    && choice.DeviceIndices.Length == 0)
                    ?? _gpus.FirstOrDefault(choice => choice.IsGemmaOpenClAuto)
                    ?? _gpus.Last()
                : _gpus.FirstOrDefault(choice => choice.DeviceIndices.Length != 0
                    && selected is not null && choice.DeviceIndices.SequenceEqual(selected.DeviceIndices))
                    ?? _gpus.FirstOrDefault(choice => choice.DeviceIndices.Length != 0);
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
            if (!Path.GetFileName(path).StartsWith("lora_", StringComparison.OrdinalIgnoreCase)
                && !File.Exists(path + ".aria2")
                && !HuggingFaceModelDownload.IsMmproj(path))
                yield return path;
    }

    private static IEnumerable<string> EnumerateAdapterFiles(string directory)
    {
        IEnumerable<string> files;
        try
        {
            files = Directory.GetFiles(directory, "*.bin", SearchOption.AllDirectories)
                .Concat(Directory.GetFiles(directory, "lora_*.gguf", SearchOption.AllDirectories));
        }
        catch (IOException) { yield break; }
        catch (UnauthorizedAccessException) { yield break; }
        foreach (string path in files.Order(StringComparer.OrdinalIgnoreCase))
            if (!HuggingFaceModelDownload.IsMmproj(path)) yield return path;
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
        App.RejectParentTraversal(path);
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
        if (ReferenceEquals(sender, ModelComboBox)) UpdateSelectedModelBackend();
        CancelAudio();
        CancelPendingImagePreparation();
        if (ReferenceEquals(sender, MmprojComboBox) && _isLoaded && SelectionMatchesLoadedBase())
        {
            await ExecuteAsync(async ct =>
            {
                await EnsureLoadedAsync(_loadedModelPath!, _loadedAdapterPath, _loadedDevices!, ct);
                await PreparePendingImageAsync(ct);
            });
            return;
        }
        if (_isLoaded && !SelectionMatchesLoaded()) await ReleaseForSelectionChangeAsync();
        UpdateControls();
    }

    private void UpdateSelectedModelBackend()
    {
        bool wasGemma = _selectedModelIsGemma;
        string? path = (ModelComboBox.SelectedItem as FileChoice)?.FilePath;
        bool gemma = false;
        if (path is not null && File.Exists(path))
        {
            try
            {
                using var reader = new GgufReader(path);
                gemma = reader.Metadata.TryGetValue("general.architecture", out object? value)
                    && value is "gemma3";
            }
            catch (IOException) { }
            catch (InvalidDataException) { }
            catch (NotSupportedException) { }
        }
        _selectedModelIsGemma = gemma;
        bool wasRefreshing = _refreshing;
        _refreshing = true;
        try
        {
            if (gemma)
            {
                bool keepGemmaChoice = wasGemma && GpuComboBox.SelectedItem is GpuChoice
                    { DeviceIndices.Length: 0 };
                if (!keepGemmaChoice)
                    GpuComboBox.SelectedItem = _gpus.FirstOrDefault(choice => choice.IsGemmaOpenClAuto)
                        ?? _gpus.FirstOrDefault(choice => choice.DeviceIndices.Length == 0);
                if (_adapters.Count != 0) AdapterComboBox.SelectedItem = _adapters[0];
                if (_mmprojs.Count != 0) MmprojComboBox.SelectedItem = _mmprojs[0];
                ThinkingCheckBox.IsChecked = false;
                _pendingImage = null;
                UpdateImageAttachment();
            }
            else if (GpuComboBox.SelectedItem is GpuChoice { DeviceIndices.Length: 0 })
                GpuComboBox.SelectedItem = _gpus.FirstOrDefault(choice => choice.DeviceIndices.Length != 0);
        }
        finally { _refreshing = wasRefreshing; }
    }

    private bool SelectionMatchesLoaded() =>
        SelectionMatchesLoadedBase() &&
        string.Equals(_loadedMmprojPath, SelectedMmprojPath, StringComparison.OrdinalIgnoreCase);

    private bool SelectionMatchesLoadedBase() =>
        string.Equals(_loadedModelPath, (ModelComboBox.SelectedItem as FileChoice)?.FilePath,
            StringComparison.OrdinalIgnoreCase) &&
        string.Equals(_loadedAdapterPath, (AdapterComboBox.SelectedItem as FileChoice)?.FilePath,
            StringComparison.OrdinalIgnoreCase) &&
        _loadedDevices is { } loadedDevices &&
        GpuComboBox.SelectedItem is GpuChoice gpu && loadedDevices.SequenceEqual(gpu.DeviceIndices) &&
        string.Equals(_loadedGemmaBackend, _selectedModelIsGemma
            ? (gpu.IsGemmaOpenClAuto ? "opencl" : "cpu") : null, StringComparison.Ordinal);

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
            if (!_isLoaded || SelectionMatchesLoaded()) return;
            await UnloadServerAsync();
            SetStatus("選択が変わったため GPU 資源を解放しました。");
        });
    }

    private void BrowseModel_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        var dialog = new OpenFileDialog
        {
            Title = "GGUF モデルを選択",
            Filter = "GGUF モデル (*.gguf)|*.gguf",
            CheckFileExists = true,
            InitialDirectory = _modelsDirectory ?? Environment.CurrentDirectory
        };
        if (dialog.ShowDialog(this) == true)
        {
            App.RejectParentTraversal(dialog.FileName);
            if (Path.GetFileName(dialog.FileName).StartsWith("lora_", StringComparison.OrdinalIgnoreCase)
                || HuggingFaceModelDownload.IsMmproj(dialog.FileName))
            {
                SetStatus("LoRA や mmproj ではなく、ベースの GGUF モデルを選んでください。");
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
            Filter = "LoRA アダプター (*.bin;*.gguf)|*.bin;*.gguf|NNtrain チェックポイント (*.bin)|*.bin|GGUF LoRA (*.gguf)|*.gguf",
            CheckFileExists = true,
            InitialDirectory = _modelsDirectory ?? Environment.CurrentDirectory
        };
        if (dialog.ShowDialog(this) == true)
        {
            App.RejectParentTraversal(dialog.FileName);
            if (HuggingFaceModelDownload.IsMmproj(dialog.FileName))
            {
                SetStatus("mmproj ではなく LoRA アダプターを選んでください。");
                return;
            }
            AdapterComboBox.SelectedItem = AddChoice(_adapters, dialog.FileName, true);
        }
    }

    private void RefreshModels_Click(object sender, RoutedEventArgs e)
    {
        if (!_busy)
        {
            RefreshGpuChoices();
            RefreshChoices();
        }
    }

    private bool TrySelectedFiles(out string? modelPath, out string? adapterPath, out GpuChoice? gpu)
    {
        modelPath = (ModelComboBox.SelectedItem as FileChoice)?.FilePath;
        adapterPath = (AdapterComboBox.SelectedItem as FileChoice)?.FilePath;
        gpu = GpuComboBox.SelectedItem as GpuChoice;
        try
        {
            if (modelPath is not null) App.RejectParentTraversal(modelPath);
            if (adapterPath is not null) App.RejectParentTraversal(adapterPath);
        }
        catch (ArgumentException ex) { SetStatus(ex.Message); return false; }
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
        if (_selectedModelIsGemma && (adapterPath is not null || SelectedMmprojPath is not null))
        {
            SetStatus("Gemma 3 では現在 LoRA と画像 mmproj を使用できません。");
            return false;
        }
        if (gpu is null)
        {
            SetStatus("使用する計算デバイスを選択してください。");
            return false;
        }
        if (!_selectedModelIsGemma && gpu.DeviceIndices.Length == 0)
        {
            SetStatus("Qwen には Arc GPU を選択してください。");
            return false;
        }
        if (_selectedModelIsGemma && gpu.DeviceIndices.Length != 0)
        {
            SetStatus("Gemma 3 には OpenCL GPU（自動）または CPU を選択してください。");
            return false;
        }
        return true;
    }

    private async void Preload_Click(object sender, RoutedEventArgs e)
    {
        if (!TrySelectedFiles(out string? modelPath, out string? adapterPath, out GpuChoice? gpu)) return;
        await ExecuteAsync(async ct =>
        {
            await EnsureLoadedAsync(modelPath!, adapterPath, gpu!.DeviceIndices, ct, gpu.IsGemmaOpenClAuto);
            await PreparePendingImageAsync(ct);
        });
    }

    public sealed record FileChoice(string Label, string? FilePath, bool IsManual = false);
    public sealed record GpuChoice(string Label, int[] DeviceIndices, bool IsGemmaOpenClAuto = false);
}
