using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using Microsoft.Win32;

namespace NNtrain.Gui;

public partial class MainWindow
{
    private readonly ObservableCollection<FileChoice> _mmprojs = [];
    private ChatImage? _pendingImage;
    private string? _loadedMmprojPath;
    private CancellationTokenSource? _imagePreparationCancellation;
    private Task? _imagePreparationTask;
    private string? _preparingImageHash;
    private string? _preparedImageHash;
    private string? SelectedMmprojPath => (MmprojComboBox.SelectedItem as FileChoice)?.FilePath;

    private void RefreshMmprojChoices()
    {
        MmprojComboBox.ItemsSource = _mmprojs;
        string? selected = SelectedMmprojPath;
        FileChoice[] manual = _mmprojs.Where(item => item.IsManual).ToArray();
        _mmprojs.Clear();
        _mmprojs.Add(new FileChoice("なし（テキストのみ）", null));
        if (_modelsDirectory is not null && Directory.Exists(_modelsDirectory))
            foreach (string path in Directory.EnumerateFiles(_modelsDirectory, "*.gguf", SearchOption.AllDirectories)
                .Where(HuggingFaceModelDownload.IsMmproj).Order(StringComparer.OrdinalIgnoreCase))
                AddChoice(_mmprojs, path, false);
        foreach (FileChoice choice in manual) AddChoice(_mmprojs, choice.FilePath!, true);
        MmprojComboBox.SelectedItem = FindChoice(_mmprojs, selected) ?? _mmprojs[0];
    }

    private void BrowseMmproj_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Title = "対応する mmproj GGUF を選択", Filter = "mmproj GGUF|*.gguf" };
        if (dialog.ShowDialog(this) == true)
            MmprojComboBox.SelectedItem = AddChoice(_mmprojs, Path.GetFullPath(dialog.FileName), true);
    }

    private async void AttachImage_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFileDialog { Title = "画像を添付", Filter = "PNG / JPEG|*.png;*.jpg;*.jpeg" };
        if (dialog.ShowDialog(this) != true) return;
        try
        {
            if (new FileInfo(dialog.FileName).Length > ChatImage.MaximumBytes)
                throw new ArgumentException("画像は 16 MiB 以下にしてください。");
            CancelPendingImagePreparation();
            _pendingImage = new ChatImage(File.ReadAllBytes(dialog.FileName), Path.GetFileName(dialog.FileName));
            UpdateImageAttachment();
            await PreparePendingImageAsync();
        }
        catch (Exception ex) { SetStatus($"画像を添付できません: {ex.Message}"); }
    }

    private void RemoveImage_Click(object sender, RoutedEventArgs e)
    {
        CancelPendingImagePreparation();
        _pendingImage = null;
        UpdateImageAttachment();
    }

    private void UpdateImageAttachment()
    {
        string preparation = _pendingImage?.Hash == _preparingImageHash && _preparingImageHash is not null
            ? " · 画像を準備中…"
            : _pendingImage?.Hash == _preparedImageHash && _preparedImageHash is not null
                ? " · 画像準備済み" : "";
        ImageAttachmentText.Text = _pendingImage is null ? "画像なし" : $"添付: {_pendingImage.Name}{preparation}";
        RemoveImageButton.IsEnabled = !_busy && _pendingImage is not null;
    }

    private void NewConversation_Click(object sender, RoutedEventArgs e)
    {
        if (_busy) return;
        CancelPendingImagePreparation();
        _conversation.Clear();
        _messages.Clear();
        _pendingImage = null;
        UpdateImageAttachment();
        SetStatus("新しい会話を開始しました。");
    }

    private void CancelPendingImagePreparation()
    {
        _imagePreparationCancellation?.Cancel();
        _preparingImageHash = null;
        _preparedImageHash = null;
    }

    private Task PreparePendingImageAsync(CancellationToken ct = default)
    {
        if (_pendingImage is not { } image || !_serverReady || !_isLoaded
            || !SelectionMatchesLoaded() || SelectedMmprojPath is null)
            return Task.CompletedTask;
        if (_preparingImageHash == image.Hash && _imagePreparationTask is { IsCompleted: false } running)
            return running.WaitAsync(ct);
        return _imagePreparationTask = PreparePendingImageCoreAsync(image, ct);
    }

    private async Task WaitForPendingImagePreparationAsync(ChatImage? image, CancellationToken ct)
    {
        // Sending consumes the same attachment. Let its serialized preparation
        // finish instead of cancelling work that generation can reuse.
        if (image is not null && _preparingImageHash == image.Hash && _imagePreparationTask is { } preparation)
            await preparation.WaitAsync(ct);
    }

    private async Task PreparePendingImageCoreAsync(ChatImage image, CancellationToken ct)
    {
        CancelPendingImagePreparation();
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(ct);
        _imagePreparationCancellation = cancellation;
        _preparingImageHash = image.Hash;
        UpdateImageAttachment();
        try
        {
            // This uses the existing NNtrain child and its serialized GPU gate.
            // Prime only known history and projected image rows. Text being
            // typed in the input box is deliberately absent from this prefix.
            ChatTurn[] preparation = _conversation.Append(new ChatTurn("user", "", Image: image)).ToArray();
            await PostAsync("internal/prepare-image", new
            {
                model = _loadedModelPath,
                lora = _loadedAdapterPath,
                mmproj = SelectedMmprojPath,
                devices = _loadedDevices,
                messages = preparation.Select(turn => new
                {
                    role = turn.Role,
                    content = MessageContent(turn),
                    assistant_prefix = turn.AssistantPrefix
                }).ToArray()
            }, cancellation.Token);
            if (ReferenceEquals(_imagePreparationCancellation, cancellation)
                && _pendingImage?.Hash == image.Hash)
                _preparedImageHash = image.Hash;
        }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested) { }
        catch (Exception ex)
        {
            if (!_busy && ReferenceEquals(_imagePreparationCancellation, cancellation)
                && _pendingImage?.Hash == image.Hash)
                SetStatus($"画像の事前処理に失敗しました。送信時に再試行します: {ex.Message}");
        }
        finally
        {
            if (ReferenceEquals(_imagePreparationCancellation, cancellation))
            {
                _imagePreparationCancellation = null;
                _preparingImageHash = null;
                UpdateImageAttachment();
            }
        }
    }

    private static object MessageContent(ChatTurn turn) => turn.Image is null ? turn.Content
        : new object[]
        {
            new { type = "image_url", image_url = new { url = turn.Image.ToDataUrl() } },
            new { type = "text", text = turn.Content }
        };
}
