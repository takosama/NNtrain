using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Net.Http;
using System.Windows;

namespace NNtrain.Gui;

public partial class MainWindow
{
    private readonly HttpClient _hfClient = new() { Timeout = Timeout.InfiniteTimeSpan };
    private readonly ObservableCollection<HfFileChoice> _hfFiles = [];
    private HuggingFaceModelDownload? _hfDownload;
    private HuggingFaceRepository? _hfRepository;
    private HuggingFaceRepository? _hfMmprojRepository;
    private CancellationTokenSource? _hfOperationCancellation;
    private bool _hfBusy;

    private void InitializeHuggingFaceTab()
    {
        _hfDownload = new HuggingFaceModelDownload(_hfClient);
        HfFilesList.ItemsSource = _hfFiles;
        HfDestinationText.Text = $"保存先: {GetModelsDownloadDirectory()}\\huggingface\\owner\\repository";
        UpdateHfControls();
    }

    private string GetModelsDownloadDirectory() =>
        _modelsDirectory ?? FindModelsDirectory() ?? Path.Combine(AppContext.BaseDirectory, "models");

    private async void HfInspect_Click(object sender, RoutedEventArgs e)
    {
        if (_hfBusy || _hfDownload is null) return;
        string input = HfRepositoryBox.Text;
        string token = HfTokenBox.Password;
        _hfRepository = null;
        foreach (HfFileChoice choice in _hfFiles.Where(choice => !choice.IsAdditionalRepository).ToArray())
            _hfFiles.Remove(choice);
        HfDownloadProgressBar.Value = 0;
        HfDownloadBytesText.Text = "";
        HfRepositoryStatusText.Text = "リポジトリのファイル一覧を取得中…";
        _hfBusy = true;
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        _hfOperationCancellation = cancellation;
        UpdateHfControls();
        try
        {
            HuggingFaceRepository repository = await _hfDownload.GetRepositoryAsync(input, token, cancellation.Token);
            _hfRepository = repository;
            foreach (HuggingFaceArtifact file in repository.Files)
            {
                HfFileChoice? existing = _hfFiles.FirstOrDefault(choice =>
                    choice.Repository.Id == repository.Id && choice.Artifact.Path == file.Path);
                if (existing is { IsAdditionalRepository: true }) _hfFiles.Remove(existing);
                if (existing is null or { IsAdditionalRepository: true })
                    _hfFiles.Add(new HfFileChoice(repository, file, false));
            }
            HfRepositoryStatusText.Text = repository.Files.Count == 0
                ? $"{repository.Id}: GGUF/mmproj は見つかりませんでした。"
                : $"{repository.Id} · {repository.Files.Count} 件 · コミット {repository.Commit[..12]}";
            HfDestinationText.Text = $"保存先: {GetModelsDownloadDirectory()}\\huggingface\\owner\\repository";
            HfDownloadStatusText.Text = "取得するファイルをチェックしてください。";
        }
        catch (OperationCanceledException)
        {
            HfRepositoryStatusText.Text = cancellation.IsCancellationRequested
                ? "一覧取得を中止しました（またはタイムアウトしました）。"
                : "一覧取得を中止しました。";
        }
        catch (Exception ex) { HfRepositoryStatusText.Text = $"一覧取得に失敗: {ex.Message}"; }
        finally
        {
            _hfOperationCancellation = null;
            _hfBusy = false;
            UpdateHfControls();
        }
    }

    private async void HfInspectMmproj_Click(object sender, RoutedEventArgs e)
    {
        if (_hfBusy || _hfDownload is null) return;
        string input = HfMmprojRepositoryBox.Text;
        string token = HfTokenBox.Password;
        _hfMmprojRepository = null;
        foreach (HfFileChoice choice in _hfFiles.Where(choice => choice.IsAdditionalRepository).ToArray())
            _hfFiles.Remove(choice);
        HfMmprojRepositoryStatusText.Text = "別リポジトリの mmproj を検索中…";
        _hfBusy = true;
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(45));
        _hfOperationCancellation = cancellation;
        UpdateHfControls();
        try
        {
            HuggingFaceRepository repository = await _hfDownload.GetRepositoryAsync(input, token, cancellation.Token);
            _hfMmprojRepository = repository;
            HuggingFaceArtifact[] mmprojs = repository.Files
                .Where(file => file.Kind == HuggingFaceArtifactKind.Mmproj).ToArray();
            foreach (HuggingFaceArtifact file in mmprojs)
                if (!_hfFiles.Any(choice => choice.Repository.Id == repository.Id && choice.Artifact.Path == file.Path))
                    _hfFiles.Add(new HfFileChoice(repository, file, true));
            HfMmprojRepositoryStatusText.Text = mmprojs.Length == 0
                ? $"{repository.Id}: mmproj は見つかりませんでした。"
                : $"{repository.Id}: mmproj {mmprojs.Length} 件 · コミット {repository.Commit[..12]}";
        }
        catch (OperationCanceledException)
        {
            HfMmprojRepositoryStatusText.Text = "mmproj 一覧取得を中止しました（またはタイムアウトしました）。";
        }
        catch (Exception ex) { HfMmprojRepositoryStatusText.Text = $"mmproj 一覧取得に失敗: {ex.Message}"; }
        finally
        {
            _hfOperationCancellation = null;
            _hfBusy = false;
            UpdateHfControls();
        }
    }

    private async void HfDownload_Click(object sender, RoutedEventArgs e)
    {
        if (_hfBusy || _hfDownload is null) return;
        HfFileChoice[] selected = _hfFiles.Where(file => file.IsChecked).ToArray();
        if (selected.Length == 0)
        {
            HfDownloadStatusText.Text = "GGUF または mmproj を1件以上チェックしてください。";
            return;
        }
        string modelsDirectory = GetModelsDownloadDirectory();
        string token = HfTokenBox.Password;
        var cancellation = new CancellationTokenSource();
        _hfOperationCancellation = cancellation;
        _hfBusy = true;
        UpdateHfControls();
        int completed = 0;
        var progress = new Progress<HuggingFaceDownloadProgress>(value =>
        {
            if (_closingFinished) return;
            HfDownloadProgressBar.IsIndeterminate = value.Total is null;
            if (value.Total is long total && total > 0)
                HfDownloadProgressBar.Value = 100d * value.Bytes / total;
            HfDownloadBytesText.Text = value.Total is long length
                ? $"{FormatBytes(value.Bytes)} / {FormatBytes(length)}"
                : FormatBytes(value.Bytes);
        });
        try
        {
            foreach (HfFileChoice file in selected)
            {
                cancellation.Token.ThrowIfCancellationRequested();
                HfDownloadProgressBar.IsIndeterminate = true;
                HfDownloadProgressBar.Value = 0;
                HfDownloadBytesText.Text = "";
                HfDownloadStatusText.Text = $"{completed + 1}/{selected.Length} を取得中: {file.Path}";
                await _hfDownload.DownloadAsync(file.Repository, file.Artifact, modelsDirectory,
                    token, progress, cancellation.Token);
                completed++;
                file.IsChecked = false;
            }
            HfDownloadStatusText.Text = $"{completed} 件を models フォルダーに保存しました。チャットタブのモデル一覧に反映しました。";
        }
        catch (OperationCanceledException)
        {
            HfDownloadStatusText.Text = $"中止しました。保存済み {completed}/{selected.Length} 件。一時ファイルは再試行時に続きから取得します。";
        }
        catch (Exception ex)
        {
            HfDownloadStatusText.Text = $"取得に失敗（保存済み {completed}/{selected.Length} 件）: {ex.Message}";
        }
        finally
        {
            HfDownloadProgressBar.IsIndeterminate = false;
            if (completed > 0) RefreshChoices();
            cancellation.Dispose();
            _hfOperationCancellation = null;
            _hfBusy = false;
            UpdateHfControls();
        }
    }

    private void HfCancel_Click(object sender, RoutedEventArgs e)
    {
        _hfOperationCancellation?.Cancel();
        HfDownloadStatusText.Text = "キャンセル中…";
    }

    private void HfOpenModels_Click(object sender, RoutedEventArgs e)
    {
        try
        {
            string path = GetModelsDownloadDirectory();
            Directory.CreateDirectory(path);
            Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
        }
        catch (Exception ex) { HfDownloadStatusText.Text = $"フォルダーを開けませんでした: {ex.Message}"; }
    }

    private void UpdateHfControls()
    {
        HfInspectButton.IsEnabled = !_hfBusy;
        HfInspectMmprojButton.IsEnabled = !_hfBusy;
        HfRepositoryBox.IsEnabled = !_hfBusy;
        HfMmprojRepositoryBox.IsEnabled = !_hfBusy;
        HfTokenBox.IsEnabled = !_hfBusy;
        HfFilesList.IsEnabled = !_hfBusy;
        HfDownloadButton.IsEnabled = !_hfBusy && _hfFiles.Count > 0;
        HfCancelButton.IsEnabled = _hfBusy;
    }

    private static string FormatBytes(long bytes) => bytes >= 1024L * 1024 * 1024
        ? $"{bytes / (1024d * 1024 * 1024):F2} GiB"
        : bytes >= 1024L * 1024 ? $"{bytes / (1024d * 1024):F1} MiB"
        : $"{bytes / 1024d:F1} KiB";

    private sealed class HfFileChoice(HuggingFaceRepository repository,
        HuggingFaceArtifact artifact, bool isAdditionalRepository) : INotifyPropertyChanged
    {
        private bool _isChecked;
        public HuggingFaceRepository Repository { get; } = repository;
        public HuggingFaceArtifact Artifact { get; } = artifact;
        public bool IsAdditionalRepository { get; } = isAdditionalRepository;
        public string Path => $"{Repository.Id} / {Artifact.Path}";
        public string Detail => $"{(Artifact.Kind == HuggingFaceArtifactKind.Mmproj ? "mmproj" : "GGUF モデル")} · "
            + (Artifact.Size is long size ? FormatBytes(size) : "サイズ不明")
            + (Artifact.Sha256 is null ? "" : " · SHA-256 検証対応");
        public bool IsChecked
        {
            get => _isChecked;
            set
            {
                if (_isChecked == value) return;
                _isChecked = value;
                PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(IsChecked)));
            }
        }
        public event PropertyChangedEventHandler? PropertyChanged;
    }
}
