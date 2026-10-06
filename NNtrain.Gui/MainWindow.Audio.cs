using System.IO;
using System.Threading.Channels;
using System.Windows;
using System.Windows.Controls;
using Microsoft.Win32;
using NNtrain.Audio;

namespace NNtrain.Gui;

public partial class MainWindow
{
    private ILocalAsrModel? _asrModel;
    private sealed record AsrModelChoice(string Label, string Folder, bool Parakeet);
    private CancellationTokenSource? _asrCancellation;
    private Task? _asrTask;
    private PcmMicrophone? _microphone;
    private long _asrEpoch;
    private long _asrCompletedEpoch = -1;
    private bool _asrActive;
    private bool _asrRecognizing;
    private long _asrProcessedSamples;
    private AudioChatDraft? _audioChatDraft;

    private void InitializeAudio()
    {
        AsrDirectoryBox.Text = Path.Combine(FindModelsDirectory() ?? Path.Combine(AppContext.BaseDirectory, "models"), "nemotron-3.5-asr-streaming-0.6b");
        GpuChoice[] devices = _gpus.Where(x => x.DeviceIndices.Length <= 1).ToArray();
        AsrGpuComboBox.ItemsSource = devices;
        AsrGpuComboBox.SelectedItem = devices.FirstOrDefault(x => x.DeviceIndices.Length == 1) ?? devices.FirstOrDefault();
        AsrModelComboBox.ItemsSource = new[]
        {
            new AsrModelChoice("Nemotron 3.5 ASR（日本語・逐次認識）", "nemotron-3.5-asr-streaming-0.6b", false),
            new AsrModelChoice("Parakeet（日本語・CTC）", "parakeet-tdt_ctc-0.6b-ja", true)
        };
        AsrModelComboBox.SelectedIndex = 0;
        UpdateAudioControls();
    }
    private void AsrBrowse_Click(object sender, RoutedEventArgs e)
    {
        var dialog = new OpenFolderDialog { Title = "音声モデルのフォルダー" };
        if (dialog.ShowDialog(this) == true) AsrDirectoryBox.Text = dialog.FolderName;
    }
    private async void AsrLoad_Click(object sender, RoutedEventArgs e)
    {
        if (_asrActive) return;
        string directory = AsrDirectoryBox.Text;
        bool parakeet = (AsrModelComboBox.SelectedItem as AsrModelChoice)?.Parakeet == true;
        int? device = (AsrGpuComboBox.SelectedItem as GpuChoice)?.DeviceIndices.FirstOrDefault(-1);
        _asrModel?.Dispose();
        _asrModel = null;
        await RunAudioAsync(async (epoch, ct) =>
        {
            AsrStatusText.Text = "FP16重みを読み込み中…";
            var model = await Task.Run(() =>
            {
                if (parakeet)
                {
                    string archive = Path.Combine(directory, "parakeet-tdt_ctc-0.6b-ja.nemo");
                    if (File.Exists(archive) && (!File.Exists(Path.Combine(directory, "model_config.yaml")) ||
                        !File.Exists(Path.Combine(directory, "tokenizer.vocab"))))
                        NemoCheckpointConverter.ExtractModelAssets(archive, directory, ct);
                    if (!File.Exists(Path.Combine(directory, "model.safetensors")))
                        NemoCheckpointConverter.ConvertToFp16(archive, Path.Combine(directory, "model.safetensors"), ct);
                }
                ILocalAsrModel loaded = parakeet ? ParakeetCtcModel.Load(directory, ct) : NemotronAsrModel.Load(directory, ct);
                try { if (device is >= 0) loaded.EnableArc(device.Value, ct); return loaded; }
                catch { loaded.Dispose(); throw; }
            }, ct);
            if (epoch != _asrEpoch) { model.Dispose(); return; }
            _asrModel = model;
            AsrStatusText.Text = $"FP16重み {model.HostWeightBytes / (1024d * 1024):N1} MiB (RAM)。{model.ExecutionDevice}。総VRAMは未測定。";
        });
    }
    private async void AsrFile_Click(object sender, RoutedEventArgs e)
    {
        if (_asrActive || _asrModel is null || _busy) return;
        var dialog = new OpenFileDialog { Title = "16bit PCM WAVを認識", Filter = "16bit PCM WAV|*.wav" };
        if (dialog.ShowDialog(this) != true) return;
        var model = _asrModel;
        await RunAudioAsync((epoch, ct) =>
        {
            BeginAudioChatInput();
            return Task.Run(() =>
            {
                using var file = File.OpenRead(dialog.FileName);
                float[] samples = Pcm16Wave.Read(file).To16Khz(ct);
                var stream = model.CreateStream();
                for (int offset = 0; offset < samples.Length; offset += 5120)
                {
                    ct.ThrowIfCancellationRequested();
                    stream.Append(samples.AsSpan(offset, Math.Min(5120, samples.Length - offset)), partial: text => ShowAudioPartial(epoch, text), ct: ct);
                }
                CompleteAudio(epoch, stream.Append([], final: true, ct: ct), stream.CacheBytes);
            }, ct);
        });
    }
    private async void AsrRecord_Click(object sender, RoutedEventArgs e)
    {
        if (_asrActive || _asrModel is null || _busy) return;
        await RunAudioAsync(async (epoch, ct) =>
        {
            // Sole microphone-start path: explicit Record click after model loading.
            using var microphone = new PcmMicrophone();
            _microphone = microphone;
            UpdateAudioControls();
            await ProcessAudioAsync(microphone.Audio, epoch, ct);
        });
    }
    // Shared by microphone capture and paced file-input integration tests.
    // Each invocation creates a fresh utterance; cache state never crosses restarts.
    private Task ProcessAudioAsync(ChannelReader<float[]> audio, long epoch, CancellationToken ct)
    {
        var model = _asrModel ?? throw new InvalidOperationException("ASR model is not loaded.");
        BeginAudioChatInput(); _asrProcessedSamples = 0;
        AsrStatusText.Text = "録音中：認識文を逐次更新します。停止で確定、キャンセルで破棄します。";
        if (_microphone is { } mic) AsrStatusText.Text += $" PCM16 {mic.CaptureSampleRate} Hz / {mic.CaptureChannels} ch -> 16000 Hz / mono";
        return Task.Run(async () =>
        {
            var stream = model.CreateStream();
            long samples = 0;
            await foreach (float[] chunk in audio.ReadAllAsync(ct))
            {
                samples += chunk.Length;
                if (samples > 16000L * 300) throw new InvalidOperationException("録音は5分までです。");
                stream.Append(chunk, partial: text => ShowAudioPartial(epoch, text), ct: ct);
                Interlocked.Exchange(ref _asrProcessedSamples, samples);
            }
            CompleteAudio(epoch, stream.Append([], true, ct: ct), stream.CacheBytes);
        }, ct);
    }
    private async Task RunAudioAsync(Func<long, CancellationToken, Task> operation)
    {
        if (_asrActive) return;
        _asrActive = true;
        long epoch = ++_asrEpoch;
        using var cancellation = new CancellationTokenSource();
        _asrCancellation = cancellation;
        AsrStatusText.Text = "音声処理中…";
        UpdateControls();
        try { _asrTask = operation(epoch, cancellation.Token); await _asrTask; }
        catch (OperationCanceledException) when (cancellation.IsCancellationRequested) { AsrStatusText.Text = "音声処理をキャンセルしました。"; }
        catch (Exception ex) { AsrStatusText.Text = $"音声処理エラー: {ex.Message}"; }
        finally
        {
            _microphone?.Dispose(); _microphone = null;
            _asrCompletedEpoch = epoch;
            _asrCancellation = null; _asrTask = null; _asrActive = false; _asrRecognizing = false;
            UpdateControls();
        }
    }
    private void ShowAudioPartial(long epoch, string text) => Dispatcher.BeginInvoke(new Action(() =>
    {
        if (epoch != _asrEpoch || epoch == _asrCompletedEpoch || _closingFinished) return;
        ApplyAudioChatText(text);
    }), System.Windows.Threading.DispatcherPriority.Background);
    private void CompleteAudio(long epoch, string text, long cacheBytes) => Dispatcher.BeginInvoke(new Action(() =>
    {
        if (epoch != _asrEpoch || _closingFinished) return;
        _asrCompletedEpoch = epoch;
        ApplyAudioChatText(text);
        _audioChatDraft = null;
        AsrStatusText.Text = $"認識完了。CPUキャッシュ {cacheBytes / 1024d:N1} KiB、Arcバッファ最大 {(_asrModel?.PeakDeviceBufferBytes ?? 0) / (1024d * 1024):N1} MiB。ドライバー領域を含む総VRAMは未測定。";
        UpdateAudioControls();
    }));
    private void AsrStop_Click(object sender, RoutedEventArgs e)
    {
        try { _microphone?.Stop(); AsrStatusText.Text = "録音を停止。残りの音声を確定中…"; AsrStopButton.IsEnabled = false; }
        catch (Exception ex) { CancelAudio(); AsrStatusText.Text = ex.Message; }
    }
    private void AsrCancel_Click(object sender, RoutedEventArgs e) => CancelAudio();
    private async void AsrModel_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (AsrModelComboBox.SelectedItem is not AsrModelChoice choice) return;
        CancelAudio();
        long epoch = _asrEpoch;
        if (_asrTask is { } task) { try { await task; } catch { } }
        if (epoch != _asrEpoch || _closingStarted) return;
        _asrModel?.Dispose(); _asrModel = null;
        AsrDirectoryBox.Text = Path.Combine(FindModelsDirectory() ?? Path.Combine(AppContext.BaseDirectory, "models"), choice.Folder);
        AsrSettingsExpander.Header = choice.Label;
        AsrStatusText.Text = choice.Parakeet
            ? "Parakeet日本語CTC：4秒ごとに発話を再認識するため途中の文が変わる場合があります。1発話は30秒まで。FP16を読み込んでください。"
            : "Nemotron日本語：キャッシュ付き逐次認識。FP16を読み込んでください。";
        UpdateAudioControls();
    }
    private async void AsrGpu_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        CancelAudio();
        long epoch = _asrEpoch;
        if (_asrTask is { } task) { try { await task; } catch { } }
        if (epoch != _asrEpoch || _closingStarted) return;
        _asrModel?.Dispose(); _asrModel = null;
        AsrStatusText.Text = "音声デバイスを変更しました。FP16重みを読み込んでください。";
        UpdateAudioControls();
    }
    private void CancelAudio()
    {
        ++_asrEpoch;
        if (_asrRecognizing && _audioChatDraft is { } draft)
        {
            MessageBox.Text = draft.Original;
            MessageBox.Select(draft.SelectionStart, draft.SelectionLength);
        }
        _audioChatDraft = null;
        _asrCancellation?.Cancel(); try { _microphone?.Stop(); } catch { }
    }
    private void BeginAudioChatInput()
    {
        _audioChatDraft = new(MessageBox.Text, MessageBox.SelectionStart, MessageBox.SelectionLength);
        _asrRecognizing = true;
        UpdateAudioControls();
    }
    private void ApplyAudioChatText(string text)
    {
        if (_audioChatDraft is not { } draft) return;
        MessageBox.Text = draft.Compose(text);
    }
    private void UpdateAudioControls()
    {
        if (AsrLoadButton is null) return;
        AsrLoadButton.IsEnabled = !_asrActive && !_busy;
        AsrBrowseButton.IsEnabled = !_asrActive;
        AsrDirectoryBox.IsEnabled = !_asrActive;
        AsrGpuComboBox.IsEnabled = !_asrActive && !_busy;
        AsrModelComboBox.IsEnabled = !_asrActive && !_busy;
        AsrFileButton.IsEnabled = !_asrActive && !_busy && _asrModel is not null;
        AsrRecordButton.IsEnabled = AsrFileButton.IsEnabled;
        AsrStopButton.IsEnabled = _asrActive && _microphone is not null;
        AsrCancelButton.IsEnabled = _asrActive;
        MessageBox.IsReadOnly = _asrRecognizing;
    }
}
