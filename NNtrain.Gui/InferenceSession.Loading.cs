using System.IO;
using NNtrain.Arc;
using NNtrain.Gemma;

namespace NNtrain.Gui;

public sealed partial class InferenceSession
{
    public async Task LoadAsync(string modelPath, string? adapterPath,
        IProgress<string>? progress, CancellationToken ct, IReadOnlyList<int>? selectedDevices = null,
        string? mmprojPath = null, string? gemmaBackend = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        modelPath = Path.GetFullPath(modelPath);
        adapterPath = string.IsNullOrWhiteSpace(adapterPath) ? null : Path.GetFullPath(adapterPath);
        mmprojPath = string.IsNullOrWhiteSpace(mmprojPath) ? null : Path.GetFullPath(mmprojPath);
        if (mmprojPath is not null && !File.Exists(mmprojPath))
            throw new FileNotFoundException("mmproj が見つかりません。", mmprojPath);
        if (!File.Exists(modelPath)) throw new FileNotFoundException("GGUF model not found.", modelPath);
        if (adapterPath is not null)
        {
            if (string.Equals(modelPath, adapterPath, StringComparison.OrdinalIgnoreCase))
                throw new ArgumentException("The model and LoRA adapter must be different files.", nameof(adapterPath));
            if (!File.Exists(adapterPath)) throw new FileNotFoundException("LoRA adapter not found.", adapterPath);
        }
        string architecture;
        using (var gguf = new GgufReader(modelPath))
            architecture = gguf.Metadata.TryGetValue("general.architecture", out object? value)
                ? value as string ?? string.Empty : string.Empty;
        if (architecture == "gemma3")
        {
            if (adapterPath is not null)
                throw new NotSupportedException("Gemma 3 の LoRA は現在サポートしていません。");
            if (mmprojPath is not null)
                throw new NotSupportedException("Gemma 3 の画像入力は現在サポートしていません。");
            gemmaBackend ??= "cpu";
            if (gemmaBackend is not ("cpu" or "opencl"))
                throw new ArgumentException("Gemma 3 backend must be cpu or opencl.", nameof(gemmaBackend));
            if (selectedDevices is { Count: > 0 })
                throw new NotSupportedException("Gemma 3 OpenCL のデバイス番号指定は未対応です。devices: [] と gemma_backend: opencl を指定してください。");
            ThrowIfDisposed();
            await _operation.WaitAsync(ct).ConfigureAwait(false);
            try
            {
                ThrowIfDisposed();
                if (_gemmaModel is not null && string.Equals(_loadedModelPath, modelPath,
                        StringComparison.OrdinalIgnoreCase)
                    && string.Equals(_loadedGemmaBackend, gemmaBackend, StringComparison.Ordinal)) return;
                await UnloadCoreAsync().ConfigureAwait(false);
                Gemma3LocalModel loaded = await Task.Run(() =>
                {
                    ct.ThrowIfCancellationRequested();
                    // The measured two-B580 profile balances the output layer's
                    // extra memory with a 53:47 split and keeps 43 layers resident
                    // at context 4096. Larger device sets retain automatic splitting.
                    IReadOnlyList<ArcDeviceInfo> arcDevices = gemmaBackend == "opencl" ? ArcDevices.Enumerate() : [];
                    bool dualB580 = gemmaBackend == "opencl" && arcDevices.Count(device =>
                        device.Name.Contains("B580", StringComparison.OrdinalIgnoreCase) &&
                        device.GlobalMemoryBytes >= 11UL * 1024 * 1024 * 1024) >= 2;
                    bool balancedPair = dualB580 && arcDevices.Count == 2;
                    string target = gemmaBackend == "opencl"
                        ? dualB580 ? "OpenCL GPU 2 台" : "OpenCL GPU" : "CPU";
                    progress?.Report($"Gemma 3 を {target} に読み込んでいます…");
                    return gemmaBackend == "opencl"
                        ? Gemma3LocalModel.Load(modelPath, gpuLayers: balancedPair ? 43 : dualB580 ? 40 : 20,
                            backend: Gemma3Backend.OpenCl, allOpenClDevices: dualB580,
                            tensorSplits: balancedPair ? [53f, 47f] : null)
                        : Gemma3LocalModel.Load(modelPath);
                }, ct).ConfigureAwait(false);
                if (ct.IsCancellationRequested)
                {
                    loaded.Dispose();
                    ct.ThrowIfCancellationRequested();
                }
                Volatile.Write(ref _gemmaModel, loaded);
                Volatile.Write(ref _loadedModelPath, modelPath);
                Volatile.Write(ref _loadedDevices, Array.Empty<int>());
                Volatile.Write(ref _loadedGemmaBackend, gemmaBackend);
                progress?.Report($"Gemma 3 を {(gemmaBackend == "opencl" ? "OpenCL GPU" : "CPU")} に読み込みました。");
                return;
            }
            finally { _operation.Release(); }
        }
        if (gemmaBackend is not null)
            throw new NotSupportedException("gemma_backend は Gemma 3 GGUF 専用です。");
        int[] devices = selectedDevices?.ToArray()
            ?? Enumerable.Range(0, Math.Min(2, ArcDevices.Enumerate().Count)).ToArray();
        if (devices.Length == 0)
            throw new NotSupportedException("推論には Intel Arc GPU が必要です。");

        ThrowIfDisposed();
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            ct.ThrowIfCancellationRequested();
            if (_model is not null
                && string.Equals(_loadedModelPath, modelPath, StringComparison.OrdinalIgnoreCase)
                && string.Equals(_loadedAdapterPath, adapterPath, StringComparison.OrdinalIgnoreCase)
                && _loadedDevices is not null && _loadedDevices.SequenceEqual(devices))
            {
                // Replacing a vision tower does not change the text model or its
                // adapter. Keep the large resident language weights in place.
                if (!string.Equals(_loadedMmprojPath, mmprojPath, StringComparison.OrdinalIgnoreCase))
                    await ReplaceVisionAsync(mmprojPath, _model, devices[0], progress, ct).ConfigureAwait(false);
                return;
            }

            // The model cannot detach an adapter. Releasing it first also avoids
            // temporarily keeping two copies of the 27B weights in Arc VRAM.
            await UnloadCoreAsync().ConfigureAwait(false);
            var loaded = await Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();
                progress?.Report($"Arc {string.Join(",", devices)} にモデルを分散して読み込みます…");
                Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(modelPath);
                Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(modelPath,
                    devices: devices,
                    progress: progress is null ? null : progress.Report,
                    options: _options.CreateQwenExecutionOptions(devices.Length, adapterPath is not null));
                try
                {
                    ct.ThrowIfCancellationRequested();
                    if (adapterPath is not null) model.LoadLora(adapterPath);
                    ct.ThrowIfCancellationRequested();
                    return (Model: model, Tokenizer: tokenizer, Devices: devices);
                }
                catch
                {
                    model.Dispose();
                    throw;
                }
            }).ConfigureAwait(false);
            if (ct.IsCancellationRequested)
            {
                loaded.Model.Dispose();
                ct.ThrowIfCancellationRequested();
            }
            // Commit the usable text model after adapter validation. A bad or
            // oversized vision tower must not discard these resident weights.
            Volatile.Write(ref _tokenizer, loaded.Tokenizer);
            Volatile.Write(ref _loadedModelPath, modelPath);
            Volatile.Write(ref _loadedAdapterPath, adapterPath);
            Volatile.Write(ref _loadedDevices, loaded.Devices);
            Volatile.Write(ref _model, loaded.Model);
            await ReplaceVisionAsync(mmprojPath, loaded.Model, devices[0], progress, ct).ConfigureAwait(false);
        }
        finally { _operation.Release(); }
    }

}
