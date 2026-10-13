using System.Diagnostics;
using System.IO;
using NNtrain.Arc;

namespace NNtrain.Gui;

public sealed partial class InferenceSession
{
    /// <summary>
    /// Prepares an attached image while the user writes the message. It only
    /// warms the image cache. The overload with conversation history also
    /// explicitly primes the language state through the final image row.
    /// </summary>
    public Task<ImagePreparationStats> PrepareImageAsync(ChatImage image, CancellationToken ct)
        => PrepareImageAsync(image, null, ct);

    public async Task<ImagePreparationStats> PrepareImageAsync(ChatImage image,
        IReadOnlyList<ChatTurn>? conversation, CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (_gemmaModel is not null)
            throw new NotSupportedException("Gemma 3 の画像入力は現在サポートしていません。");
        ChatTurn[]? turns = conversation?.ToArray();
        if (turns is not null && (turns.Length == 0 || turns[^1].Role != "user"
            || turns[^1].Content.Length != 0 || turns[^1].Image?.Hash != image.Hash))
            throw new ArgumentException("Preparation history must end with this image and an empty user message.", nameof(conversation));
        ThrowIfDisposed();
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            Qwen35QuantizedModel model = _model ?? throw new InvalidOperationException("Load a model before preparing an image.");
            LastImagePreparationMilliseconds = 0;
            return await Task.Run(() =>
            {
                bool cached = _imageEmbeddings.ContainsKey(image.Hash);
                Qwen35VisionEmbedding encoded = EncodeImageCore(image, model, ct);
                if (turns is null) return new ImagePreparationStats(encoded.GridWidth, encoded.GridHeight, cached);
                var prefixTimer = Stopwatch.StartNew();
                try
                {
                    MultimodalPrompt prefix = MultimodalPrompt.BuildAttachmentPrefix(turns,
                        text => _tokenizer!.Encode(text, model.Descriptor.ContextLength, ct), earlierImage => EncodeImageCore(earlierImage, model, ct));
                    var primed = model.PrimePromptWithEmbeddings(prefix.Tokens, ct);
                    return new ImagePreparationStats(encoded.GridWidth, encoded.GridHeight, cached,
                        prefix.Tokens.Count, primed.ReusedTokens, primed.Cached, prefixTimer.Elapsed.TotalMilliseconds);
                }
                catch
                {
                    RecoverGenerationFailure(model);
                    throw;
                }
            }, ct).ConfigureAwait(false);
        }
        finally { _operation.Release(); }
    }

    private Qwen35VisionEmbedding EncodeImageCore(ChatImage image,
        Qwen35QuantizedModel model, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        if (_loadedMmprojPath is null)
            throw new InvalidOperationException("画像を使うには対応する mmproj を選択してください。");
        if (_imageEmbeddings.TryGetValue(image.Hash, out var cached)) return cached;
        _vision ??= LoadVisionEncoder(_loadedMmprojPath, _loadedDevices![0],
            model, ct);
        var timer = Stopwatch.StartNew();
        Qwen35VisionInput input = NativeImageDecoder.Prepare(image,
            _vision.Descriptor.PatchSize, _vision.Descriptor.MergeSize);
        Qwen35VisionEmbedding encoded = _vision.Encode(input, ct);
        LastImagePreparationMilliseconds += timer.Elapsed.TotalMilliseconds;
        if (_imageEmbeddings.Count >= 8) _imageEmbeddings.Clear();
        _imageEmbeddings[image.Hash] = encoded;
        return encoded;
    }

    private async Task ReplaceVisionAsync(string? path, Qwen35QuantizedModel model,
        int device, IProgress<string>? progress, CancellationToken ct)
    {
        // Validate both geometry and the combined GPU footprint before releasing
        // the current tower. Invalid selections leave the current pairing usable.
        if (path is not null)
            await Task.Run(() => ValidateVisionMemoryBudget(path, device, model, ct), ct).ConfigureAwait(false);
        ct.ThrowIfCancellationRequested();
        model.Reset();
        _imageEmbeddings.Clear();
        _vision?.Dispose();
        _vision = null;
        model.SetExternalDeviceMemoryReservation(0, 0);
        VisionInitializationMilliseconds = 0;
        Volatile.Write(ref _loadedMmprojPath, null);
        if (path is not null && _options.PrepareVisionOnLoad)
        {
            progress?.Report("画像モデルをプリロードしています…");
            _vision = await Task.Run(() => LoadVisionEncoder(path, device, model, ct), ct).ConfigureAwait(false);
        }
        Volatile.Write(ref _loadedMmprojPath, path);
    }

    private Qwen35VisionEncoder LoadVisionEncoder(string path, int device,
        Qwen35QuantizedModel model, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        var timer = Stopwatch.StartNew();
        long reservation = ValidateVisionMemoryBudget(path, device, model, ct);
        model.SetExternalDeviceMemoryReservation(0, reservation);
        Qwen35VisionEncoder vision;
        try
        {
            vision = Qwen35VisionEncoder.Load(path, device,
                useOptimizedKernels: _options.UseOptimizedVisionKernels,
                useXmxLinear: _options.UseXmxVisionLinear, useXmxAttention: _options.UseXmxVisionAttention,
                useFlashAttention: _options.UseFlashVisionAttention);
        }
        catch
        {
            model.SetExternalDeviceMemoryReservation(0, 0);
            throw;
        }
        if (ct.IsCancellationRequested)
        {
            vision.Dispose();
            model.SetExternalDeviceMemoryReservation(0, 0);
            ct.ThrowIfCancellationRequested();
        }
        VisionInitializationMilliseconds = timer.Elapsed.TotalMilliseconds;
        return vision;
    }

    private long ValidateVisionMemoryBudget(string path, int deviceIndex,
        Qwen35QuantizedModel model, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        Qwen35VisionDescriptor vision = Qwen35VisionEncoder.Inspect(path);
        if (vision.ProjectionDimension != model.Descriptor.EmbeddingLength)
            throw new InvalidDataException("mmproj の出力次元とモデルが一致しません。対応する組み合わせを選んでください。");
        ArcDeviceInfo device = ArcDevices.Enumerate()[deviceIndex];
        long weights = 0;
        using (var reader = new GgufReader(path))
        {
            foreach (GgufTensorInfo tensor in reader.Tensors)
            {
                ct.ThrowIfCancellationRequested();
                long elements = 1;
                foreach (ulong width in tensor.Shape) elements = checked(elements * checked((long)width));
                long bytes = checked(elements * (tensor.Type == Qwen2Gguf.F16Type
                    && tensor.Shape.Count > 1 && tensor.Name != "v.position_embd.weight" ? 2 : sizeof(float)));
                if ((ulong)bytes > device.MaximumAllocationBytes)
                    throw new NotSupportedException($"mmproj のテンソル {tensor.Name} は Arc {deviceIndex} の確保上限を超えます。言語モデルは保持しています。");
                weights = checked(weights + bytes);
            }
        }
        long visionWorkspace = VisionWorkspaceBytes(vision, _options.UseXmxVisionAttention);
        long languageWorkspace = LanguageWorkspaceBytes(model.Descriptor, _options.InferencePrefillChunkTokens,
            _options.InferenceXmxPrefill, _options.InferenceBatchMixedAttention);
        long live = model.LiveDeviceBytes[0], physical = model.TotalNativeDeviceBytes[0];
        // Include either the already retained buffers or the configured future
        // retention, whichever is larger. Do not double-count an occupied pool.
        long retention = Math.Max(physical - live,
            checked((long)(_options.InferenceBufferPoolMiB + _options.InferenceDeferredReleaseMiB) * 1024 * 1024));
        if (_options.InferenceXmxPackedPrefill)
            languageWorkspace = checked(languageWorkspace + 4L * model.Descriptor.EmbeddingLength * model.Descriptor.FeedForwardLength);
        if (_options.InferenceXmxFactoredPrefill)
            languageWorkspace = checked(languageWorkspace + 9L * model.Descriptor.EmbeddingLength * model.Descriptor.FeedForwardLength / 4);
        long required = checked(live + retention + languageWorkspace + weights + visionWorkspace);
        long budget = checked((long)(device.GlobalMemoryBytes / 10 * 9));
        if (required > budget)
            throw new NotSupportedException($"Arc {deviceIndex} の画像処理には言語モデルと作業領域を合わせて約 {required / 1073741824d:F2} GiB 必要ですが、予算は {budget / 1073741824d:F2} GiB です。複数 GPU を選ぶか、mmproj を外してください。言語モデルは保持しています。");
        return checked(weights + visionWorkspace);
    }

    internal static long VisionWorkspaceBytes(Qwen35VisionDescriptor descriptor, bool xmxAttention = false)
        => InferenceMemoryBudget.VisionWorkspaceBytes(descriptor, xmxAttention);

    internal static long LanguageWorkspaceBytes(Qwen35GgufDescriptor descriptor, int chunkTokens,
        bool xmx, bool batchAttention)
        => InferenceMemoryBudget.LanguageWorkspaceBytes(descriptor, chunkTokens, xmx, batchAttention);
}
