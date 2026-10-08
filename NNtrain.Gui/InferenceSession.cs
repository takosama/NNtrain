using System.IO;
using System.Diagnostics;
using System.Text;
using NNtrain.Arc;

namespace NNtrain.Gui;

public sealed record GenerationSampling(float Temperature, float TopP, int TopK);
public enum GenerationStopReason { EndOfMessage, MaximumTokens, ContextLimit }
public sealed record GenerationStats(int PromptTokens, int CompletionTokens, int ReusedPromptTokens,
    double? FirstTokenMilliseconds, int? FirstTokenId, GenerationStopReason StopReason);
public sealed record PromptPrimeStats(int PromptTokens, int ReusedPromptTokens, bool Cached);
public sealed record ImagePreparationStats(int GridWidth, int GridHeight, bool Cached,
    int PrimedPromptTokens = 0, int ReusedPromptTokens = 0, bool PrefixCached = false,
    double PrefixPreparationMilliseconds = 0);

/// <summary>
/// Owns one resident Qwen3.5 model and serializes GPU loading, generation and
/// disposal. The window controls the five-minute idle timeout via UnloadAsync.
/// </summary>
public sealed class InferenceSession : IDisposable
{
    private readonly SemaphoreSlim _operation = new(1, 1);
    private readonly int _inferencePrefillChunkTokens;
    private readonly bool _useOptimizedVisionKernels;
    private readonly bool _useXmxVisionLinear;
    private readonly bool _useXmxVisionAttention;
    private readonly bool _useFlashVisionAttention;
    private readonly bool _inferenceBatchMixedAttention;
    private readonly bool _collectKernelTimings;
    private readonly int _inferenceProjectionRows;
    private readonly bool _inferenceXmxPrefill;
    private readonly bool _inferenceXmxPackedPrefill;
    private readonly bool _inferenceXmxFactoredPrefill;
    private readonly bool _inferenceResidentIq2Panels;
    private readonly bool _inferenceXmxGgufBslmPrefill;
    private readonly bool _inferenceSubgroupRecurrentRms;
    private readonly int _inferenceBufferPoolMiB;
    private readonly int _inferenceDeferredReleaseMiB;
    private readonly bool _inferenceBatchRecurrent;
    private readonly bool _inferenceSplitOutputHead;
    private readonly bool _preloadVisionEncoder;
    private Qwen35QuantizedModel? _model;
    private Qwen2GgufTokenizer? _tokenizer;
    private string? _loadedModelPath;
    private string? _loadedAdapterPath;
    private string? _loadedMmprojPath;
    private Qwen35VisionEncoder? _vision;
    private readonly Dictionary<string, Qwen35VisionEmbedding> _imageEmbeddings = [];
    private int[]? _loadedDevices;
    private int _disposed;

    public InferenceSession(int inferencePrefillChunkTokens = 1024, bool useOptimizedVisionKernels = true,
        bool inferenceBatchMixedAttention = true, bool collectKernelTimings = false,
        int inferenceProjectionRows = 4, bool prepareVisionOnLoad = true,
        bool inferenceXmxPrefill = true, int inferenceBufferPoolMiB = 512,
        int inferenceDeferredReleaseMiB = 256, bool inferenceBatchRecurrent = true,
        bool inferenceXmxPackedPrefill = false, bool useXmxVisionLinear = true,
        bool useXmxVisionAttention = true, bool inferenceXmxFactoredPrefill = true,
        bool useFlashVisionAttention = false, bool inferenceResidentIq2Panels = false,
        bool inferenceSubgroupRecurrentRms = true, bool inferenceXmxGgufBslmPrefill = true,
        bool inferenceSplitOutputHead = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(inferencePrefillChunkTokens);
        _inferencePrefillChunkTokens = inferencePrefillChunkTokens;
        _useOptimizedVisionKernels = useOptimizedVisionKernels;
        _useXmxVisionLinear = useXmxVisionLinear;
        _useXmxVisionAttention = useXmxVisionAttention;
        _useFlashVisionAttention = useFlashVisionAttention;
        _inferenceBatchMixedAttention = inferenceBatchMixedAttention;
        _collectKernelTimings = collectKernelTimings;
        _inferenceProjectionRows = inferenceProjectionRows;
        _inferenceXmxPrefill = inferenceXmxPrefill;
        _inferenceXmxPackedPrefill = inferenceXmxPackedPrefill;
        _inferenceXmxFactoredPrefill = inferenceXmxFactoredPrefill;
        _inferenceResidentIq2Panels = inferenceResidentIq2Panels;
        _inferenceXmxGgufBslmPrefill = inferenceXmxGgufBslmPrefill;
        _inferenceSubgroupRecurrentRms = inferenceSubgroupRecurrentRms;
        _inferenceBufferPoolMiB = inferenceBufferPoolMiB;
        _inferenceDeferredReleaseMiB = inferenceDeferredReleaseMiB;
        _inferenceBatchRecurrent = inferenceBatchRecurrent;
        _inferenceSplitOutputHead = inferenceSplitOutputHead;
        _preloadVisionEncoder = prepareVisionOnLoad;
    }

    public bool IsLoaded => Volatile.Read(ref _model) is not null;
    public string? LoadedModelPath => Volatile.Read(ref _loadedModelPath);
    public string? LoadedAdapterPath => Volatile.Read(ref _loadedAdapterPath);
    public string? LoadedMmprojPath => Volatile.Read(ref _loadedMmprojPath);
    public IReadOnlyList<int>? LoadedDevices => Volatile.Read(ref _loadedDevices);
    public GenerationStats? LastGenerationStats { get; private set; }
    /// <summary>Decode, resize and encode work for the most recent generation or preparation; cache hits take zero.</summary>
    public double LastImagePreparationMilliseconds { get; private set; }
    /// <summary>Time spent validating and uploading the currently loaded vision tower.</summary>
    public double VisionInitializationMilliseconds { get; private set; }
    public IReadOnlyDictionary<string, double> KernelMilliseconds => _model?.KernelMilliseconds
        ?? new Dictionary<string, double>();

    public async Task LoadAsync(string modelPath, string? adapterPath,
        IProgress<string>? progress, CancellationToken ct, IReadOnlyList<int>? selectedDevices = null,
        string? mmprojPath = null)
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
                    options: new Qwen35ExecutionOptions
                    {
                        InferencePrefillChunkTokens = _inferencePrefillChunkTokens,
                        InferenceBatchMixedAttention = _inferenceBatchMixedAttention,
                        InferenceBatchTextAttention = true,
                        // Use both GPUs for the Q5_K vocabulary projection when
                        // the model's format, adapters and memory budget allow it.
                        InferenceSplitOutputHead = _inferenceSplitOutputHead && devices.Length > 1,
                        CollectKernelTimings = _collectKernelTimings,
                        InferenceProjectionRows = _inferenceProjectionRows,
                        InferenceXmxPrefill = _inferenceXmxPrefill,
                        InferenceXmxPackedPrefill = _inferenceXmxPackedPrefill,
                        InferenceXmxFactoredPrefill = _inferenceXmxFactoredPrefill,
                        InferenceXmxGgufBslmPrefill = _inferenceXmxGgufBslmPrefill,
                        // Resident IQ2 panels exchange extra capacity for speed.
                        // Keep single-device text models within their old budget.
                        InferenceResidentIq2Panels = _inferenceResidentIq2Panels && devices.Length > 1,
                        InferenceSubgroupRecurrentRms = _inferenceSubgroupRecurrentRms,
                        InferenceBufferPoolMiB = _inferenceBufferPoolMiB,
                        InferenceDeferredReleaseMiB = _inferenceDeferredReleaseMiB,
                        InferenceBatchRecurrent = _inferenceBatchRecurrent,
                        ComputeModelFingerprintOnLoad = adapterPath is not null
                    });
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

    /// <summary>Streams complete UTF-8 text chunks to onText on a worker thread.</summary>
    public async Task<string> GenerateAsync(IReadOnlyList<ChatTurn> conversation,
        bool thinking, int maxNewTokens, Action<string> onText, CancellationToken ct,
        GenerationSampling? sampling = null)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        ArgumentNullException.ThrowIfNull(onText);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNewTokens);
        ChatTurn[] turns = conversation.ToArray();
        bool hasImages = turns.Any(turn => turn.Image is not null);
        string? prompt = hasImages ? null : FormatPrompt(turns, thinking);
        ThrowIfDisposed();
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            LastGenerationStats = null;
            Qwen35QuantizedModel model = _model ?? throw new InvalidOperationException("Load a model before generating.");
            Qwen2GgufTokenizer tokenizer = _tokenizer!;
            return await Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();
                var firstTokenTimer = Stopwatch.StartNew();
                LastImagePreparationMilliseconds = 0;
                MultimodalPrompt? multimodal = hasImages
                    ? MultimodalPrompt.Build(turns, thinking, text => tokenizer.Encode(text, model.Descriptor.ContextLength, ct),
                        image => EncodeImageCore(image, model, ct)) : null;
                int[] promptIds = multimodal is null ? tokenizer.Encode(prompt!, model.Descriptor.ContextLength, ct)
                    : multimodal.Tokens.Select(token => token.TokenId).ToArray();
                if (promptIds.Length > model.Descriptor.ContextLength)
                    throw new ArgumentException("The conversation exceeds the model context length.", nameof(conversation));
                Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();
                var text = new StringBuilder();
                double? firstTokenMilliseconds = null;
                int? firstTokenId = null;
                try
                {
                    Action<int> onToken = token =>
                    {
                        ct.ThrowIfCancellationRequested();
                        firstTokenMilliseconds ??= firstTokenTimer.Elapsed.TotalMilliseconds;
                        firstTokenId ??= token;
                        if (token == tokenizer.EosTokenId) return;
                        WriteChunk(decoder.Append(token));
                    };
                    int[] generatedIds = multimodal is null
                        ? model.GenerateTokenIdsWithPrefixReuse(promptIds, maxNewTokens,
                            ct, tokenizer.EosTokenId, onToken, temperature: sampling?.Temperature ?? 0f,
                            topP: sampling?.TopP ?? 1f, topK: sampling?.TopK ?? 1)
                        : model.GenerateTokenIdsWithEmbeddings(multimodal.Tokens, maxNewTokens,
                            multimodal.NextPosition, ct, tokenizer.EosTokenId, onToken,
                            temperature: sampling?.Temperature ?? 0f, topP: sampling?.TopP ?? 1f,
                            topK: sampling?.TopK ?? 1);
                    ct.ThrowIfCancellationRequested();
                    WriteChunk(decoder.Complete());
                    GenerationStopReason stopReason = generatedIds.Length > promptIds.Length
                        && generatedIds[^1] == tokenizer.EosTokenId
                        ? GenerationStopReason.EndOfMessage
                        : generatedIds.Length >= model.Descriptor.ContextLength
                            ? GenerationStopReason.ContextLimit : GenerationStopReason.MaximumTokens;
                    LastGenerationStats = new GenerationStats(promptIds.Length,
                        generatedIds.Length - promptIds.Length, model.LastReusedPromptTokens,
                        firstTokenMilliseconds, firstTokenId, stopReason);
                    return text.ToString();
                }
                catch
                {
                    // A callback cancellation/exception leaves the model's
                    // sequence marked faulted. Reset preserves resident weights.
                    try { model.Reset(); }
                    catch
                    {
                        // Preserve the generation failure if the GPU is also
                        // unable to reset or release its state.
                        try { UnloadCore(); } catch { }
                    }
                    throw;
                }

                void WriteChunk(string chunk)
                {
                    if (chunk.Length == 0) return;
                    onText(chunk);
                    text.Append(chunk);
                }

            }).ConfigureAwait(false);
        }
        finally { _operation.Release(); }
    }

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
                    try { model.Reset(); }
                    catch { try { UnloadCore(); } catch { } }
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
        if (path is not null && _preloadVisionEncoder)
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
                useOptimizedKernels: _useOptimizedVisionKernels,
                useXmxLinear: _useXmxVisionLinear, useXmxAttention: _useXmxVisionAttention,
                useFlashAttention: _useFlashVisionAttention);
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
        long visionWorkspace = VisionWorkspaceBytes(vision, _useXmxVisionAttention);
        long languageWorkspace = LanguageWorkspaceBytes(model.Descriptor, _inferencePrefillChunkTokens,
            _inferenceXmxPrefill, _inferenceBatchMixedAttention);
        long live = model.LiveDeviceBytes[0], physical = model.TotalNativeDeviceBytes[0];
        // Include either the already retained buffers or the configured future
        // retention, whichever is larger. Do not double-count an occupied pool.
        long retention = Math.Max(physical - live,
            checked((long)(_inferenceBufferPoolMiB + _inferenceDeferredReleaseMiB) * 1024 * 1024));
        if (_inferenceXmxPackedPrefill)
            languageWorkspace = checked(languageWorkspace + 4L * model.Descriptor.EmbeddingLength * model.Descriptor.FeedForwardLength);
        if (_inferenceXmxFactoredPrefill)
            languageWorkspace = checked(languageWorkspace + 9L * model.Descriptor.EmbeddingLength * model.Descriptor.FeedForwardLength / 4);
        long required = checked(live + retention + languageWorkspace + weights + visionWorkspace);
        long budget = checked((long)(device.GlobalMemoryBytes / 10 * 9));
        if (required > budget)
            throw new NotSupportedException($"Arc {deviceIndex} の画像処理には言語モデルと作業領域を合わせて約 {required / 1073741824d:F2} GiB 必要ですが、予算は {budget / 1073741824d:F2} GiB です。複数 GPU を選ぶか、mmproj を外してください。言語モデルは保持しています。");
        return checked(weights + visionWorkspace);
    }

    internal static long VisionWorkspaceBytes(Qwen35VisionDescriptor d, bool xmxAttention = false)
    {
        // The learned position table is interpolated to the actual image grid;
        // ImageSize does not bound a 768-pixel-cap input's attention workspace.
        long rows = 768L * 768 / ((long)d.PatchSize * d.PatchSize);
        long mergedRows = rows / ((long)d.MergeSize * d.MergeSize);
        long attention = checked(rows * 7 * d.EmbeddingLength);
        long feedForward = checked(rows * (3L * d.EmbeddingLength + d.FeedForwardLength));
        long merger = checked(rows * 3 * d.EmbeddingLength + mergedRows * d.ProjectionDimension);
        long scratch = checked((rows * rows * d.HeadCount
            + rows * 3 * d.PatchSize * d.PatchSize + Math.Max(attention, Math.Max(feedForward, merger))) * sizeof(float));
        if (xmxAttention)
            scratch = checked(scratch + rows * 3 * d.EmbeddingLength * sizeof(float));
        // The vision lane retains a 256 MiB pool plus 128 MiB retired buffers.
        // The extra 64 MiB covers program/driver allocations and alignment.
        return checked(scratch + 448L * 1024 * 1024);
    }

    internal static long LanguageWorkspaceBytes(Qwen35GgufDescriptor d, int chunkTokens,
        bool xmx, bool batchAttention)
    {
        long rows = Math.Max(1, chunkTokens);
        long query = checked((long)d.HeadCount * d.HeadWidth);
        long kv = checked((long)d.KvHeadCount * d.HeadWidth);
        long values = checked((long)d.LinearValueHeads * d.LinearHeadWidth);
        long qkv = checked((2L * d.LinearKeyHeads + d.LinearValueHeads) * d.LinearHeadWidth);
        long feedForward = checked(5L * d.EmbeddingLength + 3L * d.FeedForwardLength);
        long recurrent = checked(3L * d.EmbeddingLength + 2 * qkv + 2 * values + 2L * d.LinearValueHeads);
        long attention = checked(3L * d.EmbeddingLength + 4 * query + 3 * kv);
        long scratch = checked(rows * Math.Max(feedForward, Math.Max(recurrent, attention)) * sizeof(float));
        if (xmx) scratch = checked(scratch + rows * Math.Max(d.FeedForwardLength, d.EmbeddingLength) * sizeof(float));
        if (batchAttention && rows > 1) scratch = checked(scratch + 128L * 1024 * 1024);
        return Math.Max(scratch, 64L * 1024 * 1024);
    }

    public async Task UnloadAsync()
    {
        ThrowIfDisposed();
        await _operation.WaitAsync().ConfigureAwait(false);
        try { await UnloadCoreAsync().ConfigureAwait(false); }
        finally { _operation.Release(); }
    }

    public async Task<PromptPrimeStats> PrimeHistoryAsync(IReadOnlyList<ChatTurn> conversation,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        if (conversation.Count == 0 || conversation[^1].Role != "assistant")
            throw new ArgumentException("History must end with an assistant turn.", nameof(conversation));
        if (conversation.Any(turn => turn.Image is not null)) return new PromptPrimeStats(0, 0, false);
        string prompt = FormatHistory(conversation);
        ThrowIfDisposed();
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            Qwen35QuantizedModel model = _model ?? throw new InvalidOperationException("Load a model before priming.");
            Qwen2GgufTokenizer tokenizer = _tokenizer!;
            return await Task.Run(() =>
            {
                int[] ids = tokenizer.Encode(prompt, model.Descriptor.ContextLength, ct);
                if (ids.Length >= model.Descriptor.ContextLength)
                    throw new ArgumentException("The conversation exceeds the model context length.", nameof(conversation));
                try
                {
                    var result = model.PrimePromptPrefix(ids, ct);
                    return new PromptPrimeStats(ids.Length, result.ReusedTokens, result.Cached);
                }
                catch
                {
                    try { model.Reset(); }
                    catch { try { UnloadCore(); } catch { } }
                    throw;
                }
            }, ct).ConfigureAwait(false);
        }
        finally { _operation.Release(); }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;
        _operation.Wait();
        try { UnloadCore(); }
        finally
        {
            _operation.Release();
            _operation.Dispose();
        }
    }

    private Task UnloadCoreAsync() => Task.Run(UnloadCore);

    private void UnloadCore()
    {
        Qwen35QuantizedModel? model = Interlocked.Exchange(ref _model, null);
        Volatile.Write(ref _tokenizer, null);
        Volatile.Write(ref _loadedModelPath, null);
        Volatile.Write(ref _loadedAdapterPath, null);
        Volatile.Write(ref _loadedMmprojPath, null);
        _imageEmbeddings.Clear();
        _vision?.Dispose();
        _vision = null;
        LastImagePreparationMilliseconds = 0;
        VisionInitializationMilliseconds = 0;
        Volatile.Write(ref _loadedDevices, null);
        model?.Dispose();
    }

    private void ThrowIfDisposed() => ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);

    private static string FormatPrompt(IReadOnlyList<ChatTurn> conversation, bool thinking)
    {
        if (conversation.Count == 0 || conversation[^1] is not { Role: "user" })
            throw new ArgumentException("The conversation must end with a user turn.", nameof(conversation));
        var prompt = new StringBuilder();
        AppendTurns(prompt, conversation);
        prompt.Append("<|im_start|>assistant\n<think>\n");
        if (!thinking) prompt.Append("\n</think>\n\n");
        return prompt.ToString();
    }

    private static string FormatHistory(IReadOnlyList<ChatTurn> conversation)
    {
        var prompt = new StringBuilder();
        AppendTurns(prompt, conversation);
        return prompt.ToString();
    }

    private static void AppendTurns(StringBuilder prompt, IReadOnlyList<ChatTurn> conversation)
    {
        foreach (ChatTurn turn in conversation)
        {
            if (turn is null || turn.Role is not ("system" or "user" or "assistant"))
                throw new ArgumentException("Chat roles must be system, user or assistant.", nameof(conversation));
            prompt.Append("<|im_start|>").Append(turn.Role).Append('\n')
                .Append(turn.Role == "assistant" ? turn.AssistantPrefix : null)
                .Append(EscapeMarkers(turn.Content ?? string.Empty)).Append("<|im_end|>\n");
        }
    }

    private static string EscapeMarkers(string content) => content
        .Replace("<|", "<\u200b|", StringComparison.Ordinal)
        .Replace("<think>", "<\u200bthink>", StringComparison.Ordinal)
        .Replace("</think>", "<\u200b/think>", StringComparison.Ordinal);
}
