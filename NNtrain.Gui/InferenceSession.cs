using System.IO;
using System.Diagnostics;
using System.Text;
using NNtrain.Arc;

namespace NNtrain.Gui;

public sealed record ChatTurn(string Role, string Content, string? AssistantPrefix = null);
public sealed record GenerationSampling(float Temperature, float TopP, int TopK);
public enum GenerationStopReason { EndOfMessage, MaximumTokens, ContextLimit }
public sealed record GenerationStats(int PromptTokens, int CompletionTokens, int ReusedPromptTokens,
    double? FirstTokenMilliseconds, int? FirstTokenId, GenerationStopReason StopReason);
public sealed record PromptPrimeStats(int PromptTokens, int ReusedPromptTokens, bool Cached);

/// <summary>
/// Owns one resident Qwen3.5 model and serializes GPU loading, generation and
/// disposal. The window controls the five-minute idle timeout via UnloadAsync.
/// </summary>
public sealed class InferenceSession : IDisposable
{
    private readonly SemaphoreSlim _operation = new(1, 1);
    private Qwen35QuantizedModel? _model;
    private Qwen2GgufTokenizer? _tokenizer;
    private string? _loadedModelPath;
    private string? _loadedAdapterPath;
    private int[]? _loadedDevices;
    private int _disposed;

    public bool IsLoaded => Volatile.Read(ref _model) is not null;
    public string? LoadedModelPath => Volatile.Read(ref _loadedModelPath);
    public string? LoadedAdapterPath => Volatile.Read(ref _loadedAdapterPath);
    public IReadOnlyList<int>? LoadedDevices => Volatile.Read(ref _loadedDevices);
    public GenerationStats? LastGenerationStats { get; private set; }

    public async Task LoadAsync(string modelPath, string? adapterPath,
        IProgress<string>? progress, CancellationToken ct, IReadOnlyList<int>? selectedDevices = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        modelPath = Path.GetFullPath(modelPath);
        adapterPath = string.IsNullOrWhiteSpace(adapterPath) ? null : Path.GetFullPath(adapterPath);
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
                return;

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
                        InferencePrefillChunkTokens = 16,
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
            // Assign only after loading and adapter validation both succeeded.
            Volatile.Write(ref _tokenizer, loaded.Tokenizer);
            Volatile.Write(ref _loadedModelPath, modelPath);
            Volatile.Write(ref _loadedAdapterPath, adapterPath);
            Volatile.Write(ref _loadedDevices, loaded.Devices);
            Volatile.Write(ref _model, loaded.Model);
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
        string prompt = FormatPrompt(conversation.ToArray(), thinking);
        ThrowIfDisposed();
        LastGenerationStats = null;
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            Qwen35QuantizedModel model = _model ?? throw new InvalidOperationException("Load a model before generating.");
            Qwen2GgufTokenizer tokenizer = _tokenizer!;
            return await Task.Run(() =>
            {
                ct.ThrowIfCancellationRequested();
                int[] promptIds = tokenizer.Encode(prompt);
                if (promptIds.Length > model.Descriptor.ContextLength)
                    throw new ArgumentException("The conversation exceeds the model context length.", nameof(conversation));
                Qwen2GgufTokenizer.StreamingDecoder decoder = tokenizer.CreateStreamingDecoder();
                var text = new StringBuilder();
                var firstTokenTimer = Stopwatch.StartNew();
                double? firstTokenMilliseconds = null;
                int? firstTokenId = null;
                try
                {
                    int[] generatedIds = model.GenerateTokenIdsWithPrefixReuse(promptIds, maxNewTokens,
                        ct, tokenizer.EosTokenId, token =>
                    {
                        ct.ThrowIfCancellationRequested();
                        firstTokenMilliseconds ??= firstTokenTimer.Elapsed.TotalMilliseconds;
                        firstTokenId ??= token;
                        if (token == tokenizer.EosTokenId) return;
                        WriteChunk(decoder.Append(token));
                    }, temperature: sampling?.Temperature ?? 0f,
                        topP: sampling?.TopP ?? 1f, topK: sampling?.TopK ?? 1);
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
        string prompt = FormatHistory(conversation);
        ThrowIfDisposed();
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            Qwen35QuantizedModel model = _model ?? throw new InvalidOperationException("Load a model before priming.");
            Qwen2GgufTokenizer tokenizer = _tokenizer!;
            return await Task.Run(() =>
            {
                int[] ids = tokenizer.Encode(prompt);
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
