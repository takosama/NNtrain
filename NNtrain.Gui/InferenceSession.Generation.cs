using System.Diagnostics;
using System.Text;
using NNtrain.Gemma;

namespace NNtrain.Gui;

public sealed partial class InferenceSession
{
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
        ThrowIfDisposed();
        await _operation.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            ThrowIfDisposed();
            LastGenerationStats = null;
            if (_gemmaModel is { } gemma)
            {
                if (hasImages)
                    throw new NotSupportedException("Gemma 3 の画像入力は現在サポートしていません。");
                Gemma3ChatTurn[] gemmaTurns = turns.Select(turn =>
                    new Gemma3ChatTurn(turn.Role, turn.Content)).ToArray();
                string gemmaPrompt = Gemma3LocalModel.FormatConversation(gemmaTurns);
                Gemma3GenerationResult result = await gemma.GenerateAsync(gemmaPrompt,
                    maxNewTokens, onText, ct, sampling?.Temperature ?? 0f,
                    sampling?.TopP ?? 1f, sampling?.TopK ?? 1).ConfigureAwait(false);
                LastGenerationStats = new GenerationStats(result.PromptTokens, result.CompletionTokens, 0,
                    result.FirstTokenMilliseconds, null,
                    result.EndedByStop
                        ? GenerationStopReason.EndOfMessage : GenerationStopReason.MaximumTokens);
                return result.Text;
            }
            Qwen35QuantizedModel model = _model ?? throw new InvalidOperationException("Load a model before generating.");
            Qwen2GgufTokenizer tokenizer = _tokenizer!;
            string? prompt = hasImages ? null : ChatPromptFormatter.FormatPrompt(turns, thinking);
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
                    // Callback cancellation invalidates sequence state, not resident weights.
                    RecoverGenerationFailure(model);
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

    public async Task<PromptPrimeStats> PrimeHistoryAsync(IReadOnlyList<ChatTurn> conversation,
        CancellationToken ct)
    {
        ArgumentNullException.ThrowIfNull(conversation);
        if (_gemmaModel is not null) return new PromptPrimeStats(0, 0, false);
        if (conversation.Count == 0 || conversation[^1].Role != "assistant")
            throw new ArgumentException("History must end with an assistant turn.", nameof(conversation));
        if (conversation.Any(turn => turn.Image is not null)) return new PromptPrimeStats(0, 0, false);
        string prompt = ChatPromptFormatter.FormatHistory(conversation);
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
                    RecoverGenerationFailure(model);
                    throw;
                }
            }, ct).ConfigureAwait(false);
        }
        finally { _operation.Release(); }
    }

}
