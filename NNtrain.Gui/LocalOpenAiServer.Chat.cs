using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using Microsoft.AspNetCore.Http;

namespace NNtrain.Gui;

public sealed partial class LocalOpenAiServer
{
    private async Task ChatAsync(HttpContext context, CancellationToken ct)
    {
        Stopwatch requestTimer = Stopwatch.StartNew();
        JsonElement json = await LocalApiRequests.ReadObjectAsync(context, ct).ConfigureAwait(false);
        LocalChatRequest request = LocalApiRequests.ParseChat(json, _initialLora);
        string model = request.Model;
        string? lora = request.Lora;
        ChatTurn[] messages = request.Messages;
        int maxTokens = request.MaxTokens;
        bool stream = request.Stream, think = request.Thinking, primeHistory = request.PrimeHistory;
        GenerationSampling sampling = request.Sampling;
        string id = "chatcmpl-" + Guid.NewGuid().ToString("N");
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();

        await _requestGate.WaitAsync(ct).ConfigureAwait(false);
        double gateWaitMilliseconds = requestTimer.Elapsed.TotalMilliseconds;
        try
        {
            await LoadCoreAsync(model, lora, request.Devices, ct, request.Mmproj, request.GemmaBackend).ConfigureAwait(false);
            Log($"Chat {id}: model={model}, lora={lora ?? "none"}, stream={stream}, think={think}, " +
                $"max_tokens={maxTokens}, temperature={sampling.Temperature.ToString(CultureInfo.InvariantCulture)}, " +
                $"top_p={sampling.TopP.ToString(CultureInfo.InvariantCulture)}, top_k={sampling.TopK}");
            Log($"Chat {id} history: turns={messages.Length}, characters={messages.Sum(message => message.Content.Length)}");
            Log($"Chat {id} input [{messages[^1].Role}]: {messages[^1].Content}");
            if (stream)
            {
                context.Response.StatusCode = 200;
                context.Response.ContentType = "text/event-stream; charset=utf-8";
                context.Response.Headers.CacheControl = "no-cache";
                context.Response.Headers.Append("X-Accel-Buffering", "no");
                await LocalApiResponses.WriteEventAsync(context.Response,
                    LocalApiResponses.RoleChunk(id, created, model), ct).ConfigureAwait(false);
            }
            Log($"Chat {id} output: gate_wait_ms={gateWaitMilliseconds:F1}");
            Stopwatch generationTimer = Stopwatch.StartNew();
            bool firstText = true;
            double? firstTextFromRequestMilliseconds = null;
            string raw = await _session.GenerateAsync(messages, think, maxTokens, chunk =>
            {
                if (firstText)
                {
                    firstText = false;
                    firstTextFromRequestMilliseconds = requestTimer.Elapsed.TotalMilliseconds;
                    Log($"Chat {id} first text after {firstTextFromRequestMilliseconds / 1000:F2}s " +
                        $"from request ({generationTimer.Elapsed.TotalSeconds:F2}s generating).");
                }
                Console.Write(chunk);
                Console.Out.Flush();
                if (stream)
                    LocalApiResponses.WriteEventAsync(context.Response,
                        LocalApiResponses.ContentChunk(id, created, model, chunk), ct).GetAwaiter().GetResult();
            }, ct, sampling).ConfigureAwait(false);
            Console.WriteLine();
            GenerationStats? stats = _session.LastGenerationStats;
            (string finishReason, _) = LocalApiResponses.Stop(stats);
            Log($"Chat {id} completed: prompt_tokens={stats?.PromptTokens}, " +
                $"completion_tokens={stats?.CompletionTokens}, reused={stats?.ReusedPromptTokens}, " +
                $"uncached={(stats is null ? null : stats.PromptTokens - stats.ReusedPromptTokens)}, " +
                $"gate_wait_ms={gateWaitMilliseconds:F1}, " +
                $"first_text_ms_from_request={firstTextFromRequestMilliseconds:F1}, " +
                $"first_token_ms_from_generation={stats?.FirstTokenMilliseconds:F1}, " +
                $"finish_reason={finishReason}");
            MarkCompleted();
            if (stream)
            {
                await LocalApiResponses.WriteEventAsync(context.Response,
                    LocalApiResponses.FinishChunk(id, created, model, stats), ct).ConfigureAwait(false);
                await LocalApiResponses.WriteDoneAsync(context.Response, ct).ConfigureAwait(false);
                if (primeHistory)
                {
                    try
                    {
                        await PrimeHistoryAsync(messages, raw, think, stats?.StopReason,
                            _lifetime.Token).ConfigureAwait(false);
                    }
                    catch (OperationCanceledException) when (_lifetime.IsCancellationRequested) { }
                    catch (Exception error) { Log($"Chat {id} history prefill failed: {error}"); }
                }
            }
            else
            {
                await context.Response.WriteAsJsonAsync(
                    LocalApiResponses.Completion(id, created, model, raw, stats), ct).ConfigureAwait(false);
            }
        }
        finally
        {
            if (_session.IsLoaded) MarkCompleted();
            _requestGate.Release();
        }
    }

    private async Task PrimeHistoryAsync(ChatTurn[] messages, string raw, bool thinking,
        GenerationStopReason? stopReason, CancellationToken ct)
    {
        var parser = new ThinkingStreamParser(thinking);
        parser.Append(raw);
        ThinkingStreamSnapshot snapshot = parser.Complete(stopReason == GenerationStopReason.EndOfMessage);
        if (string.IsNullOrWhiteSpace(snapshot.AnswerText)) return;
        string prefix = thinking ? "<think>\n</think>\n" : "<think>\n\n</think>\n\n";
        ChatTurn[] history = [.. messages, new ChatTurn("assistant", snapshot.AnswerText, prefix)];
        Stopwatch timer = Stopwatch.StartNew();
        Log("Priming sanitized conversation history for the next turn.");
        PromptPrimeStats result = await _session.PrimeHistoryAsync(history, ct).ConfigureAwait(false);
        Log($"History prefill: prompt_tokens={result.PromptTokens}, reused={result.ReusedPromptTokens}, " +
            $"uncached={result.PromptTokens - result.ReusedPromptTokens}, cached={result.Cached}, " +
            $"elapsed={timer.Elapsed.TotalSeconds:F2}s.");
        MarkCompleted();
    }

}
