using System.IO;
using System.Text.Json;
using Microsoft.AspNetCore.Http;

namespace NNtrain.Gui;

/// <summary>Defines the response wire contract independently of model execution.</summary>
internal static class LocalApiResponses
{
    internal static int ErrorStatus(Exception error) => error switch
    {
        LocalApiException api => api.StatusCode,
        JsonException or BadHttpRequestException or ArgumentException
            or FileNotFoundException or InvalidDataException or NotSupportedException
            or System.Text.RegularExpressions.RegexMatchTimeoutException => 400,
        _ => 500
    };

    internal static object Error(Exception error, int status) => new
    {
        error = new
        {
            message = error.Message,
            type = status == 500 ? "server_error" : "invalid_request_error",
            param = error is LocalApiException requestError ? requestError.Parameter : null,
            code = status == 500 ? "internal_error" : "invalid_request"
        }
    };

    internal static (string FinishReason, string StopDetail) Stop(GenerationStats? stats)
        => (stats?.StopReason == GenerationStopReason.EndOfMessage ? "stop" : "length",
            stats?.StopReason switch
            {
                GenerationStopReason.EndOfMessage => "end_of_message",
                GenerationStopReason.ContextLimit => "context_limit",
                _ => "maximum_tokens"
            });

    internal static object RoleChunk(string id, long created, string model) => new
    {
        id, @object = "chat.completion.chunk", created, model,
        choices = new[] { new { index = 0, delta = new { role = "assistant" }, finish_reason = (string?)null } }
    };

    internal static object ContentChunk(string id, long created, string model, string content) => new
    {
        id, @object = "chat.completion.chunk", created, model,
        choices = new[] { new { index = 0, delta = new { content }, finish_reason = (string?)null } }
    };

    internal static object FinishChunk(string id, long created, string model, GenerationStats? stats)
    {
        (string finish, string detail) = Stop(stats);
        return new
        {
            id, @object = "chat.completion.chunk", created, model,
            choices = new[] { new { index = 0, delta = new { }, finish_reason = finish, nntrain_stop_reason = detail } }
        };
    }

    internal static object Completion(string id, long created, string model, string content, GenerationStats? stats)
    {
        (string finish, string detail) = Stop(stats);
        int promptTokens = stats?.PromptTokens ?? 0;
        int completionTokens = stats?.CompletionTokens ?? 0;
        return new
        {
            id, @object = "chat.completion", created, model,
            choices = new[] { new
            {
                index = 0,
                message = new { role = "assistant", content },
                finish_reason = finish,
                nntrain_stop_reason = detail
            } },
            usage = new
            {
                prompt_tokens = promptTokens,
                completion_tokens = completionTokens,
                total_tokens = promptTokens + completionTokens
            }
        };
    }

    internal static async Task WriteEventAsync(HttpResponse response, object value, CancellationToken ct)
    {
        await response.WriteAsync("data: " + JsonSerializer.Serialize(value) + "\n\n", ct).ConfigureAwait(false);
        await response.Body.FlushAsync(ct).ConfigureAwait(false);
    }

    internal static async Task WriteDoneAsync(HttpResponse response, CancellationToken ct)
    {
        await response.WriteAsync("data: [DONE]\n\n", ct).ConfigureAwait(false);
        await response.Body.FlushAsync(ct).ConfigureAwait(false);
    }
}
