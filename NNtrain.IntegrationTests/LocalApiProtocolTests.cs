using System.Net;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using NNtrain.Gui;
using Xunit;

namespace NNtrain.IntegrationTests;

public sealed class LocalApiProtocolTests
{
    private const string Token = "0123456789ABCDEF0123456789ABCDEF";

    [Theory]
    [InlineData("127.0.0.1", "127.0.0.1", false, true, "application/json", 200)]
    [InlineData("::1", "localhost", false, true, "application/json; charset=utf-8", 200)]
    [InlineData("192.0.2.1", "127.0.0.1", false, true, "application/json", 403)]
    [InlineData("127.0.0.1", "untrusted.example", false, true, "application/json", 403)]
    [InlineData("127.0.0.1", "127.0.0.1", true, false, "text/plain", 403)]
    [InlineData("127.0.0.1", "127.0.0.1", false, false, "text/plain", 401)]
    [InlineData("127.0.0.1", "127.0.0.1", false, true, "text/plain", 415)]
    public async Task AccessPolicyRetainsBoundaryAndRejectionPrecedence(string remote, string host,
        bool origin, bool authorized, string contentType, int expected)
    {
        using var policy = new LocalApiAccessPolicy(Token);
        DefaultHttpContext context = Context();
        context.Connection.RemoteIpAddress = IPAddress.Parse(remote);
        context.Request.Host = new HostString(host);
        context.Request.ContentType = contentType;
        if (origin) context.Request.Headers.Origin = "https://untrusted.example";
        if (!authorized) context.Request.Headers.Authorization = "Bearer incorrect";
        bool called = false;
        await policy.InvokeAsync(context, _ => { called = true; return Task.CompletedTask; });
        Assert.Equal(expected, context.Response.StatusCode);
        Assert.Equal(expected == 200, called);
    }

    [Fact]
    public async Task AdmissionLimitReleasesAllSlotsAfterEndpointExceptions()
    {
        using var policy = new LocalApiAccessPolicy(Token);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        int admitted = 0;
        Task[] occupied = Enumerable.Range(0, 4).Select(_ => policy.InvokeAsync(Context(), async _ =>
        {
            Interlocked.Increment(ref admitted);
            await release.Task;
            throw new InvalidOperationException("endpoint failure");
        })).ToArray();
        Assert.Equal(4, admitted);
        DefaultHttpContext overflow = Context();
        await policy.InvokeAsync(overflow, _ => throw new InvalidOperationException("must not run"));
        Assert.Equal(429, overflow.Response.StatusCode);
        Assert.Equal("1", overflow.Response.Headers.RetryAfter.ToString());
        release.SetResult();
        foreach (Task task in occupied) await Assert.ThrowsAsync<InvalidOperationException>(() => task);
        DefaultHttpContext next = Context();
        await policy.InvokeAsync(next, _ => Task.CompletedTask);
        Assert.Equal(200, next.Response.StatusCode);
    }

    [Fact]
    public void ChatDefaultsTokenAliasAndExplicitLoraNullRemainCompatible()
    {
        using var model = new ModelFile();
        JsonElement json = JsonSerializer.SerializeToElement(new
        {
            model = model.Path, messages = new[] { new { role = "user", content = "hi" } }
        });
        LocalChatRequest defaults = LocalApiRequests.ParseChat(json, "default.bin");
        Assert.Equal("default.bin", defaults.Lora);
        Assert.Equal(512, defaults.MaxTokens);
        Assert.Equal(new GenerationSampling(.6f, .95f, 20), defaults.Sampling);
        Assert.True(defaults.Thinking);
        Assert.False(defaults.Stream);
        Assert.False(defaults.PrimeHistory);
        LocalChatRequest explicitValues = LocalApiRequests.ParseChat(JsonSerializer.SerializeToElement(new
        {
            model = model.Path, lora = (string?)null, devices = Array.Empty<int>(),
            max_completion_tokens = 7, max_tokens = "ignored alias", stream = true,
            think = false, prime_history = true, temperature = 0, top_p = 1, top_k = 1,
            messages = new[] { new { role = "user", content = "hi" } }
        }), "default.bin");
        Assert.Null(explicitValues.Lora);
        Assert.NotNull(explicitValues.Devices);
        Assert.Empty(explicitValues.Devices);
        Assert.Equal(7, explicitValues.MaxTokens);
        Assert.True(explicitValues.Stream);
        Assert.False(explicitValues.Thinking);
        Assert.True(explicitValues.PrimeHistory);
    }

    [Theory]
    [InlineData("max_tokens", "0", "max_tokens")]
    [InlineData("max_completion_tokens", "8193", "max_tokens")]
    [InlineData("temperature", "2.01", "temperature")]
    [InlineData("top_p", "0", "top_p")]
    [InlineData("top_k", "257", "top_k")]
    [InlineData("stream", "1", "stream")]
    [InlineData("devices", "[0,0]", "devices")]
    [InlineData("devices", "[-1]", "devices")]
    [InlineData("devices", "[0,1,2]", "devices")]
    [InlineData("gemma_backend", "\"cuda\"", "gemma_backend")]
    public void InvalidChatOptionsKeepClientErrorParameter(string key, string value, string parameter)
    {
        using var model = new ModelFile();
        using var document = JsonDocument.Parse("{\"model\":" + JsonSerializer.Serialize(model.Path) +
            ",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"" + key + "\":" + value + "}");
        LocalApiException error = Assert.Throws<LocalApiException>(() => LocalApiRequests.ParseChat(document.RootElement, null));
        Assert.Equal(400, error.StatusCode);
        Assert.Equal(parameter, error.Parameter);
    }

    [Theory]
    [InlineData("\\\\untrusted.example\\share\\model.gguf")]
    [InlineData("\\\\?\\UNC\\untrusted.example\\share\\model.gguf")]
    [InlineData("//untrusted.example/share/model.gguf")]
    [InlineData("models/../model.gguf")]
    public void FileValidationRejectsNetworkAndTraversalPathsBeforeAccess(string path)
    {
        LocalApiException error = Assert.Throws<LocalApiException>(() => LocalApiRequests.ValidateFilePath(path, ".gguf", "model"));
        Assert.Equal(400, error.StatusCode);
        Assert.Equal("model", error.Parameter);
    }

    [Fact]
    public void MessageParsingKeepsDeveloperMappingAndLocalImageRestriction()
    {
        var image = new ChatImage([137, 80, 78, 71, 13, 10, 26, 10]);
        JsonElement json = JsonSerializer.SerializeToElement(new
        {
            messages = new object[]
            {
                new { role = "developer", content = "instruction" },
                new { role = "user", content = new object[] { new { type = "text", text = "describe" },
                    new { type = "image_url", image_url = new { url = image.ToDataUrl() } } } }
            }
        });
        ChatTurn[] messages = LocalApiRequests.RequiredMessages(json);
        Assert.Equal("system", messages[0].Role);
        Assert.Equal("describe", messages[1].Content);
        Assert.Equal(image.Hash, messages[1].Image?.Hash);
        JsonElement remote = JsonSerializer.SerializeToElement(new
        {
            messages = new[] { new { role = "user", content = new[] {
                new { type = "image_url", image_url = new { url = "https://untrusted.example/image.png" } } } } }
        });
        Assert.Equal("messages", Assert.Throws<LocalApiException>(() => LocalApiRequests.RequiredMessages(remote)).Parameter);
    }

    [Theory]
    [InlineData(GenerationStopReason.EndOfMessage, "stop", "end_of_message")]
    [InlineData(GenerationStopReason.ContextLimit, "length", "context_limit")]
    [InlineData(GenerationStopReason.MaximumTokens, "length", "maximum_tokens")]
    public async Task CompletionAndSseShareExistingStopReasonAndUsageContract(
        GenerationStopReason stop, string reason, string detail)
    {
        var stats = new GenerationStats(11, 3, 5, 1, 7, stop);
        JsonElement completion = JsonSerializer.SerializeToElement(LocalApiResponses.Completion("id", 10, "model", "text", stats));
        JsonElement choice = completion.GetProperty("choices")[0];
        Assert.Equal(reason, choice.GetProperty("finish_reason").GetString());
        Assert.Equal(detail, choice.GetProperty("nntrain_stop_reason").GetString());
        Assert.Equal(14, completion.GetProperty("usage").GetProperty("total_tokens").GetInt32());
        var context = new DefaultHttpContext();
        await using var body = new MemoryStream();
        context.Response.Body = body;
        await LocalApiResponses.WriteEventAsync(context.Response, LocalApiResponses.RoleChunk("id", 10, "model"), TestContext.Current.CancellationToken);
        await LocalApiResponses.WriteEventAsync(context.Response, LocalApiResponses.ContentChunk("id", 10, "model", "text"), TestContext.Current.CancellationToken);
        await LocalApiResponses.WriteEventAsync(context.Response, LocalApiResponses.FinishChunk("id", 10, "model", stats), TestContext.Current.CancellationToken);
        await LocalApiResponses.WriteDoneAsync(context.Response, TestContext.Current.CancellationToken);
        string[] events = Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(4, events.Length);
        Assert.Equal("data: [DONE]", events[^1]);
        using var final = JsonDocument.Parse(events[2][6..]);
        Assert.Equal(reason, final.RootElement.GetProperty("choices")[0].GetProperty("finish_reason").GetString());
        Assert.Equal(JsonValueKind.Object, final.RootElement.GetProperty("choices")[0].GetProperty("delta").ValueKind);
    }

    [Fact]
    public void ErrorEnvelopeRetainsRequestParameterAndServerClassification()
    {
        var error = new LocalApiException(408, "Request body read timed out.", "messages");
        Assert.Equal(408, LocalApiResponses.ErrorStatus(error));
        JsonElement envelope = JsonSerializer.SerializeToElement(LocalApiResponses.Error(error, 408)).GetProperty("error");
        Assert.Equal("messages", envelope.GetProperty("param").GetString());
        Assert.Equal("invalid_request_error", envelope.GetProperty("type").GetString());
        Assert.Equal("invalid_request", envelope.GetProperty("code").GetString());
        Assert.Equal(500, LocalApiResponses.ErrorStatus(new InvalidOperationException("GPU failure")));
        Assert.Equal(400, LocalApiResponses.ErrorStatus(new FileNotFoundException("missing")));
    }

    private static DefaultHttpContext Context()
    {
        var context = new DefaultHttpContext();
        context.Connection.RemoteIpAddress = IPAddress.Loopback;
        context.Request.Host = new HostString("127.0.0.1");
        context.Request.Method = "POST";
        context.Request.ContentType = "application/json";
        context.Request.Headers.Authorization = "Bearer " + Token;
        return context;
    }

    private sealed class ModelFile : IDisposable
    {
        public string Path { get; } = "gui-contract-" + Guid.NewGuid().ToString("N") + ".gguf";
        public ModelFile() => File.WriteAllBytes(Path, []);
        public void Dispose() => File.Delete(Path);
    }
}
