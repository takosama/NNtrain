using System.Net;
using System.Text;
using System.Text.Json;
using NNtrain.Gui;
using Xunit;

namespace NNtrain.IntegrationTests;

public sealed class LocalInferenceClientRefactorTests
{
    private static LocalGenerationRequest Request(bool stream = false) => new(
        [new ChatTurn("user", "質問"), new ChatTurn("assistant", "前の回答", "prefix"),
            new ChatTurn("user", "画像", Image: new ChatImage([255, 216, 255], "test.jpg"))],
        "model.gguf", "lora.bin", "mmproj.gguf", [0, 1], null, true, stream, 512,
        new GenerationSampling(0.7f, 0.9f, 40));

    [Fact]
    public async Task ChatRequestKeepsAuthenticationMultimodalContentAndGenerationOptions()
    {
        using var client = CreateClient(async (request, ct) =>
        {
            Assert.Equal("/v1/chat/completions", request.RequestUri!.AbsolutePath);
            Assert.Equal("Bearer", request.Headers.Authorization!.Scheme);
            Assert.Equal("session-token", request.Headers.Authorization.Parameter);
            Assert.Equal("application/json", request.Content!.Headers.ContentType!.MediaType);
            using JsonDocument document = JsonDocument.Parse(await request.Content.ReadAsStringAsync(ct));
            JsonElement body = document.RootElement;
            Assert.Equal("model.gguf", body.GetProperty("model").GetString());
            Assert.Equal("lora.bin", body.GetProperty("lora").GetString());
            Assert.Equal("mmproj.gguf", body.GetProperty("mmproj").GetString());
            Assert.Equal(JsonValueKind.Null, body.GetProperty("gemma_backend").ValueKind);
            Assert.Equal(512, body.GetProperty("max_tokens").GetInt32());
            Assert.Equal(0.7f, body.GetProperty("temperature").GetSingle());
            Assert.Equal(0.9f, body.GetProperty("top_p").GetSingle());
            Assert.Equal(40, body.GetProperty("top_k").GetInt32());
            Assert.True(body.GetProperty("think").GetBoolean());
            Assert.False(body.GetProperty("stream").GetBoolean());
            Assert.False(body.GetProperty("prime_history").GetBoolean());
            Assert.Equal(new[] { 0, 1 }, body.GetProperty("devices").EnumerateArray().Select(x => x.GetInt32()));
            JsonElement messages = body.GetProperty("messages");
            Assert.Equal("質問", messages[0].GetProperty("content").GetString());
            Assert.Equal("prefix", messages[1].GetProperty("assistant_prefix").GetString());
            Assert.Equal("image_url", messages[2].GetProperty("content")[0].GetProperty("type").GetString());
            Assert.Equal("data:image/jpeg;base64,/9j/", messages[2].GetProperty("content")[0]
                .GetProperty("image_url").GetProperty("url").GetString());
            Assert.Equal("画像", messages[2].GetProperty("content")[1].GetProperty("text").GetString());
            return Json("{\"choices\":[{\"message\":{\"content\":\"回答\"},\"finish_reason\":\"stop\"}]}");
        });
        var chunks = new List<string>();
        LocalGeneration result = await client.GenerateAsync(Request(), chunks.Add, TestContext.Current.CancellationToken);
        Assert.Equal("回答", result.Text);
        Assert.Equal(GenerationStopReason.EndOfMessage, result.StopReason);
        Assert.Equal(new[] { "回答" }, chunks);
    }

    [Fact]
    public async Task StreamConsumesMultilineEventsAndCustomStopReasonWithDoneAtEndOfFile()
    {
        const string response = ": heartbeat\n\ndata: {\"choices\":\n" +
            "data: [{\"delta\":{\"content\":\"前\"}}]}\n\n" +
            "data: {\"choices\":[{\"delta\":{\"content\":\"後\"},\"finish_reason\":\"length\",\"nntrain_stop_reason\":\"context_limit\"}]}\n\n" +
            "data: [DONE]";
        using var client = CreateClient(async (request, ct) =>
        {
            using JsonDocument json = JsonDocument.Parse(await request.Content!.ReadAsStringAsync(ct));
            Assert.True(json.RootElement.GetProperty("stream").GetBoolean());
            Assert.True(json.RootElement.GetProperty("prime_history").GetBoolean());
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(response) };
        });
        var chunks = new List<string>();
        LocalGeneration result = await client.GenerateAsync(Request(true), chunks.Add, TestContext.Current.CancellationToken);
        Assert.Equal("前後", result.Text);
        Assert.Equal(GenerationStopReason.ContextLimit, result.StopReason);
        Assert.Equal(new[] { "前", "後" }, chunks);
    }

    [Theory]
    [InlineData("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"},\"finish_reason\":\"stop\"}]}\n\n")]
    [InlineData("data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: [DONE]\n\n")]
    public async Task PrematureStreamTerminationPreservesObservedTextAndReportsFailure(string response)
    {
        using var client = CreateClient((_, _) => Task.FromResult(Json(response)));
        var chunks = new List<string>();
        await Assert.ThrowsAsync<IOException>(() => client.GenerateAsync(Request(true), chunks.Add, TestContext.Current.CancellationToken));
        Assert.Equal(new[] { "partial" }, chunks);
    }

    [Fact]
    public async Task StreamApiErrorAfterPartialTextIsNotReportedAsSuccessfulCompletion()
    {
        const string response = "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\n" +
            "data: {\"error\":{\"message\":\"GPU failure\"}}\n\n";
        using var client = CreateClient((_, _) => Task.FromResult(Json(response)));
        var chunks = new List<string>();
        InvalidOperationException error = await Assert.ThrowsAsync<InvalidOperationException>(() =>
            client.GenerateAsync(Request(true), chunks.Add, TestContext.Current.CancellationToken));
        Assert.Equal("GPU failure", error.Message);
        Assert.Equal(new[] { "partial" }, chunks);
    }

    [Theory]
    [InlineData("{\"error\":{\"message\":\"denied\"}}", "denied")]
    [InlineData("plain failure", "plain failure")]
    public async Task FailedHttpResponseKeepsStatusAndServerError(string response, string detail)
    {
        using var client = CreateClient((_, _) => Task.FromResult(new HttpResponseMessage(HttpStatusCode.Forbidden)
            { Content = new StringContent(response) }));
        HttpRequestException error = await Assert.ThrowsAsync<HttpRequestException>(() =>
            client.PostAsync("internal/load", new { }, TestContext.Current.CancellationToken));
        Assert.Equal($"サーバー応答 403: {detail}", error.Message);
    }

    [Fact]
    public async Task NonStreamingResponseRequiresARecognizedStopReason()
    {
        using var client = CreateClient((_, _) => Task.FromResult(Json("{\"choices\":[{\"message\":{\"content\":\"answer\"}}]}")));
        var chunks = new List<string>();
        await Assert.ThrowsAsync<InvalidDataException>(() => client.GenerateAsync(Request(), chunks.Add, TestContext.Current.CancellationToken));
        Assert.Empty(chunks);
    }

    [Fact]
    public async Task CancellationDuringStreamingRemainsCancellation()
    {
        using var cancellation = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var client = CreateClient((_, _) => Task.FromResult(Json(
            "data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}\n\ndata: [DONE]\n\n")));
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() =>
            client.GenerateAsync(Request(true), _ => cancellation.Cancel(), cancellation.Token));
    }

    [Fact]
    public async Task StatePreservesLoadedDeviceAndBackendMetadataAndOptionalIdleTime()
    {
        using var client = CreateClient((request, _) =>
        {
            Assert.Equal("/internal/state", request.RequestUri!.AbsolutePath);
            return Task.FromResult(Json("{\"is_loaded\":true,\"model\":\"model\",\"lora\":null,\"mmproj\":\"vision\",\"devices\":[],\"gemma_backend\":\"opencl\",\"idle_seconds_remaining\":12.5}"));
        });
        LocalInferenceState state = await client.ReadStateAsync(TestContext.Current.CancellationToken);
        Assert.True(state.IsLoaded);
        Assert.Equal("model", state.ModelPath);
        Assert.Null(state.AdapterPath);
        Assert.Equal("vision", state.MmprojPath);
        Assert.Empty(state.Devices!);
        Assert.Equal("opencl", state.GemmaBackend);
        Assert.Equal(12.5, state.IdleSecondsRemaining);
    }

    [Fact]
    public void ServerLaunchKeepsCredentialOutOfArgumentsAndPassesOwnedParentAndAdapter()
    {
        var start = LocalInferenceServerProcess.CreateStartInfo("/runtime/dotnet", "/application", 12345,
            456, "adapter with spaces.bin", "TOKEN_NAME", "secret-token");
        Assert.False(start.UseShellExecute);
        Assert.Equal("secret-token", start.Environment["TOKEN_NAME"]);
        Assert.Equal(new[] { "/application/NNtrain.Gui.dll", "--server", "--port", "12345",
            "--lora", "adapter with spaces.bin", "--parent-pid", "456" }, start.ArgumentList);
        Assert.DoesNotContain("secret-token", start.ArgumentList);
    }

    [Fact]
    public void AppHostLaunchDoesNotAddASecondManagedEntrypoint()
    {
        var start = LocalInferenceServerProcess.CreateStartInfo("/application/NNtrain.Gui.exe", "/application",
            12345, 456, null, "TOKEN_NAME", "secret-token");
        Assert.Equal(new[] { "--server", "--port", "12345", "--parent-pid", "456" }, start.ArgumentList);
    }

    private static LocalInferenceClient CreateClient(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send)
    {
        var client = new LocalInferenceClient("session-token", new Handler(send));
        client.SetAddress(new Uri("http://127.0.0.1:12345/"));
        return client;
    }

    private static HttpResponseMessage Json(string json) => new(HttpStatusCode.OK)
        { Content = new StringContent(json, Encoding.UTF8, "application/json") };

    private sealed class Handler(Func<HttpRequestMessage, CancellationToken, Task<HttpResponseMessage>> send) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken ct) => send(request, ct);
    }
}
