using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using Xunit;

namespace NNtrain.Gui.Tests;

public sealed class LocalApiSecurityTests
{
    [Fact]
    public async Task ManagementRequiresSessionAuthenticationAndRejectsBrowserAndNetworkPaths()
    {
        await using var server = new LocalOpenAiServer();
        Uri address = await server.StartAsync(0, null, TestContext.Current.CancellationToken);
        using var client = new HttpClient { BaseAddress = address };
        using (var response = await client.GetAsync("health", TestContext.Current.CancellationToken))
            Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
        foreach (string endpoint in new[] { "internal/unload", "internal/shutdown", "internal/load" })
        {
            using var response = await client.PostAsync(endpoint, Json("{}"), TestContext.Current.CancellationToken);
            Assert.Equal(HttpStatusCode.Unauthorized, response.StatusCode);
        }
        client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", server.AuthenticationToken);
        using (var response = await client.GetAsync("health", TestContext.Current.CancellationToken))
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var stalled = new List<System.Net.Sockets.TcpClient>();
        try
        {
            for (int i = 0; i < 4; i++)
            {
                var connection = new System.Net.Sockets.TcpClient();
                stalled.Add(connection);
                await connection.ConnectAsync(IPAddress.Loopback, address.Port, TestContext.Current.CancellationToken);
                string header = $"POST /internal/load HTTP/1.1\r\nHost: 127.0.0.1:{address.Port}\r\nAuthorization: Bearer {server.AuthenticationToken}\r\nContent-Type: application/json\r\nContent-Length: 2\r\n\r\n{{";
                await connection.GetStream().WriteAsync(Encoding.ASCII.GetBytes(header), TestContext.Current.CancellationToken);
            }
            await Task.Delay(200, TestContext.Current.CancellationToken);
            using var response = await client.GetAsync("health", TestContext.Current.CancellationToken);
            Assert.Equal(HttpStatusCode.TooManyRequests, response.StatusCode);
        }
        finally
        {
            foreach (var connection in stalled)
            {
                await connection.GetStream().WriteAsync(new byte[] { (byte)'}' }, TestContext.Current.CancellationToken);
                await connection.GetStream().ReadAsync(new byte[4096], TestContext.Current.CancellationToken);
                connection.Dispose();
            }
        }
        using (var response = await client.PostAsync("internal/shutdown", new StringContent("{}"), TestContext.Current.CancellationToken))
            Assert.Equal(HttpStatusCode.UnsupportedMediaType, response.StatusCode);
        using (var request = new HttpRequestMessage(HttpMethod.Post, "internal/shutdown") { Content = Json("{}") })
        {
            request.Headers.Add("Origin", "https://untrusted.example");
            using var response = await client.SendAsync(request, TestContext.Current.CancellationToken);
            Assert.Equal(HttpStatusCode.Forbidden, response.StatusCode);
        }
        using (var response = await client.PostAsync("internal/shutdown", Json("["), TestContext.Current.CancellationToken))
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
        foreach (string path in new[] { @"\\untrusted.example\share\model.gguf", @"\\?\UNC\untrusted.example\share\model.gguf", "//untrusted.example/share/model.gguf" })
        {
            using var response = await client.PostAsync("internal/load", Json(System.Text.Json.JsonSerializer.Serialize(new { model = path })), TestContext.Current.CancellationToken);
            Assert.Equal(HttpStatusCode.BadRequest, response.StatusCode);
            Assert.Contains("local drive", await response.Content.ReadAsStringAsync(TestContext.Current.CancellationToken));
        }
        using (var response = await client.GetAsync("health", TestContext.Current.CancellationToken))
            Assert.Equal(HttpStatusCode.OK, response.StatusCode);
    }

    private static StringContent Json(string text) => new(text, Encoding.UTF8, "application/json");
}
