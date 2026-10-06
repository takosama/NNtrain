using System.Net;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NNtrain.Gui;
using Xunit;

public sealed class HuggingFaceModelDownloadTests
{
    private static readonly string Commit = new('a', 40);
    private static readonly byte[] ModelBytes = Encoding.ASCII.GetBytes("GGUFmodel payload");

    [Theory]
    [InlineData("owner/model", "owner/model")]
    [InlineData(" https://huggingface.co/owner/model/tree/main ", "owner/model")]
    [InlineData("https://huggingface.co/owner/model/blob/main/weights/model.gguf", "owner/model")]
    public void RepositoryIdAcceptsModelIdsAndHuggingFaceLinks(string input, string expected)
        => Assert.Equal(expected, HuggingFaceModelDownload.ParseRepositoryId(input));

    [Theory]
    [InlineData("http://huggingface.co/owner/model")]
    [InlineData("https://huggingface.co.evil.test/owner/model")]
    [InlineData("https://user@huggingface.co/owner/model")]
    [InlineData("https://huggingface.co/owner/model/resolve/main/file.gguf")]
    [InlineData("owner/../model")]
    [InlineData("owner/CON")]
    [InlineData("owner/model%2Fother")]
    public void RepositoryIdRejectsUntrustedLocations(string input)
        => Assert.Throws<ArgumentException>(() => HuggingFaceModelDownload.ParseRepositoryId(input));

    [Fact]
    public async Task RepositoryListingKeepsOnlySafeModelsAndMmprojFiles()
    {
        string json = JsonSerializer.Serialize(new
        {
            sha = Commit,
            siblings = new object[]
            {
                new { rfilename = "weights/Z-model.GGUF", size = 16, lfs = new { sha256 = Hash(ModelBytes), size = 16 } },
                new { rfilename = "vision/mmproj-f16.bin", lfs = new { sha256 = Hash(ModelBytes), size = 9 } },
                new { rfilename = "weights/a-model.gguf", size = 4 },
                new { rfilename = "lora_adapter.gguf", size = 7 },
                new { rfilename = "README.md", size = 1 },
                new { rfilename = "../escape.gguf", size = 7 },
                new { rfilename = "folder\\escape.gguf", size = 7 },
                new { rfilename = "CON/model.gguf", size = 7 },
                new { rfilename = "bad-hash.gguf", lfs = new { sha256 = "not a hash", size = 5 } }
            }
        });
        using var client = new HttpClient(new FakeHandler(request =>
        {
            Assert.Equal("https://huggingface.co/api/models/owner/model?blobs=true", request.RequestUri!.ToString());
            Assert.Equal("Bearer", request.Headers.Authorization?.Scheme);
            Assert.Equal("secret", request.Headers.Authorization?.Parameter);
            return new HttpResponseMessage(HttpStatusCode.OK) { Content = new StringContent(json) };
        }));

        HuggingFaceRepository repository = await new HuggingFaceModelDownload(client)
            .GetRepositoryAsync("owner/model", " secret ", CancellationToken.None);

        Assert.Equal("owner/model", repository.Id);
        Assert.Equal(Commit, repository.Commit);
        Assert.Equal(new[] { "bad-hash.gguf", "weights/a-model.gguf", "weights/Z-model.GGUF", "vision/mmproj-f16.bin" },
            repository.Files.Select(file => file.Path));
        Assert.Equal(HuggingFaceArtifactKind.Mmproj, repository.Files[^1].Kind);
        Assert.Equal(9, repository.Files[^1].Size);
        Assert.Null(repository.Files[0].Sha256);
        Assert.Equal(Hash(ModelBytes), repository.Files[2].Sha256);
    }

    [Fact]
    public async Task DownloadUsesPinnedCommitAndPublishesOnlyVerifiedBytes()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("weights/model.gguf", ModelBytes);
        using var client = new HttpClient(new FakeHandler(request =>
        {
            Assert.Equal($"https://huggingface.co/owner/model/resolve/{Commit}/weights/model.gguf",
                request.RequestUri!.ToString());
            Assert.Equal("secret", request.Headers.Authorization?.Parameter);
            Assert.Null(request.Headers.Range);
            return Bytes(HttpStatusCode.OK, ModelBytes);
        }));

        string target = await new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, "secret", null, CancellationToken.None);

        Assert.Equal(Path.Combine(folder.Path, "huggingface", "owner", "model", "weights", "model.gguf"), target);
        Assert.Equal(ModelBytes, await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken));
        Assert.False(File.Exists(Partial(target)));
    }

    [Fact]
    public async Task HashMismatchKeepsPartialAndDoesNotPublishModel()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes);
        byte[] corrupt = Encoding.ASCII.GetBytes("GGUFwrong payload");
        using var client = new HttpClient(new FakeHandler(_ => Bytes(HttpStatusCode.OK, corrupt)));
        string target = Target(folder.Path, "model.gguf");

        await Assert.ThrowsAsync<InvalidDataException>(() => new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None));

        Assert.False(File.Exists(target));
        Assert.Equal(corrupt, await File.ReadAllBytesAsync(Partial(target), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task CompletePartialMustPassHashBeforeItCanBePublished()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes);
        string target = Target(folder.Path, "model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        byte[] corrupt = (byte[])ModelBytes.Clone();
        corrupt[^1] ^= 1;
        await File.WriteAllBytesAsync(Partial(target), corrupt, TestContext.Current.CancellationToken);
        using var client = new HttpClient(new FakeHandler(_ => throw new Xunit.Sdk.XunitException("Complete partial should not request more bytes.")));

        await Assert.ThrowsAsync<InvalidDataException>(() => new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None));

        Assert.False(File.Exists(target));
        Assert.Equal(corrupt, await File.ReadAllBytesAsync(Partial(target), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ExistingTargetIsNeverOverwritten()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes);
        string target = Target(folder.Path, "model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        byte[] previous = Encoding.ASCII.GetBytes("existing file");
        await File.WriteAllBytesAsync(target, previous, TestContext.Current.CancellationToken);
        using var client = new HttpClient(new FakeHandler(_ => throw new Xunit.Sdk.XunitException("Existing target should not request bytes.")));

        await Assert.ThrowsAsync<IOException>(() => new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None));

        Assert.Equal(previous, await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task InvalidGgufHeaderKeepsPartialEvenWhenHashMatches()
    {
        using var folder = new TemporaryDirectory();
        byte[] nonGguf = Encoding.ASCII.GetBytes("bad header");
        HuggingFaceRepository repository = Repository("model.gguf", nonGguf);
        using var client = new HttpClient(new FakeHandler(_ => Bytes(HttpStatusCode.OK, nonGguf)));
        string target = Target(folder.Path, "model.gguf");

        await Assert.ThrowsAsync<InvalidDataException>(() => new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None));

        Assert.False(File.Exists(target));
        Assert.True(File.Exists(Partial(target)));
    }

    [Fact]
    public async Task UnknownLengthWithoutHashCannotPublishAnUnverifiableDownload()
    {
        using var folder = new TemporaryDirectory();
        var artifact = new HuggingFaceArtifact("model.gguf", HuggingFaceArtifactKind.Model, null, null);
        var repository = new HuggingFaceRepository("owner/model", Commit, new[] { artifact });
        string target = Target(folder.Path, artifact.Path);
        using var client = new HttpClient(new FakeHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new UnknownLengthContent(ModelBytes)
        }));

        await Assert.ThrowsAsync<InvalidDataException>(() => new HuggingFaceModelDownload(client).DownloadAsync(
            repository, artifact, folder.Path, null, null, CancellationToken.None));

        Assert.False(File.Exists(target));
        Assert.Equal(ModelBytes, await File.ReadAllBytesAsync(Partial(target), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task HashAloneCanVerifyAResponseWithoutLength()
    {
        using var folder = new TemporaryDirectory();
        var artifact = new HuggingFaceArtifact("model.gguf", HuggingFaceArtifactKind.Model, null, Hash(ModelBytes));
        var repository = new HuggingFaceRepository("owner/model", Commit, new[] { artifact });
        using var client = new HttpClient(new FakeHandler(_ => new HttpResponseMessage(HttpStatusCode.OK)
        {
            Content = new UnknownLengthContent(ModelBytes)
        }));

        string target = await new HuggingFaceModelDownload(client).DownloadAsync(
            repository, artifact, folder.Path, null, null, CancellationToken.None);

        Assert.Equal(ModelBytes, await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task DownloadRejectsUnlistedOrUnsafeArtifactBeforeSendingRequest()
    {
        using var folder = new TemporaryDirectory();
        int requests = 0;
        using var client = new HttpClient(new FakeHandler(_ =>
        {
            requests++;
            return Bytes(HttpStatusCode.OK, ModelBytes);
        }));
        var service = new HuggingFaceModelDownload(client);
        HuggingFaceRepository listed = Repository("model.gguf", ModelBytes);
        var forged = listed.Files[0] with { Sha256 = new string('0', 64) };
        var unsafeArtifact = new HuggingFaceArtifact("../escape.gguf", HuggingFaceArtifactKind.Model,
            ModelBytes.Length, Hash(ModelBytes));
        var unsafeRepository = listed with { Files = new[] { unsafeArtifact } };

        await Assert.ThrowsAsync<ArgumentException>(() => service.DownloadAsync(
            listed, forged, folder.Path, null, null, CancellationToken.None));
        await Assert.ThrowsAsync<ArgumentException>(() => service.DownloadAsync(
            unsafeRepository, unsafeArtifact, folder.Path, null, null, CancellationToken.None));
        Assert.Equal(0, requests);
        Assert.False(File.Exists(Path.Combine(folder.Path, "escape.gguf")));
    }

    [Fact]
    public async Task ResumeRequestsRemainingRangeAndAppendsOnlyMatchingPartialContent()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes);
        string target = Target(folder.Path, "model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        await File.WriteAllBytesAsync(Partial(target), ModelBytes[..4], TestContext.Current.CancellationToken);
        using var client = new HttpClient(new FakeHandler(request =>
        {
            Assert.Equal("bytes=4-", request.Headers.Range?.ToString());
            var response = Bytes(HttpStatusCode.PartialContent, ModelBytes[4..]);
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(4, ModelBytes.Length - 1, ModelBytes.Length);
            return response;
        }));

        await new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None);

        Assert.Equal(ModelBytes, await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task PartialFromAnotherCommitIsNeverResumed()
    {
        using var folder = new TemporaryDirectory();
        string currentCommit = new('b', 40);
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes) with { Commit = currentCommit };
        string target = Target(folder.Path, "model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        byte[] oldPartial = Encoding.ASCII.GetBytes("GGUFold revision");
        await File.WriteAllBytesAsync(Partial(target), oldPartial, TestContext.Current.CancellationToken);
        using var client = new HttpClient(new FakeHandler(request =>
        {
            Assert.Equal($"https://huggingface.co/owner/model/resolve/{currentCommit}/model.gguf",
                request.RequestUri!.ToString());
            Assert.Null(request.Headers.Range);
            return Bytes(HttpStatusCode.OK, ModelBytes);
        }));

        await new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None);

        Assert.Equal(ModelBytes, await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken));
        Assert.Equal(oldPartial, await File.ReadAllBytesAsync(Partial(target), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task IgnoredRangeRestartsPartialInsteadOfDuplicatingItsBytes()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes);
        string target = Target(folder.Path, "model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        await File.WriteAllBytesAsync(Partial(target), ModelBytes[..4], TestContext.Current.CancellationToken);
        using var client = new HttpClient(new FakeHandler(request =>
        {
            Assert.Equal("bytes=4-", request.Headers.Range?.ToString());
            return Bytes(HttpStatusCode.OK, ModelBytes);
        }));

        await new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None);

        Assert.Equal(ModelBytes, await File.ReadAllBytesAsync(target, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task MismatchedContentRangeDoesNotModifyExistingPartial()
    {
        using var folder = new TemporaryDirectory();
        HuggingFaceRepository repository = Repository("model.gguf", ModelBytes);
        string target = Target(folder.Path, "model.gguf");
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        await File.WriteAllBytesAsync(Partial(target), ModelBytes[..4], TestContext.Current.CancellationToken);
        using var client = new HttpClient(new FakeHandler(request =>
        {
            Assert.Equal("bytes=4-", request.Headers.Range?.ToString());
            var response = Bytes(HttpStatusCode.PartialContent, ModelBytes[4..]);
            response.Content.Headers.ContentRange = new ContentRangeHeaderValue(3, ModelBytes.Length - 1, ModelBytes.Length);
            return response;
        }));

        await Assert.ThrowsAsync<InvalidDataException>(() => new HuggingFaceModelDownload(client).DownloadAsync(
            repository, repository.Files[0], folder.Path, null, null, CancellationToken.None));

        Assert.False(File.Exists(target));
        Assert.Equal(ModelBytes[..4], await File.ReadAllBytesAsync(Partial(target), TestContext.Current.CancellationToken));
    }

    private static HuggingFaceRepository Repository(string path, byte[] contents)
    {
        var artifact = new HuggingFaceArtifact(path, HuggingFaceArtifactKind.Model, contents.Length, Hash(contents));
        return new HuggingFaceRepository("owner/model", Commit, new[] { artifact });
    }

    private static string Hash(byte[] contents) => Convert.ToHexString(SHA256.HashData(contents)).ToLowerInvariant();

    private static string Target(string root, string relativePath) =>
        Path.Combine(root, "huggingface", "owner", "model", relativePath);

    private static string Partial(string target) => target + "." + Commit + ".part";

    private static HttpResponseMessage Bytes(HttpStatusCode status, byte[] contents) =>
        new(status) { Content = new ByteArrayContent(contents) };

    private sealed class FakeHandler(Func<HttpRequestMessage, HttpResponseMessage> respond) : HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
            => Task.FromResult(respond(request));
    }

    private sealed class UnknownLengthContent(byte[] bytes) : HttpContent
    {
        protected override bool TryComputeLength(out long length)
        {
            length = 0;
            return false;
        }

        protected override Task SerializeToStreamAsync(Stream stream, TransportContext? context)
            => stream.WriteAsync(bytes).AsTask();
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(),
            "nntrain-hf-test-" + Guid.NewGuid().ToString("N"));

        public TemporaryDirectory() => Directory.CreateDirectory(Path);

        public void Dispose() => Directory.Delete(Path, recursive: true);
    }
}
