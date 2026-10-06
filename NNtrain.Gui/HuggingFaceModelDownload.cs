using System.Buffers;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace NNtrain.Gui;

internal enum HuggingFaceArtifactKind { Model, Mmproj }

internal sealed record HuggingFaceArtifact(string Path, HuggingFaceArtifactKind Kind,
    long? Size, string? Sha256);

internal sealed record HuggingFaceRepository(string Id, string Commit,
    IReadOnlyList<HuggingFaceArtifact> Files);

internal sealed record HuggingFaceDownloadProgress(string Path, long Bytes, long? Total);

// Only public/model-repository endpoints are used; an optional token is kept in memory
// for the duration of a request and is never stored in the downloaded files.
internal sealed class HuggingFaceModelDownload(HttpClient client)
{
    private static readonly Regex RepositoryPart = new("^[A-Za-z0-9][A-Za-z0-9._-]*$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly Regex HexSha = new("^[a-fA-F0-9]{40,64}$",
        RegexOptions.Compiled | RegexOptions.CultureInvariant);
    private static readonly HashSet<string> WindowsReservedNames = new(StringComparer.OrdinalIgnoreCase)
    {
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5",
        "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4",
        "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"
    };

    internal static string ParseRepositoryId(string input)
    {
        string value = input.Trim();
        if (Uri.TryCreate(value, UriKind.Absolute, out Uri? uri))
        {
            if (uri.Scheme != Uri.UriSchemeHttps || !uri.Host.Equals("huggingface.co", StringComparison.OrdinalIgnoreCase)
                || uri.Port != 443 || !string.IsNullOrEmpty(uri.UserInfo))
                throw new ArgumentException("https://huggingface.co のモデルリポジトリURLを指定してください。");
            string[] urlParts = uri.AbsolutePath.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
            if (urlParts.Length < 2 || (urlParts.Length > 2 && urlParts[2] is not ("tree" or "blob")))
                throw new ArgumentException("モデルリポジトリのURLを指定してください。");
            value = string.Join('/', urlParts.Take(2).Select(Uri.UnescapeDataString));
        }
        string[] parts = value.Split('/');
        if (parts.Length != 2 || parts.Any(part => !IsSafeRepositoryPart(part)))
            throw new ArgumentException("リポジトリは owner/name 形式で指定してください。");
        return value;
    }

    internal async Task<HuggingFaceRepository> GetRepositoryAsync(string input,
        string? token, CancellationToken ct)
    {
        string id = ParseRepositoryId(input);
        using var request = new HttpRequestMessage(HttpMethod.Get,
            new Uri($"https://huggingface.co/api/models/{id}?blobs=true"));
        AddToken(request, token);
        using HttpResponseMessage response = await client.SendAsync(request,
            HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
        await EnsureSuccessAsync(response, ct).ConfigureAwait(false);
        await using Stream stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
        using JsonDocument json = await JsonDocument.ParseAsync(stream,
            new JsonDocumentOptions { MaxDepth = 32 }, ct).ConfigureAwait(false);
        JsonElement root = json.RootElement;
        string commit = root.GetProperty("sha").GetString() ?? "";
        if (!HexSha.IsMatch(commit))
            throw new InvalidDataException("リポジトリのコミットSHAを取得できませんでした。");
        var files = new List<HuggingFaceArtifact>();
        if (root.TryGetProperty("siblings", out JsonElement siblings) && siblings.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement sibling in siblings.EnumerateArray())
            {
                if (!sibling.TryGetProperty("rfilename", out JsonElement nameElement)
                    || nameElement.ValueKind != JsonValueKind.String) continue;
                string path = nameElement.GetString() ?? "";
                if (!IsSafeRelativePath(path) || !TryClassify(path, out HuggingFaceArtifactKind kind)) continue;
                long? size = ReadNonnegativeLong(sibling, "size");
                string? sha256 = null;
                if (sibling.TryGetProperty("lfs", out JsonElement lfs) && lfs.ValueKind == JsonValueKind.Object)
                {
                    size ??= ReadNonnegativeLong(lfs, "size");
                    sha256 = ReadString(lfs, "sha256");
                }
                if (sha256 is not null && (sha256.Length != 64 || !sha256.All(Uri.IsHexDigit)))
                    sha256 = null;
                files.Add(new HuggingFaceArtifact(path, kind, size, sha256));
            }
        }
        return new HuggingFaceRepository(id, commit,
            files.OrderBy(file => file.Kind).ThenBy(file => file.Path, StringComparer.OrdinalIgnoreCase).ToArray());
    }

    internal async Task<string> DownloadAsync(HuggingFaceRepository repository,
        HuggingFaceArtifact artifact, string modelsDirectory, string? token,
        IProgress<HuggingFaceDownloadProgress>? progress, CancellationToken ct)
    {
        if (ParseRepositoryId(repository.Id) != repository.Id || !HexSha.IsMatch(repository.Commit)
            || !repository.Files.Contains(artifact) || !IsSafeRelativePath(artifact.Path)
            || !TryClassify(artifact.Path, out HuggingFaceArtifactKind kind) || kind != artifact.Kind)
            throw new ArgumentException("リポジトリの一覧にないファイルは取得できません。");

        string root = Path.GetFullPath(modelsDirectory);
        string[] idParts = repository.Id.Split('/');
        string target = Path.GetFullPath(Path.Combine(
            [root, "huggingface", idParts[0], idParts[1], .. artifact.Path.Split('/')]));
        if (!Path.IsPathFullyQualified(target)
            || !target.StartsWith(root.TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar,
                StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException("保存先が models フォルダーの外です。");
        string directory = Path.GetDirectoryName(target)!;
        EnsureNoLinks(root, directory);
        Directory.CreateDirectory(directory);
        if (File.Exists(target))
            throw new IOException($"既に存在します: {target}");
        // The partial belongs to this immutable commit. A later repo revision
        // must never append bytes from a different revision of the same name.
        string partial = target + "." + repository.Commit + ".part";
        if (File.Exists(partial) && (File.GetAttributes(partial) & FileAttributes.ReparsePoint) != 0)
            throw new IOException("一時ファイルがリンクです。安全のため使用しません。");
        long offset = File.Exists(partial) ? new FileInfo(partial).Length : 0;
        if (artifact.Size is long knownSize && offset > knownSize)
            throw new InvalidDataException("既存の一時ファイルが公開サイズを超えています。");

        long? expected = artifact.Size;
        if (!(expected is long completeSize && offset == completeSize && offset > 0))
        {
            string encodedPath = string.Join('/', artifact.Path.Split('/').Select(Uri.EscapeDataString));
            using var request = new HttpRequestMessage(HttpMethod.Get,
                new Uri($"https://huggingface.co/{repository.Id}/resolve/{repository.Commit}/{encodedPath}"));
            AddToken(request, token);
            if (offset > 0) request.Headers.Range = new RangeHeaderValue(offset, null);
            using HttpResponseMessage response = await client.SendAsync(request,
                HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false);
            await EnsureSuccessAsync(response, ct).ConfigureAwait(false);
            if (offset > 0 && response.StatusCode == HttpStatusCode.PartialContent)
            {
                if (response.Content.Headers.ContentRange?.From != offset)
                    throw new InvalidDataException("再開位置がサーバー応答と一致しません。");
                if (artifact.Size is long total && response.Content.Headers.ContentRange?.Length is long receivedTotal
                    && total != receivedTotal)
                    throw new InvalidDataException("公開サイズと再開応答のサイズが一致しません。");
            }
            else if (response.StatusCode == HttpStatusCode.OK)
            {
                // A server may ignore Range. Restart this application's partial file instead of appending duplicates.
                offset = 0;
            }
            else
            {
                throw new InvalidDataException("予期しないダウンロード応答です。");
            }
            expected = artifact.Size ?? response.Content.Headers.ContentRange?.Length
                ?? (response.Content.Headers.ContentLength is long length ? checked(offset + length) : null);
            await using (var output = new FileStream(partial, offset == 0 ? FileMode.Create : FileMode.Append,
                FileAccess.Write, FileShare.None, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan))
            await using (Stream input = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false))
            {
                byte[] buffer = ArrayPool<byte>.Shared.Rent(1024 * 1024);
                try
                {
                    progress?.Report(new HuggingFaceDownloadProgress(artifact.Path, offset, expected));
                    int read;
                    while ((read = await input.ReadAsync(buffer, ct).ConfigureAwait(false)) != 0)
                    {
                        await output.WriteAsync(buffer.AsMemory(0, read), ct).ConfigureAwait(false);
                        offset += read;
                        progress?.Report(new HuggingFaceDownloadProgress(artifact.Path, offset, expected));
                    }
                    await output.FlushAsync(ct).ConfigureAwait(false);
                }
                finally { ArrayPool<byte>.Shared.Return(buffer); }
            }
        }
        else progress?.Report(new HuggingFaceDownloadProgress(artifact.Path, offset, expected));
        if (expected is long finalSize && offset != finalSize)
            throw new InvalidDataException($"ファイルサイズが一致しません（{offset}/{finalSize} バイト）。途中ファイルを保持しました。");
        if (offset == 0) throw new InvalidDataException("空のファイルは保存しません。");
        if (expected is null && artifact.Sha256 is null)
            throw new InvalidDataException("ファイルサイズもSHA-256も確認できません。一時ファイルを保持しました。");
        if (artifact.Sha256 is string sha256)
        {
            await using var data = new FileStream(partial, FileMode.Open, FileAccess.Read,
                FileShare.Read, 1024 * 1024, FileOptions.Asynchronous | FileOptions.SequentialScan);
            string actual = Convert.ToHexString(await SHA256.HashDataAsync(data, ct).ConfigureAwait(false));
            if (!actual.Equals(sha256, StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("公開SHA-256と一致しません。一時ファイルを保持しました。");
        }
        if (Path.GetExtension(artifact.Path).Equals(".gguf", StringComparison.OrdinalIgnoreCase))
        {
            await using var data = new FileStream(partial, FileMode.Open, FileAccess.Read,
                FileShare.Read, 4, FileOptions.Asynchronous);
            byte[] magic = new byte[4];
            if (await data.ReadAsync(magic, ct).ConfigureAwait(false) != 4 || !magic.AsSpan().SequenceEqual("GGUF"u8))
                throw new InvalidDataException("GGUFヘッダーがありません。一時ファイルを保持しました。");
        }
        File.Move(partial, target, overwrite: false);
        return target;
    }

    internal static bool IsMmproj(string path) =>
        Path.GetFileName(path).Contains("mmproj", StringComparison.OrdinalIgnoreCase);

    private static bool TryClassify(string path, out HuggingFaceArtifactKind kind)
    {
        string extension = Path.GetExtension(path);
        if (IsMmproj(path) && (extension.Equals(".gguf", StringComparison.OrdinalIgnoreCase)
            || extension.Equals(".bin", StringComparison.OrdinalIgnoreCase)))
        {
            kind = HuggingFaceArtifactKind.Mmproj;
            return true;
        }
        kind = HuggingFaceArtifactKind.Model;
        return extension.Equals(".gguf", StringComparison.OrdinalIgnoreCase)
            && !Path.GetFileName(path).StartsWith("lora_", StringComparison.OrdinalIgnoreCase);
    }

    private static bool IsSafeRelativePath(string path) =>
        path.Length is > 0 and < 1024 && path.Split('/').All(IsSafeFileSegment);

    private static bool IsSafeRepositoryPart(string segment) =>
        IsSafeFileSegment(segment) && RepositoryPart.IsMatch(segment);

    private static bool IsSafeFileSegment(string segment) =>
        segment.Length is > 0 and <= 255 && segment is not ("." or "..")
        && segment[^1] is not ('.' or ' ')
        && !WindowsReservedNames.Contains(segment.Split('.')[0])
        && !segment.Any(c => c < 32 || c is '/' or '\\' or ':' or '<' or '>' or '"' or '|' or '?' or '*');

    private static void EnsureNoLinks(string root, string directory)
    {
        string current = root;
        if (Directory.Exists(current) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
            throw new IOException("models フォルダーがリンクです。安全のため保存しません。");
        string relative = Path.GetRelativePath(root, directory);
        foreach (string part in relative.Split(Path.DirectorySeparatorChar))
        {
            current = Path.Combine(current, part);
            if (Directory.Exists(current) && (File.GetAttributes(current) & FileAttributes.ReparsePoint) != 0)
                throw new IOException("保存先にディレクトリリンクが含まれています。");
        }
    }

    private static void AddToken(HttpRequestMessage request, string? token)
    {
        if (!string.IsNullOrWhiteSpace(token))
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token.Trim());
    }

    private static string? ReadString(JsonElement item, string property) =>
        item.TryGetProperty(property, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() : null;

    private static long? ReadNonnegativeLong(JsonElement item, string property) =>
        item.TryGetProperty(property, out JsonElement value) && value.ValueKind == JsonValueKind.Number
        && value.TryGetInt64(out long number) && number >= 0
            ? number : null;

    private static async Task EnsureSuccessAsync(HttpResponseMessage response, CancellationToken ct)
    {
        if (response.IsSuccessStatusCode) return;
        string detail = response.StatusCode switch
        {
            HttpStatusCode.Unauthorized or HttpStatusCode.Forbidden => "アクセスできません。制限付きモデルでは利用許諾とHugging Faceトークンを確認してください。",
            HttpStatusCode.NotFound => "リポジトリまたはファイルが見つかりません。",
            (HttpStatusCode)429 => "Hugging Faceのアクセス制限に達しました。時間をおいて再試行してください。",
            _ => $"HTTP {(int)response.StatusCode}"
        };
        await Task.CompletedTask;
        ct.ThrowIfCancellationRequested();
        throw new HttpRequestException(detail, null, response.StatusCode);
    }
}
