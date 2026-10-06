using System.Security.Cryptography;

namespace NNtrain.Gui;

/// <summary>An encoded local image. No remote image URL is fetched by inference.</summary>
public sealed class ChatImage
{
    public const int MaximumBytes = 16 * 1024 * 1024;
    private readonly byte[] _bytes;
    public string Name { get; }
    public string MimeType { get; }
    public string Hash { get; }
    public ReadOnlyMemory<byte> Bytes => _bytes;

    public ChatImage(byte[] bytes, string name = "image")
    {
        ArgumentNullException.ThrowIfNull(bytes);
        if (bytes.Length is < 3 or > MaximumBytes)
            throw new ArgumentException("画像は 16 MiB 以下の PNG / JPEG を選んでください。", nameof(bytes));
        MimeType = bytes.AsSpan().StartsWith(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }) ? "image/png"
            : bytes[0] == 255 && bytes[1] == 216 && bytes[2] == 255 ? "image/jpeg"
            : throw new ArgumentException("PNG / JPEG 画像が必要です。", nameof(bytes));
        _bytes = (byte[])bytes.Clone();
        Name = name;
        Hash = Convert.ToHexStringLower(SHA256.HashData(_bytes));
    }

    public string ToDataUrl() => $"data:{MimeType};base64,{Convert.ToBase64String(_bytes)}";

    public static ChatImage FromDataUrl(string url)
    {
        ArgumentNullException.ThrowIfNull(url);
        const string png = "data:image/png;base64,";
        const string jpeg = "data:image/jpeg;base64,";
        string prefix = url.StartsWith(png, StringComparison.OrdinalIgnoreCase) ? png
            : url.StartsWith(jpeg, StringComparison.OrdinalIgnoreCase) ? jpeg
            : throw new ArgumentException("image_url はローカル画像の PNG / JPEG data URL が必要です。");
        if (url.Length - prefix.Length > ((MaximumBytes + 2) / 3) * 4)
            throw new ArgumentException("画像は 16 MiB 以下にしてください。");
        var image = new ChatImage(Convert.FromBase64String(url[prefix.Length..]));
        if (!string.Equals(image.MimeType, prefix[5..prefix.IndexOf(';')], StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("画像の形式と data URL が一致しません。");
        return image;
    }
}
