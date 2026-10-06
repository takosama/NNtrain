using System.IO;
using System.Runtime.InteropServices;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace NNtrain.Gui;

internal static class NativeImageDecoder
{
    public static Qwen35VisionInput Prepare(ChatImage image, int patchSize, int mergeSize)
    {
        // ChatImage owns an immutable array. Decode directly from its read-only
        // stream rather than cloning up to 16 MiB for every new image.
        using var stream = MemoryMarshal.TryGetArray(image.Bytes, out ArraySegment<byte> encoded)
            ? new MemoryStream(encoded.Array!, encoded.Offset, encoded.Count,
                writable: false, publiclyVisible: false)
            : new MemoryStream(image.Bytes.ToArray(), writable: false);
        BitmapFrame frame = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat,
            BitmapCacheOption.OnLoad).Frames[0];
        int width = frame.PixelWidth, height = frame.PixelHeight;
        if (width <= 0 || height <= 0 || (long)width * height > 40_000_000)
            throw new ArgumentException("画像は 4000 万画素以下にしてください。");
        var rgb = new FormatConvertedBitmap(frame, PixelFormats.Rgb24, null, 0);
        int stride = checked(width * 3);
        var pixels = new byte[checked(stride * height)];
        rgb.CopyPixels(pixels, stride, 0);
        return Qwen35VisionPreprocessor.Prepare(pixels, width, height, patchSize, mergeSize);
    }
}
