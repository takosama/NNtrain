namespace NNtrain;

/// <summary>
/// Flattened spatial patches in Qwen's merged-block order. Grid dimensions count
/// patches before the 2x2 spatial merge, not text tokens after the merge.
/// </summary>
public sealed record Qwen35VisionInput(float[] Patches, int GridHeight, int GridWidth);

/// <summary>Vision features in merged-grid row order.</summary>
public sealed record Qwen35VisionEmbedding(float[] Values, int EmbeddingLength, int GridHeight, int GridWidth);

public static class Qwen35VisionPreprocessor
{
    // Qwen3-VL's image processor config uses 256x256 as its minimum image area.
    private const int MinimumPixels = 256 * 256;

    /// <summary>
    /// Converts packed, row-major RGB bytes into spatial patches. Within each
    /// merged block, patches are top-left, top-right, bottom-left, bottom-right;
    /// within each patch, values are channel, row, column. Static images need
    /// one spatial copy: the encoder handles the two equal temporal planes.
    /// </summary>
    public static Qwen35VisionInput Prepare(byte[] rgb, int width, int height,
        int patchSize = 16, int mergeSize = 2, int maximumPixels = 768 * 768)
        => Prepare(rgb, width, height, patchSize, mergeSize, maximumPixels,
            Math.Min(8, Environment.ProcessorCount));

    // A single-worker route supports pixel-for-pixel checks of the independent
    // row work. Keep the worker count bounded: decoded images and filter
    // intermediates are shared, rather than duplicated for each worker.
    internal static Qwen35VisionInput Prepare(byte[] rgb, int width, int height,
        int patchSize, int mergeSize, int maximumPixels, int maximumDegreeOfParallelism)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (patchSize <= 0) throw new ArgumentOutOfRangeException(nameof(patchSize));
        if (mergeSize <= 0) throw new ArgumentOutOfRangeException(nameof(mergeSize));
        if (maximumPixels <= 0) throw new ArgumentOutOfRangeException(nameof(maximumPixels));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maximumDegreeOfParallelism);
        long inputPixels = (long)width * height;
        if (rgb.Length % 3 != 0 || inputPixels != rgb.Length / 3)
            throw new ArgumentException("RGB data length must equal width × height × 3.", nameof(rgb));

        long factorLong = (long)patchSize * mergeSize;
        if (factorLong > int.MaxValue || factorLong * factorLong > maximumPixels)
            throw new ArgumentOutOfRangeException(nameof(maximumPixels),
                "The pixel cap must fit at least one merged patch block.");
        int factor = (int)factorLong;
        (int resizedHeight, int resizedWidth) = SmartResize(height, width, factor, maximumPixels);
        long outputPixels = (long)resizedHeight * resizedWidth;
        long valuesLength = outputPixels * 3;
        if (valuesLength > Array.MaxLength)
            throw new ArgumentOutOfRangeException(nameof(maximumPixels), "The patch tensor is too large.");

        byte[] resized = resizedHeight == height && resizedWidth == width
            ? rgb : ResizeBicubic(rgb, width, height, resizedWidth, resizedHeight,
                maximumDegreeOfParallelism);
        int gridHeight = resizedHeight / patchSize;
        int gridWidth = resizedWidth / patchSize;
        float[] patches = new float[(int)valuesLength];
        ForRows(gridHeight / mergeSize, outputPixels, maximumDegreeOfParallelism, groupRow =>
        {
            int groupY = groupRow * mergeSize;
            int cursor = checked(groupRow * gridWidth * mergeSize * patchSize * patchSize * 3);
            for (int groupX = 0; groupX < gridWidth; groupX += mergeSize)
            for (int localY = 0; localY < mergeSize; localY++)
            for (int localX = 0; localX < mergeSize; localX++)
            for (int channel = 0; channel < 3; channel++)
            for (int patchY = 0; patchY < patchSize; patchY++)
            for (int patchX = 0; patchX < patchSize; patchX++)
            {
                int pixelY = (groupY + localY) * patchSize + patchY;
                int pixelX = (groupX + localX) * patchSize + patchX;
                byte value = resized[(pixelY * resizedWidth + pixelX) * 3 + channel];
                patches[cursor++] = value * (2f / 255f) - 1f;
            }
        });
        return new Qwen35VisionInput(patches, gridHeight, gridWidth);
    }

    private static (int Height, int Width) SmartResize(int height, int width, int factor, int maximumPixels)
    {
        if ((double)Math.Max(height, width) / Math.Min(height, width) > 200)
            throw new ArgumentOutOfRangeException(nameof(width), "Image aspect ratio exceeds Qwen's 200:1 limit.");

        // Qwen smart_resize: round both sides to patch*merge, then scale the
        // original aspect ratio if the rounded area lies outside the budget.
        int resizedHeight = checked((int)Math.Round((double)height / factor) * factor);
        int resizedWidth = checked((int)Math.Round((double)width / factor) * factor);
        int minimumPixels = (int)Math.Max((long)factor * factor, Math.Min(MinimumPixels, maximumPixels));
        long roundedArea = (long)resizedHeight * resizedWidth;
        if (roundedArea > maximumPixels)
        {
            double beta = Math.Sqrt((double)height * width / maximumPixels);
            resizedHeight = Math.Max(factor, checked((int)Math.Floor(height / beta / factor) * factor));
            resizedWidth = Math.Max(factor, checked((int)Math.Floor(width / beta / factor) * factor));
        }
        else if (roundedArea < minimumPixels)
        {
            double beta = Math.Sqrt((double)minimumPixels / ((long)height * width));
            resizedHeight = checked((int)Math.Ceiling(height * beta / factor) * factor);
            resizedWidth = checked((int)Math.Ceiling(width * beta / factor) * factor);
        }

        // Rounding the two axes independently can exceed a caller's tight cap.
        // Keep the cap strict while changing the source aspect as little as the
        // required factor permits.
        double sourceAspect = (double)width / height;
        while ((long)resizedHeight * resizedWidth > maximumPixels)
        {
            bool canReduceHeight = resizedHeight > factor;
            bool canReduceWidth = resizedWidth > factor;
            if (!canReduceHeight && !canReduceWidth)
                throw new ArgumentOutOfRangeException(nameof(maximumPixels));
            double heightError = canReduceHeight
                ? Math.Abs(Math.Log((double)resizedWidth / (resizedHeight - factor) / sourceAspect))
                : double.PositiveInfinity;
            double widthError = canReduceWidth
                ? Math.Abs(Math.Log((double)(resizedWidth - factor) / resizedHeight / sourceAspect))
                : double.PositiveInfinity;
            if (heightError <= widthError) resizedHeight -= factor;
            else resizedWidth -= factor;
        }
        if (resizedHeight <= 0 || resizedWidth <= 0)
            throw new ArgumentOutOfRangeException(nameof(maximumPixels));
        return (resizedHeight, resizedWidth);
    }

    private static byte[] ResizeBicubic(byte[] rgb, int sourceWidth, int sourceHeight,
        int targetWidth, int targetHeight, int maximumDegreeOfParallelism)
    {
        // Torchvision's antialiased bicubic resize uses pixel-center coordinates
        // and the Pillow cubic kernel (a=-0.5). Compute the two axes separately
        // to keep downsampling work linear in image size.
        Contribution[] xWeights = BuildContributions(sourceWidth, targetWidth);
        Contribution[] yWeights = BuildContributions(sourceHeight, targetHeight);
        float[] horizontal = new float[checked(sourceHeight * targetWidth * 3)];
        ForRows(sourceHeight, (long)sourceHeight * targetWidth, maximumDegreeOfParallelism, y =>
        {
            for (int x = 0; x < targetWidth; x++)
            {
                Contribution weights = xWeights[x];
                int output = (y * targetWidth + x) * 3;
                float red = 0, green = 0, blue = 0;
                for (int tap = 0; tap < weights.Indices.Length; tap++)
                {
                    int input = (y * sourceWidth + weights.Indices[tap]) * 3;
                    float weight = weights.Weights[tap];
                    red += rgb[input] * weight;
                    green += rgb[input + 1] * weight;
                    blue += rgb[input + 2] * weight;
                }
                horizontal[output] = red;
                horizontal[output + 1] = green;
                horizontal[output + 2] = blue;
            }
        });

        byte[] resized = new byte[checked(targetWidth * targetHeight * 3)];
        ForRows(targetHeight, (long)targetHeight * targetWidth, maximumDegreeOfParallelism, y =>
        {
            Contribution weights = yWeights[y];
            for (int x = 0; x < targetWidth; x++)
            {
                int output = (y * targetWidth + x) * 3;
                float red = 0, green = 0, blue = 0;
                for (int tap = 0; tap < weights.Indices.Length; tap++)
                {
                    int input = (weights.Indices[tap] * targetWidth + x) * 3;
                    float weight = weights.Weights[tap];
                    red += horizontal[input] * weight;
                    green += horizontal[input + 1] * weight;
                    blue += horizontal[input + 2] * weight;
                }
                resized[output] = ToByte(red);
                resized[output + 1] = ToByte(green);
                resized[output + 2] = ToByte(blue);
            }
        });
        return resized;
    }

    private static void ForRows(int count, long pixels, int maximumDegreeOfParallelism, Action<int> body)
    {
        // Small images stay sequential so scheduling overhead does not slow
        // thumbnails. Independent rows retain each pixel's tap/reduction order.
        if (maximumDegreeOfParallelism > 1 && count >= 4 && pixels >= 128_000)
            Parallel.For(0, count, new ParallelOptions
            {
                MaxDegreeOfParallelism = maximumDegreeOfParallelism
            }, body);
        else
            for (int row = 0; row < count; row++) body(row);
    }

    private static Contribution[] BuildContributions(int sourceLength, int targetLength)
    {
        var contributions = new Contribution[targetLength];
        if (sourceLength == targetLength)
        {
            for (int i = 0; i < targetLength; i++)
                contributions[i] = new Contribution([i], [1f]);
            return contributions;
        }
        double scale = (double)sourceLength / targetLength;
        double filterScale = Math.Max(1, scale);
        double radius = 2 * filterScale;
        for (int output = 0; output < targetLength; output++)
        {
            double center = (output + 0.5) * scale - 0.5;
            int first = Math.Max(0, (int)Math.Ceiling(center - radius));
            int last = Math.Min(sourceLength - 1, (int)Math.Floor(center + radius));
            int count = last - first + 1;
            var indices = new int[count];
            var weights = new float[count];
            double sum = 0;
            for (int tap = 0; tap < count; tap++)
            {
                indices[tap] = first + tap;
                double weight = Cubic((indices[tap] - center) / filterScale);
                weights[tap] = (float)weight;
                sum += weight;
            }
            for (int tap = 0; tap < count; tap++) weights[tap] = (float)(weights[tap] / sum);
            contributions[output] = new Contribution(indices, weights);
        }
        return contributions;
    }

    private static double Cubic(double value)
    {
        double x = Math.Abs(value);
        if (x < 1) return (1.5 * x - 2.5) * x * x + 1;
        if (x < 2) return ((-0.5 * x + 2.5) * x - 4) * x + 2;
        return 0;
    }

    private static byte ToByte(float value) => (byte)Math.Clamp((int)MathF.Round(value), 0, 255);

    private sealed record Contribution(int[] Indices, float[] Weights);
}
