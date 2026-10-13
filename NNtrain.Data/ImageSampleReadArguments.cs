namespace NNtrain;

internal static class ImageSampleReadArguments
{
    internal static void Validate(
        int index,
        int count,
        int imageSize,
        Span<float> destination)
    {
        if ((uint)index >= (uint)count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                index,
                $"Sample index must be between 0 and {count - 1}.");
        }

        if (destination.Length < imageSize)
        {
            throw new ArgumentException(
                $"Image destination must contain at least {imageSize} " +
                "elements.",
                nameof(destination));
        }
    }
}
