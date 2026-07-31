namespace NNtrain;

public class Mnist : IImageClassificationDataset
{
    private readonly byte[] _imageData;
    private readonly byte[] _labelData;

    private const int ImageMagicNumber = 2051;
    private const int LabelMagicNumber = 2049;
    private const int ImageOffset = 16;
    private const int LabelOffset = 8;

    private const int ExpectedRows = 28;
    private const int ExpectedColumns = 28;
    private const int ExpectedImageSize = ExpectedRows * ExpectedColumns;
    private const int ExpectedClassCount = 10;

    public Mnist(string imagePath, string labelPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(imagePath);
        ArgumentException.ThrowIfNullOrWhiteSpace(labelPath);

        _imageData = File.ReadAllBytes(imagePath);
        _labelData = File.ReadAllBytes(labelPath);

        Count = ValidateHeaders(_imageData, _labelData);
    }

    public int Count { get; }
    public int Rows => ExpectedRows;
    public int Columns => ExpectedColumns;
    public int ImageSize => ExpectedImageSize;
    public int ClassCount => ExpectedClassCount;

    public int ReadSample(int index, Span<float> destination)
    {
        if ((uint)index >= (uint)Count)
        {
            throw new ArgumentOutOfRangeException(
                nameof(index),
                index,
                $"Sample index must be between 0 and {Count - 1}.");
        }

        if (destination.Length < ImageSize)
        {
            throw new ArgumentException(
                $"Image destination must contain at least {ImageSize} " +
                "elements.",
                nameof(destination));
        }

        int offset = ImageOffset + index * ImageSize;
        for (int i = 0; i < ImageSize; i++)
            destination[i] = _imageData[offset + i] / 255.0f;

        int label = _labelData[LabelOffset + index];
        if ((uint)label >= (uint)ClassCount)
        {
            throw new InvalidDataException(
                $"MNIST label '{label}' at index '{index}' is outside the " +
                $"valid range 0..{ClassCount - 1}.");
        }

        return label;
    }

    private static int ValidateHeaders(byte[] images, byte[] labels)
    {
        if (images.Length < ImageOffset)
        {
            throw new InvalidDataException(
                $"Image IDX header must be at least {ImageOffset} bytes.");
        }

        if (labels.Length < LabelOffset)
        {
            throw new InvalidDataException(
                $"Label IDX header must be at least {LabelOffset} bytes.");
        }

        int imageMagic = ReadInt32(images, 0);
        if (imageMagic != ImageMagicNumber)
        {
            throw new InvalidDataException(
                $"Invalid image IDX magic number '{imageMagic}'. Expected " +
                $"'{ImageMagicNumber}'.");
        }

        int labelMagic = ReadInt32(labels, 0);
        if (labelMagic != LabelMagicNumber)
        {
            throw new InvalidDataException(
                $"Invalid label IDX magic number '{labelMagic}'. Expected " +
                $"'{LabelMagicNumber}'.");
        }

        int imageCount = ReadInt32(images, 4);
        int labelCount = ReadInt32(labels, 4);
        if (imageCount <= 0 || labelCount <= 0)
        {
            throw new InvalidDataException(
                "Image and label IDX counts must both be positive.");
        }

        if (imageCount != labelCount)
        {
            throw new InvalidDataException(
                $"Image IDX count '{imageCount}' does not match label IDX " +
                $"count '{labelCount}'.");
        }

        int rows = ReadInt32(images, 8);
        int columns = ReadInt32(images, 12);
        if (rows != ExpectedRows || columns != ExpectedColumns)
        {
            throw new InvalidDataException(
                $"MNIST images must be {ExpectedRows}x{ExpectedColumns}, " +
                "but the IDX header " +
                $"declares {rows}x{columns}.");
        }

        long expectedImageLength =
            ImageOffset + (long)imageCount * ExpectedImageSize;
        if (images.LongLength != expectedImageLength)
        {
            throw new InvalidDataException(
                $"Image IDX length '{images.LongLength}' does not match the " +
                $"declared count. Expected '{expectedImageLength}' bytes.");
        }

        long expectedLabelLength = LabelOffset + (long)labelCount;
        if (labels.LongLength != expectedLabelLength)
        {
            throw new InvalidDataException(
                $"Label IDX length '{labels.LongLength}' does not match the " +
                $"declared count. Expected '{expectedLabelLength}' bytes.");
        }

        return imageCount;
    }

    private static int ReadInt32(byte[] data, int offset)
    {
        return System.Buffers.Binary.BinaryPrimitives.ReadInt32BigEndian(
            data.AsSpan(offset, sizeof(int)));
    }
}
