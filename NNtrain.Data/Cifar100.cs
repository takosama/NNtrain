namespace NNtrain;

public class Cifar100 : IImageClassificationDataset
{
    private const int ImageRows = 32;
    private const int ImageColumns = 32;
    private const int ChannelCount = 3;
    private const int ChannelSize = ImageRows * ImageColumns;
    private const int LabelBytes = 2;
    private const int RecordSize = LabelBytes + ChannelCount * ChannelSize;
    private const int FineLabelOffset = 1;
    private const int ImageOffset = LabelBytes;
    private const int ExpectedClassCount = 100;

    private readonly byte[] _data;

    public Cifar100(string dataPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataPath);

        _data = File.ReadAllBytes(dataPath);
        if (_data.Length == 0 || _data.Length % RecordSize != 0)
        {
            throw new InvalidDataException(
                $"CIFAR-100 binary length '{_data.Length}' must be a " +
                $"positive multiple of the {RecordSize}-byte record size.");
        }

        Count = _data.Length / RecordSize;
    }

    public int Count { get; }

    public int Rows => ImageRows;

    public int Columns => ImageColumns * ChannelCount;

    public int ImageSize => ChannelCount * ChannelSize;

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

        int recordOffset = index * RecordSize;
        int label = _data[recordOffset + FineLabelOffset];
        if ((uint)label >= (uint)ClassCount)
        {
            throw new InvalidDataException(
                $"CIFAR-100 fine label '{label}' at index '{index}' is " +
                $"outside the valid range 0..{ClassCount - 1}.");
        }

        int imageOffset = recordOffset + ImageOffset;
        for (int row = 0; row < ImageRows; row++)
        {
            int sourceRow = imageOffset + row * ImageColumns;
            int destinationRow = row * Columns;

            for (int column = 0; column < ImageColumns; column++)
            {
                int sourcePixel = sourceRow + column;
                destination[destinationRow + column] =
                    _data[sourcePixel] / 255f;
                destination[destinationRow + ImageColumns + column] =
                    _data[sourcePixel + ChannelSize] / 255f;
                destination[
                    destinationRow + 2 * ImageColumns + column] =
                    _data[sourcePixel + 2 * ChannelSize] / 255f;
            }
        }

        return label;
    }
}
