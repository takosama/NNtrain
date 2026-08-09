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
    private static readonly float[] ChannelMeans =
        [0.50707516f, 0.48654887f, 0.44091784f];
    private static readonly float[] ChannelStandardDeviations =
        [0.26733429f, 0.25643846f, 0.27615047f];

    private readonly byte[] _data;
    private readonly Cifar100Options _options;

    public Cifar100(
        string dataPath,
        Cifar100Options? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataPath);

        _options = options ?? new Cifar100Options();
        if (_options.PatchSize <= 0
            || ImageRows % _options.PatchSize != 0
            || ImageColumns % _options.PatchSize != 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                _options.PatchSize,
                $"CIFAR-100 patch size must be a positive divisor of " +
                $"{ImageRows}.");
        }

        if (_options.RandomCropPadding < 0
            || _options.RandomCropPadding > ImageRows)
        {
            throw new ArgumentOutOfRangeException(
                nameof(options),
                _options.RandomCropPadding,
                $"CIFAR-100 random crop padding must be between 0 and " +
                $"{ImageRows}.");
        }

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

    public int Rows
        => (ImageRows / _options.PatchSize)
            * (ImageColumns / _options.PatchSize);

    public int Columns
        => _options.PatchSize * _options.PatchSize * ChannelCount;

    public int ImageSize => ChannelCount * ChannelSize;

    public int ClassCount => ExpectedClassCount;

    public Cifar100Options Options => _options with { };

    public int ReadSample(int index, Span<float> destination)
        => ReadTransformedSample(
            index,
            destination,
            cropTop: _options.RandomCropPadding,
            cropLeft: _options.RandomCropPadding,
            flipHorizontal: false,
            flipVertical: false);

    public int ReadTrainingSample(
        int index,
        Span<float> destination,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(random);
        ValidateReadArguments(index, destination);

        int cropRange = 2 * _options.RandomCropPadding + 1;
        int cropTop = random.Next(cropRange);
        int cropLeft = random.Next(cropRange);
        bool flipHorizontal =
            _options.HorizontalFlip && random.Next(2) == 1;
        bool flipVertical =
            _options.VerticalFlip && random.Next(2) == 1;

        return ReadTransformedSample(
            index,
            destination,
            cropTop,
            cropLeft,
            flipHorizontal,
            flipVertical);
    }

    private int ReadTransformedSample(
        int index,
        Span<float> destination,
        int cropTop,
        int cropLeft,
        bool flipHorizontal,
        bool flipVertical)
    {
        ValidateReadArguments(index, destination);

        int recordOffset = index * RecordSize;
        int label = ReadFineLabel(index, recordOffset);
        int imageOffset = recordOffset + ImageOffset;

        int patchSize = _options.PatchSize;
        int patchRows = ImageRows / patchSize;
        int patchColumns = ImageColumns / patchSize;
        int destinationIndex = 0;

        for (int patchRow = 0; patchRow < patchRows; patchRow++)
        {
            for (int patchColumn = 0;
                patchColumn < patchColumns;
                patchColumn++)
            {
                for (int channel = 0; channel < ChannelCount; channel++)
                {
                    for (int rowInPatch = 0;
                        rowInPatch < patchSize;
                        rowInPatch++)
                    {
                        int outputRow = patchRow * patchSize + rowInPatch;
                        int croppedRow = flipVertical
                            ? ImageRows - 1 - outputRow
                            : outputRow;
                        int sourceRow = croppedRow
                            + cropTop
                            - _options.RandomCropPadding;

                        for (int columnInPatch = 0;
                            columnInPatch < patchSize;
                            columnInPatch++)
                        {
                            int outputColumn =
                                patchColumn * patchSize + columnInPatch;
                            int croppedColumn = flipHorizontal
                                ? ImageColumns - 1 - outputColumn
                                : outputColumn;
                            int sourceColumn = croppedColumn
                                + cropLeft
                                - _options.RandomCropPadding;

                            if ((uint)sourceRow >= ImageRows
                                || (uint)sourceColumn >= ImageColumns)
                            {
                                destination[destinationIndex++] =
                                    ConvertPixel(value: 0, channel);
                                continue;
                            }

                            int sourceIndex = imageOffset
                                + channel * ChannelSize
                                + sourceRow * ImageColumns
                                + sourceColumn;
                            destination[destinationIndex++] = ConvertPixel(
                                _data[sourceIndex],
                                channel);
                        }
                    }
                }
            }
        }

        return label;
    }

    private void ValidateReadArguments(int index, Span<float> destination)
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
    }

    private int ReadFineLabel(int index, int recordOffset)
    {
        int label = _data[recordOffset + FineLabelOffset];
        if ((uint)label >= (uint)ClassCount)
        {
            throw new InvalidDataException(
                $"CIFAR-100 fine label '{label}' at index '{index}' is " +
                $"outside the valid range 0..{ClassCount - 1}.");
        }

        return label;
    }

    private float ConvertPixel(byte value, int channel)
    {
        float scaled = value / 255f;
        if (!_options.Normalize)
            return scaled;

        return (scaled - ChannelMeans[channel])
            / ChannelStandardDeviations[channel];
    }
}
