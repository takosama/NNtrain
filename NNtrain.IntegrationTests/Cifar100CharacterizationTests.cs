using NNtrain;
using Xunit;

public sealed class Cifar100CharacterizationTests
{
    private const int Rows = 32;
    private const int ColumnsPerChannel = 32;
    private const int ChannelSize = Rows * ColumnsPerChannel;
    private const int ImageSize = 3 * ChannelSize;
    private const int RecordSize = 2 + ImageSize;

    [Fact]
    public void ReadsFineLabelAndArrangesImageAsSixtyFourFourByFourPatches()
    {
        using var directory = new TemporaryDirectory();
        var record = new byte[RecordSize];
        record[0] = 19;
        record[1] = 99;
        record[2] = 10;
        record[2 + ChannelSize] = 20;
        record[2 + 2 * ChannelSize] = 30;

        int lastPixel = ChannelSize - 1;
        record[2 + lastPixel] = 40;
        record[2 + ChannelSize + lastPixel] = 50;
        record[2 + 2 * ChannelSize + lastPixel] = 60;

        string path = directory.WriteData(record);
        IImageClassificationDataset dataset = new Cifar100(path);
        var destination = new float[dataset.ImageSize];

        int label = dataset.ReadSample(0, destination);

        Assert.Equal(99, label);
        Assert.Equal(1, dataset.Count);
        Assert.Equal(64, dataset.Rows);
        Assert.Equal(48, dataset.Columns);
        Assert.Equal(3072, dataset.ImageSize);
        Assert.Equal(100, dataset.ClassCount);
        AssertClose(10f / 255f, destination[0]);
        AssertClose(20f / 255f, destination[16]);
        AssertClose(30f / 255f, destination[32]);
        AssertClose(40f / 255f, destination[63 * 48 + 15]);
        AssertClose(50f / 255f, destination[63 * 48 + 31]);
        AssertClose(60f / 255f, destination[63 * 48 + 47]);
    }

    [Fact]
    public void TrainingReadAppliesHorizontalAndVerticalFlipsToEveryChannel()
    {
        using var directory = new TemporaryDirectory();
        var record = new byte[RecordSize];
        record[1] = 7;
        SetPixel(record, row: 31, column: 31, red: 11, green: 22, blue: 33);
        SetPixel(record, row: 0, column: 0, red: 44, green: 55, blue: 66);
        IImageClassificationDataset dataset = new Cifar100(
            directory.WriteData(record),
            new Cifar100Options
            {
                HorizontalFlip = true,
                VerticalFlip = true,
            });
        var destination = new float[dataset.ImageSize];
        var random = new SequenceRandom(4, 4, 1, 1);

        int label = dataset.ReadTrainingSample(0, destination, random);

        Assert.Equal(7, label);
        AssertRgb(destination, 0, 0, 11, 22, 33);
        AssertRgb(destination, 31, 31, 44, 55, 66);
    }

    [Fact]
    public void TrainingRandomCropZeroPadsPixelsOutsideTheImage()
    {
        using var directory = new TemporaryDirectory();
        var record = new byte[RecordSize];
        record[1] = 8;
        SetPixel(record, row: 0, column: 0, red: 21, green: 42, blue: 84);
        IImageClassificationDataset dataset = new Cifar100(
            directory.WriteData(record));
        var destination = new float[dataset.ImageSize];
        var random = new SequenceRandom(0, 0, 0, 0);

        dataset.ReadTrainingSample(0, destination, random);

        AssertRgb(destination, 0, 0, 0, 0, 0);
        AssertRgb(destination, 4, 4, 21, 42, 84);
    }

    [Fact]
    public void NormalizationUsesCifar100TrainingChannelStatistics()
    {
        using var directory = new TemporaryDirectory();
        var record = new byte[RecordSize];
        record[1] = 9;
        SetPixel(record, row: 0, column: 0, red: 0, green: 128, blue: 255);
        IImageClassificationDataset dataset = new Cifar100(
            directory.WriteData(record),
            new Cifar100Options
            {
                Normalize = true,
                RandomCropPadding = 0,
                HorizontalFlip = false,
                VerticalFlip = false,
            });
        var destination = new float[dataset.ImageSize];

        dataset.ReadSample(0, destination);

        AssertClose(
            (0f - 0.50707516f) / 0.26733429f,
            destination[0]);
        AssertClose(
            (128f / 255f - 0.48654887f) / 0.25643846f,
            destination[16]);
        AssertClose(
            (1f - 0.44091784f) / 0.27615047f,
            destination[32]);
    }

    [Fact]
    public void CountsEveryBinaryRecord()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.WriteData(new byte[2 * RecordSize]);

        var dataset = new Cifar100(path);

        Assert.Equal(2, dataset.Count);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(33)]
    public void RejectsPatchSizesThatDoNotDivideTheImage(int patchSize)
    {
        using var directory = new TemporaryDirectory();
        string path = directory.WriteData(new byte[RecordSize]);

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => new Cifar100(
                path,
                new Cifar100Options { PatchSize = patchSize }));

        Assert.Equal("options", exception.ParamName);
        Assert.Contains("patch size", exception.Message);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3073)]
    [InlineData(3075)]
    public void RejectsInvalidBinaryLengths(int length)
    {
        using var directory = new TemporaryDirectory();
        string path = directory.WriteData(new byte[length]);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Cifar100(path));

        Assert.Contains("3074-byte record size", exception.Message);
    }

    [Fact]
    public void RejectsFineLabelOutsideTheClassRange()
    {
        using var directory = new TemporaryDirectory();
        var record = new byte[RecordSize];
        record[1] = 100;
        var dataset = new Cifar100(directory.WriteData(record));
        var destination = new float[dataset.ImageSize];

        var exception = Assert.Throws<InvalidDataException>(
            () => dataset.ReadSample(0, destination));

        Assert.Contains("fine label", exception.Message);
        Assert.Contains("valid range", exception.Message);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    public void ReadSampleRejectsAnOutOfRangeIndex(int index)
    {
        using var directory = new TemporaryDirectory();
        var dataset = new Cifar100(
            directory.WriteData(new byte[RecordSize]));
        var destination = new float[dataset.ImageSize];

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => dataset.ReadSample(index, destination));

        Assert.Equal("index", exception.ParamName);
    }

    [Fact]
    public void ReadSampleRejectsAShortDestination()
    {
        using var directory = new TemporaryDirectory();
        var dataset = new Cifar100(
            directory.WriteData(new byte[RecordSize]));
        var destination = new float[dataset.ImageSize - 1];

        var exception = Assert.Throws<ArgumentException>(
            () => dataset.ReadSample(0, destination));

        Assert.Equal("destination", exception.ParamName);
    }

    private static void AssertClose(
        float expected,
        float actual,
        float tolerance = 1e-5f)
    {
        Assert.InRange(actual, expected - tolerance, expected + tolerance);
    }

    private static void SetPixel(
        byte[] record,
        int row,
        int column,
        byte red,
        byte green,
        byte blue)
    {
        int pixel = row * ColumnsPerChannel + column;
        record[2 + pixel] = red;
        record[2 + ChannelSize + pixel] = green;
        record[2 + 2 * ChannelSize + pixel] = blue;
    }

    private static void AssertRgb(
        float[] image,
        int row,
        int column,
        byte red,
        byte green,
        byte blue)
    {
        const int patchSize = 4;
        const int patchesPerRow = ColumnsPerChannel / patchSize;
        const int featuresPerPatch = 3 * patchSize * patchSize;
        int patch = (row / patchSize) * patchesPerRow
            + column / patchSize;
        int position = (row % patchSize) * patchSize
            + column % patchSize;
        int patchOffset = patch * featuresPerPatch;

        AssertClose(red / 255f, image[patchOffset + position]);
        AssertClose(
            green / 255f,
            image[patchOffset + patchSize * patchSize + position]);
        AssertClose(
            blue / 255f,
            image[patchOffset + 2 * patchSize * patchSize + position]);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.Cifar100Tests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        internal string WriteData(byte[] data)
        {
            string path = Path.Combine(Root, "cifar-100.bin");
            File.WriteAllBytes(path, data);
            return path;
        }

        public void Dispose()
        {
            Directory.Delete(Root, recursive: true);
        }
    }

    private sealed class SequenceRandom(params int[] values) : Random
    {
        private int _index;

        public override int Next(int maxValue)
        {
            if (_index >= values.Length)
                throw new InvalidOperationException("No random value remains.");

            int value = values[_index++];
            if ((uint)value >= (uint)maxValue)
            {
                throw new InvalidOperationException(
                    $"Random value '{value}' is outside 0..{maxValue - 1}.");
            }

            return value;
        }
    }
}
