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
    public void ReadsFineLabelAndArrangesRgbChannelsAs32By96()
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
        Assert.Equal(32, dataset.Rows);
        Assert.Equal(96, dataset.Columns);
        Assert.Equal(3072, dataset.ImageSize);
        Assert.Equal(100, dataset.ClassCount);
        AssertClose(10f / 255f, destination[0]);
        AssertClose(20f / 255f, destination[32]);
        AssertClose(30f / 255f, destination[64]);
        AssertClose(40f / 255f, destination[31 * 96 + 31]);
        AssertClose(50f / 255f, destination[31 * 96 + 63]);
        AssertClose(60f / 255f, destination[31 * 96 + 95]);
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
}
