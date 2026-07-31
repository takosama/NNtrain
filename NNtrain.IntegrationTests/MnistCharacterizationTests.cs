using System.Buffers.Binary;
using Xunit;
using NNtrain;

public sealed class MnistCharacterizationTests
{
    private const int ExpectedRows = 28;
    private const int ExpectedColumns = 28;
    private const int ExpectedImageSize = ExpectedRows * ExpectedColumns;

    [Fact]
    public void ReadsPixelsAndLabelsUsingCurrentIdxOffsets()
    {
        using var directory = new TemporaryDirectory();
        var images = new byte[16 + ExpectedImageSize];
        images[16] = 0;
        images[17] = 127;
        images[18] = 255;
        var labels = new byte[9];
        labels[8] = 7;
        WriteInt32(images, 0, 2051);
        WriteInt32(images, 4, 1);
        WriteInt32(images, 8, ExpectedRows);
        WriteInt32(images, 12, ExpectedColumns);
        WriteInt32(labels, 0, 2049);
        WriteInt32(labels, 4, 1);
        string imagePath = Path.Combine(
            directory.Root,
            "custom-images.idx3-ubyte");
        string labelPath = Path.Combine(
            directory.Root,
            "custom-labels.idx1-ubyte");

        File.WriteAllBytes(imagePath, images);
        File.WriteAllBytes(labelPath, labels);

        IImageClassificationDataset dataset = new Mnist(
            imagePath,
            labelPath);
        var destination = new float[dataset.ImageSize];
        int label = dataset.ReadSample(0, destination);

        AssertClose(0f, destination[0]);
        AssertClose(127f / 255f, destination[1]);
        AssertClose(1f, destination[2]);
        Assert.Equal(7, label);
        Assert.Equal(1, dataset.Count);
        Assert.Equal(ExpectedRows, dataset.Rows);
        Assert.Equal(ExpectedColumns, dataset.Columns);
        Assert.Equal(ExpectedImageSize, dataset.ImageSize);
        Assert.Equal(10, dataset.ClassCount);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(1)]
    public void ReadSampleRejectsAnOutOfRangeIndex(int index)
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(directory);
        IImageClassificationDataset dataset = new Mnist(
            files.ImagePath,
            files.LabelPath);
        var destination = new float[dataset.ImageSize];

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => dataset.ReadSample(index, destination));

        Assert.Equal("index", exception.ParamName);
    }

    [Fact]
    public void ReadSampleRejectsAShortDestination()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(directory);
        IImageClassificationDataset dataset = new Mnist(
            files.ImagePath,
            files.LabelPath);
        var destination = new float[dataset.ImageSize - 1];

        var exception = Assert.Throws<ArgumentException>(
            () => dataset.ReadSample(0, destination));

        Assert.Equal("destination", exception.ParamName);
    }

    [Fact]
    public void ReadSampleRejectsALabelOutsideTheClassRange()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(directory, label: 10);
        IImageClassificationDataset dataset = new Mnist(
            files.ImagePath,
            files.LabelPath);
        var destination = new float[dataset.ImageSize];

        var exception = Assert.Throws<InvalidDataException>(
            () => dataset.ReadSample(0, destination));

        Assert.Contains("valid range", exception.Message);
    }

    [Fact]
    public void MnistDoesNotExposeSplitImageAndLabelReads()
    {
        Assert.Null(typeof(Mnist).GetMethod("GetDataFloat"));
        Assert.Null(typeof(Mnist).GetMethod("GetLabel"));
    }

    [Fact]
    public void RejectsInvalidImageMagicNumber()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(directory, imageMagic: 999);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("image IDX magic number", exception.Message);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void RejectsTruncatedHeaders(bool truncateImages)
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(directory);
        if (truncateImages)
            File.WriteAllBytes(files.ImagePath, new byte[15]);
        else
            File.WriteAllBytes(files.LabelPath, new byte[7]);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("header must be at least", exception.Message);
    }

    [Fact]
    public void RejectsInvalidLabelMagicNumber()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(directory, labelMagic: 999);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("label IDX magic number", exception.Message);
    }

    [Fact]
    public void RejectsNonPositiveCounts()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(
            directory,
            imageCount: 0,
            labelCount: 0);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("counts must both be positive", exception.Message);
    }

    [Fact]
    public void RejectsMismatchedImageAndLabelCounts()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(
            directory,
            imageCount: 2,
            labelCount: 1);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("does not match", exception.Message);
    }

    [Theory]
    [InlineData(27, 28)]
    [InlineData(28, 27)]
    public void RejectsNonMnistImageDimensions(int rows, int columns)
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(
            directory,
            rows: rows,
            columns: columns);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("must be 28x28", exception.Message);
    }

    [Fact]
    public void RejectsImagePayloadThatDoesNotMatchDeclaredCount()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(
            directory,
            imagePayloadLength: ExpectedImageSize - 1);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("Image IDX length", exception.Message);
    }

    [Fact]
    public void RejectsLabelPayloadThatDoesNotMatchDeclaredCount()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles files = WriteDataset(
            directory,
            labelPayloadLength: 0);

        var exception = Assert.Throws<InvalidDataException>(
            () => new Mnist(files.ImagePath, files.LabelPath));

        Assert.Contains("Label IDX length", exception.Message);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void RejectsMissingImagePath(string? imagePath)
    {
        var exception = Assert.ThrowsAny<ArgumentException>(
            () => new Mnist(imagePath!, "labels.idx1-ubyte"));

        Assert.Equal("imagePath", exception.ParamName);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    public void RejectsMissingLabelPath(string? labelPath)
    {
        var exception = Assert.ThrowsAny<ArgumentException>(
            () => new Mnist("images.idx3-ubyte", labelPath!));

        Assert.Equal("labelPath", exception.ParamName);
    }

    private static DatasetFiles WriteDataset(
        TemporaryDirectory directory,
        int imageMagic = 2051,
        int labelMagic = 2049,
        int imageCount = 1,
        int labelCount = 1,
        int rows = ExpectedRows,
        int columns = ExpectedColumns,
        byte label = 0,
        int? imagePayloadLength = null,
        int? labelPayloadLength = null)
    {
        int actualImagePayloadLength = imagePayloadLength
            ?? imageCount * ExpectedImageSize;
        int actualLabelPayloadLength = labelPayloadLength ?? labelCount;
        var images = new byte[16 + actualImagePayloadLength];
        var labels = new byte[8 + actualLabelPayloadLength];
        WriteInt32(images, 0, imageMagic);
        WriteInt32(images, 4, imageCount);
        WriteInt32(images, 8, rows);
        WriteInt32(images, 12, columns);
        WriteInt32(labels, 0, labelMagic);
        WriteInt32(labels, 4, labelCount);
        if (labels.Length > 8)
            labels[8] = label;

        string imagePath = Path.Combine(
            directory.Root,
            "images.idx3-ubyte");
        string labelPath = Path.Combine(
            directory.Root,
            "labels.idx1-ubyte");
        File.WriteAllBytes(imagePath, images);
        File.WriteAllBytes(labelPath, labels);
        return new DatasetFiles(imagePath, labelPath);
    }

    private static void WriteInt32(byte[] destination, int offset, int value)
    {
        BinaryPrimitives.WriteInt32BigEndian(
            destination.AsSpan(offset, sizeof(int)),
            value);
    }

    private static void AssertClose(
        float expected,
        float actual,
        float tolerance = 1e-5f)
    {
        Assert.InRange(actual, expected - tolerance, expected + tolerance);
    }

    private readonly record struct DatasetFiles(
        string ImagePath,
        string LabelPath);

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.Tests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose()
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
