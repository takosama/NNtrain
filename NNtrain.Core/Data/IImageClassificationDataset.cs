namespace NNtrain;

public interface IImageClassificationDataset
{
    int Count { get; }

    int Rows { get; }

    int Columns { get; }

    int ImageSize { get; }

    int ClassCount { get; }

    int count => Count;

    int rows => Rows;

    int columns => Columns;

    int image_size => ImageSize;

    int class_count => ClassCount;

    int ReadSample(int index, Span<float> destination);

    int getitem(int index, Span<float> destination)
        => ReadSample(index, destination);

    int ReadTrainingSample(
        int index,
        Span<float> destination,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return ReadSample(index, destination);
    }
}
