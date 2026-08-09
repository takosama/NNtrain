namespace NNtrain;

public interface IImageClassificationDataset
{
    int Count { get; }

    int Rows { get; }

    int Columns { get; }

    int ImageSize { get; }

    int ClassCount { get; }

    int ReadSample(int index, Span<float> destination);

    int ReadTrainingSample(
        int index,
        Span<float> destination,
        Random random)
    {
        ArgumentNullException.ThrowIfNull(random);
        return ReadSample(index, destination);
    }
}
