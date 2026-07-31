namespace NNtrain;

public interface IImageClassificationDataset
{
    int Count { get; }

    int Rows { get; }

    int Columns { get; }

    int ImageSize { get; }

    int ClassCount { get; }

    int ReadSample(int index, Span<float> destination);
}
