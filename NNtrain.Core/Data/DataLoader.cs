namespace NNtrain;

public sealed record DataBatch(Tensor Input, int[] Target)
{
    public Tensor input => Input;
    public int[] target => Target;
}

/// <summary>
/// Batches an image-classification dataset using PyTorch DataLoader semantics.
/// </summary>
public sealed class DataLoader : IEnumerable<DataBatch>
{
    private readonly IImageClassificationDataset _dataset;
    private readonly bool _shuffle;
    private readonly bool _dropLast;
    private readonly bool _useTrainingAugmentation;
    private readonly Random _shuffleRandom;
    private readonly Random _augmentationRandom;

    public DataLoader(
        IImageClassificationDataset dataset,
        int batch_size = 1,
        bool shuffle = false,
        bool drop_last = false,
        bool training = false,
        Random? generator = null,
        Random? augmentation_generator = null)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        if (batch_size <= 0)
            throw new ArgumentOutOfRangeException(nameof(batch_size));
        if (dataset.Count <= 0)
            throw new ArgumentException("Dataset must contain samples.", nameof(dataset));

        _dataset = dataset;
        BatchSize = batch_size;
        _shuffle = shuffle;
        _dropLast = drop_last;
        _useTrainingAugmentation = training;
        _shuffleRandom = generator ?? torch.generator();
        _augmentationRandom = augmentation_generator
            ?? _shuffleRandom;
    }

    public IImageClassificationDataset Dataset => _dataset;
    public IImageClassificationDataset dataset => Dataset;
    public int BatchSize { get; }
    public int batch_size => BatchSize;
    public int Count => _dropLast
        ? _dataset.Count / BatchSize
        : (_dataset.Count + BatchSize - 1) / BatchSize;

    public IEnumerator<DataBatch> GetEnumerator()
    {
        int[] order = Enumerable.Range(0, _dataset.Count).ToArray();
        if (_shuffle)
            Shuffle(order, _shuffleRandom);

        for (int start = 0; start < order.Length; start += BatchSize)
        {
            int count = Math.Min(BatchSize, order.Length - start);
            if (_dropLast && count < BatchSize)
                yield break;
            yield return ReadBatch(order, start, count);
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator()
        => GetEnumerator();

    private DataBatch ReadBatch(int[] order, int start, int count)
    {
        var input = new float[checked(count * _dataset.ImageSize)];
        var target = new int[count];
        for (int offset = 0; offset < count; offset++)
        {
            Span<float> destination = input.AsSpan(
                offset * _dataset.ImageSize,
                _dataset.ImageSize);
            int sampleIndex = order[start + offset];
            target[offset] = _useTrainingAugmentation
                ? _dataset.ReadTrainingSample(
                    sampleIndex,
                    destination,
                    _augmentationRandom)
                : _dataset.ReadSample(sampleIndex, destination);
        }

        return new DataBatch(
            Tensor.FromOwnedData(
                input,
                [count, _dataset.Rows, _dataset.Columns],
                "input"),
            target);
    }

    private static void Shuffle(int[] values, Random random)
    {
        for (int index = values.Length - 1; index > 0; index--)
        {
            int other = random.Next(index + 1);
            (values[index], values[other]) = (values[other], values[index]);
        }
    }
}
