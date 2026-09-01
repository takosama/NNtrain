#pragma warning disable CS8981

namespace NNtrain;

/// <summary>PyTorch-style dataset factories.</summary>
public static class datasets
{
    public static IImageClassificationDataset mnist(
        string images,
        string labels)
        => new Mnist(images, labels);

    public static IImageClassificationDataset cifar100(
        string data,
        int patch_size = 1,
        bool normalize = true,
        int random_crop_padding = 0,
        bool horizontal_flip = false,
        bool vertical_flip = false)
        => new Cifar100(
            data,
            new Cifar100Options
            {
                PatchSize = patch_size,
                Normalize = normalize,
                RandomCropPadding = random_crop_padding,
                HorizontalFlip = horizontal_flip,
                VerticalFlip = vertical_flip,
            });

    public static IAsyncEnumerable<string> wikipedia(
        string root,
        string text_column = "text",
        int? max_documents = null,
        CancellationToken cancellation_token = default,
        int? shuffle_seed = null)
        => WikiParquetCorpus.ReadTextsAsync(
            root,
            text_column,
            max_documents,
            cancellation_token,
            shuffle_seed);

    public static IAsyncEnumerable<string> fineweb(
        string root,
        string text_column = "text",
        int? max_documents = null,
        CancellationToken cancellation_token = default,
        int? shuffle_seed = null)
        => FineWebParquetCorpus.ReadTextsAsync(
            root,
            text_column,
            max_documents,
            cancellation_token,
            shuffle_seed);
}
