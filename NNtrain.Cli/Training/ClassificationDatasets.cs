namespace NNtrain;

internal static class ClassificationDatasets
{
    internal static IImageClassificationDataset Create(
        DatasetConfiguration configuration,
        string role)
    {
        if (configuration.IsType(DatasetConfiguration.MnistType))
        {
            EnsureDataFile(configuration.ImagePath, $"{role} image");
            EnsureDataFile(configuration.LabelPath, $"{role} label");
            return datasets.mnist(
                images: configuration.ImagePath,
                labels: configuration.LabelPath);
        }

        if (configuration.IsType(DatasetConfiguration.Cifar100Type))
        {
            EnsureDataFile(configuration.DataPath, $"{role} CIFAR-100");
            return datasets.cifar100(
                data: configuration.DataPath,
                patch_size: configuration.PatchSize,
                normalize: configuration.Normalize,
                random_crop_padding:
                    configuration.Augmentation.RandomCropPadding,
                horizontal_flip:
                    configuration.Augmentation.HorizontalFlip,
                vertical_flip:
                    configuration.Augmentation.VerticalFlip);
        }

        throw new ArgumentException(
            $"Unsupported {role.ToLowerInvariant()} dataset type " +
            $"'{configuration.Type}'.",
            nameof(configuration));
    }

    internal static void ValidateCompatibility(
        IImageClassificationDataset training,
        IImageClassificationDataset evaluation)
    {
        if (training.Rows != evaluation.Rows
            || training.Columns != evaluation.Columns)
        {
            throw new ArgumentException(
                $"Training input shape '{training.Rows}x" +
                $"{training.Columns}' does not match evaluation input " +
                $"shape '{evaluation.Rows}x{evaluation.Columns}'.");
        }

        if (training.ClassCount != evaluation.ClassCount)
        {
            throw new ArgumentException(
                $"Training class count '{training.ClassCount}' does not " +
                $"match evaluation class count " +
                $"'{evaluation.ClassCount}'.");
        }
    }

    internal static string DescribeAugmentation(
        Cifar100Options options)
    {
        var operations = new List<string>();
        if (options.RandomCropPadding > 0)
        {
            operations.Add(
                $"random crop (padding {options.RandomCropPadding})");
        }
        if (options.HorizontalFlip)
            operations.Add("horizontal flip");
        if (options.VerticalFlip)
            operations.Add("vertical flip");

        return operations.Count == 0
            ? "augmentation = disabled"
            : $"augmentation = {string.Join(", ", operations)}";
    }

    private static void EnsureDataFile(string path, string role)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"{role} data file was not found at '{path}'. " +
                "Check the corresponding path in the training " +
                "configuration.",
                path);
        }
    }

}
