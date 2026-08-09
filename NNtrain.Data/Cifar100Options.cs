namespace NNtrain;

public sealed record Cifar100Options
{
    public int PatchSize { get; init; } = 4;

    public bool Normalize { get; init; }

    public int RandomCropPadding { get; init; } = 4;

    public bool HorizontalFlip { get; init; } = true;

    public bool VerticalFlip { get; init; }
}
