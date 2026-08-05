namespace NNtrain;

public sealed record TrainerOptions
{
    public int Epochs { get; init; } = 200;

    public int StepsPerEpoch { get; init; } = 256;

    public int RandomSeed { get; init; } = 1234;

    public float LabelSmoothing { get; init; } = 0.1f;
}
