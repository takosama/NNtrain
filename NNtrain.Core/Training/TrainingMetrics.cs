namespace NNtrain;

public sealed record TrainingMetrics(
    float Loss,
    float Accuracy,
    TimeSpan Elapsed);
