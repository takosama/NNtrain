namespace NNtrain;

public sealed record TrainingBatchResult(
    int Epoch,
    int Batch,
    int TotalBatches,
    float Loss,
    bool IsCorrect);
