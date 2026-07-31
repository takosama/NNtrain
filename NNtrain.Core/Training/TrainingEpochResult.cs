namespace NNtrain;

public sealed record TrainingEpochResult(
    int Epoch,
    int TrainingSteps,
    int EvaluationSamples,
    TrainingMetrics Training,
    TrainingMetrics Evaluation);
