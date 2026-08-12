namespace NNtrain;

internal sealed record ClassificationTrainingCheckpoint(
    int FormatVersion,
    int CompletedEpoch,
    ModuleState Model,
    OptimizerStateDictionary Optimizer,
    LRSchedulerStateDictionary Scheduler,
    ModuleState? BestModel,
    int BestEpoch,
    float BestEvaluationLoss,
    float EarlyStoppingReferenceLoss,
    int EpochsWithoutImprovement)
{
    internal const int CurrentFormatVersion = 1;
}

internal static class ClassificationCheckpoint
{
    internal static void Save(
        string path,
        ClassificationTrainingCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        torch.save(checkpoint, path);
    }

    internal static ClassificationTrainingCheckpoint Load(string path)
    {
        ClassificationTrainingCheckpoint checkpoint =
            torch.load<ClassificationTrainingCheckpoint>(path);
        if (checkpoint.FormatVersion
                != ClassificationTrainingCheckpoint.CurrentFormatVersion
            || checkpoint.CompletedEpoch < 0
            || checkpoint.Model is null
            || checkpoint.Optimizer is null
            || checkpoint.Scheduler is null)
        {
            throw new InvalidDataException(
                "Classification training checkpoint is incompatible.");
        }
        return checkpoint;
    }
}
