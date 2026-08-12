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
    int EpochsWithoutImprovement,
    int CurrentEpoch = 0,
    int CompletedUpdatesInEpoch = 0,
    double CurrentTrainingLossSum = 0d,
    int CurrentTrainingCorrect = 0,
    int CurrentTrainingSamples = 0)
{
    internal const int CurrentFormatVersion = 2;
}

internal static class ClassificationCheckpoint
{
    internal static void Save(
        string path,
        ClassificationTrainingCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        safetensors.torch.save_file(
            checkpoint.Model,
            GetSafeTensorsPath(path));
        torch.save(checkpoint, path);
    }

    internal static ClassificationTrainingCheckpoint Load(string path)
    {
        ClassificationTrainingCheckpoint checkpoint =
            torch.load<ClassificationTrainingCheckpoint>(path);
        if (checkpoint.FormatVersion is < 1
                or > ClassificationTrainingCheckpoint.CurrentFormatVersion
            || checkpoint.CompletedEpoch < 0
            || checkpoint.CurrentEpoch < 0
            || checkpoint.CompletedUpdatesInEpoch < 0
            || !double.IsFinite(checkpoint.CurrentTrainingLossSum)
            || checkpoint.CurrentTrainingLossSum < 0d
            || checkpoint.CurrentTrainingCorrect < 0
            || checkpoint.CurrentTrainingSamples < 0
            || checkpoint.CurrentTrainingCorrect
                > checkpoint.CurrentTrainingSamples
            || checkpoint.Model is null
            || checkpoint.Optimizer is null
            || checkpoint.Scheduler is null)
        {
            throw new InvalidDataException(
                "Classification training checkpoint is incompatible.");
        }
        string safeTensorsPath = GetSafeTensorsPath(path);
        if (!File.Exists(safeTensorsPath))
            return checkpoint;

        ModuleState safeModel = safetensors.torch.load_file(safeTensorsPath);
        return ModuleStatesEqual(safeModel, checkpoint.Model)
            ? checkpoint with { Model = safeModel }
            : checkpoint;
    }

    internal static string GetSafeTensorsPath(string checkpointPath)
        => Path.ChangeExtension(
            Path.GetFullPath(checkpointPath),
            ".safetensors");

    private static bool ModuleStatesEqual(ModuleState left, ModuleState right)
    {
        if (left.FormatVersion != right.FormatVersion
            || left.Parameters.Length != right.Parameters.Length)
        {
            return false;
        }
        for (int index = 0; index < left.Parameters.Length; index++)
        {
            ModuleParameterState first = left.Parameters[index];
            ModuleParameterState second = right.Parameters[index];
            if (first.Index != second.Index
                || first.Name != second.Name
                || !first.Shape.AsSpan().SequenceEqual(second.Shape)
                || !first.Values.AsSpan().SequenceEqual(second.Values))
            {
                return false;
            }
        }
        return true;
    }
}
