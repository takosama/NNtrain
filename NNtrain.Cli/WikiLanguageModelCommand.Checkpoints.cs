namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private const int CheckpointFormatVersion = 4;

    private static void SaveCheckpoint(
        string path,
        WikiModelCheckpoint checkpoint)
    {
        safetensors.torch.save_file(
            checkpoint.CurrentModel ?? checkpoint.Model,
            GetSafeTensorsPath(path));
        torch.save(checkpoint, path);
    }

    private static void SaveBestModelSafeTensors(
        string checkpointPath,
        ModuleState state)
        => safetensors.torch.save_file(
            state,
            GetBestSafeTensorsPath(checkpointPath));

    private static WikiModelCheckpoint LoadCheckpoint(string path)
    {
        WikiModelCheckpoint checkpoint = torch.load<WikiModelCheckpoint>(path);
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion
            || checkpoint.Model is null)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }
        string safeTensorsPath = GetSafeTensorsPath(path);
        if (!File.Exists(safeTensorsPath))
            return checkpoint;
        ModuleState safeModel = safetensors.torch.load_file(safeTensorsPath);
        ModuleState expected = checkpoint.CurrentModel ?? checkpoint.Model;
        return ModuleStatesEqual(safeModel, expected)
            ? checkpoint with { CurrentModel = safeModel }
            : checkpoint;
    }

    internal static WikiResumePosition RestoreTrainingCheckpoint(
        WikiTrainingConfiguration config,
        IWikiLanguageModel model,
        IOptimizer optimizer,
        WarmupCosineProgressLRScheduler scheduler,
        ref ModuleState? bestState,
        ref float bestLoss,
        ref int bestEpoch,
        ref long globalStep,
        TextWriter output)
    {
        if (!config.ResumeFromCheckpoint)
            return new WikiResumePosition(1, 0, 0d, 0, 0, []);
        if (!File.Exists(config.CheckpointPath))
        {
            throw new FileNotFoundException(
                "Wiki training checkpoint was not found.",
                config.CheckpointPath);
        }

        WikiModelCheckpoint checkpoint = LoadCheckpoint(config.CheckpointPath);
        if (!CheckpointArchitectureMatchesConfiguration(checkpoint, config))
        {
            throw new InvalidDataException(
                "Checkpoint model architecture does not match the current " +
                "Wiki training configuration.");
        }

        int completedEpoch = checkpoint.CompletedEpoch == 0
            ? checkpoint.Epoch
            : checkpoint.CompletedEpoch;
        bool hasPartialEpoch = checkpoint.CurrentEpoch > completedEpoch;
        if (!hasPartialEpoch && completedEpoch >= config.Epochs)
        {
            throw new InvalidDataException(
                $"Checkpoint already completed epoch {completedEpoch}, " +
                $"but the configured epoch count is {config.Epochs}.");
        }

        model.load_state_dict(checkpoint.CurrentModel ?? checkpoint.Model);
        if (checkpoint.Optimizer is not null
            && checkpoint.Scheduler is not null)
        {
            optimizer.load_state_dict(checkpoint.Optimizer);
            scheduler.load_state_dict(checkpoint.Scheduler);
        }
        else
        {
            output.WriteLine(
                "checkpoint contains model weights only; optimizer and " +
                "scheduler start from their configured initial state");
        }

        bestState = checkpoint.Model;
        bestLoss = checkpoint.ValidationLoss;
        bestEpoch = checkpoint.Epoch;
        globalStep = checkpoint.GlobalStep;
        output.WriteLine(
            $"resumed checkpoint = {config.CheckpointPath}, next epoch " +
            $"{(hasPartialEpoch ? checkpoint.CurrentEpoch : completedEpoch + 1)}, " +
            $"global step {globalStep:N0}");
        return new WikiResumePosition(
            hasPartialEpoch ? checkpoint.CurrentEpoch : completedEpoch + 1,
            hasPartialEpoch ? checkpoint.CompletedBatchesInEpoch : 0,
            hasPartialEpoch ? checkpoint.CurrentLossSum : 0d,
            hasPartialEpoch ? checkpoint.CurrentTargetCount : 0,
            hasPartialEpoch ? checkpoint.CompletedDocumentsInEpoch : 0,
            hasPartialEpoch ? checkpoint.CurrentTokenBuffer ?? [] : []);
    }

    internal static void SaveTrainingCheckpoint(
        WikiTrainingConfiguration config,
        int vocabularySize,
        int completedEpoch,
        ModuleState bestState,
        float bestLoss,
        int bestEpoch,
        IWikiLanguageModel model,
        IOptimizer optimizer,
        WarmupCosineProgressLRScheduler scheduler,
        long globalStep,
        int currentEpoch = 0,
        int completedBatchesInEpoch = 0,
        double currentLossSum = 0d,
        long currentTargetCount = 0,
        long completedDocumentsInEpoch = 0,
        int[]? currentTokenBuffer = null)
    {
        SaveCheckpoint(
            config.CheckpointPath,
            new WikiModelCheckpoint(
                CheckpointFormatVersion,
                bestEpoch,
                bestLoss,
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.Heads,
                config.HiddenSize,
                config.Layers,
                config.Dropout,
                config.InitializationScale,
                bestState,
                config.ModelArchitecture,
                config.HyenaFilterWidth,
                config.ForgetMemoryKeyWidth,
                config.ForgetMemoryValueWidth,
                config.ForgetMemoryRetentionMinimum,
                config.ForgetMemoryRetentionMaximum,
                completedEpoch,
                model.state_dict(),
                optimizer.state_dict(),
                scheduler.state_dict(),
                globalStep,
                currentEpoch,
                completedBatchesInEpoch,
                currentLossSum,
                currentTargetCount,
                completedDocumentsInEpoch,
                currentTokenBuffer));
    }

    internal static string GetSafeTensorsPath(string checkpointPath)
        => Path.ChangeExtension(
            Path.GetFullPath(checkpointPath),
            ".safetensors");

    internal static string GetBestSafeTensorsPath(string checkpointPath)
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        return Path.Combine(
            directory,
            $"{Path.GetFileNameWithoutExtension(fullPath)}.best.safetensors");
    }

    private static bool ModuleStatesEqual(ModuleState left, ModuleState right)
    {
        if (left.Parameters.Length != right.Parameters.Length)
            return false;
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

    internal sealed record WikiResumePosition(
        int Epoch,
        int CompletedBatches,
        double LossSum,
        long TargetCount,
        long CompletedDocuments,
        int[] TokenBuffer);

    internal sealed record WikiModelCheckpoint(
        int FormatVersion,
        int Epoch,
        float ValidationLoss,
        int VocabularySize,
        int ContextLength,
        int ModelWidth,
        int Heads,
        int HiddenSize,
        int Layers,
        float Dropout,
        float InitializationScale,
        ModuleState Model,
        string? ModelArchitecture = null,
        int HyenaFilterWidth = 64,
        int ForgetMemoryKeyWidth = 16,
        int ForgetMemoryValueWidth = 16,
        float ForgetMemoryRetentionMinimum = 0.5f,
        float ForgetMemoryRetentionMaximum = 0.99f,
        int CompletedEpoch = 0,
        ModuleState? CurrentModel = null,
        OptimizerStateDictionary? Optimizer = null,
        LRSchedulerStateDictionary? Scheduler = null,
        long GlobalStep = 0,
        int CurrentEpoch = 0,
        int CompletedBatchesInEpoch = 0,
        double CurrentLossSum = 0d,
        long CurrentTargetCount = 0,
        long CompletedDocumentsInEpoch = 0,
        int[]? CurrentTokenBuffer = null);
}
