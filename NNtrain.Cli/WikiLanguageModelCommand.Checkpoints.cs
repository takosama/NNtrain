namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private const int CheckpointFormatVersion = 3;

    private static void SaveCheckpoint(
        string path,
        WikiModelCheckpoint checkpoint)
    {
        torch.save(checkpoint, path);
    }

    private static WikiModelCheckpoint LoadCheckpoint(string path)
    {
        WikiModelCheckpoint checkpoint = torch.load<WikiModelCheckpoint>(path);
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion
            || checkpoint.Model is null)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }
        return checkpoint;
    }

    internal static int RestoreTrainingCheckpoint(
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
            return 1;
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
        if (completedEpoch >= config.Epochs)
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
            $"{completedEpoch + 1}, global step {globalStep:N0}");
        return completedEpoch + 1;
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
        long globalStep)
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
                globalStep));
    }

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
        long GlobalStep = 0);
}
