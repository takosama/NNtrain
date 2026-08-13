using NNtrain;
using Xunit;

public sealed class CheckpointDirectoryCreationTests
{
    [Fact]
    public void WikiTrainingCheckpointCreatesMissingParentDirectory()
    {
        string temporaryRoot = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-wiki-checkpoint-directory-{Guid.NewGuid():N}");
        string checkpointPath = Path.Combine(
            temporaryRoot,
            "configured",
            "nested",
            "training.json");
        try
        {
            var configuration = new WikiTrainingConfiguration
            {
                CheckpointPath = checkpointPath,
                VocabularySize = 8,
                ContextLength = 2,
                ModelWidth = 4,
                Heads = 1,
                HiddenSize = 8,
                Layers = 1,
                ModelArchitecture =
                    WikiTrainingConfiguration.TransformerArchitecture,
                Optimizer = WikiTrainingConfiguration.AdamWOptimizer,
                Dropout = 0f,
            };
            IWikiLanguageModel model = WikiLanguageModelCommand.CreateModel(
                configuration,
                configuration.VocabularySize);
            IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
                model,
                configuration);
            WarmupCosineProgressLRScheduler scheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    optimizer,
                    configuration.WarmupPercent);
            ModuleState state = model.state_dict();

            Assert.False(Directory.Exists(Path.GetDirectoryName(checkpointPath)));

            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                configuration,
                configuration.VocabularySize,
                completedEpoch: 0,
                state,
                bestLoss: 1f,
                bestEpoch: 0,
                model,
                optimizer,
                scheduler,
                globalStep: 0);

            Assert.True(File.Exists(checkpointPath));
            Assert.True(File.Exists(
                WikiLanguageModelCommand.GetSafeTensorsPath(checkpointPath)));
        }
        finally
        {
            if (Directory.Exists(temporaryRoot))
                Directory.Delete(temporaryRoot, recursive: true);
        }
    }

    [Fact]
    public void ClassificationCheckpointCreatesMissingParentDirectory()
    {
        string temporaryRoot = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-classification-checkpoint-directory-{Guid.NewGuid():N}");
        string checkpointPath = Path.Combine(
            temporaryRoot,
            "configured",
            "nested",
            "training.json");
        try
        {
            var modelState = new ModuleState(
                ModuleState.CurrentFormatVersion,
                [
                    new ModuleParameterState(
                        0,
                        "weight",
                        [1],
                        [0.25f]),
                ]);
            var checkpoint = new ClassificationTrainingCheckpoint(
                ClassificationTrainingCheckpoint.CurrentFormatVersion,
                CompletedEpoch: 0,
                modelState,
                new OptimizerStateDictionary("AdamW", "{}", []),
                new LRSchedulerStateDictionary("CosineAnnealingLR", 0),
                BestModel: null,
                BestEpoch: 0,
                BestEvaluationLoss: 1f,
                EarlyStoppingReferenceLoss: 1f,
                EpochsWithoutImprovement: 0);

            Assert.False(Directory.Exists(Path.GetDirectoryName(checkpointPath)));

            ClassificationCheckpoint.Save(checkpointPath, checkpoint);

            Assert.True(File.Exists(checkpointPath));
            Assert.True(File.Exists(
                ClassificationCheckpoint.GetSafeTensorsPath(checkpointPath)));
            ClassificationTrainingCheckpoint restored =
                ClassificationCheckpoint.Load(checkpointPath);
            Assert.Equal(
                checkpoint.Model.Parameters[0].Values,
                restored.Model.Parameters[0].Values);
        }
        finally
        {
            if (Directory.Exists(temporaryRoot))
                Directory.Delete(temporaryRoot, recursive: true);
        }
    }

    [Fact]
    public void ClassificationBestModelUsesConfiguredCheckpointDirectory()
    {
        string checkpointPath = Path.Combine(
            Path.GetTempPath(),
            "configured-checkpoints",
            "latest.json");

        Assert.Equal(
            Path.Combine(
                Path.GetTempPath(),
                "configured-checkpoints",
                "latest.best-model.json"),
            Program.GetBestModelCheckpointPath(checkpointPath));
    }

    [Fact]
    public void ClassificationBestModelKeepsLegacyDefaultFileName()
    {
        string checkpointPath = Path.Combine(
            Path.GetTempPath(),
            "training.checkpoint.json");

        Assert.Equal(
            Path.Combine(Path.GetTempPath(), "training.best-model.json"),
            Program.GetBestModelCheckpointPath(checkpointPath));
    }
}
