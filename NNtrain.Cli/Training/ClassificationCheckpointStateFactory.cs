namespace NNtrain;

internal static class ClassificationCheckpointStateFactory
{
    internal static ClassificationTrainingCheckpoint
        Create(
            int completedEpoch,
            int currentEpoch,
            int completedUpdatesInEpoch,
            ILRScheduler scheduler,
            int bestEpoch,
            float bestEvaluationLoss,
            float earlyStoppingReferenceLoss,
            int epochsWithoutImprovement,
            double currentTrainingLossSum,
            int currentTrainingCorrect,
            int currentTrainingSamples)
        => new(
            ClassificationTrainingCheckpoint.CurrentFormatVersion,
            completedEpoch,
            new ModuleState(ModuleState.CurrentFormatVersion, []),
            new OptimizerStateDictionary("ArtifactStream", null, []),
            scheduler.state_dict(),
            null,
            bestEpoch,
            bestEvaluationLoss,
            earlyStoppingReferenceLoss,
            epochsWithoutImprovement,
            currentEpoch,
            completedUpdatesInEpoch,
            currentTrainingLossSum,
            currentTrainingCorrect,
            currentTrainingSamples);

}
