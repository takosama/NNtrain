namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static IWikiLanguageModel CreateModel(
        WikiTrainingConfiguration config,
        int vocabularySize)
    {
        if (config.IsForgetMemoryV2Architecture())
        {
            return new FrogetMemoryV2Gpt(
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.HiddenSize,
                config.Layers,
                config.ForgetMemoryKeyWidth,
                config.ForgetMemoryValueWidth,
                config.ForgetMemoryRetentionMinimum,
                config.ForgetMemoryRetentionMaximum,
                new Random(config.Seed),
                config.InitializationScale,
                config.Dropout);
        }

        if (config.IsArchitecture(
            WikiTrainingConfiguration.ForgetScanArchitecture))
        {
            return new ForgetScanGpt(
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.HiddenSize,
                config.Layers,
                new Random(config.Seed),
                config.InitializationScale,
                config.Dropout);
        }

        if (config.IsArchitecture(WikiTrainingConfiguration.HyenaArchitecture))
        {
            return new HyenaGpt(
                vocabularySize,
                config.ContextLength,
                config.ModelWidth,
                config.HiddenSize,
                config.Layers,
                new Random(config.Seed),
                config.InitializationScale,
                config.Dropout,
                config.HyenaFilterWidth,
                config.GetHyenaConvolutionAlgorithm());
        }

        return new GptRinWikiJp(
            vocabularySize,
            config.ContextLength,
            config.ModelWidth,
            config.Heads,
            config.HiddenSize,
            config.Layers,
            new Random(config.Seed),
            config.InitializationScale,
            config.Dropout);
    }

    private static string GetCheckpointArchitecture(
        WikiModelCheckpoint checkpoint)
        => string.IsNullOrWhiteSpace(checkpoint.ModelArchitecture)
            ? WikiTrainingConfiguration.TransformerArchitecture
            : checkpoint.ModelArchitecture;

    private static bool CheckpointArchitectureMatchesConfiguration(
        WikiModelCheckpoint checkpoint,
        WikiTrainingConfiguration config)
        => checkpoint.VocabularySize == config.VocabularySize
            && checkpoint.ContextLength == config.ContextLength
            && checkpoint.ModelWidth == config.ModelWidth
            && checkpoint.Heads == config.Heads
            && checkpoint.HiddenSize == config.HiddenSize
            && checkpoint.Layers == config.Layers
            && (config.IsForgetMemoryV2Architecture()
                ? IsCheckpointForgetMemoryV2(checkpoint)
                : string.Equals(
                    GetCheckpointArchitecture(checkpoint),
                    config.ModelArchitecture,
                    StringComparison.OrdinalIgnoreCase))
            && (!string.Equals(
                    GetCheckpointArchitecture(checkpoint),
                    WikiTrainingConfiguration.HyenaArchitecture,
                    StringComparison.OrdinalIgnoreCase)
                || checkpoint.HyenaFilterWidth == config.HyenaFilterWidth)
            && (!IsCheckpointForgetMemoryV2(checkpoint)
                || (checkpoint.ForgetMemoryKeyWidth
                        == config.ForgetMemoryKeyWidth
                    && checkpoint.ForgetMemoryValueWidth
                        == config.ForgetMemoryValueWidth
                    && checkpoint.ForgetMemoryRetentionMinimum
                        == config.ForgetMemoryRetentionMinimum
                    && checkpoint.ForgetMemoryRetentionMaximum
                        == config.ForgetMemoryRetentionMaximum));

    private static bool IsCheckpointForgetMemoryV2(
        WikiModelCheckpoint checkpoint)
        => string.Equals(
                GetCheckpointArchitecture(checkpoint),
                WikiTrainingConfiguration.ForgetMemoryV2Architecture,
                StringComparison.OrdinalIgnoreCase)
            || string.Equals(
                GetCheckpointArchitecture(checkpoint),
                WikiTrainingConfiguration.FrogetMemoryV2ArchitectureAlias,
                StringComparison.OrdinalIgnoreCase);

    private static IWikiLanguageModel CreateModel(
        WikiModelCheckpoint checkpoint,
        int seed)
    {
        if (IsCheckpointForgetMemoryV2(checkpoint))
        {
            return new FrogetMemoryV2Gpt(
                checkpoint.VocabularySize,
                checkpoint.ContextLength,
                checkpoint.ModelWidth,
                checkpoint.HiddenSize,
                checkpoint.Layers,
                checkpoint.ForgetMemoryKeyWidth,
                checkpoint.ForgetMemoryValueWidth,
                checkpoint.ForgetMemoryRetentionMinimum,
                checkpoint.ForgetMemoryRetentionMaximum,
                new Random(seed),
                checkpoint.InitializationScale,
                checkpoint.Dropout);
        }

        if (string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.ForgetScanArchitecture,
            StringComparison.OrdinalIgnoreCase))
        {
            return new ForgetScanGpt(
                checkpoint.VocabularySize,
                checkpoint.ContextLength,
                checkpoint.ModelWidth,
                checkpoint.HiddenSize,
                checkpoint.Layers,
                new Random(seed),
                checkpoint.InitializationScale,
                checkpoint.Dropout);
        }

        if (string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.HyenaArchitecture,
            StringComparison.OrdinalIgnoreCase))
        {
            return new HyenaGpt(
                checkpoint.VocabularySize,
                checkpoint.ContextLength,
                checkpoint.ModelWidth,
                checkpoint.HiddenSize,
                checkpoint.Layers,
                new Random(seed),
                checkpoint.InitializationScale,
                checkpoint.Dropout,
                checkpoint.HyenaFilterWidth);
        }

        return new GptRinWikiJp(
            checkpoint.VocabularySize,
            checkpoint.ContextLength,
            checkpoint.ModelWidth,
            checkpoint.Heads,
            checkpoint.HiddenSize,
            checkpoint.Layers,
            new Random(seed),
            checkpoint.InitializationScale,
            checkpoint.Dropout);
    }
}
