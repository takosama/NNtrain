namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static IWikiLanguageModel CreateModel(
        WikiTrainingConfiguration config,
        int vocabularySize)
        => CreateModel(config, vocabularySize, config.GetModelDType());

    internal static IWikiLanguageModel CreateModel(
        WikiTrainingConfiguration config,
        int vocabularySize,
        TensorDType modelDType)
    {
        if (config.IsForgetMemoryV2Architecture())
        {
            return nn.forget_memory_v2_lm(
                vocab_size: vocabularySize,
                context_length: config.ContextLength,
                d_model: config.ModelWidth,
                dim_feedforward: config.HiddenSize,
                num_layers: config.Layers,
                key_width: config.ForgetMemoryKeyWidth,
                value_width: config.ForgetMemoryValueWidth,
                retention_min: config.ForgetMemoryRetentionMinimum,
                retention_max: config.ForgetMemoryRetentionMaximum,
                generator: new Random(config.Seed),
                init_scale: config.InitializationScale,
                dropout: config.Dropout,
                dtype: modelDType);
        }

        if (modelDType != TensorDType.Float32)
        {
            throw new InvalidOperationException(
                $"Model dtype '{modelDType}' is not supported for " +
                $"architecture '{config.ModelArchitecture}'.");
        }

        if (config.IsArchitecture(
            WikiTrainingConfiguration.ForgetScanArchitecture))
        {
            return nn.forget_scan_lm(
                vocab_size: vocabularySize,
                context_length: config.ContextLength,
                d_model: config.ModelWidth,
                dim_feedforward: config.HiddenSize,
                num_layers: config.Layers,
                generator: new Random(config.Seed),
                init_scale: config.InitializationScale,
                dropout: config.Dropout);
        }

        if (config.IsArchitecture(WikiTrainingConfiguration.HyenaArchitecture))
        {
            return nn.hyena_lm(
                vocab_size: vocabularySize,
                context_length: config.ContextLength,
                d_model: config.ModelWidth,
                dim_feedforward: config.HiddenSize,
                num_layers: config.Layers,
                generator: new Random(config.Seed),
                init_scale: config.InitializationScale,
                dropout: config.Dropout,
                filter_width: config.HyenaFilterWidth,
                convolution: config.GetHyenaConvolutionAlgorithm());
        }

        return nn.transformer_lm(
            vocab_size: vocabularySize,
            context_length: config.ContextLength,
            d_model: config.ModelWidth,
            num_heads: config.Heads,
            dim_feedforward: config.HiddenSize,
            num_layers: config.Layers,
            generator: new Random(config.Seed),
            init_scale: config.InitializationScale,
            dropout: config.Dropout);
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
                WikiTrainingConfiguration.ForgetMemoryV2ArchitectureAlias,
                StringComparison.OrdinalIgnoreCase);

    internal static IWikiLanguageModel CreateModel(
        WikiModelCheckpoint checkpoint,
        int seed)
    {
        TensorDType modelDType = GetCheckpointModelDType(checkpoint);
        if (IsCheckpointForgetMemoryV2(checkpoint))
        {
            return new ForgetMemoryV2Gpt(
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
                checkpoint.Dropout,
                modelDType);
        }

        if (modelDType != TensorDType.Float32)
        {
            throw new InvalidDataException(
                $"Checkpoint model dtype '{modelDType}' is not supported " +
                $"for architecture '{GetCheckpointArchitecture(checkpoint)}'.");
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

    internal static TensorDType GetCheckpointModelDType(
        WikiModelCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (checkpoint.FormatVersion < DTypeCheckpointFormatVersion)
            return TensorDType.Float32;

        TensorDType dtype = checkpoint.ModelDType
            ?? throw new InvalidDataException(
                "Wiki model checkpoint does not declare its model dtype.");
        if (dtype is not TensorDType.Float32
            and not TensorDType.Float16
            and not TensorDType.BFloat16)
        {
            throw new InvalidDataException(
                $"Wiki model checkpoint declares unsupported model dtype " +
                $"'{dtype}'.");
        }
        return dtype;
    }
}
