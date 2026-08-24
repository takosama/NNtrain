namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    internal static LanguageModel CreateModel(
        WikiTrainingConfiguration config,
        int vocabularySize)
        => CreateModel(config, vocabularySize, config.GetModelDType());

    internal static LanguageModel CreateModel(
        WikiTrainingConfiguration config,
        int vocabularySize,
        TensorDType modelDType)
    {
        if (config.IsForgetMemoryDrnArchitecture())
        {
            return nn.forget_memory_drn_lm(
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

        if (config.IsForgetMemoryV3Architecture())
        {
            return nn.forget_memory_v3_lm(
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

        if (config.IsArchitecture(
            WikiTrainingConfiguration.ForgetScanArchitecture))
        {
            RequireFloat32ModelDType(config.ModelArchitecture, modelDType);
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
            RequireFloat32ModelDType(config.ModelArchitecture, modelDType);
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
            dropout: config.Dropout,
            dtype: modelDType,
            tie_word_embeddings: config.TieWordEmbeddings);
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
            && checkpoint.TieWordEmbeddings == config.TieWordEmbeddings
            && (config.IsForgetMemoryArchitecture()
                ? CheckpointForgetMemoryVersionMatches(checkpoint, config)
                : string.Equals(
                    GetCheckpointArchitecture(checkpoint),
                    config.ModelArchitecture,
                    StringComparison.OrdinalIgnoreCase))
            && (!string.Equals(
                    GetCheckpointArchitecture(checkpoint),
                    WikiTrainingConfiguration.HyenaArchitecture,
                    StringComparison.OrdinalIgnoreCase)
                || checkpoint.HyenaFilterWidth == config.HyenaFilterWidth)
            && (!IsCheckpointForgetMemory(checkpoint)
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

    private static bool IsCheckpointForgetMemoryV3(
        WikiModelCheckpoint checkpoint)
        => string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.ForgetMemoryV3Architecture,
            StringComparison.OrdinalIgnoreCase);

    private static bool IsCheckpointForgetMemoryDrn(
        WikiModelCheckpoint checkpoint)
        => string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.ForgetMemoryDrnArchitecture,
            StringComparison.OrdinalIgnoreCase);

    private static bool IsCheckpointForgetMemory(
        WikiModelCheckpoint checkpoint)
        => IsCheckpointForgetMemoryV2(checkpoint)
            || IsCheckpointForgetMemoryV3(checkpoint)
            || IsCheckpointForgetMemoryDrn(checkpoint);

    private static bool CheckpointForgetMemoryVersionMatches(
        WikiModelCheckpoint checkpoint,
        WikiTrainingConfiguration config)
    {
        if (config.IsForgetMemoryDrnArchitecture())
            return IsCheckpointForgetMemoryDrn(checkpoint);
        return config.IsForgetMemoryV3Architecture()
            ? IsCheckpointForgetMemoryV3(checkpoint)
            : IsCheckpointForgetMemoryV2(checkpoint);
    }

    internal static LanguageModel CreateModel(
        WikiModelCheckpoint checkpoint,
        int seed)
    {
        TensorDType modelDType = GetCheckpointModelDType(checkpoint);
        if (IsCheckpointForgetMemoryDrn(checkpoint))
        {
            return new ForgetMemoryDRNGpt(
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

        if (IsCheckpointForgetMemoryV3(checkpoint))
        {
            return new ForgetMemoryV3Gpt(
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

        if (string.Equals(
            GetCheckpointArchitecture(checkpoint),
            WikiTrainingConfiguration.ForgetScanArchitecture,
            StringComparison.OrdinalIgnoreCase))
        {
            RequireFloat32CheckpointDType(checkpoint, modelDType);
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
            RequireFloat32CheckpointDType(checkpoint, modelDType);
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
            checkpoint.Dropout,
            modelDType,
            checkpoint.TieWordEmbeddings);
    }

    private static void RequireFloat32ModelDType(
        string architecture,
        TensorDType modelDType)
    {
        if (modelDType != TensorDType.Float32)
        {
            throw new InvalidOperationException(
                $"Model dtype '{modelDType}' is not supported for " +
                $"architecture '{architecture}'.");
        }
    }

    private static void RequireFloat32CheckpointDType(
        WikiModelCheckpoint checkpoint,
        TensorDType modelDType)
    {
        if (modelDType != TensorDType.Float32)
        {
            throw new InvalidDataException(
                $"Checkpoint model dtype '{modelDType}' is not supported " +
                $"for architecture '{GetCheckpointArchitecture(checkpoint)}'.");
        }
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
