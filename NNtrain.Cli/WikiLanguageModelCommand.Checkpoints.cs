namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private const int CheckpointFormatVersion = 6;
    private const int DTypeCheckpointFormatVersion = 5;
    private const int PrecisionModeCheckpointFormatVersion = 6;

    private static void SaveCheckpoint(
        string path,
        WikiModelCheckpoint checkpoint)
    {
        ValidateCheckpoint(checkpoint);
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        safetensors.torch.save_file(
            checkpoint.CurrentModel ?? checkpoint.Model,
            GetSafeTensorsPath(fullPath));
        torch.save(checkpoint, fullPath);
    }

    private static void SaveBestModelSafeTensors(
        string checkpointPath,
        ModuleState state)
    {
        string path = GetBestSafeTensorsPath(checkpointPath);
        string? directory = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        safetensors.torch.save_file(state, path);
    }

    private static WikiModelCheckpoint LoadCheckpoint(string path)
    {
        WikiModelCheckpoint checkpoint = torch.load<WikiModelCheckpoint>(path);
        ValidateCheckpoint(checkpoint);
        string safeTensorsPath = GetSafeTensorsPath(path);
        if (!File.Exists(safeTensorsPath))
            return checkpoint;
        ModuleState safeModel = safetensors.torch.load_file(safeTensorsPath);
        ModuleState expected = checkpoint.CurrentModel ?? checkpoint.Model;
        // The JSON state is authoritative because it retains the exact FP32
        // master weights. SafeTensors intentionally stores the quantized
        // physical representation and is used only as a validated artifact.
        if (!ModuleStatesEqual(safeModel, expected))
        {
            throw new InvalidDataException(
                "Wiki checkpoint SafeTensors sidecar does not match the " +
                "JSON model metadata or quantized weights.");
        }
        return checkpoint;
    }

    internal static ModuleState LoadGenerationModelState(
        WikiModelCheckpoint checkpoint,
        string checkpointPath)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ValidateCheckpoint(checkpoint);
        string bestSafeTensorsPath = GetBestSafeTensorsPath(checkpointPath);
        if (!File.Exists(bestSafeTensorsPath))
            return checkpoint.Model;

        ModuleState safeModel = safetensors.torch.load_file(
            bestSafeTensorsPath);
        if (!ModuleStatesEqual(safeModel, checkpoint.Model))
        {
            throw new InvalidDataException(
                "Wiki best-model SafeTensors sidecar does not match the " +
                "checkpoint model metadata or quantized weights.");
        }
        return safeModel;
    }

    internal static TensorDType ResolveModelDTypeForTraining(
        WikiTrainingConfiguration config)
        => ResolvePrecisionForTraining(config).StorageDType;

    internal static TensorPrecisionMode ResolvePrecisionModeForTraining(
        WikiTrainingConfiguration config)
        => ResolvePrecisionForTraining(config).Mode;

    internal static WikiPrecisionSelection ResolvePrecisionForTraining(
        WikiTrainingConfiguration config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (!config.ResumeFromCheckpoint)
        {
            TensorPrecisionMode newRunMode = config.GetPrecisionMode();
            return new WikiPrecisionSelection(
                newRunMode,
                newRunMode.ToStorageDType());
        }
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

        TensorPrecisionMode checkpointMode =
            GetCheckpointPrecisionMode(checkpoint);
        TensorPrecisionMode? configuredMode = config.GetExplicitPrecisionMode();
        if (configuredMode is not null
            && configuredMode.Value != checkpointMode)
        {
            throw new InvalidDataException(
                $"Configured precisionMode '{FormatPrecisionMode(configuredMode.Value)}' " +
                $"does not match checkpoint precision mode " +
                $"'{FormatPrecisionMode(checkpointMode)}'. Remove precisionMode " +
                "to inherit the checkpoint mode, or use a matching value.");
        }
        return new WikiPrecisionSelection(
            checkpointMode,
            GetCheckpointModelDType(checkpoint));
    }

    internal static WikiResumePosition RestoreTrainingCheckpoint(
        WikiTrainingConfiguration config,
        LanguageModel model,
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
        TensorPrecisionMode checkpointMode =
            GetCheckpointPrecisionMode(checkpoint);
        if (model is Module module && module.PrecisionMode != checkpointMode)
        {
            throw new InvalidDataException(
                $"Checkpoint precision mode '{FormatPrecisionMode(checkpointMode)}' " +
                $"does not match the constructed model precision mode " +
                $"'{FormatPrecisionMode(module.PrecisionMode)}'.");
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
        LanguageModel model,
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
                currentTokenBuffer,
                model is Module module
                    ? module.DType
                    : config.GetModelDType(),
                config.TieWordEmbeddings,
                model is Module precisionModule
                    ? precisionModule.PrecisionMode
                    : config.GetPrecisionMode()));
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
        if (left.FormatVersion != right.FormatVersion
            || left.Parameters.Length != right.Parameters.Length)
            return false;
        for (int index = 0; index < left.Parameters.Length; index++)
        {
            ModuleParameterState first = left.Parameters[index];
            ModuleParameterState second = right.Parameters[index];
            if (first.Index != second.Index
                || first.Name != second.Name
                || !first.Shape.AsSpan().SequenceEqual(second.Shape)
                || first.DType != second.DType
                || first.Values.Length != second.Values.Length)
            {
                return false;
            }
            for (int valueIndex = 0;
                valueIndex < first.Values.Length;
                valueIndex++)
            {
                float expected = second.DType switch
                {
                    TensorDType.Float16 =>
                        (float)(Half)second.Values[valueIndex],
                    TensorDType.BFloat16 =>
                        QuantizeBFloat16(second.Values[valueIndex]),
                    _ => second.Values[valueIndex],
                };
                if (BitConverter.SingleToInt32Bits(first.Values[valueIndex])
                    != BitConverter.SingleToInt32Bits(expected))
                {
                    return false;
                }
            }
        }
        return true;
    }

    private static float QuantizeBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint absolute = bits & 0x7FFFFFFFu;
        if (absolute > 0x7F800000u)
            return BitConverter.UInt32BitsToSingle((bits >> 16 | 0x40u) << 16);
        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle((rounded >> 16) << 16);
    }

    private static void ValidateCheckpoint(WikiModelCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion
            || checkpoint.Model is null)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }

        TensorDType modelDType = GetCheckpointModelDType(checkpoint);
        _ = GetCheckpointPrecisionMode(checkpoint);
        ValidateCheckpointModelState(
            checkpoint.Model,
            modelDType,
            "best model");
        if (checkpoint.CurrentModel is not null)
        {
            ValidateCheckpointModelState(
                checkpoint.CurrentModel,
                modelDType,
                "current model");
        }
    }

    private static void ValidateCheckpointModelState(
        ModuleState state,
        TensorDType modelDType,
        string stateName)
    {
        if (state.FormatVersion != ModuleState.CurrentFormatVersion
            || state.Parameters is null)
        {
            throw new InvalidDataException(
                $"Wiki checkpoint {stateName} state has an unsupported " +
                "format.");
        }

        for (int index = 0; index < state.Parameters.Length; index++)
        {
            ModuleParameterState? parameter = state.Parameters[index];
            if (parameter is null
                || parameter.Index != index
                || parameter.Shape is null
                || parameter.Values is null
                || parameter.DType != modelDType)
            {
                throw new InvalidDataException(
                    $"Wiki checkpoint {stateName} parameter {index} does " +
                    $"not match model dtype " +
                    $"'{FormatPrecisionMode(modelDType)}'.");
            }
        }
    }

    private static string FormatPrecisionMode(TensorDType dtype)
        => FormatPrecisionMode(dtype.ToPrecisionMode());

    private static string FormatPrecisionMode(TensorPrecisionMode mode)
        => mode switch
        {
            TensorPrecisionMode.Mix16_32 =>
                WikiTrainingConfiguration.Mix16_32PrecisionMode,
            TensorPrecisionMode.BFloat16 =>
                WikiTrainingConfiguration.BFloat16PrecisionMode,
            TensorPrecisionMode.Float32 =>
                WikiTrainingConfiguration.Float32PrecisionMode,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    internal sealed record WikiResumePosition(
        int Epoch,
        int CompletedBatches,
        double LossSum,
        long TargetCount,
        long CompletedDocuments,
        int[] TokenBuffer);

    internal readonly record struct WikiPrecisionSelection(
        TensorPrecisionMode Mode,
        TensorDType StorageDType);

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
        int[]? CurrentTokenBuffer = null,
        TensorDType? ModelDType = null,
        bool TieWordEmbeddings = false,
        TensorPrecisionMode? PrecisionMode = null);
}
