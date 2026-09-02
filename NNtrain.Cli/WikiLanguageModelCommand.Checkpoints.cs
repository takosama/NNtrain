using NNtrain.Training.Persistence;
using NNtrain.Training.Optimization;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private const int CheckpointFormatVersion = 8;
    private const int DTypeCheckpointFormatVersion = 5;
    private const int PrecisionModeCheckpointFormatVersion = 6;

    private static void SaveCheckpoint(
        string path,
        WikiModelCheckpoint checkpoint,
        IOptimizer optimizer,
        Module? currentModelSource,
        ICheckpointFaultInjector? faultInjector = null)
        => WikiCheckpointCompatibilityAdapter.Save(
            path,
            checkpoint,
            optimizer,
            currentModelSource,
            faultInjector);

    private static WikiModelCheckpoint LoadCheckpoint(
        string path,
        bool validateSafeTensors = true)
    {
        WikiModelCheckpoint checkpoint =
            WikiCheckpointCompatibilityAdapter.Load(path);
        ValidateCheckpoint(checkpoint);
        if (checkpoint.FormatVersion >= 7)
            return checkpoint;
        if (!validateSafeTensors)
            return checkpoint;
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

    private static WikiModelCheckpoint LoadCheckpointMetadata(string path)
    {
        WikiCheckpointMetadata metadata =
            WikiCheckpointCompatibilityAdapter.LoadMetadata(path);
        WikiModelCheckpoint checkpoint = metadata.ToCheckpointShell();
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }
        _ = GetCheckpointModelDType(checkpoint);
        ValidateBfp8BlockMetadata(checkpoint);
        return checkpoint;
    }

    private static WikiModelCheckpoint LoadCheckpointForResume(
        string path,
        Module model)
    {
        WikiResumeCheckpointData resume =
            WikiCheckpointCompatibilityAdapter.LoadResume(path);
        if (resume.FormatVersion >= 7)
        {
            if (resume.ArtifactSlot is < 0 or > 1)
            {
                throw new InvalidDataException(
                    "Checkpoint artifact slot is invalid.");
            }
            string bestArtifactPath = GetBestModelArtifactPath(
                path,
                GetBestArtifactSlot(
                    resume.ArtifactSlot,
                    resume.BestArtifactSlot));
            if (!File.Exists(bestArtifactPath))
            {
                throw new FileNotFoundException(
                    "Checkpoint best-model artifact was not found.",
                    bestArtifactPath);
            }
            _ = resume.ModelDType
                ?? throw new InvalidDataException(
                    "Checkpoint model dtype is missing.");
            SafeTensorFile.LoadModel(
                GetCurrentModelArtifactPath(path, resume.ArtifactSlot),
                model);
            WikiModelCheckpoint artifactCheckpoint =
                resume.ToCheckpointShell();
            ValidateCheckpoint(artifactCheckpoint);
            return artifactCheckpoint;
        }
        if (resume.CurrentModel is null)
        {
            // Formats predating mid-epoch checkpoints did not carry a
            // current-model state. They are normally small enough for the
            // legacy loader and still retain their exact behavior.
            return LoadCheckpoint(path, validateSafeTensors: false);
        }

        ModuleState currentModel = resume.CurrentModel;
        ModuleState bestModel;
        string bestSafeTensorsPath = GetBestSafeTensorsPath(path);
        if (File.Exists(bestSafeTensorsPath))
        {
            bestModel = safetensors.torch.load_file(bestSafeTensorsPath);
        }
        else if (resume.Epoch == 0)
        {
            // Before the first completed epoch, SaveTrainingCheckpoint writes
            // the same state as both best and current. Reuse the one exact
            // JSON state instead of allocating a duplicate hundreds-of-MB
            // object graph.
            bestModel = currentModel;
        }
        else
        {
            // A legacy checkpoint can have a completed best epoch without a
            // best-model sidecar. Preserve exact compatibility for that rare
            // case instead of silently substituting the current model.
            return LoadCheckpoint(path, validateSafeTensors: false);
        }

        WikiModelCheckpoint checkpoint = resume.ToCheckpoint(
            bestModel,
            currentModel);
        ValidateCheckpoint(checkpoint);
        return checkpoint;
    }

    internal static ModuleState LoadGenerationModelState(
        WikiModelCheckpoint checkpoint,
        string checkpointPath)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ValidateCheckpoint(checkpoint);
        if (checkpoint.FormatVersion >= 7)
        {
            return safetensors.torch.load_file(
                GetBestModelArtifactPath(
                    checkpointPath,
                    GetBestArtifactSlot(
                        checkpoint.ArtifactSlot,
                        checkpoint.BestArtifactSlot)));
        }
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

    internal static void LoadGenerationModelInto(
        WikiModelCheckpoint checkpoint,
        string checkpointPath,
        Module model)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        ArgumentNullException.ThrowIfNull(model);
        ValidateCheckpoint(checkpoint);
        if (checkpoint.FormatVersion >= 7)
        {
            SafeTensorFile.LoadModel(
                GetBestModelArtifactPath(
                    checkpointPath,
                    GetBestArtifactSlot(
                        checkpoint.ArtifactSlot,
                        checkpoint.BestArtifactSlot)),
                model);
            return;
        }
        model.load_state_dict(
            LoadGenerationModelState(checkpoint, checkpointPath));
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
                newRunMode.ToStorageDType(),
                config.Bfp8BlockSize);
        }
        if (!File.Exists(config.CheckpointPath))
        {
            throw new FileNotFoundException(
                "Wiki training checkpoint was not found.",
                config.CheckpointPath);
        }

        // Do not deserialize the multi-gigabyte model and optimizer payload
        // merely to determine the resume dtype. Unknown JSON properties are
        // skipped by System.Text.Json while the small metadata shell is read.
        WikiModelCheckpoint checkpoint =
            LoadCheckpointMetadata(config.CheckpointPath);
        if (!CheckpointArchitectureMatchesConfiguration(checkpoint, config))
        {
            throw new InvalidDataException(
                "Checkpoint model architecture does not match the current " +
                "Wiki training configuration.");
        }
        if (checkpoint.TrainingSeed is int checkpointSeed
            && checkpointSeed != config.Seed)
        {
            throw new InvalidDataException(
                $"Configured training seed '{config.Seed}' does not match " +
                $"checkpoint seed '{checkpointSeed}'. Restore the original " +
                "seed so the mid-epoch shuffle and stochastic operations " +
                "remain reproducible.");
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

        int bfp8BlockSize = config.Bfp8BlockSize;
        if (checkpointMode == TensorPrecisionMode.Mix8_32
            && checkpoint.Bfp8BlockSize is int checkpointBlockSize)
        {
            // An omitted setting preserves the checkpoint's exact payload
            // contract. An explicit setting is a requested precision
            // migration: the FP32 master stored in the checkpoint is loaded
            // into a model created with the requested descriptor and safely
            // requantized once, before CUDA residency is prepared.
            bfp8BlockSize = config.HasExplicitBfp8BlockSize
                ? config.Bfp8BlockSize
                : checkpointBlockSize;
        }
        return new WikiPrecisionSelection(
            checkpointMode,
            GetCheckpointModelDType(checkpoint),
            bfp8BlockSize);
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

        // The regular checkpoint contains best model, current model, and a
        // SafeTensors duplicate. Load only the exact current state plus the
        // optimizer; best weights come from their compact sidecar (or share
        // current state before the first completed epoch).
        WikiModelCheckpoint checkpoint =
            LoadCheckpointForResume(config.CheckpointPath, model);
        if (!CheckpointArchitectureMatchesConfiguration(checkpoint, config))
        {
            throw new InvalidDataException(
                "Checkpoint model architecture does not match the current " +
                "Wiki training configuration.");
        }
        if (checkpoint.TrainingSeed is int checkpointSeed
            && checkpointSeed != config.Seed)
        {
            throw new InvalidDataException(
                $"Configured training seed '{config.Seed}' does not match " +
                $"checkpoint seed '{checkpointSeed}'. Restore the original " +
                "seed so the mid-epoch shuffle and stochastic operations " +
                "remain reproducible.");
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

        if (checkpoint.FormatVersion < 7)
        {
            ModuleState currentModel =
                checkpoint.CurrentModel ?? checkpoint.Model;
            model.load_state_dict(currentModel);
        }
        model.RestoreTrainingRandomState(checkpoint.TrainingRandomState);

        LRSchedulerStateDictionary? schedulerState = checkpoint.Scheduler;
        // Stop the returned checkpoint shell from retaining the duplicate
        // current state after it has moved to the model. This makes it
        // collectible before streamed optimizer state and the first resumed
        // forward allocate their moment/activation buffers.
        checkpoint = checkpoint with
        {
            CurrentModel = null,
            Optimizer = null,
            Scheduler = null,
        };
        GC.Collect(
            GC.MaxGeneration,
            GCCollectionMode.Aggressive,
            blocking: true,
            compacting: true);
        bool optimizerRestored =
            CheckpointOptimizerStateStream.TryLoad(
                config.CheckpointPath,
                checkpoint,
                optimizer,
                output);
        if (optimizerRestored)
        {
            ApplyOrdinaryMuonPolicyAfterResume(
                config,
                optimizer,
                output);
            ApplyNekoMuonNewtonSchulzDepthPolicyOverride(
                config,
                optimizer,
                output);
        }
        if (optimizerRestored && schedulerState is not null)
        {
            scheduler.load_state_dict(schedulerState);
            schedulerState = null;
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);
        }
        else
        {
            output.WriteLine(
                "checkpoint contains model weights only; optimizer and " +
                "scheduler start from their configured initial state");
        }

        // Artifact checkpoints keep the best weights on disk. Loading them
        // here would retain a second complete model for the rest of training.
        // Legacy checkpoints still need their embedded best state until the
        // first format-8 save migrates it to an artifact.
        bestState = checkpoint.FormatVersion >= 7
            ? null
            : checkpoint.Model;
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
            hasPartialEpoch ? checkpoint.CurrentTokenBuffer ?? [] : [],
            checkpoint.AdaptiveCudaShardState);
    }

    internal static void ApplyOrdinaryMuonPolicyAfterResume(
        WikiTrainingConfiguration config,
        IOptimizer optimizer,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(output);
        if (!config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer))
            return;

        int applied = 0;
        foreach (NekoMuon muon in OptimizerBundle
            .GetCheckpointLeafOptimizers(optimizer)
            .OfType<NekoMuon>())
        {
            // State restoration also restores the historical NekoMuon
            // options. Reasserting the ordinary-Muon contract here prevents
            // an old checkpoint from silently changing momentum or NS
            // cadence while preserving its accumulated fast moment and step.
            muon.SetOrdinaryMuonPolicy();
            applied++;
        }

        if (applied == 0)
        {
            throw new InvalidDataException(
                "Muon was configured, but the restored optimizer has no " +
                "compatible matrix-optimizer state.");
        }

        output.WriteLine(
            "resume Muon policy = momentum 0.95, Nesterov, " +
            "fixed NS5 every step (runtime override)");
    }

    internal static void ApplyNekoMuonNewtonSchulzDepthPolicyOverride(
        WikiTrainingConfiguration config,
        IOptimizer optimizer,
        TextWriter output)
    {
        if (!config.HasNekoMuonNewtonSchulzDepthPolicyOverride)
            return;

        NekoMuonNewtonSchulzDepthMode mode =
            config.GetNekoMuonNewtonSchulzDepthMode();
        float depth = config.GetNekoMuonNewtonSchulzDepth();
        int applied = 0;
        foreach (NekoMuon nekoMuon in OptimizerBundle
            .GetCheckpointLeafOptimizers(optimizer)
            .OfType<NekoMuon>())
        {
            nekoMuon.SetNewtonSchulzDepthPolicy(mode, depth);
            applied++;
        }

        if (applied == 0)
        {
            throw new InvalidDataException(
                "A NekoMuon Newton-Schulz depth policy was configured, but " +
                "the restored optimizer has no NekoMuon state.");
        }

        output.WriteLine(mode == NekoMuonNewtonSchulzDepthMode.Adaptive
            ? "resume NekoMuon Newton-Schulz depth policy = adaptive " +
                "(runtime override)"
            : $"resume NekoMuon Newton-Schulz depth policy = " +
                $"{mode.ToString().ToLowerInvariant()} {depth:G} " +
                "(runtime override)");
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
        int[]? currentTokenBuffer = null,
        ModuleState? currentStateOverride = null,
        CudaAdaptiveShardState? adaptiveCudaShardState = null,
        ICheckpointFaultInjector? checkpointFaultInjector = null)
    {
        Module? currentModelSource = currentStateOverride is null
            ? model
            : null;
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
                currentStateOverride ?? EmptyModuleState(),
                Optimizer: null,
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
                    : config.GetPrecisionMode(),
                 Bfp8BlockSize: GetCheckpointBfp8BlockSize(model, config),
                 TrainingSeed: config.Seed,
                 TrainingRandomState: model.CaptureTrainingRandomState(),
                 AdaptiveCudaShardState: adaptiveCudaShardState),
            optimizer,
            currentModelSource,
            checkpointFaultInjector);
    }

    internal static ModuleState LoadBestTrainingModelState(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        WikiModelCheckpoint checkpoint = LoadCheckpoint(checkpointPath);
        return LoadGenerationModelState(checkpoint, checkpointPath);
    }

    internal static void LoadBestTrainingModelInto(
        string checkpointPath,
        Module model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentNullException.ThrowIfNull(model);
        WikiModelCheckpoint checkpoint = LoadCheckpoint(checkpointPath);
        LoadGenerationModelInto(checkpoint, checkpointPath, model);
    }

    internal static string GetCurrentModelArtifactPath(
        string checkpointPath,
        int artifactSlot)
        => GetArtifactPath(
            checkpointPath,
            $"current.{ValidateArtifactSlot(artifactSlot)}.safetensors");

    internal static string GetBestModelArtifactPath(
        string checkpointPath,
        int artifactSlot)
        => GetArtifactPath(
            checkpointPath,
            $"best.{ValidateArtifactSlot(artifactSlot)}.safetensors");

    internal static string GetOptimizerArtifactPath(
        string checkpointPath,
        int artifactSlot,
        int optimizerIndex)
    {
        if (optimizerIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(optimizerIndex));
        return GetArtifactPath(
            checkpointPath,
            $"optimizer.{ValidateArtifactSlot(artifactSlot)}." +
            $"{optimizerIndex}.json");
    }

    internal static string GetOptimizerBinaryArtifactPath(
        string checkpointPath,
        int artifactSlot,
        int optimizerIndex)
    {
        if (optimizerIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(optimizerIndex));
        return GetArtifactPath(
            checkpointPath,
            $"optimizer.{ValidateArtifactSlot(artifactSlot)}." +
            $"{optimizerIndex}.bin");
    }

    private static int GetBestArtifactSlot(
        int currentArtifactSlot,
        int bestArtifactSlot)
        => bestArtifactSlot is 0 or 1
            ? bestArtifactSlot
            : ValidateArtifactSlot(currentArtifactSlot);

    private static string GetArtifactPath(
        string checkpointPath,
        string artifactSuffix)
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        return Path.Combine(
            directory,
            $"{Path.GetFileNameWithoutExtension(fullPath)}." +
            artifactSuffix);
    }

    private static int ValidateArtifactSlot(int artifactSlot)
        => artifactSlot is 0 or 1
            ? artifactSlot
            : throw new ArgumentOutOfRangeException(nameof(artifactSlot));

    private static ModuleState EmptyModuleState()
        => new(ModuleState.CurrentFormatVersion, []);

    private static ModuleState RelabelStateDType(
        ModuleState state,
        TensorDType dtype)
        => state with
        {
            Parameters = state.Parameters
                .Select(parameter => parameter.DType == dtype
                    ? parameter
                    : parameter with { DType = dtype })
                .ToArray(),
        };

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

    private static int? GetCheckpointBfp8BlockSize(
        LanguageModel model,
        WikiTrainingConfiguration config)
    {
        TensorPrecisionMode precisionMode = model is Module module
            ? module.PrecisionMode
            : config.GetPrecisionMode();
        if (precisionMode != TensorPrecisionMode.Mix8_32)
            return null;

        int? blockSize = null;
        foreach (Parameter parameter in model.parameters())
        {
            Bfp8QuantizationDescriptor? descriptor =
                parameter.T.Bfp8Quantization;
            if (descriptor is not
                { Granularity: Bfp8ScaleGranularity.Block })
            {
                throw new InvalidOperationException(
                    "A mix8_32 checkpoint requires block-scaled BFP8 " +
                    "parameter storage.");
            }
            if (blockSize is int existing
                && existing != descriptor.BlockSize)
            {
                throw new InvalidOperationException(
                    "A mix8_32 checkpoint cannot contain multiple BFP8 " +
                    "parameter block sizes.");
            }
            blockSize = descriptor.BlockSize;
        }
        return blockSize ?? config.Bfp8BlockSize;
    }

    private static void ValidateBfp8BlockMetadata(
        WikiModelCheckpoint checkpoint)
    {
        TensorPrecisionMode precisionMode =
            GetCheckpointPrecisionMode(checkpoint);
        if (checkpoint.Bfp8BlockSize is null)
            return;
        if (checkpoint.Bfp8BlockSize <= 0
            || precisionMode != TensorPrecisionMode.Mix8_32)
        {
            throw new InvalidDataException(
                "Wiki checkpoint BFP8 block-size metadata is invalid.");
        }
    }

    private static void ValidateCheckpoint(
        WikiModelCheckpoint checkpoint,
        bool requireArtifactMetadata = true)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion
            || checkpoint.Model is null)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }

        TensorDType modelDType = GetCheckpointModelDType(checkpoint);
        ValidateBfp8BlockMetadata(checkpoint);
        if (checkpoint.FormatVersion >= 7 && requireArtifactMetadata)
        {
            if (checkpoint.ArtifactSlot is < 0 or > 1
                || (checkpoint.FormatVersion >= 8
                    && checkpoint.BestArtifactSlot is < 0 or > 1)
                || checkpoint.OptimizerStateTypes is null
                || checkpoint.OptimizerStateTypes.Length == 0
                || checkpoint.OptimizerStateTypes.Any(
                    string.IsNullOrWhiteSpace))
            {
                throw new InvalidDataException(
                    "Wiki checkpoint artifact metadata is invalid.");
            }
            if (checkpoint.Model.Parameters is { Length: 0 }
                && checkpoint.CurrentModel is null)
            {
                return;
            }
        }
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
            TensorPrecisionMode.Bfp8 =>
                WikiTrainingConfiguration.Bfp8PrecisionMode,
            TensorPrecisionMode.Mix8_32 =>
                WikiTrainingConfiguration.Mix8_32PrecisionMode,
            _ => throw new ArgumentOutOfRangeException(nameof(mode)),
        };

    internal sealed record WikiResumePosition(
        int Epoch,
        int CompletedBatches,
        double LossSum,
        long TargetCount,
        long CompletedDocuments,
        int[] TokenBuffer,
        CudaAdaptiveShardState? AdaptiveCudaShardState = null);

    internal readonly record struct WikiPrecisionSelection(
        TensorPrecisionMode Mode,
        TensorDType StorageDType,
        int Bfp8BlockSize);

    /// <summary>
    /// Scalar-only view of a Wiki checkpoint. System.Text.Json skips the
    /// omitted model and optimizer properties while streaming through the
    /// file, so resume precision detection does not allocate their payloads.
    /// </summary>
    private sealed record WikiCheckpointMetadata(
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
        string? ModelArchitecture = null,
        int HyenaFilterWidth = 64,
        int ForgetMemoryKeyWidth = 16,
        int ForgetMemoryValueWidth = 16,
        float ForgetMemoryRetentionMinimum = 0.5f,
        float ForgetMemoryRetentionMaximum = 0.99f,
        int CompletedEpoch = 0,
        long GlobalStep = 0,
        int CurrentEpoch = 0,
        int CompletedBatchesInEpoch = 0,
        double CurrentLossSum = 0d,
        long CurrentTargetCount = 0,
        long CompletedDocumentsInEpoch = 0,
        int[]? CurrentTokenBuffer = null,
        TensorDType? ModelDType = null,
        bool TieWordEmbeddings = false,
        TensorPrecisionMode? PrecisionMode = null,
        int ArtifactSlot = -1,
        int BestArtifactSlot = -1,
        string[]? OptimizerStateTypes = null,
        int? Bfp8BlockSize = null,
        int? TrainingSeed = null,
        TrainingRandomState? TrainingRandomState = null,
        CudaAdaptiveShardState? AdaptiveCudaShardState = null)
    {
        internal WikiModelCheckpoint ToCheckpointShell()
            => new(
                FormatVersion,
                Epoch,
                ValidationLoss,
                VocabularySize,
                ContextLength,
                ModelWidth,
                Heads,
                HiddenSize,
                Layers,
                Dropout,
                InitializationScale,
                new ModuleState(ModuleState.CurrentFormatVersion, []),
                ModelArchitecture,
                HyenaFilterWidth,
                ForgetMemoryKeyWidth,
                ForgetMemoryValueWidth,
                ForgetMemoryRetentionMinimum,
                ForgetMemoryRetentionMaximum,
                CompletedEpoch,
                CurrentModel: null,
                Optimizer: null,
                Scheduler: null,
                GlobalStep,
                CurrentEpoch,
                CompletedBatchesInEpoch,
                CurrentLossSum,
                CurrentTargetCount,
                CompletedDocumentsInEpoch,
                CurrentTokenBuffer,
                ModelDType,
                TieWordEmbeddings,
                PrecisionMode,
                ArtifactSlot,
                BestArtifactSlot,
                 OptimizerStateTypes,
                 Bfp8BlockSize,
                 TrainingSeed,
                 TrainingRandomState,
                 AdaptiveCudaShardState);
    }

    /// <summary>
    /// Resume-oriented checkpoint view. The duplicated best-model JSON field
    /// is deliberately absent; the compact best SafeTensors sidecar supplies
    /// it after an epoch, while a partial first epoch shares CurrentModel.
    /// </summary>
    private sealed record WikiResumeCheckpointData(
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
        string? ModelArchitecture = null,
        int HyenaFilterWidth = 64,
        int ForgetMemoryKeyWidth = 16,
        int ForgetMemoryValueWidth = 16,
        float ForgetMemoryRetentionMinimum = 0.5f,
        float ForgetMemoryRetentionMaximum = 0.99f,
        int CompletedEpoch = 0,
        ModuleState? CurrentModel = null,
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
        TensorPrecisionMode? PrecisionMode = null,
        int ArtifactSlot = -1,
        int BestArtifactSlot = -1,
        string[]? OptimizerStateTypes = null,
        int? Bfp8BlockSize = null,
        int? TrainingSeed = null,
        TrainingRandomState? TrainingRandomState = null,
        CudaAdaptiveShardState? AdaptiveCudaShardState = null)
    {
        internal WikiModelCheckpoint ToCheckpointShell()
            => new(
                FormatVersion,
                Epoch,
                ValidationLoss,
                VocabularySize,
                ContextLength,
                ModelWidth,
                Heads,
                HiddenSize,
                Layers,
                Dropout,
                InitializationScale,
                EmptyModuleState(),
                ModelArchitecture,
                HyenaFilterWidth,
                ForgetMemoryKeyWidth,
                ForgetMemoryValueWidth,
                ForgetMemoryRetentionMinimum,
                ForgetMemoryRetentionMaximum,
                CompletedEpoch,
                CurrentModel: null,
                Optimizer: null,
                Scheduler,
                GlobalStep,
                CurrentEpoch,
                CompletedBatchesInEpoch,
                CurrentLossSum,
                CurrentTargetCount,
                CompletedDocumentsInEpoch,
                CurrentTokenBuffer,
                ModelDType,
                TieWordEmbeddings,
                PrecisionMode,
                ArtifactSlot,
                BestArtifactSlot,
                 OptimizerStateTypes,
                 Bfp8BlockSize,
                 TrainingSeed,
                 TrainingRandomState,
                 AdaptiveCudaShardState);

        internal WikiModelCheckpoint ToCheckpoint(ModuleState currentModel)
            => ToCheckpoint(EmptyModuleState(), currentModel);

        internal WikiModelCheckpoint ToCheckpoint(
            ModuleState bestModel,
            ModuleState currentModel)
            => new(
                FormatVersion,
                Epoch,
                ValidationLoss,
                VocabularySize,
                ContextLength,
                ModelWidth,
                Heads,
                HiddenSize,
                Layers,
                Dropout,
                InitializationScale,
                bestModel,
                ModelArchitecture,
                HyenaFilterWidth,
                ForgetMemoryKeyWidth,
                ForgetMemoryValueWidth,
                ForgetMemoryRetentionMinimum,
                ForgetMemoryRetentionMaximum,
                CompletedEpoch,
                currentModel,
                Optimizer: null,
                Scheduler,
                GlobalStep,
                CurrentEpoch,
                CompletedBatchesInEpoch,
                CurrentLossSum,
                CurrentTargetCount,
                CompletedDocumentsInEpoch,
                CurrentTokenBuffer,
                ModelDType,
                TieWordEmbeddings,
                PrecisionMode,
                ArtifactSlot,
                BestArtifactSlot,
                 OptimizerStateTypes,
                 Bfp8BlockSize,
                 TrainingSeed,
                 TrainingRandomState,
                 AdaptiveCudaShardState);
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
        long GlobalStep = 0,
        int CurrentEpoch = 0,
        int CompletedBatchesInEpoch = 0,
        double CurrentLossSum = 0d,
        long CurrentTargetCount = 0,
        long CompletedDocumentsInEpoch = 0,
        int[]? CurrentTokenBuffer = null,
        TensorDType? ModelDType = null,
        bool TieWordEmbeddings = false,
        TensorPrecisionMode? PrecisionMode = null,
        int ArtifactSlot = -1,
        int BestArtifactSlot = -1,
        string[]? OptimizerStateTypes = null,
        int? Bfp8BlockSize = null,
        int? TrainingSeed = null,
        TrainingRandomState? TrainingRandomState = null,
        CudaAdaptiveShardState? AdaptiveCudaShardState = null);
}
