namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    private const int CheckpointFormatVersion = 8;
    private const int DTypeCheckpointFormatVersion = 5;
    private const int PrecisionModeCheckpointFormatVersion = 6;

    private static void SaveCheckpoint(
        string path,
        WikiModelCheckpoint checkpoint,
        IOptimizer optimizer)
    {
        ValidateCheckpoint(checkpoint, requireArtifactMetadata: false);
        ArgumentNullException.ThrowIfNull(optimizer);
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        WikiCheckpointMetadata? previous = File.Exists(fullPath)
            ? torch.load<WikiCheckpointMetadata>(fullPath)
            : null;
        int artifactSlot = previous is { FormatVersion: >= 7, ArtifactSlot: 0 }
            ? 1
            : 0;
        int previousBestSlot = previous is { FormatVersion: >= 7 }
            ? GetBestArtifactSlot(
                previous.ArtifactSlot,
                previous.BestArtifactSlot)
            : -1;
        bool bestMetadataChanged = previous is null
            || previous.Epoch != checkpoint.Epoch
            || BitConverter.SingleToInt32Bits(previous.ValidationLoss)
                != BitConverter.SingleToInt32Bits(
                    checkpoint.ValidationLoss);
        bool writeBestArtifact = previousBestSlot < 0
            || !File.Exists(GetBestModelArtifactPath(
                fullPath,
                previousBestSlot))
            || bestMetadataChanged;
        int bestArtifactSlot = writeBestArtifact
            ? previousBestSlot is 0 ? 1 : 0
            : previousBestSlot;
        if (writeBestArtifact && checkpoint.Model.Parameters.Length == 0)
        {
            throw new InvalidDataException(
                "The best-model artifact is missing or its metadata changed, " +
                "but the in-memory best state has already been released. " +
                "Refusing to replace it with the current model.");
        }
        ModuleState currentState = checkpoint.CurrentModel ?? checkpoint.Model;
        TensorPrecisionMode precisionMode =
            GetCheckpointPrecisionMode(checkpoint);
        ModuleState currentArtifact = precisionMode
            == TensorPrecisionMode.Mix16_32
            ? RelabelStateDType(currentState, TensorDType.Float32)
            : currentState;
        safetensors.torch.save_file(
            currentArtifact,
            GetCurrentModelArtifactPath(fullPath, artifactSlot));
        if (writeBestArtifact)
        {
            safetensors.torch.save_file(
                checkpoint.Model,
                GetBestModelArtifactPath(fullPath, bestArtifactSlot));
        }

        IReadOnlyList<IOptimizer> leaves =
            OptimizerStateStream.GetLeafOptimizers(optimizer);
        var optimizerTypes = new string[leaves.Count];
        for (int index = 0; index < leaves.Count; index++)
        {
            optimizerTypes[index] =
                OptimizerStateStream.GetStateType(leaves[index]);
            SaveOptimizerBinaryArtifact(
                GetOptimizerBinaryArtifactPath(
                    fullPath,
                    artifactSlot,
                    index),
                leaves[index]);
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);
        }

        // The manifest is committed last. Until this atomic move succeeds,
        // readers continue to use the other complete artifact slot.
        var manifest = checkpoint with
        {
            Model = EmptyModuleState(),
            CurrentModel = null,
            Optimizer = null,
            ArtifactSlot = artifactSlot,
            BestArtifactSlot = bestArtifactSlot,
            OptimizerStateTypes = optimizerTypes,
        };
        torch.save(manifest, fullPath);

        // Drop the exact current-model snapshot before the caller writes an
        // optional human-facing epoch snapshot. At production width this is
        // hundreds of MB and otherwise remains in Gen2 until memory pressure.
        checkpoint = manifest;
        currentState = null!;
        currentArtifact = null!;
        GC.Collect(
            GC.MaxGeneration,
            GCCollectionMode.Aggressive,
            blocking: true,
            compacting: true);
    }

    private static void SaveOptimizerBinaryArtifact(
        string path,
        IOptimizer optimizer)
    {
        string temporaryPath = path + ".tmp";
        try
        {
            using (var stream = new FileStream(
                temporaryPath,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 4 * 1024 * 1024,
                FileOptions.SequentialScan))
            {
                OptimizerStateStream.SaveStateBinary(optimizer, stream);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporaryPath, path, overwrite: true);
        }
        catch
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
            throw;
        }
    }

    private static WikiModelCheckpoint LoadCheckpoint(
        string path,
        bool validateSafeTensors = true)
    {
        WikiModelCheckpoint checkpoint = torch.load<WikiModelCheckpoint>(path);
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
            torch.load<WikiCheckpointMetadata>(path);
        WikiModelCheckpoint checkpoint = metadata.ToCheckpointShell();
        if (checkpoint.FormatVersion is < 1 or > CheckpointFormatVersion)
        {
            throw new InvalidDataException(
                "Wiki model checkpoint has an unsupported format.");
        }
        _ = GetCheckpointModelDType(checkpoint);
        _ = GetCheckpointPrecisionMode(checkpoint);
        return checkpoint;
    }

    private static WikiModelCheckpoint LoadCheckpointForResume(string path)
    {
        WikiResumeCheckpointData resume =
            torch.load<WikiResumeCheckpointData>(path);
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
            TensorDType modelDType = resume.ModelDType
                ?? throw new InvalidDataException(
                    "Checkpoint model dtype is missing.");
            ModuleState artifactCurrentModel = safetensors.torch.load_file(
                GetCurrentModelArtifactPath(path, resume.ArtifactSlot));
            artifactCurrentModel = RelabelStateDType(
                artifactCurrentModel,
                modelDType);
            WikiModelCheckpoint artifactCheckpoint = resume.ToCheckpoint(
                artifactCurrentModel);
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

        // The regular checkpoint contains best model, current model, and a
        // SafeTensors duplicate. Load only the exact current state plus the
        // optimizer; best weights come from their compact sidecar (or share
        // current state before the first completed epoch).
        WikiModelCheckpoint checkpoint =
            LoadCheckpointForResume(config.CheckpointPath);
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

        ModuleState? currentModel =
            checkpoint.CurrentModel ?? checkpoint.Model;
        model.load_state_dict(currentModel);
        currentModel = null;

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
            hasPartialEpoch ? checkpoint.CurrentTokenBuffer ?? [] : []);
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
        foreach (NekoMuon nekoMuon in OptimizerStateStream
            .GetLeafOptimizers(optimizer)
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
        ModuleState? currentStateOverride = null)
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
                currentStateOverride ?? model.state_dict(),
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
                    : config.GetPrecisionMode()),
            optimizer);
    }

    internal static ModuleState LoadBestTrainingModelState(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        WikiModelCheckpoint checkpoint = LoadCheckpoint(checkpointPath);
        return LoadGenerationModelState(checkpoint, checkpointPath);
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
        _ = GetCheckpointPrecisionMode(checkpoint);
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
        string[]? OptimizerStateTypes = null)
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
                OptimizerStateTypes);
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
        string[]? OptimizerStateTypes = null)
    {
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
                OptimizerStateTypes);
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
        string[]? OptimizerStateTypes = null);
}
