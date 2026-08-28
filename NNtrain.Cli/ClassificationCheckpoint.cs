using NNtrain.Training.Optimization;
using NNtrain.Training.Persistence;

namespace NNtrain;

internal sealed record ClassificationTrainingCheckpoint(
    int FormatVersion,
    int CompletedEpoch,
    ModuleState Model,
    OptimizerStateDictionary Optimizer,
    LRSchedulerStateDictionary Scheduler,
    ModuleState? BestModel,
    int BestEpoch,
    float BestEvaluationLoss,
    float EarlyStoppingReferenceLoss,
    int EpochsWithoutImprovement,
    int CurrentEpoch = 0,
    int CompletedUpdatesInEpoch = 0,
    double CurrentTrainingLossSum = 0d,
    int CurrentTrainingCorrect = 0,
    int CurrentTrainingSamples = 0,
    int ArtifactSlot = -1,
    int BestArtifactSlot = -1,
    string[]? OptimizerStateTypes = null,
    string[]? OptimizerLeafNames = null)
{
    internal const int CurrentFormatVersion = 2;
}

internal static class ClassificationCheckpoint
{
    private static readonly CheckpointVersionRegistry<
        ClassificationTrainingCheckpoint> Formats = CreateFormats();

    internal static void Save(
        string path,
        ClassificationTrainingCheckpoint checkpoint,
        ICheckpointFaultInjector? faultInjector = null)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        string fullPath = Path.GetFullPath(path);
        var repository = new CheckpointRepository(fullPath, faultInjector);
        Formats.Write(
            repository,
            checkpoint.FormatVersion,
            checkpoint);
    }

    /// <summary>
    /// Writes the current model directly from parameter storage. The legacy
    /// checkpoint overload remains available for callers that already own a
    /// ModuleState, while the training path avoids constructing one.
    /// </summary>
    internal static void Save(
        string path,
        Module currentModel,
        ClassificationTrainingCheckpoint checkpoint,
        ICheckpointFaultInjector? faultInjector = null)
    {
        ArgumentNullException.ThrowIfNull(currentModel);
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (checkpoint.FormatVersion
            != ClassificationTrainingCheckpoint.CurrentFormatVersion)
        {
            throw new NotSupportedException(
                $"No streaming classification checkpoint writer is " +
                $"registered for format version {checkpoint.FormatVersion}.");
        }

        string fullPath = Path.GetFullPath(path);
        var repository = new CheckpointRepository(fullPath, faultInjector);
        using CheckpointWriteTransaction transaction = repository.BeginWrite();
        WriteStreamingCheckpoint(
            transaction,
            new ClassificationWriteRequest(currentModel, checkpoint));
        if (!transaction.IsCommitted)
        {
            throw new InvalidOperationException(
                "The classification checkpoint manifest was not committed.");
        }
    }

    /// <summary>
    /// Publishes a bounded-memory v2 training checkpoint. Current/best model
    /// tensors and each optimizer leaf are artifacts; the JSON manifest owns
    /// only scalar resume metadata and is committed last.
    /// </summary>
    internal static void Save(
        string path,
        Module currentModel,
        IOptimizer optimizer,
        ClassificationTrainingCheckpoint checkpoint,
        bool publishCurrentAsBest,
        ICheckpointFaultInjector? faultInjector = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(currentModel);
        ArgumentNullException.ThrowIfNull(optimizer);
        ArgumentNullException.ThrowIfNull(checkpoint);
        if (checkpoint.FormatVersion
            != ClassificationTrainingCheckpoint.CurrentFormatVersion)
        {
            throw new NotSupportedException(
                $"No artifact classification checkpoint writer is " +
                $"registered for format version {checkpoint.FormatVersion}.");
        }

        string fullPath = Path.GetFullPath(path);
        ClassificationArtifactMetadata? previous = File.Exists(fullPath)
            ? torch.load<ClassificationArtifactMetadata>(fullPath)
            : null;
        int artifactSlot = SelectCurrentArtifactSlot(fullPath, previous);
        int previousBestSlot = ResolveExistingBestArtifactSlot(
            fullPath,
            previous?.BestArtifactSlot ?? -1);
        int bestArtifactSlot = previousBestSlot;
        bool writeBestArtifact = publishCurrentAsBest;
        if (writeBestArtifact)
        {
            bestArtifactSlot = previousBestSlot is 0 ? 1 : 0;
        }
        else if (checkpoint.BestEpoch > 0 && bestArtifactSlot < 0)
        {
            throw new InvalidDataException(
                "Best-model metadata exists, but its streaming artifact is " +
                "missing. Refusing to silently replace it with the current model.");
        }

        OptimizerBundle bundle = OptimizerBundle.Wrap(optimizer);
        IReadOnlyList<IOptimizer> leaves = bundle.LeafOptimizers;
        var optimizerTypes = new string[leaves.Count];
        var optimizerLeafNames = new string[leaves.Count];
        for (int index = 0; index < leaves.Count; index++)
        {
            optimizerTypes[index] = OptimizerStateStream.GetStateType(leaves[index]);
            optimizerLeafNames[index] = bundle.Leaves[index].Name;
        }

        var manifest = checkpoint with
        {
            Model = EmptyModuleState(),
            Optimizer = EmptyOptimizerState(),
            BestModel = null,
            ArtifactSlot = artifactSlot,
            BestArtifactSlot = bestArtifactSlot,
            OptimizerStateTypes = optimizerTypes,
            OptimizerLeafNames = optimizerLeafNames,
        };

        var repository = new CheckpointRepository(fullPath, faultInjector);
        using CheckpointWriteTransaction transaction = repository.BeginWrite();
        string currentArtifactPath = GetCurrentModelArtifactPath(
            fullPath,
            artifactSlot);
        transaction.PublishArtifact(
            currentArtifactPath,
            stagingPath => SafeTensorFile.SaveModel(
                currentModel,
                stagingPath,
                artifactDTypeOverride: GetArtifactDTypeOverride(currentModel)),
            preservePreviousOnRollback: false);

        if (writeBestArtifact)
        {
            transaction.PublishArtifact(
                GetBestModelArtifactPath(fullPath, bestArtifactSlot),
                stagingPath => File.Copy(
                    currentArtifactPath,
                    stagingPath,
                    overwrite: false),
                preservePreviousOnRollback: false);
        }

        for (int index = 0; index < leaves.Count; index++)
        {
            IOptimizer leaf = leaves[index];
            transaction.PublishArtifact(
                GetOptimizerArtifactPath(fullPath, artifactSlot, index),
                stagingPath => SaveOptimizerArtifact(stagingPath, leaf),
                preservePreviousOnRollback: false);
        }

        transaction.CommitManifest(
            stagingPath => torch.save(manifest, stagingPath));
        if (!transaction.IsCommitted)
        {
            throw new InvalidOperationException(
                "The classification checkpoint manifest was not committed.");
        }
    }

    internal static ClassificationTrainingCheckpoint Load(string path)
    {
        ClassificationTrainingCheckpoint checkpoint;
        try
        {
            checkpoint = Formats.Read(path);
        }
        catch (NotSupportedException exception)
        {
            throw new InvalidDataException(
                "Classification training checkpoint is incompatible.",
                exception);
        }
        if (checkpoint.FormatVersion is < 1
                or > ClassificationTrainingCheckpoint.CurrentFormatVersion
            || checkpoint.CompletedEpoch < 0
            || checkpoint.CurrentEpoch < 0
            || checkpoint.CompletedUpdatesInEpoch < 0
            || !double.IsFinite(checkpoint.CurrentTrainingLossSum)
            || checkpoint.CurrentTrainingLossSum < 0d
            || checkpoint.CurrentTrainingCorrect < 0
            || checkpoint.CurrentTrainingSamples < 0
            || checkpoint.CurrentTrainingCorrect
                > checkpoint.CurrentTrainingSamples
            || checkpoint.Model is null
            || checkpoint.Optimizer is null
            || checkpoint.Scheduler is null)
        {
            throw new InvalidDataException(
                "Classification training checkpoint is incompatible.");
        }
        string safeTensorsPath = checkpoint.ArtifactSlot is 0 or 1
            ? GetCurrentModelArtifactPath(path, checkpoint.ArtifactSlot)
            : GetSafeTensorsPath(path);
        if (!File.Exists(safeTensorsPath))
            return checkpoint;

        ModuleState safeModel = safetensors.torch.load_file(safeTensorsPath);
        return checkpoint.Model.Parameters.Length == 0
            || ModuleStatesEqual(safeModel, checkpoint.Model)
            ? checkpoint with { Model = safeModel }
            : checkpoint;
    }

    /// <summary>
    /// Restores the current model directly into an existing model while the
    /// manifest is parsed. This keeps exact FP32 master values for mixed
    /// precision; the byte-compatible SafeTensors artifact remains the
    /// transaction artifact. Legacy readers stay available when it is absent.
    /// </summary>
    internal static ClassificationTrainingCheckpoint LoadIntoModel(
        string path,
        Module model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(model);
        string fullPath = Path.GetFullPath(path);
        ClassificationResumeData resume =
            torch.load<ClassificationResumeData>(fullPath);
        if (resume.ArtifactSlot is 0 or 1)
        {
            string artifactPath = GetCurrentModelArtifactPath(
                fullPath,
                resume.ArtifactSlot);
            if (!File.Exists(artifactPath))
            {
                throw new FileNotFoundException(
                    "Classification current-model artifact was not found.",
                    artifactPath);
            }
            SafeTensorFile.LoadModel(artifactPath, model);
            ClassificationTrainingCheckpoint streamed = resume.ToCheckpoint();
            ValidateResumeMetadata(streamed);
            return streamed;
        }

        string safeTensorsPath = GetSafeTensorsPath(fullPath);
        if (!File.Exists(safeTensorsPath))
        {
            ClassificationTrainingCheckpoint legacy = Load(fullPath);
            model.load_state_dict(legacy.Model);
            return legacy with { Model = EmptyModuleState() };
        }

        ClassificationTrainingCheckpoint checkpoint =
            LoadStreamingManifest(fullPath, model, safeTensorsPath);
        ValidateResumeMetadata(checkpoint);
        return checkpoint;
    }

    /// <summary>
    /// Restores the current model and optimizer without constructing either
    /// aggregate state dictionary. Legacy v1/v2 JSON payloads are consumed as
    /// bounded streams and upgraded lazily to a best-model sidecar.
    /// </summary>
    internal static ClassificationTrainingCheckpoint LoadIntoModel(
        string path,
        Module model,
        IOptimizer optimizer,
        TextWriter? output = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(optimizer);
        string fullPath = Path.GetFullPath(path);
        ClassificationResumeData resume =
            torch.load<ClassificationResumeData>(fullPath);
        ClassificationTrainingCheckpoint checkpoint = resume.ToCheckpoint();
        ValidateResumeMetadata(checkpoint);

        if (resume.ArtifactSlot is 0 or 1)
        {
            RestoreCurrentArtifact(fullPath, resume.ArtifactSlot, model);
            RestoreOptimizerArtifacts(
                fullPath,
                resume.ArtifactSlot,
                resume.OptimizerStateTypes,
                resume.OptimizerLeafNames,
                optimizer,
                output ?? TextWriter.Null);
            return checkpoint;
        }

        RestoreLegacyCurrentModel(fullPath, model);
        if (!CheckpointOptimizerStateStream.TryLoadLegacyJson(
                fullPath,
                optimizer,
                output ?? TextWriter.Null))
        {
            ClassificationTrainingCheckpoint legacy = Load(fullPath);
            optimizer.load_state_dict(legacy.Optimizer);
        }

        int bestArtifactSlot = ResolveExistingBestArtifactSlot(fullPath, -1);
        if (checkpoint.BestEpoch > 0 && bestArtifactSlot < 0)
        {
            // Preserve an old embedded best state without retaining its full
            // ModuleState. The live model is temporarily overwritten, saved
            // through the bounded checkpoint staging block, then restored.
            StreamingModuleStateJsonReader.Restore(
                fullPath,
                nameof(ClassificationTrainingCheckpoint.BestModel),
                model);
            bestArtifactSlot = 0;
            SafeTensorFile.SaveModel(
                model,
                GetBestModelArtifactPath(fullPath, bestArtifactSlot),
                artifactDTypeOverride: GetArtifactDTypeOverride(model));
            RestoreLegacyCurrentModel(fullPath, model);
        }

        return checkpoint with { BestArtifactSlot = bestArtifactSlot };
    }

    internal static void LoadBestModel(string path, Module model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        ArgumentNullException.ThrowIfNull(model);
        string fullPath = Path.GetFullPath(path);
        ClassificationResumeData resume =
            torch.load<ClassificationResumeData>(fullPath);
        int slot = ResolveExistingBestArtifactSlot(
            fullPath,
            resume.BestArtifactSlot);
        if (slot is 0 or 1)
        {
            SafeTensorFile.LoadModel(
                GetBestModelArtifactPath(fullPath, slot),
                model);
            return;
        }
        if (resume.BestEpoch <= 0)
        {
            throw new InvalidOperationException(
                "The classification checkpoint has no best model.");
        }
        StreamingModuleStateJsonReader.Restore(
            fullPath,
            nameof(ClassificationTrainingCheckpoint.BestModel),
            model);
    }

    private static ClassificationTrainingCheckpoint LoadStreamingManifest(
        string path,
        Module model,
        string safeTensorsPath)
    {
        ClassificationCompatibilityResumeData resume =
            torch.load<ClassificationCompatibilityResumeData>(path);
        if (model.DType == TensorDType.Bfp8)
        {
            SafeTensorFile.LoadModel(safeTensorsPath, model);
        }
        else
        {
            StreamingModuleStateJsonReader.Restore(
                path,
                nameof(ClassificationTrainingCheckpoint.Model),
                model);
        }
        return resume.ToCheckpoint();
    }

    internal static string GetSafeTensorsPath(string checkpointPath)
        => Path.ChangeExtension(
            Path.GetFullPath(checkpointPath),
            ".safetensors");

    private static CheckpointVersionRegistry<
        ClassificationTrainingCheckpoint> CreateFormats()
    {
        var registry = new CheckpointVersionRegistry<
            ClassificationTrainingCheckpoint>(DetectFormatVersion);
        for (int version = 1;
            version <= ClassificationTrainingCheckpoint.CurrentFormatVersion;
            version++)
        {
            registry.RegisterReader(
                new DelegateCheckpointVersionReader<
                    ClassificationTrainingCheckpoint>(
                    version,
                    torch.load<ClassificationTrainingCheckpoint>));
            registry.RegisterWriter(
                new DelegateCheckpointVersionWriter<
                    ClassificationTrainingCheckpoint>(
                    version,
                    WriteCompatibilityCheckpoint));
        }
        return registry;
    }

    private static int DetectFormatVersion(string path)
        => torch.load<ClassificationCheckpointHeader>(path).FormatVersion;

    private static void WriteCompatibilityCheckpoint(
        CheckpointWriteTransaction transaction,
        ClassificationTrainingCheckpoint checkpoint)
    {
        transaction.PublishArtifact(
            GetSafeTensorsPath(transaction.ManifestPath),
            stagingPath => safetensors.torch.save_file(
                GetSafeTensorArtifactState(checkpoint.Model),
                stagingPath));
        transaction.CommitManifest(
            stagingPath => torch.save(checkpoint, stagingPath));
    }

    private static void WriteStreamingCheckpoint(
        CheckpointWriteTransaction transaction,
        ClassificationWriteRequest request)
    {
        transaction.PublishArtifact(
            new CheckpointArtifactWriteRequest(
                GetSafeTensorsPath(transaction.ManifestPath),
                stagingPath => SafeTensorFile.SaveModel(
                    request.CurrentModel,
                    stagingPath,
                    artifactDTypeOverride:
                        request.CurrentModel.DType == TensorDType.Bfp8
                            ? TensorDType.Float32
                            : null)));
        transaction.CommitManifest(
            stagingPath => SaveStreamingManifest(
                request.Checkpoint,
                request.CurrentModel,
                stagingPath));
    }

    private static void SaveStreamingManifest(
        ClassificationTrainingCheckpoint checkpoint,
        Module currentModel,
        string path)
    {
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string temporaryPath = fullPath + ".tmp";
        try
        {
            using (var stream = new FileStream(
                temporaryPath,
                FileMode.Create,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 1024 * 1024,
                FileOptions.SequentialScan))
            using (var writer = new System.Text.Json.Utf8JsonWriter(stream))
            using (var staging = new CheckpointFloatStagingBuffer())
            {
                writer.WriteStartObject();
                WriteProperty(writer, nameof(checkpoint.FormatVersion), checkpoint.FormatVersion);
                WriteProperty(writer, nameof(checkpoint.CompletedEpoch), checkpoint.CompletedEpoch);
                writer.WritePropertyName(nameof(checkpoint.Model));
                StreamingModuleStateJsonWriter.Write(
                    writer,
                    currentModel,
                    staging);
                WriteProperty(writer, nameof(checkpoint.Optimizer), checkpoint.Optimizer);
                WriteProperty(writer, nameof(checkpoint.Scheduler), checkpoint.Scheduler);
                WriteProperty(writer, nameof(checkpoint.BestModel), checkpoint.BestModel);
                WriteProperty(writer, nameof(checkpoint.BestEpoch), checkpoint.BestEpoch);
                WriteProperty(writer, nameof(checkpoint.BestEvaluationLoss), checkpoint.BestEvaluationLoss);
                WriteProperty(writer, nameof(checkpoint.EarlyStoppingReferenceLoss), checkpoint.EarlyStoppingReferenceLoss);
                WriteProperty(writer, nameof(checkpoint.EpochsWithoutImprovement), checkpoint.EpochsWithoutImprovement);
                WriteProperty(writer, nameof(checkpoint.CurrentEpoch), checkpoint.CurrentEpoch);
                WriteProperty(writer, nameof(checkpoint.CompletedUpdatesInEpoch), checkpoint.CompletedUpdatesInEpoch);
                WriteProperty(writer, nameof(checkpoint.CurrentTrainingLossSum), checkpoint.CurrentTrainingLossSum);
                WriteProperty(writer, nameof(checkpoint.CurrentTrainingCorrect), checkpoint.CurrentTrainingCorrect);
                WriteProperty(writer, nameof(checkpoint.CurrentTrainingSamples), checkpoint.CurrentTrainingSamples);
                WriteProperty(writer, nameof(checkpoint.ArtifactSlot), checkpoint.ArtifactSlot);
                WriteProperty(writer, nameof(checkpoint.BestArtifactSlot), checkpoint.BestArtifactSlot);
                WriteProperty(writer, nameof(checkpoint.OptimizerStateTypes), checkpoint.OptimizerStateTypes);
                WriteProperty(writer, nameof(checkpoint.OptimizerLeafNames), checkpoint.OptimizerLeafNames);
                writer.WriteEndObject();
                writer.Flush();
                stream.Flush(flushToDisk: true);
            }
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        catch
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
            throw;
        }
    }

    private static void WriteProperty<T>(
        System.Text.Json.Utf8JsonWriter writer,
        string name,
        T value)
    {
        writer.WritePropertyName(name);
        torch.SerializeJsonValue(writer, value);
    }

    private static void ValidateResumeMetadata(
        ClassificationTrainingCheckpoint checkpoint)
    {
        if (checkpoint.FormatVersion is < 1
                or > ClassificationTrainingCheckpoint.CurrentFormatVersion
            || checkpoint.CompletedEpoch < 0
            || checkpoint.CurrentEpoch < 0
            || checkpoint.CompletedUpdatesInEpoch < 0
            || !double.IsFinite(checkpoint.CurrentTrainingLossSum)
            || checkpoint.CurrentTrainingLossSum < 0d
            || checkpoint.CurrentTrainingCorrect < 0
            || checkpoint.CurrentTrainingSamples < 0
            || checkpoint.CurrentTrainingCorrect
                > checkpoint.CurrentTrainingSamples
            || checkpoint.Optimizer is null
            || checkpoint.Scheduler is null)
        {
            throw new InvalidDataException(
                "Classification training checkpoint is incompatible.");
        }
    }

    private static int SelectCurrentArtifactSlot(
        string checkpointPath,
        ClassificationArtifactMetadata? previous)
    {
        if (previous?.ArtifactSlot == 0)
            return 1;
        if (previous?.ArtifactSlot == 1)
            return 0;

        // A legacy checkpoint may already own the historical .safetensors
        // path. Publish the first artifact-format checkpoint to slot 1 so a
        // failed migration cannot invalidate the old manifest.
        return File.Exists(GetSafeTensorsPath(checkpointPath)) ? 1 : 0;
    }

    private static int ResolveExistingBestArtifactSlot(
        string checkpointPath,
        int preferredSlot)
    {
        if (preferredSlot is 0 or 1
            && File.Exists(GetBestModelArtifactPath(
                checkpointPath,
                preferredSlot)))
        {
            return preferredSlot;
        }
        for (int slot = 0; slot <= 1; slot++)
        {
            if (File.Exists(GetBestModelArtifactPath(checkpointPath, slot)))
                return slot;
        }
        return -1;
    }

    private static void RestoreCurrentArtifact(
        string checkpointPath,
        int artifactSlot,
        Module model)
    {
        string artifactPath = GetCurrentModelArtifactPath(
            checkpointPath,
            artifactSlot);
        if (!File.Exists(artifactPath))
        {
            throw new FileNotFoundException(
                "Classification current-model artifact was not found.",
                artifactPath);
        }
        SafeTensorFile.LoadModel(artifactPath, model);
    }

    private static void RestoreLegacyCurrentModel(
        string checkpointPath,
        Module model)
    {
        if (model.DType == TensorDType.Bfp8)
        {
            string safeTensorsPath = GetSafeTensorsPath(checkpointPath);
            if (!File.Exists(safeTensorsPath))
            {
                throw new FileNotFoundException(
                    "Legacy BFP8 classification model artifact was not found.",
                    safeTensorsPath);
            }
            SafeTensorFile.LoadModel(safeTensorsPath, model);
            return;
        }
        StreamingModuleStateJsonReader.Restore(
            checkpointPath,
            nameof(ClassificationTrainingCheckpoint.Model),
            model);
    }

    private static void RestoreOptimizerArtifacts(
        string checkpointPath,
        int artifactSlot,
        string[]? serializedTypes,
        string[]? serializedNames,
        IOptimizer optimizer,
        TextWriter output)
    {
        OptimizerBundle bundle = OptimizerBundle.Wrap(optimizer);
        IReadOnlyList<IOptimizer> leaves = bundle.LeafOptimizers;
        if (serializedTypes is null || serializedTypes.Length != leaves.Count)
        {
            throw new InvalidDataException(
                "Classification optimizer artifact metadata is invalid.");
        }
        if (serializedNames is not null
            && serializedNames.Length != leaves.Count)
        {
            throw new InvalidDataException(
                "Classification optimizer leaf-name metadata is invalid.");
        }

        for (int index = 0; index < leaves.Count; index++)
        {
            IOptimizer leaf = leaves[index];
            string expectedType = OptimizerStateStream.GetStateType(leaf);
            string expectedName = bundle.Leaves[index].Name;
            if (serializedNames is not null
                && !string.Equals(
                    expectedName,
                    serializedNames[index],
                    StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Classification optimizer leaf {index} is named " +
                    $"'{serializedNames[index]}', but the configured " +
                    $"optimizer expects '{expectedName}'.");
            }
            if (!string.Equals(
                    expectedType,
                    serializedTypes[index],
                    StringComparison.Ordinal))
            {
                throw new InvalidDataException(
                    $"Classification optimizer leaf {index} is " +
                    $"'{serializedTypes[index]}', but the configured " +
                    $"optimizer expects '{expectedType}'.");
            }

            string artifactPath = GetOptimizerArtifactPath(
                checkpointPath,
                artifactSlot,
                index);
            if (!File.Exists(artifactPath))
            {
                throw new FileNotFoundException(
                    "Classification optimizer artifact was not found.",
                    artifactPath);
            }
            using var stream = new FileStream(
                artifactPath,
                FileMode.Open,
                FileAccess.Read,
                FileShare.Read,
                bufferSize: 4 * 1024 * 1024,
                FileOptions.SequentialScan);
            OptimizerStateStream.LoadStateBinary(leaf, stream);
            output.WriteLine(
                $"streamed optimizer state = {expectedType} " +
                $"({index + 1}/{leaves.Count})");
        }
    }

    private static void SaveOptimizerArtifact(
        string path,
        IOptimizer optimizer)
    {
        using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 4 * 1024 * 1024,
            FileOptions.SequentialScan);
        OptimizerStateStream.SaveStateBinary(optimizer, stream);
        stream.Flush(flushToDisk: true);
    }

    private static TensorDType? GetArtifactDTypeOverride(Module model)
        => model.PrecisionMode is TensorPrecisionMode.Mix16_32
                or TensorPrecisionMode.Mix8_32
                or TensorPrecisionMode.Bfp8
            ? TensorDType.Float32
            : null;

    internal static string GetCurrentModelArtifactPath(
        string checkpointPath,
        int artifactSlot)
        => artifactSlot switch
        {
            0 => GetSafeTensorsPath(checkpointPath),
            1 => Path.ChangeExtension(
                Path.GetFullPath(checkpointPath),
                ".current.1.safetensors"),
            _ => throw new ArgumentOutOfRangeException(nameof(artifactSlot)),
        };

    internal static string GetBestModelArtifactPath(
        string checkpointPath,
        int artifactSlot)
    {
        if (artifactSlot is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(artifactSlot));
        return Path.ChangeExtension(
            Path.GetFullPath(checkpointPath),
            $".best.{artifactSlot}.safetensors");
    }

    internal static string GetOptimizerArtifactPath(
        string checkpointPath,
        int artifactSlot,
        int leafIndex)
    {
        if (artifactSlot is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(artifactSlot));
        ArgumentOutOfRangeException.ThrowIfNegative(leafIndex);
        return Path.ChangeExtension(
            Path.GetFullPath(checkpointPath),
            $".optimizer.{artifactSlot}.{leafIndex}.nnopt");
    }

    private static ModuleState EmptyModuleState()
        => new(ModuleState.CurrentFormatVersion, []);

    private static OptimizerStateDictionary EmptyOptimizerState()
        => new("ArtifactStream", null, []);

    internal static ModuleState GetSafeTensorArtifactState(ModuleState state)
        => state.Parameters.Any(
            parameter => parameter.DType == TensorDType.Bfp8)
                ? state with
                {
                    Parameters = state.Parameters
                        .Select(parameter => parameter.DType == TensorDType.Bfp8
                            ? parameter with { DType = TensorDType.Float32 }
                            : parameter)
                        .ToArray(),
                }
                : state;

    private static bool ModuleStatesEqual(ModuleState left, ModuleState right)
    {
        if (left.FormatVersion != right.FormatVersion
            || left.Parameters.Length != right.Parameters.Length)
        {
            return false;
        }
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

    private sealed record ClassificationCheckpointHeader(int FormatVersion);

    private sealed record ClassificationWriteRequest(
        Module CurrentModel,
        ClassificationTrainingCheckpoint Checkpoint);

    private sealed record ClassificationResumeData(
        int FormatVersion,
        int CompletedEpoch,
        LRSchedulerStateDictionary Scheduler,
        int BestEpoch,
        float BestEvaluationLoss,
        float EarlyStoppingReferenceLoss,
        int EpochsWithoutImprovement,
        int CurrentEpoch = 0,
        int CompletedUpdatesInEpoch = 0,
        double CurrentTrainingLossSum = 0d,
        int CurrentTrainingCorrect = 0,
        int CurrentTrainingSamples = 0,
        int ArtifactSlot = -1,
        int BestArtifactSlot = -1,
        string[]? OptimizerStateTypes = null,
        string[]? OptimizerLeafNames = null)
    {
        internal ClassificationTrainingCheckpoint ToCheckpoint()
            => new(
                FormatVersion,
                CompletedEpoch,
                EmptyModuleState(),
                EmptyOptimizerState(),
                Scheduler,
                null,
                BestEpoch,
                BestEvaluationLoss,
                EarlyStoppingReferenceLoss,
                EpochsWithoutImprovement,
                CurrentEpoch,
                CompletedUpdatesInEpoch,
                CurrentTrainingLossSum,
                CurrentTrainingCorrect,
                CurrentTrainingSamples,
                ArtifactSlot,
                BestArtifactSlot,
                OptimizerStateTypes,
                OptimizerLeafNames);
    }

    private sealed record ClassificationCompatibilityResumeData(
        int FormatVersion,
        int CompletedEpoch,
        OptimizerStateDictionary Optimizer,
        LRSchedulerStateDictionary Scheduler,
        ModuleState? BestModel,
        int BestEpoch,
        float BestEvaluationLoss,
        float EarlyStoppingReferenceLoss,
        int EpochsWithoutImprovement,
        int CurrentEpoch = 0,
        int CompletedUpdatesInEpoch = 0,
        double CurrentTrainingLossSum = 0d,
        int CurrentTrainingCorrect = 0,
        int CurrentTrainingSamples = 0,
        int ArtifactSlot = -1,
        int BestArtifactSlot = -1,
        string[]? OptimizerStateTypes = null,
        string[]? OptimizerLeafNames = null)
    {
        internal ClassificationTrainingCheckpoint ToCheckpoint()
            => new(
                FormatVersion,
                CompletedEpoch,
                EmptyModuleState(),
                Optimizer,
                Scheduler,
                BestModel,
                BestEpoch,
                BestEvaluationLoss,
                EarlyStoppingReferenceLoss,
                EpochsWithoutImprovement,
                CurrentEpoch,
                CompletedUpdatesInEpoch,
                CurrentTrainingLossSum,
                CurrentTrainingCorrect,
                CurrentTrainingSamples,
                ArtifactSlot,
                BestArtifactSlot,
                OptimizerStateTypes,
                OptimizerLeafNames);
    }

    private sealed record ClassificationArtifactMetadata(
        int FormatVersion,
        int BestEpoch,
        float BestEvaluationLoss,
        int ArtifactSlot = -1,
        int BestArtifactSlot = -1,
        string[]? OptimizerStateTypes = null,
        string[]? OptimizerLeafNames = null);

}
