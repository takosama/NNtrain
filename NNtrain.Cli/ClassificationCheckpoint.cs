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
    int CurrentTrainingSamples = 0)
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
        string safeTensorsPath = GetSafeTensorsPath(path);
        if (!File.Exists(safeTensorsPath))
            return checkpoint;

        ModuleState safeModel = safetensors.torch.load_file(safeTensorsPath);
        return ModuleStatesEqual(safeModel, checkpoint.Model)
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

    private static ClassificationTrainingCheckpoint LoadStreamingManifest(
        string path,
        Module model,
        string safeTensorsPath)
    {
        ClassificationResumeData resume =
            torch.load<ClassificationResumeData>(path);
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
                WriteModuleState(writer, currentModel, staging);
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

    private static void WriteModuleState(
        System.Text.Json.Utf8JsonWriter writer,
        Module model,
        CheckpointFloatStagingBuffer staging)
    {
        Parameter[] parameters = model.parameters().ToArray();
        writer.WriteStartObject();
        WriteProperty(
            writer,
            nameof(ModuleState.FormatVersion),
            ModuleState.CurrentFormatVersion);
        writer.WritePropertyName(nameof(ModuleState.Parameters));
        writer.WriteStartArray();
        for (int index = 0; index < parameters.Length; index++)
        {
            Parameter parameter = parameters[index];
            Tensor tensor = parameter.T;
            writer.WriteStartObject();
            WriteProperty(writer, nameof(ModuleParameterState.Index), index);
            WriteProperty(writer, nameof(ModuleParameterState.Name), parameter.Name);
            WriteProperty(
                writer,
                nameof(ModuleParameterState.Shape),
                tensor.Shape.ToArray());
            writer.WritePropertyName(nameof(ModuleParameterState.Values));
            writer.WriteStartArray();
            int maximumChunkElements = tensor.DType == TensorDType.Bfp8
                ? CheckpointFloatStagingBuffer.MaximumElementCount / 2
                : CheckpointFloatStagingBuffer.MaximumElementCount;
            for (int offset = 0; offset < tensor.Numel;)
            {
                int count = Math.Min(
                    maximumChunkElements,
                    tensor.Numel - offset);
                ReadOnlySpan<float> values = tensor.CopyCheckpointRangeTo(
                    offset,
                    count,
                    staging,
                    preferMaster: true);
                foreach (float value in values)
                    torch.SerializeJsonValue(writer, value);
                offset += count;
            }
            writer.WriteEndArray();
            WriteProperty(writer, nameof(ModuleParameterState.DType), tensor.DType);
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
        writer.WriteEndObject();
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

    private static ModuleState EmptyModuleState()
        => new(ModuleState.CurrentFormatVersion, []);

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
        int CurrentTrainingSamples = 0)
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
                CurrentTrainingSamples);
    }

}
