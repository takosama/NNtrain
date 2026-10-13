using System.Text.Json;

namespace NNtrain;

internal static class ClassificationModelCheckpoint
{
    internal static void Save(
        string path,
        int epoch,
        float evaluationLoss,
        Module model)
    {
        string fullPath = Path.GetFullPath(path);
        var repository = new NNtrain.Training.Persistence.CheckpointRepository(
            fullPath);
        using NNtrain.Training.Persistence.CheckpointWriteTransaction
            transaction = repository.BeginWrite();
        transaction.PublishArtifact(
            Path.ChangeExtension(fullPath, ".safetensors"),
            stagingPath => SafeTensorFile.SaveModel(
                model,
                stagingPath,
                artifactDTypeOverride:
                    model.PrecisionMode is TensorPrecisionMode.Mix16_32
                            or TensorPrecisionMode.Mix8_32
                            or TensorPrecisionMode.Bfp8
                        ? TensorDType.Float32
                        : null));
        transaction.CommitManifest(
            stagingPath => SaveManifest(
                stagingPath,
                epoch,
                evaluationLoss,
                model));
    }

    private static void SaveManifest(
        string path,
        int epoch,
        float evaluationLoss,
        Module model)
    {
        using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1024 * 1024,
            FileOptions.SequentialScan);
        using var writer = new Utf8JsonWriter(stream);
        using var staging = new CheckpointFloatStagingBuffer();
        writer.WriteStartObject();
        StreamingModuleStateJsonWriter.WriteProperty(
            writer,
            nameof(ModelCheckpoint.FormatVersion),
            ModelCheckpoint.CurrentFormatVersion);
        StreamingModuleStateJsonWriter.WriteProperty(
            writer,
            nameof(ModelCheckpoint.Epoch),
            epoch);
        StreamingModuleStateJsonWriter.WriteProperty(
            writer,
            nameof(ModelCheckpoint.EvaluationLoss),
            evaluationLoss);
        writer.WritePropertyName(nameof(ModelCheckpoint.Model));
        StreamingModuleStateJsonWriter.Write(writer, model, staging);
        writer.WriteEndObject();
        writer.Flush();
        stream.Flush(flushToDisk: true);
    }

    internal static string GetBestModelCheckpointPath(
        string trainingCheckpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(trainingCheckpointPath);
        string fullPath = Path.GetFullPath(trainingCheckpointPath);
        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        string fileName = Path.GetFileName(fullPath);
        const string defaultSuffix = ".checkpoint.json";
        string stem = fileName.EndsWith(
            defaultSuffix,
            StringComparison.OrdinalIgnoreCase)
            ? fileName[..^defaultSuffix.Length]
            : Path.GetFileNameWithoutExtension(fileName);
        return Path.Combine(directory, $"{stem}.best-model.json");
    }

    private sealed record ModelCheckpoint(
        int FormatVersion,
        int Epoch,
        float EvaluationLoss,
        ModuleState Model)
    {
        internal const int CurrentFormatVersion = 1;
    }
}
