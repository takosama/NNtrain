using System.Globalization;

namespace NNtrain;

internal static class CheckpointSnapshot
{
    internal static string Save(
        string checkpointPath,
        string modelName,
        double epochPosition,
        Module model,
        DateTimeOffset? timestamp = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        string path = PreparePath(
            checkpointPath,
            modelName,
            epochPosition,
            timestamp);
        SafeTensorFile.SaveModel(
            model,
            path,
            artifactDTypeOverride: model.DType == TensorDType.Bfp8
                ? TensorDType.Float32
                : null);
        return path;
    }

    internal static string Save(
        string checkpointPath,
        string modelName,
        double epochPosition,
        ModuleState modelState,
        DateTimeOffset? timestamp = null)
    {
        ArgumentNullException.ThrowIfNull(modelState);
        string path = PreparePath(
            checkpointPath,
            modelName,
            epochPosition,
            timestamp);
        ModuleState artifact = modelState.Parameters.Any(
            parameter => parameter.DType == TensorDType.Bfp8)
                ? modelState with
                {
                    Parameters = modelState.Parameters
                        .Select(parameter => parameter.DType == TensorDType.Bfp8
                            ? parameter with { DType = TensorDType.Float32 }
                            : parameter)
                        .ToArray(),
                }
                : modelState;
        safetensors.torch.save_file(artifact, path);
        return path;
    }

    private static string PreparePath(
        string checkpointPath,
        string modelName,
        double epochPosition,
        DateTimeOffset? timestamp)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        if (!double.IsFinite(epochPosition) || epochPosition < 0d)
            throw new ArgumentOutOfRangeException(nameof(epochPosition));
        return GetPath(
            checkpointPath,
            modelName,
            epochPosition,
            timestamp ?? DateTimeOffset.Now);
    }

    internal static string GetPath(
        string checkpointPath,
        string modelName,
        double epochPosition,
        DateTimeOffset timestamp)
    {
        string fullCheckpointPath = Path.GetFullPath(checkpointPath);
        string directory = Path.GetDirectoryName(fullCheckpointPath)
            ?? Environment.CurrentDirectory;
        string safeModelName = SanitizeFileName(modelName);
        string epoch = epochPosition.ToString(
            "0.0",
            CultureInfo.InvariantCulture);
        return Path.Combine(
            directory,
            $"{safeModelName}_{epoch}_epoch_" +
            $"{timestamp:yyyyMMdd_HHmm}.safetensors");
    }

    private static string SanitizeFileName(string value)
    {
        char[] invalid = Path.GetInvalidFileNameChars();
        var result = value
            .Select(character => invalid.Contains(character) ? '_' : character)
            .ToArray();
        return new string(result);
    }
}
