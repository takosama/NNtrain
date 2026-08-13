using System.Globalization;

namespace NNtrain;

internal static class CheckpointSnapshot
{
    internal static string Save(
        string checkpointPath,
        string modelName,
        double epochPosition,
        ModuleState modelState,
        DateTimeOffset? timestamp = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(modelName);
        ArgumentNullException.ThrowIfNull(modelState);
        if (!double.IsFinite(epochPosition) || epochPosition < 0d)
            throw new ArgumentOutOfRangeException(nameof(epochPosition));

        string path = GetPath(
            checkpointPath,
            modelName,
            epochPosition,
            timestamp ?? DateTimeOffset.Now);
        safetensors.torch.save_file(modelState, path);
        return path;
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
