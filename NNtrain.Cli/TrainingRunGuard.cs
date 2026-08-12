using System.Text.Json;

namespace NNtrain;

/// <summary>
/// Holds an exclusive training-run lease and leaves a durable marker when a
/// process exits without reporting successful completion.
/// </summary>
internal sealed class TrainingRunGuard : IDisposable
{
    private readonly string _markerPath;
    private FileStream? _lease;
    private bool _completed;

    private TrainingRunGuard(
        string markerPath,
        FileStream lease,
        bool interrupted)
    {
        _markerPath = markerPath;
        _lease = lease;
        WasInterrupted = interrupted;
    }

    internal bool WasInterrupted { get; }

    internal string MarkerPath => _markerPath;

    internal static TrainingRunGuard Begin(
        string configurationPath,
        string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(configurationPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        string markerPath = GetMarkerPath(checkpointPath);
        string? directory = Path.GetDirectoryName(markerPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        FileStream lease;
        try
        {
            lease = new FileStream(
                markerPath,
                FileMode.OpenOrCreate,
                FileAccess.ReadWrite,
                FileShare.Read,
                bufferSize: 4096,
                FileOptions.WriteThrough);
        }
        catch (IOException exception)
        {
            throw new IOException(
                $"Another training process is already using checkpoint " +
                $"'{Path.GetFullPath(checkpointPath)}'.",
                exception);
        }

        bool interrupted = lease.Length > 0;
        try
        {
            lease.SetLength(0);
            var marker = new TrainingRunMarker(
                FormatVersion: 1,
                ProcessId: Environment.ProcessId,
                StartedAtUtc: DateTimeOffset.UtcNow,
                ConfigurationPath: Path.GetFullPath(configurationPath),
                CheckpointPath: Path.GetFullPath(checkpointPath));
            JsonSerializer.Serialize(lease, marker);
            lease.Flush(flushToDisk: true);
            lease.Position = 0;
            return new TrainingRunGuard(markerPath, lease, interrupted);
        }
        catch
        {
            lease.Dispose();
            throw;
        }
    }

    internal void Complete() => _completed = true;

    public void Dispose()
    {
        FileStream? lease = Interlocked.Exchange(ref _lease, null);
        if (lease is null)
            return;
        lease.Dispose();
        if (_completed && File.Exists(_markerPath))
            File.Delete(_markerPath);
    }

    internal static string GetMarkerPath(string checkpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(checkpointPath);
        string fullPath = Path.GetFullPath(checkpointPath);
        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        return Path.Combine(
            directory,
            $"{Path.GetFileName(fullPath)}.running.json");
    }

    private sealed record TrainingRunMarker(
        int FormatVersion,
        int ProcessId,
        DateTimeOffset StartedAtUtc,
        string ConfigurationPath,
        string CheckpointPath);
}
