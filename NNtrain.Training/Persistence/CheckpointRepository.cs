namespace NNtrain.Training.Persistence;

public enum CheckpointFaultPoint
{
    BeforeArtifactStage = 0,
    AfterArtifactStage = 1,
    AfterArtifactPublish = 2,
    BeforeManifestStage = 3,
    AfterManifestStage = 4,
    BeforeManifestPublish = 5,
    AfterManifestPublish = 6,
}

public sealed record CheckpointFaultContext(
    CheckpointFaultPoint Point,
    string Path,
    int ArtifactIndex);

/// <summary>
/// Deferred artifact payload. The callback receives the transaction-owned
/// staging path, allowing a model to stream directly into its existing
/// artifact name without first constructing an in-memory payload.
/// </summary>
public sealed record CheckpointArtifactWriteRequest(
    string ArtifactPath,
    Action<string> WriteStagedArtifact,
    bool PreservePreviousOnRollback = true);

/// <summary>Test and diagnostics hook called at durable publication boundaries.</summary>
public interface ICheckpointFaultInjector
{
    void OnCheckpointFaultPoint(CheckpointFaultContext context);
}

/// <summary>
/// Creates checkpoint write transactions for one stable manifest path.
/// Artifacts are published first and the manifest is the sole final commit.
/// </summary>
public sealed class CheckpointRepository
{
    private readonly ICheckpointFaultInjector? _faultInjector;
    private int _activeWriter;

    public CheckpointRepository(
        string manifestPath,
        ICheckpointFaultInjector? faultInjector = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        ManifestPath = Path.GetFullPath(manifestPath);
        _faultInjector = faultInjector;
    }

    public string ManifestPath { get; }

    public CheckpointWriteTransaction BeginWrite()
    {
        if (Interlocked.CompareExchange(ref _activeWriter, 1, 0) != 0)
        {
            throw new InvalidOperationException(
                "A checkpoint write transaction is already active for this repository.");
        }
        return new CheckpointWriteTransaction(
            this,
            _faultInjector,
            ReleaseWriter);
    }

    private void ReleaseWriter() => Volatile.Write(ref _activeWriter, 0);
}

/// <summary>
/// A single artifact-first, manifest-last checkpoint publication. Writers
/// receive a staging path in the same directory as the final target.
/// </summary>
public sealed class CheckpointWriteTransaction : IDisposable
{
    private readonly CheckpointRepository _repository;
    private readonly ICheckpointFaultInjector? _faultInjector;
    private readonly Action _releaseWriter;
    private readonly HashSet<string> _publishedArtifacts =
        new(StringComparer.OrdinalIgnoreCase);
    private readonly List<PublishedArtifact> _artifactRollbacks = [];
    private readonly List<Exception> _rollbackErrors = [];
    private readonly string _transactionId = Guid.NewGuid().ToString("N");
    private int _artifactIndex;
    private int _disposed;
    private int _committed;

    internal CheckpointWriteTransaction(
        CheckpointRepository repository,
        ICheckpointFaultInjector? faultInjector,
        Action releaseWriter)
    {
        _repository = repository;
        _faultInjector = faultInjector;
        _releaseWriter = releaseWriter;
    }

    public string ManifestPath => _repository.ManifestPath;

    public bool IsCommitted => Volatile.Read(ref _committed) != 0;

    public IReadOnlyList<Exception> RollbackErrors
    {
        get
        {
            lock (_rollbackErrors)
                return _rollbackErrors.ToArray();
        }
    }

    public void PublishArtifact(
        CheckpointArtifactWriteRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        PublishArtifact(
            request.ArtifactPath,
            request.WriteStagedArtifact,
            request.PreservePreviousOnRollback);
    }

    public void PublishArtifact(
        string artifactPath,
        Action<string> writeStagedArtifact,
        bool preservePreviousOnRollback = true)
    {
        ThrowIfClosed();
        ArgumentException.ThrowIfNullOrWhiteSpace(artifactPath);
        ArgumentNullException.ThrowIfNull(writeStagedArtifact);
        string fullPath = Path.GetFullPath(artifactPath);
        if (string.Equals(
            fullPath,
            ManifestPath,
            StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException(
                "The manifest must be committed with CommitManifest.",
                nameof(artifactPath));
        }
        if (!_publishedArtifacts.Add(fullPath))
        {
            throw new InvalidOperationException(
                $"Artifact '{fullPath}' was already published by this transaction.");
        }

        int index = _artifactIndex++;
        PublishFile(
            fullPath,
            writeStagedArtifact,
            CheckpointFaultPoint.BeforeArtifactStage,
            CheckpointFaultPoint.AfterArtifactStage,
            CheckpointFaultPoint.AfterArtifactPublish,
            index,
            onPublishing: preservePreviousOnRollback
                ? () => PrepareArtifactRollback(fullPath)
                : null);
    }

    public void CommitManifest(Action<string> writeStagedManifest)
    {
        ThrowIfClosed();
        ArgumentNullException.ThrowIfNull(writeStagedManifest);
        if (IsCommitted)
            throw new InvalidOperationException("The manifest is already committed.");

        PublishFile(
            ManifestPath,
            writeStagedManifest,
            CheckpointFaultPoint.BeforeManifestStage,
            CheckpointFaultPoint.AfterManifestStage,
            CheckpointFaultPoint.AfterManifestPublish,
            artifactIndex: -1,
            beforePublish: CheckpointFaultPoint.BeforeManifestPublish,
            onPublished: MarkCommitted);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        try
        {
            if (IsCommitted)
                DeleteBackups();
            else
                RollbackArtifacts();
        }
        finally
        {
            _releaseWriter();
        }
    }

    private void PublishFile(
        string finalPath,
        Action<string> writeStagedFile,
        CheckpointFaultPoint beforeStage,
        CheckpointFaultPoint afterStage,
        CheckpointFaultPoint afterPublish,
        int artifactIndex,
        CheckpointFaultPoint? beforePublish = null,
        Action? onPublishing = null,
        Action? onPublished = null)
    {
        string? directory = Path.GetDirectoryName(finalPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string stagingPath = finalPath + $".transaction.{Guid.NewGuid():N}.tmp";
        try
        {
            Inject(beforeStage, finalPath, artifactIndex);
            writeStagedFile(stagingPath);
            if (!File.Exists(stagingPath))
            {
                throw new InvalidDataException(
                    $"Checkpoint writer did not create staged file '{stagingPath}'.");
            }
            FlushToDisk(stagingPath);
            Inject(afterStage, finalPath, artifactIndex);
            if (beforePublish.HasValue)
                Inject(beforePublish.Value, finalPath, artifactIndex);
            onPublishing?.Invoke();
            File.Move(stagingPath, finalPath, overwrite: true);
            onPublished?.Invoke();
            Inject(afterPublish, finalPath, artifactIndex);
        }
        finally
        {
            if (File.Exists(stagingPath))
                File.Delete(stagingPath);
        }
    }

    private void Inject(
        CheckpointFaultPoint point,
        string path,
        int artifactIndex)
        => _faultInjector?.OnCheckpointFaultPoint(
            new CheckpointFaultContext(point, path, artifactIndex));

    private static void FlushToDisk(string path)
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.ReadWrite,
            FileShare.Read,
            bufferSize: 4096,
            FileOptions.WriteThrough);
        stream.Flush(flushToDisk: true);
    }

    private void PrepareArtifactRollback(string finalPath)
    {
        string? backupPath = null;
        bool existed = File.Exists(finalPath);
        if (existed)
        {
            backupPath = finalPath +
                $".transaction.{_transactionId}.backup";
            File.Copy(finalPath, backupPath, overwrite: false);
        }
        var rollback = new PublishedArtifact(
            finalPath,
            backupPath,
            existed);
        _artifactRollbacks.Add(rollback);
        if (backupPath is not null)
            FlushToDisk(backupPath);
    }

    private void MarkCommitted()
    {
        // The manifest move is the commit point. Mark it before cleanup and
        // before the after-publish fault hook so post-commit exceptions never
        // roll artifacts back underneath the new manifest.
        Volatile.Write(ref _committed, 1);
    }

    private void RollbackArtifacts()
    {
        for (int index = _artifactRollbacks.Count - 1; index >= 0; index--)
        {
            PublishedArtifact artifact = _artifactRollbacks[index];
            try
            {
                if (artifact.ExistedBefore)
                {
                    if (artifact.BackupPath is null
                        || !File.Exists(artifact.BackupPath))
                    {
                        throw new IOException(
                            $"Checkpoint artifact backup is missing for " +
                            $"'{artifact.FinalPath}'.");
                    }
                    File.Move(
                        artifact.BackupPath,
                        artifact.FinalPath,
                        overwrite: true);
                }
                else if (File.Exists(artifact.FinalPath))
                {
                    File.Delete(artifact.FinalPath);
                }
            }
            catch (Exception exception)
            {
                lock (_rollbackErrors)
                    _rollbackErrors.Add(exception);
            }
        }
    }

    private void DeleteBackups()
    {
        foreach (PublishedArtifact artifact in _artifactRollbacks)
        {
            if (artifact.BackupPath is null
                || !File.Exists(artifact.BackupPath))
            {
                continue;
            }
            try
            {
                File.Delete(artifact.BackupPath);
            }
            catch (Exception exception)
            {
                lock (_rollbackErrors)
                    _rollbackErrors.Add(exception);
            }
        }
    }

    private void ThrowIfClosed()
    {
        ObjectDisposedException.ThrowIf(
            Volatile.Read(ref _disposed) != 0,
            this);
        if (IsCommitted)
        {
            throw new InvalidOperationException(
                "No files can be published after the manifest commit.");
        }
    }

    private sealed record PublishedArtifact(
        string FinalPath,
        string? BackupPath,
        bool ExistedBefore);
}
