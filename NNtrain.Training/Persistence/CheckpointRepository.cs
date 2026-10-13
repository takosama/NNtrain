namespace NNtrain.Training.Persistence;

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
