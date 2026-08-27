namespace NNtrain.Training.Persistence;

/// <summary>Reads one checkpoint manifest version without changing it.</summary>
public interface ICheckpointVersionReader<TCheckpoint>
{
    int FormatVersion { get; }
    TCheckpoint Read(string manifestPath);
}

/// <summary>Writes one checkpoint version through a publication transaction.</summary>
public interface ICheckpointVersionWriter<TCheckpoint>
{
    int FormatVersion { get; }
    void Write(
        CheckpointWriteTransaction transaction,
        TCheckpoint checkpoint);
}

/// <summary>
/// Maps stable on-disk format versions to compatibility readers and writers.
/// Reader and writer registration are independent so legacy formats can stay
/// readable without remaining writable.
/// </summary>
public sealed class CheckpointVersionRegistry<TCheckpoint>
{
    private readonly object _sync = new();
    private readonly Func<string, int>? _detectVersion;
    private readonly Dictionary<int, ICheckpointVersionReader<TCheckpoint>>
        _readers = [];
    private readonly Dictionary<int, ICheckpointVersionWriter<TCheckpoint>>
        _writers = [];

    public CheckpointVersionRegistry(
        Func<string, int>? detectVersion = null)
        => _detectVersion = detectVersion;

    public CheckpointVersionRegistry<TCheckpoint> RegisterReader(
        ICheckpointVersionReader<TCheckpoint> reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        ValidateVersion(reader.FormatVersion);
        lock (_sync)
        {
            if (!_readers.TryAdd(reader.FormatVersion, reader))
            {
                throw new InvalidOperationException(
                    $"A checkpoint reader for format {reader.FormatVersion} " +
                    "is already registered.");
            }
        }
        return this;
    }

    public CheckpointVersionRegistry<TCheckpoint> RegisterWriter(
        ICheckpointVersionWriter<TCheckpoint> writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ValidateVersion(writer.FormatVersion);
        lock (_sync)
        {
            if (!_writers.TryAdd(writer.FormatVersion, writer))
            {
                throw new InvalidOperationException(
                    $"A checkpoint writer for format {writer.FormatVersion} " +
                    "is already registered.");
            }
        }
        return this;
    }

    public bool CanRead(int formatVersion)
    {
        lock (_sync)
            return _readers.ContainsKey(formatVersion);
    }

    public bool CanWrite(int formatVersion)
    {
        lock (_sync)
            return _writers.ContainsKey(formatVersion);
    }

    public TCheckpoint Read(string manifestPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        if (_detectVersion is null)
        {
            throw new InvalidOperationException(
                "This checkpoint registry has no format-version detector.");
        }
        return Read(manifestPath, _detectVersion(manifestPath));
    }

    public TCheckpoint Read(string manifestPath, int formatVersion)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(manifestPath);
        ICheckpointVersionReader<TCheckpoint> reader;
        lock (_sync)
        {
            if (!_readers.TryGetValue(formatVersion, out reader!))
                throw UnsupportedVersion(formatVersion, "reader");
        }
        return reader.Read(Path.GetFullPath(manifestPath));
    }

    public void Write(
        CheckpointRepository repository,
        int formatVersion,
        TCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(repository);
        ICheckpointVersionWriter<TCheckpoint> writer;
        lock (_sync)
        {
            if (!_writers.TryGetValue(formatVersion, out writer!))
                throw UnsupportedVersion(formatVersion, "writer");
        }

        using CheckpointWriteTransaction transaction =
            repository.BeginWrite();
        writer.Write(transaction, checkpoint);
        if (!transaction.IsCommitted)
        {
            throw new InvalidOperationException(
                $"Checkpoint writer for format {formatVersion} returned " +
                "without committing its manifest.");
        }
    }

    private static void ValidateVersion(int formatVersion)
    {
        if (formatVersion <= 0)
            throw new ArgumentOutOfRangeException(nameof(formatVersion));
    }

    private static NotSupportedException UnsupportedVersion(
        int formatVersion,
        string operation)
        => new(
            $"No checkpoint {operation} is registered for format " +
            $"version {formatVersion}.");
}

/// <summary>Delegate-backed compatibility reader.</summary>
public sealed class DelegateCheckpointVersionReader<TCheckpoint>(
    int formatVersion,
    Func<string, TCheckpoint> read)
    : ICheckpointVersionReader<TCheckpoint>
{
    private readonly Func<string, TCheckpoint> _read = read
        ?? throw new ArgumentNullException(nameof(read));

    public int FormatVersion { get; } = formatVersion > 0
        ? formatVersion
        : throw new ArgumentOutOfRangeException(nameof(formatVersion));

    public TCheckpoint Read(string manifestPath) => _read(manifestPath);
}

/// <summary>Delegate-backed compatibility writer.</summary>
public sealed class DelegateCheckpointVersionWriter<TCheckpoint>(
    int formatVersion,
    Action<CheckpointWriteTransaction, TCheckpoint> write)
    : ICheckpointVersionWriter<TCheckpoint>
{
    private readonly Action<CheckpointWriteTransaction, TCheckpoint> _write =
        write ?? throw new ArgumentNullException(nameof(write));

    public int FormatVersion { get; } = formatVersion > 0
        ? formatVersion
        : throw new ArgumentOutOfRangeException(nameof(formatVersion));

    public void Write(
        CheckpointWriteTransaction transaction,
        TCheckpoint checkpoint)
        => _write(transaction, checkpoint);
}
