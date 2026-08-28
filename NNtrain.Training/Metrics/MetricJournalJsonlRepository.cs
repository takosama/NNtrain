using System.Text;
using System.Text.Json;

namespace NNtrain.Training.Metrics;

/// <summary>Result of reading or recovering an append-only metric journal.</summary>
public sealed record MetricJournalLoadResult(
    MetricJournal Journal,
    bool IgnoredCorruptTail,
    int RemovedAfterCheckpoint);

/// <summary>
/// JSONL persistence for metric journals. Appends are exclusive, performed as
/// one record write and flushed to stable storage. Recovery rewrites through
/// an atomically renamed file in the same directory.
/// </summary>
public sealed class MetricJournalJsonlRepository
{
    private static readonly UTF8Encoding StrictUtf8 = new(
        encoderShouldEmitUTF8Identifier: false,
        throwOnInvalidBytes: true);
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = false,
    };

    private readonly object _sync = new();
    private MetricJournalEntry? _lastKnownEntry;
    private bool _appendStateInitialized;

    public MetricJournalJsonlRepository(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Path = System.IO.Path.GetFullPath(path);
    }

    public string Path { get; }

    public bool Exists => File.Exists(Path);

    public void AppendAndFlush(MetricJournalEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        entry.Validate();
        byte[] json = JsonSerializer.SerializeToUtf8Bytes(entry, JsonOptions);
        byte[] record = new byte[json.Length + 1];
        json.CopyTo(record, 0);
        record[^1] = (byte)'\n';

        lock (_sync)
        {
            InitializeAppendState();
            if (_lastKnownEntry is not null
                && MetricJournal.ComparePosition(entry, _lastKnownEntry) < 0)
            {
                throw new ArgumentException(
                    "Metric journal positions must be appended in nondecreasing order.",
                    nameof(entry));
            }
            EnsureDirectory();
            using var stream = new FileStream(
                Path,
                FileMode.OpenOrCreate,
                FileAccess.ReadWrite,
                FileShare.Read,
                bufferSize: 4096,
                FileOptions.WriteThrough);
            bool needsSeparator = false;
            if (stream.Length > 0)
            {
                stream.Position = stream.Length - 1;
                needsSeparator = stream.ReadByte() != '\n';
                if (needsSeparator)
                {
                    MetricJournalLoadResult existing = LoadCore();
                    if (existing.IgnoredCorruptTail)
                    {
                        throw new InvalidDataException(
                            "The metric journal has a corrupt tail. Recover it before appending.");
                    }
                }
            }

            stream.Position = stream.Length;
            if (needsSeparator)
                stream.WriteByte((byte)'\n');
            stream.Write(record);
            stream.Flush(flushToDisk: true);
            _lastKnownEntry = entry;
        }
    }

    public MetricJournalLoadResult Load()
    {
        lock (_sync)
            return LoadCore();
    }

    /// <summary>
    /// Safely ignores a partial final record, removes observations newer than
    /// the checkpoint, then atomically publishes the recovered journal.
    /// </summary>
    public MetricJournalLoadResult RecoverThrough(long checkpointGlobalStep)
    {
        if (checkpointGlobalStep < -1)
            throw new ArgumentOutOfRangeException(nameof(checkpointGlobalStep));
        lock (_sync)
        {
            MetricJournalLoadResult loaded = LoadCore();
            int removed = loaded.Journal.TruncateAfter(checkpointGlobalStep);
            if (loaded.IgnoredCorruptTail || removed > 0)
                ReplaceAtomicallyCore(loaded.Journal.Entries);
            else
                SetAppendState(loaded.Journal.Entries);
            return loaded with { RemovedAfterCheckpoint = removed };
        }
    }

    public void ReplaceAtomically(IEnumerable<MetricJournalEntry> entries)
    {
        ArgumentNullException.ThrowIfNull(entries);
        var journal = new MetricJournal();
        foreach (MetricJournalEntry entry in entries)
            journal.Append(entry);
        lock (_sync)
            ReplaceAtomicallyCore(journal.Entries);
    }

    private MetricJournalLoadResult LoadCore()
    {
        var journal = new MetricJournal();
        if (!File.Exists(Path))
            return new MetricJournalLoadResult(journal, false, 0);

        byte[] contents = File.ReadAllBytes(Path);
        List<ArraySegment<byte>> lines = SplitLines(contents);
        int lastContentLine = lines.FindLastIndex(
            static line => !IsWhiteSpace(line));
        bool ignoredCorruptTail = false;

        for (int index = 0; index <= lastContentLine; index++)
        {
            ArraySegment<byte> line = TrimCarriageReturn(lines[index]);
            try
            {
                if (IsWhiteSpace(line))
                    throw new InvalidDataException("Blank JSONL records are not valid.");
                ReadOnlySpan<byte> bytes = line.AsSpan();
                if (index == 0
                    && bytes.StartsWith(StrictUtf8.GetPreamble()))
                {
                    bytes = bytes[StrictUtf8.GetPreamble().Length..];
                }
                _ = StrictUtf8.GetString(bytes);
                MetricJournalEntry? entry = JsonSerializer.Deserialize<
                    MetricJournalEntry>(bytes, JsonOptions);
                if (entry is null)
                    throw new InvalidDataException("The metric record was null.");
                journal.Append(entry);
            }
            catch (Exception exception) when (IsRecordFailure(exception))
            {
                if (index == lastContentLine)
                {
                    ignoredCorruptTail = true;
                    break;
                }
                throw new InvalidDataException(
                    $"Metric journal record {index + 1} is corrupt.",
                    exception);
            }
        }

        return new MetricJournalLoadResult(
            journal,
            ignoredCorruptTail,
            RemovedAfterCheckpoint: 0);
    }

    private void ReplaceAtomicallyCore(
        IReadOnlyList<MetricJournalEntry> entries)
    {
        EnsureDirectory();
        string directory = System.IO.Path.GetDirectoryName(Path)!;
        string temporaryPath = System.IO.Path.Combine(
            directory,
            $".{System.IO.Path.GetFileName(Path)}.{Guid.NewGuid():N}.tmp");
        try
        {
            using (var stream = new FileStream(
                temporaryPath,
                FileMode.CreateNew,
                FileAccess.Write,
                FileShare.None,
                bufferSize: 4096,
                FileOptions.WriteThrough))
            {
                foreach (MetricJournalEntry entry in entries)
                {
                    entry.Validate();
                    byte[] json = JsonSerializer.SerializeToUtf8Bytes(
                        entry,
                        JsonOptions);
                    stream.Write(json);
                    stream.WriteByte((byte)'\n');
                }
                stream.Flush(flushToDisk: true);
            }

            File.Move(temporaryPath, Path, overwrite: true);
            SetAppendState(entries);
        }
        finally
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }

    private void EnsureDirectory()
    {
        string? directory = System.IO.Path.GetDirectoryName(Path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
    }

    private void InitializeAppendState()
    {
        if (_appendStateInitialized)
            return;
        MetricJournalLoadResult loaded = LoadCore();
        if (loaded.IgnoredCorruptTail)
        {
            throw new InvalidDataException(
                "The metric journal has a corrupt tail. Recover it before appending.");
        }
        SetAppendState(loaded.Journal.Entries);
    }

    private void SetAppendState(IReadOnlyList<MetricJournalEntry> entries)
    {
        _lastKnownEntry = entries.Count == 0 ? null : entries[^1];
        _appendStateInitialized = true;
    }

    private static List<ArraySegment<byte>> SplitLines(byte[] contents)
    {
        var lines = new List<ArraySegment<byte>>();
        int start = 0;
        for (int index = 0; index < contents.Length; index++)
        {
            if (contents[index] != (byte)'\n')
                continue;
            lines.Add(new ArraySegment<byte>(contents, start, index - start));
            start = index + 1;
        }
        lines.Add(new ArraySegment<byte>(contents, start, contents.Length - start));
        return lines;
    }

    private static ArraySegment<byte> TrimCarriageReturn(
        ArraySegment<byte> line)
        => line.Count > 0
            && line.Array![line.Offset + line.Count - 1] == (byte)'\r'
                ? new ArraySegment<byte>(line.Array, line.Offset, line.Count - 1)
                : line;

    private static bool IsWhiteSpace(ArraySegment<byte> line)
    {
        for (int index = 0; index < line.Count; index++)
        {
            byte value = line.Array![line.Offset + index];
            if (value is not ((byte)' ' or (byte)'\t' or (byte)'\r'))
                return false;
        }
        return true;
    }

    private static bool IsRecordFailure(Exception exception)
        => exception is JsonException
            or DecoderFallbackException
            or ArgumentException
            or InvalidDataException;
}
