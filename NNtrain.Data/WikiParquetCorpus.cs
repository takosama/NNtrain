using System.Runtime.CompilerServices;
using Parquet;
using Parquet.Schema;

namespace NNtrain;

/// <summary>
/// Streams a text column from sharded Wikipedia Parquet files.
/// </summary>
public static class WikiParquetCorpus
{
    internal const int ShuffledReaderCacheCapacity = 8;

    public static async Task<long> CountRowsAsync(
        string directoryPath,
        CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);
        string fullPath = Path.GetFullPath(directoryPath);
        if (!Directory.Exists(fullPath))
        {
            throw new DirectoryNotFoundException(
                $"Wikipedia data directory was not found at '{fullPath}'.");
        }

        string[] files = OrderFiles(
            Directory.GetFiles(fullPath, "*.parquet"),
            shuffleSeed: null);
        if (files.Length == 0)
        {
            throw new FileNotFoundException(
                $"No .parquet files were found in '{fullPath}'.");
        }

        long count = 0;
        foreach (string file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            await using ParquetReader reader = await ParquetReader.CreateAsync(
                file,
                cancellationToken: cancellationToken);
            for (int group = 0; group < reader.RowGroupCount; group++)
            {
                using ParquetRowGroupReader rowGroup =
                    reader.OpenRowGroupReader(group);
                count = checked(count + rowGroup.RowCount);
            }
        }
        return count;
    }

    public static async IAsyncEnumerable<string> ReadTextsAsync(
        string directoryPath,
        string textColumn = "text",
        int? maxDocuments = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default,
        int? shuffleSeed = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directoryPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(textColumn);
        if (maxDocuments.HasValue && maxDocuments.Value <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(maxDocuments),
                maxDocuments,
                "Maximum document count must be positive when specified.");
        }

        string fullPath = Path.GetFullPath(directoryPath);
        if (!Directory.Exists(fullPath))
        {
            throw new DirectoryNotFoundException(
                $"Wikipedia data directory was not found at '{fullPath}'.");
        }

        string[] files = OrderFiles(
            Directory.GetFiles(fullPath, "*.parquet"),
            shuffleSeed);
        if (files.Length == 0)
        {
            throw new FileNotFoundException(
                $"No .parquet files were found in '{fullPath}'.");
        }

        int emitted = 0;
        if (shuffleSeed.HasValue)
        {
            await foreach (string value in ReadShuffledRowGroupsAsync(
                files,
                textColumn,
                maxDocuments,
                shuffleSeed.Value,
                cancellationToken))
            {
                yield return value;
            }
            yield break;
        }

        foreach (string file in files)
        {
            cancellationToken.ThrowIfCancellationRequested();
            (ParquetReader openedReader, DataField field) =
                await OpenTextReaderAsync(
                    file,
                    textColumn,
                    cancellationToken);
            await using ParquetReader reader = openedReader;

            for (int group = 0; group < reader.RowGroupCount; group++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                using ParquetRowGroupReader rowGroup =
                    reader.OpenRowGroupReader(group);
                if (rowGroup.RowCount > int.MaxValue)
                {
                    throw new InvalidDataException(
                        $"Parquet row group {group} in '{file}' is too large.");
                }

                var values = new string?[(int)rowGroup.RowCount];
                await rowGroup.ReadAsync(
                    field,
                    values,
                    cancellationToken: cancellationToken);
                foreach (string? value in values)
                {
                    if (string.IsNullOrEmpty(value))
                        continue;
                    yield return value;
                    emitted++;
                    if (maxDocuments.HasValue
                        && emitted >= maxDocuments.Value)
                    {
                        yield break;
                    }
                }
            }
        }
    }

    /// <summary>
    /// Produces a stable baseline order and optionally permutes whole shards.
    /// Shard shuffling removes the article-length discontinuity between the
    /// end of one Wikipedia pass and the beginning of the next without
    /// retaining the complete text corpus in memory.
    /// </summary>
    internal static string[] OrderFiles(
        IEnumerable<string> paths,
        int? shuffleSeed)
    {
        ArgumentNullException.ThrowIfNull(paths);
        string[] files = paths
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (!shuffleSeed.HasValue)
            return files;

        var random = new Random(shuffleSeed.Value);
        for (int index = files.Length - 1; index > 0; index--)
        {
            int swapIndex = random.Next(index + 1);
            (files[index], files[swapIndex]) =
                (files[swapIndex], files[index]);
        }
        return files;
    }

    /// <summary>
    /// Preserves the original global permutation for checkpoint cursor replay.
    /// Only compact locations are retained, not shard readers or metadata.
    /// </summary>
    internal static RowGroupLocation[] OrderRowGroups(
        IReadOnlyList<int> groupCounts,
        int shuffleSeed)
    {
        ArgumentNullException.ThrowIfNull(groupCounts);
        if (groupCounts.Any(count => count < 0))
            throw new ArgumentOutOfRangeException(nameof(groupCounts));
        var groups = new RowGroupLocation[groupCounts.Sum()];
        int offset = 0;
        for (int fileIndex = 0; fileIndex < groupCounts.Count; fileIndex++)
        {
            int count = groupCounts[fileIndex];
            for (int groupIndex = 0; groupIndex < count; groupIndex++)
                groups[offset++] = new RowGroupLocation(fileIndex, groupIndex);
        }

        var random = new Random(shuffleSeed);
        for (int index = groups.Length - 1; index > 0; index--)
        {
            int swapIndex = random.Next(index + 1);
            (groups[index], groups[swapIndex]) =
                (groups[swapIndex], groups[index]);
        }
        return groups;
    }

    private static async IAsyncEnumerable<string>
        ReadShuffledRowGroupsAsync(
            string[] files,
            string textColumn,
            int? maxDocuments,
            int shuffleSeed,
            [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        // Read metadata one shard at a time. Keep the historical permutation:
        // changing to windowed shuffle would silently invalidate resume cursors.
        var groupCounts = new int[files.Length];
        for (int index = 0; index < files.Length; index++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            (ParquetReader reader, _) = await OpenTextReaderAsync(
                files[index], textColumn, cancellationToken);
            await using (reader)
                groupCounts[index] = reader.RowGroupCount;
        }

        RowGroupLocation[] groups = OrderRowGroups(groupCounts, shuffleSeed);
        var readers = new Dictionary<int, (ParquetReader Reader, DataField Field)>();
        var recency = new LinkedList<int>();
        int emitted = 0;
        try
        {
            foreach (RowGroupLocation location in groups)
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!readers.TryGetValue(location.FileIndex, out var entry))
                {
                    if (readers.Count == ShuffledReaderCacheCapacity)
                    {
                        int evicted = recency.First!.Value;
                        var old = readers[evicted];
                        readers.Remove(evicted);
                        recency.RemoveFirst();
                        await old.Reader.DisposeAsync();
                    }
                    entry = await OpenTextReaderAsync(
                        files[location.FileIndex], textColumn, cancellationToken);
                    readers.Add(location.FileIndex, entry);
                }
                recency.Remove(location.FileIndex);
                recency.AddLast(location.FileIndex);
                {
                    (ParquetReader reader, DataField field) = entry;
                    using ParquetRowGroupReader rowGroup =
                        reader.OpenRowGroupReader(location.GroupIndex);
                    if (rowGroup.RowCount > int.MaxValue)
                    {
                        throw new InvalidDataException(
                            $"Parquet row group {location.GroupIndex} in " +
                            $"'{files[location.FileIndex]}' is too large.");
                    }

                    var values = new string?[(int)rowGroup.RowCount];
                    await rowGroup.ReadAsync(
                        field,
                        values,
                        cancellationToken: cancellationToken);
                    foreach (string? value in values)
                    {
                        if (string.IsNullOrEmpty(value))
                            continue;
                        yield return value;
                        emitted++;
                        if (maxDocuments.HasValue
                            && emitted >= maxDocuments.Value)
                        {
                            yield break;
                        }
                    }
                }
            }
        }
        finally
        {
            // Start every disposal even if another reader fails to close.
            await Task.WhenAll(readers.Values.Select(async entry =>
                await entry.Reader.DisposeAsync()));
        }
    }

    private static async Task<(ParquetReader Reader, DataField Field)>
        OpenTextReaderAsync(
            string file,
            string textColumn,
            CancellationToken cancellationToken)
    {
        ParquetReader reader = await ParquetReader.CreateAsync(
            file,
            cancellationToken: cancellationToken);
        try
        {
            DataField[] fields = reader.Schema.GetDataFields();
            DataField? field = fields.FirstOrDefault(candidate =>
                string.Equals(
                    candidate.Name,
                    textColumn,
                    StringComparison.OrdinalIgnoreCase));
            if (field is null)
            {
                throw new InvalidDataException(
                    $"Parquet file '{file}' does not contain text column " +
                    $"'{textColumn}'. Available columns: " +
                    string.Join(", ", fields.Select(
                        candidate => candidate.Name)));
            }
            if (field.ClrType != typeof(string))
            {
                throw new InvalidDataException(
                    $"Parquet column '{textColumn}' in '{file}' has CLR " +
                    $"type '{field.ClrType.Name}', not String.");
            }
            return (reader, field);
        }
        catch
        {
            await reader.DisposeAsync();
            throw;
        }
    }

    internal readonly record struct RowGroupLocation(
        int FileIndex,
        int GroupIndex);
}
