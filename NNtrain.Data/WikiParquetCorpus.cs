using System.Runtime.CompilerServices;
using Parquet;
using Parquet.Schema;

namespace NNtrain;

/// <summary>
/// Streams a text column from sharded Wikipedia Parquet files.
/// </summary>
public static class WikiParquetCorpus
{
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
            await using ParquetReader reader = await ParquetReader.CreateAsync(
                file,
                cancellationToken: cancellationToken);
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
                    string.Join(", ", fields.Select(candidate => candidate.Name)));
            }
            if (field.ClrType != typeof(string))
            {
                throw new InvalidDataException(
                    $"Parquet column '{textColumn}' in '{file}' has CLR type " +
                    $"'{field.ClrType.Name}', not String.");
            }

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
    /// Builds a global row-group permutation. Wikipedia shards in the current
    /// corpus contain small row groups, so this mixes long-article and
    /// short-article shards while only materializing one row group at a time.
    /// </summary>
    internal static RowGroupLocation[] OrderRowGroups(
        IReadOnlyList<int> groupCounts,
        int shuffleSeed)
    {
        ArgumentNullException.ThrowIfNull(groupCounts);
        if (groupCounts.Any(count => count < 0))
            throw new ArgumentOutOfRangeException(nameof(groupCounts));
        var groups = new List<RowGroupLocation>(groupCounts.Sum());
        for (int fileIndex = 0; fileIndex < groupCounts.Count; fileIndex++)
        {
            int count = groupCounts[fileIndex];
            for (int groupIndex = 0; groupIndex < count; groupIndex++)
                groups.Add(new RowGroupLocation(fileIndex, groupIndex));
        }

        var random = new Random(shuffleSeed);
        for (int index = groups.Count - 1; index > 0; index--)
        {
            int swapIndex = random.Next(index + 1);
            (groups[index], groups[swapIndex]) =
                (groups[swapIndex], groups[index]);
        }
        return groups.ToArray();
    }

    private static async IAsyncEnumerable<string>
        ReadShuffledRowGroupsAsync(
            string[] files,
            string textColumn,
            int? maxDocuments,
            int shuffleSeed,
            [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var readers = new List<(ParquetReader Reader, DataField Field)>(
            files.Length);
        try
        {
            foreach (string file in files)
            {
                cancellationToken.ThrowIfCancellationRequested();
                ParquetReader reader = await ParquetReader.CreateAsync(
                    file,
                    cancellationToken: cancellationToken);
                DataField[] fields = reader.Schema.GetDataFields();
                DataField? field = fields.FirstOrDefault(candidate =>
                    string.Equals(
                        candidate.Name,
                        textColumn,
                        StringComparison.OrdinalIgnoreCase));
                if (field is null)
                {
                    await reader.DisposeAsync();
                    throw new InvalidDataException(
                        $"Parquet file '{file}' does not contain text column " +
                        $"'{textColumn}'. Available columns: " +
                        string.Join(", ", fields.Select(candidate => candidate.Name)));
                }
                if (field.ClrType != typeof(string))
                {
                    await reader.DisposeAsync();
                    throw new InvalidDataException(
                        $"Parquet column '{textColumn}' in '{file}' has CLR " +
                        $"type '{field.ClrType.Name}', not String.");
                }
                readers.Add((reader, field));
            }

            RowGroupLocation[] groups = OrderRowGroups(
                readers.Select(item => item.Reader.RowGroupCount).ToArray(),
                shuffleSeed);
            int emitted = 0;
            foreach (RowGroupLocation location in groups)
            {
                cancellationToken.ThrowIfCancellationRequested();
                (ParquetReader reader, DataField field) =
                    readers[location.FileIndex];
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
        finally
        {
            foreach ((ParquetReader reader, _) in readers)
                await reader.DisposeAsync();
        }
    }

    internal readonly record struct RowGroupLocation(
        int FileIndex,
        int GroupIndex);
}
