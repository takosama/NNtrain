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

        string[] files = Directory.GetFiles(fullPath, "*.parquet")
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
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
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
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

        string[] files = Directory.GetFiles(fullPath, "*.parquet")
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToArray();
        if (files.Length == 0)
        {
            throw new FileNotFoundException(
                $"No .parquet files were found in '{fullPath}'.");
        }

        int emitted = 0;
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
}
