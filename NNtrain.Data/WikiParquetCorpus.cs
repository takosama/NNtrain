namespace NNtrain;

/// <summary>Streams a text column from sharded Wikipedia Parquet files.</summary>
public static class WikiParquetCorpus
{
    internal const int ShuffledReaderCacheCapacity =
        ParquetTextCorpus.ShuffledReaderCacheCapacity;

    public static Task<long> CountRowsAsync(
        string directoryPath,
        CancellationToken cancellationToken = default)
        => ParquetTextCorpus.CountRowsAsync(directoryPath, cancellationToken);

    public static IAsyncEnumerable<string> ReadTextsAsync(
        string directoryPath,
        string textColumn = "text",
        int? maxDocuments = null,
        CancellationToken cancellationToken = default,
        int? shuffleSeed = null)
        => ParquetTextCorpus.ReadTextsAsync(
            directoryPath, textColumn, maxDocuments, cancellationToken, shuffleSeed);

    internal static string[] OrderFiles(IEnumerable<string> paths, int? shuffleSeed)
        => ParquetTextCorpus.OrderFiles(paths, shuffleSeed);

    internal static RowGroupLocation[] OrderRowGroups(
        IReadOnlyList<int> groupCounts,
        int shuffleSeed)
        => ParquetTextCorpus.OrderRowGroups(groupCounts, shuffleSeed)
            .Select(location => new RowGroupLocation(location.FileIndex, location.GroupIndex))
            .ToArray();

    internal readonly record struct RowGroupLocation(int FileIndex, int GroupIndex);
}
