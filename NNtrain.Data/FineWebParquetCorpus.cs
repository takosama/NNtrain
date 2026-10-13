namespace NNtrain;

/// <summary>
/// Streams the text field from local FineWeb Parquet shards. FineWeb uses
/// the same columnar text contract as the existing Wikipedia reader, while
/// retaining a distinct public entry point and configuration identity.
/// </summary>
public static class FineWebParquetCorpus
{
    public static Task<long> CountRowsAsync(
        string directoryPath,
        CancellationToken cancellationToken = default)
        => ParquetTextCorpus.CountRowsAsync(
            directoryPath,
            cancellationToken);

    public static IAsyncEnumerable<string> ReadTextsAsync(
        string directoryPath,
        string textColumn = "text",
        int? maxDocuments = null,
        CancellationToken cancellationToken = default,
        int? shuffleSeed = null)
        => ParquetTextCorpus.ReadTextsAsync(
            directoryPath,
            textColumn,
            maxDocuments,
            cancellationToken,
            shuffleSeed);
}
