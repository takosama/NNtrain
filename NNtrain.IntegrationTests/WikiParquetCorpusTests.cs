using NNtrain;
using Parquet;
using Parquet.Schema;
using Xunit;

public sealed class WikiParquetCorpusTests
{
    [Fact]
    public void ShardOrderIsSortedWithoutSeedAndDeterministicWithSeed()
    {
        string[] paths = Enumerable.Range(0, 20)
            .Select(index => $"train-{19 - index:D5}.parquet")
            .ToArray();

        string[] sorted = WikiParquetCorpus.OrderFiles(paths, null);
        string[] first = WikiParquetCorpus.OrderFiles(paths, 1234);
        string[] repeated = WikiParquetCorpus.OrderFiles(paths, 1234);
        string[] nextEpoch = WikiParquetCorpus.OrderFiles(paths, 5678);

        Assert.Equal(paths.Order(StringComparer.OrdinalIgnoreCase), sorted);
        Assert.Equal(first, repeated);
        Assert.Equal(sorted, first.Order(StringComparer.OrdinalIgnoreCase));
        Assert.NotEqual(sorted, first);
        Assert.NotEqual(first, nextEpoch);
    }

    [Fact]
    public void RowGroupsAreGloballyPermutedAcrossEveryShard()
    {
        int[] counts = [3, 2, 4, 1];

        WikiParquetCorpus.RowGroupLocation[] first =
            WikiParquetCorpus.OrderRowGroups(counts, 2468);
        WikiParquetCorpus.RowGroupLocation[] repeated =
            WikiParquetCorpus.OrderRowGroups(counts, 2468);
        WikiParquetCorpus.RowGroupLocation[] nextEpoch =
            WikiParquetCorpus.OrderRowGroups(counts, 8642);

        Assert.Equal(first, repeated);
        Assert.Equal(counts.Sum(), first.Length);
        Assert.Equal(
            Enumerable.Range(0, counts.Length),
            first.Select(group => group.FileIndex).Distinct().Order());
        Assert.False(first.SequenceEqual(nextEpoch));
        for (int file = 0; file < counts.Length; file++)
        {
            Assert.Equal(
                Enumerable.Range(0, counts[file]),
                first
                    .Where(group => group.FileIndex == file)
                    .Select(group => group.GroupIndex)
                    .Order());
        }
    }

    [Fact]
    public async Task SeededReaderUsesGlobalOrderAndDisposesAllReaders()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain.WikiParquetCorpusTests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            await WriteShard(
                Path.Combine(directory, "train-00000.parquet"),
                ["a0", "a1"]);
            await WriteShard(
                Path.Combine(directory, "train-00001.parquet"),
                ["b0", "b1"]);
            await WriteShard(
                Path.Combine(directory, "train-00002.parquet"),
                ["c0", "c1"]);

            string[] first = await ReadAll(directory, 31415);
            string[] repeated = await ReadAll(directory, 31415);

            Assert.Equal(first, repeated);
            Assert.Equal(
                new[] { "a0", "a1", "b0", "b1", "c0", "c1" },
                first.Order(StringComparer.Ordinal));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task FineWebFactoryReadsParquetTextColumn()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain.FineWebParquetCorpusTests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            await WriteShard(
                Path.Combine(directory, "train-00000.parquet"),
                ["fineweb-a", "fineweb-b"]);

            long count = await FineWebParquetCorpus.CountRowsAsync(
                directory,
                TestContext.Current.CancellationToken);
            var values = new List<string>();
            await foreach (string value in datasets.fineweb(
                directory,
                cancellation_token: TestContext.Current.CancellationToken))
                values.Add(value);

            Assert.Equal(2, count);
            Assert.Equal(["fineweb-a", "fineweb-b"], values);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    private static async Task WriteShard(string path, string[] values)
    {
        var field = new DataField<string>("text");
        var schema = new ParquetSchema(field);
        await using Stream stream = File.Create(path);
        await using ParquetWriter writer =
            await ParquetWriter.CreateAsync(schema, stream);
        using ParquetRowGroupWriter rowGroup = writer.CreateRowGroup();
        await rowGroup.WriteAsync(field, values);
    }

    private static async Task<string[]> ReadAll(
        string directory,
        int seed)
    {
        var values = new List<string>();
        await foreach (string value in WikiParquetCorpus.ReadTextsAsync(
            directory,
            shuffleSeed: seed))
        {
            values.Add(value);
        }
        return values.ToArray();
    }
}
