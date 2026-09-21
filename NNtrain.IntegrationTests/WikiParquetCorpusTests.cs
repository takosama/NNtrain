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
    public async Task SeededReaderPreservesGlobalOrderAndDisposesAllReaders()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain.WikiParquetCorpusTests-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            int shardCount = WikiParquetCorpus.ShuffledReaderCacheCapacity * 2
                + 3;
            for (int index = 0; index < shardCount; index++)
            {
                await WriteShard(
                    Path.Combine(directory, $"train-{index:D5}.parquet"),
                    [$"doc-{index:D5}"]);
            }

            string[] first = await ReadAll(directory, 31415);
            string[] repeated = await ReadAll(directory, 31415);

            Assert.Equal(first, repeated);
            string[] files = WikiParquetCorpus.OrderFiles(
                Directory.GetFiles(directory, "*.parquet"), 31415);
            string[] expected = WikiParquetCorpus.OrderRowGroups(
                Enumerable.Repeat(1, shardCount).ToArray(), 31415)
                .Select(group => "doc-" + Path.GetFileNameWithoutExtension(
                    files[group.FileIndex])[6..])
                .ToArray();
            Assert.Equal(expected, first);
            Assert.Equal(
                Enumerable.Range(0, shardCount)
                    .Select(index => $"doc-{index:D5}"),
                first.Order(StringComparer.Ordinal));
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public async Task SeededReaderBoundsOpenReadersAndClosesOnEarlyExit()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain.WikiParquetBounded-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            int shardCount = WikiParquetCorpus.ShuffledReaderCacheCapacity * 2 + 1;
            string[] paths = Enumerable.Range(0, shardCount)
                .Select(index => Path.Combine(
                    directory,
                    $"train-{index:D5}.parquet"))
                .ToArray();
            const int Seed = 27182;
            foreach (string path in paths)
            {
                await WriteShard(
                    path,
                    [Path.GetFileNameWithoutExtension(path)]);
            }

            var values = new List<string>();
            await foreach (string value in WikiParquetCorpus.ReadTextsAsync(
                directory,
                maxDocuments: shardCount - 1,
                cancellationToken: TestContext.Current.CancellationToken,
                shuffleSeed: Seed))
            {
                values.Add(value);
                Assert.InRange(CountLockedFiles(paths), 1,
                    WikiParquetCorpus.ShuffledReaderCacheCapacity);
            }

            Assert.Equal(shardCount - 1, values.Count);
            Assert.Equal(0, CountLockedFiles(paths));
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

    private static async Task WriteShard(
        string path,
        string[] values,
        string fieldName = "text")
    {
        var field = new DataField<string>(fieldName);
        var schema = new ParquetSchema(field);
        await using Stream stream = File.Create(path);
        await using ParquetWriter writer =
            await ParquetWriter.CreateAsync(schema, stream);
        using ParquetRowGroupWriter rowGroup = writer.CreateRowGroup();
        await rowGroup.WriteAsync(field, values);
    }

    private static int CountLockedFiles(IEnumerable<string> paths)
    {
        int locked = 0;
        foreach (string path in paths)
        {
            try
            {
                using FileStream stream = File.Open(
                    path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            }
            catch (IOException)
            {
                locked++;
            }
        }
        return locked;
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
