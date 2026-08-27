using System.Text;
using NNtrain.Training.Metrics;
using Xunit;

public sealed class MetricJournalRepositoryTests
{
    [Fact]
    public void AppendsFlushesAndLoadsCanonicalJsonLines()
    {
        using var directory = new TemporaryDirectory();
        string path = System.IO.Path.Combine(directory.Root, "metrics.jsonl");
        var repository = new MetricJournalJsonlRepository(path);
        MetricJournalEntry first = Entry(1, 0.25, MetricKinds.TrainLoss, 5.5);
        MetricJournalEntry second = Entry(1, 0.25, MetricKinds.EvaluationLoss, 5.75);

        repository.AppendAndFlush(first);
        repository.AppendAndFlush(second);
        MetricJournalLoadResult loaded = repository.Load();

        Assert.False(loaded.IgnoredCorruptTail);
        Assert.Equal(2, loaded.Journal.Count);
        Assert.Equal(first, loaded.Journal.Entries[0]);
        Assert.Equal(second, loaded.Journal.Entries[1]);
        string persisted = File.ReadAllText(path);
        Assert.Contains("\"globalStep\":1", persisted);
        Assert.Contains("\"kind\":\"train_loss\"", persisted);
        Assert.EndsWith("\n", persisted, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(
            () => repository.AppendAndFlush(
                Entry(0, 0.1, MetricKinds.TrainLoss, 6)));
    }

    [Fact]
    public void RecoveryIgnoresCorruptTailAndTruncatesAfterCheckpoint()
    {
        using var directory = new TemporaryDirectory();
        string path = System.IO.Path.Combine(directory.Root, "metrics.jsonl");
        var repository = new MetricJournalJsonlRepository(path);
        repository.AppendAndFlush(Entry(1, 0.1, MetricKinds.TrainLoss, 5));
        repository.AppendAndFlush(Entry(2, 0.2, MetricKinds.TrainLoss, 4));
        repository.AppendAndFlush(Entry(3, 0.3, MetricKinds.TrainLoss, 3));
        File.AppendAllText(path, "{\"globalStep\":4", new UTF8Encoding(false));

        MetricJournalLoadResult recovered = repository.RecoverThrough(2);

        Assert.True(recovered.IgnoredCorruptTail);
        Assert.Equal(1, recovered.RemovedAfterCheckpoint);
        Assert.Equal([1L, 2L], recovered.Journal.Entries.Select(x => x.GlobalStep));
        MetricJournalLoadResult reloaded = repository.Load();
        Assert.False(reloaded.IgnoredCorruptTail);
        Assert.Equal(2, reloaded.Journal.Count);
    }

    [Fact]
    public void CorruptionBeforeFinalRecordIsNotSilentlyIgnored()
    {
        using var directory = new TemporaryDirectory();
        string path = System.IO.Path.Combine(directory.Root, "metrics.jsonl");
        var repository = new MetricJournalJsonlRepository(path);
        repository.AppendAndFlush(Entry(1, 0.1, MetricKinds.TrainLoss, 5));
        File.AppendAllText(path, "not json\n", new UTF8Encoding(false));
        File.AppendAllText(
            path,
            "{\"globalStep\":2,\"epoch\":0.2,\"progress\":0.2," +
            "\"kind\":\"train_loss\",\"value\":4," +
            "\"timestamp\":\"2026-08-27T00:00:00+00:00\"}\n",
            new UTF8Encoding(false));

        Assert.Throws<InvalidDataException>(() => repository.Load());
    }

    private static MetricJournalEntry Entry(
        long step,
        double epoch,
        string kind,
        double value)
        => new(
            step,
            epoch,
            Math.Clamp(epoch / 5d, 0d, 1d),
            kind,
            value,
            DateTimeOffset.Parse("2026-08-27T00:00:00+00:00"));

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                $"NNtrain.MetricJournalRepositoryTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
