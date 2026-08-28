using NNtrain;
using NNtrain.Training.Metrics;
using Xunit;

public sealed class MetricJournalTests
{
    [Fact]
    public void JournalRendersWithExistingCircleAndTitleContract()
    {
        using var directory = new TemporaryDirectory();
        string htmlPath = System.IO.Path.Combine(directory.Root, "loss.html");
        var journal = new MetricJournal();
        journal.Append(Entry(10, 0.5, MetricKinds.TrainLoss, 5));
        journal.Append(Entry(20, 1.0, MetricKinds.TrainLoss, 4));
        journal.Append(Entry(20, 1.0, MetricKinds.EvaluationLoss, 4.25));

        LossGraphMetricAdapter.RenderFromJournal(journal, htmlPath, 2);

        string html = File.ReadAllText(htmlPath);
        Assert.Equal(2, Count(html, "class=\"train-point\""));
        Assert.Equal(1, Count(html, "class=\"eval-point\""));
        Assert.Contains("<title>epoch 0.5: 5.000000</title>", html);
        Assert.Contains("<title>epoch 1: 4.000000</title>", html);
        Assert.Contains("<title>epoch 1: 4.250000</title>", html);
    }

    [Fact]
    public void ImportsLegacyHtmlOnlyWhenSidecarIsMissing()
    {
        using var directory = new TemporaryDirectory();
        string htmlPath = System.IO.Path.Combine(directory.Root, "loss.html");
        string sidecarPath = System.IO.Path.Combine(directory.Root, "metrics.jsonl");
        var legacy = new LossGraph(htmlPath, totalEpochs: 3);
        legacy.AddPoint(0.5f, 5f);
        legacy.AddPoint(1f, 4f, 4.25f);
        legacy.AddPoint(1.5f, 3f);
        legacy.Write();

        MetricJournal imported = LossGraphMetricAdapter.LoadSidecarOrImportLegacy(
            sidecarPath,
            htmlPath,
            totalEpochs: 3,
            checkpointGlobalStep: 100,
            checkpointEpoch: 1f,
            importTimestamp: DateTimeOffset.UnixEpoch);

        Assert.True(File.Exists(sidecarPath));
        Assert.Equal(3, imported.Count);
        Assert.DoesNotContain(imported.Entries, entry => entry.Epoch > 1d);
        File.WriteAllText(htmlPath, "broken legacy graph");
        MetricJournal reloaded = LossGraphMetricAdapter.LoadSidecarOrImportLegacy(
            sidecarPath,
            htmlPath,
            totalEpochs: 3,
            checkpointGlobalStep: 100,
            checkpointEpoch: 1f);
        Assert.Equal(imported.Entries.ToArray(), reloaded.Entries.ToArray());
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

    private static int Count(string text, string value)
    {
        int count = 0;
        int offset = 0;
        while ((offset = text.IndexOf(value, offset, StringComparison.Ordinal)) >= 0)
        {
            count++;
            offset += value.Length;
        }
        return count;
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = System.IO.Path.Combine(
                System.IO.Path.GetTempPath(),
                $"NNtrain.MetricJournalTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
