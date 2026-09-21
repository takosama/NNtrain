using NNtrain;
using NNtrain.Training.Metrics;
using Xunit;

public sealed class DpoMetricReporterTests
{
    [Fact]
    public void HtmlUsesStepsAndResumeKeepsCommittedHistoryOnly()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"NNtrain.DpoMetrics-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string html = Path.Combine(directory, "dpo.loss.html");
            using (var reporter = new DpoMetricReporter(html, 100, 10, false, 0))
            {
                reporter.Append(1, 1, .69);
                Assert.Contains("<title>step 1:", File.ReadAllText(html));
                reporter.Append(2, 1, .62);
                reporter.Append(3, 1, .61);
                Assert.DoesNotContain("<title>step 2:", File.ReadAllText(html));
            }
            Assert.Contains("<title>step 3:", File.ReadAllText(html));
            var repository = new MetricJournalJsonlRepository(TrainingMetricReporter.GetSidecarPath(html));
            var original = repository.Load().Journal.Entries;
            using (var resumed = new DpoMetricReporter(html, 200, 10, true, 2))
            {
                Assert.DoesNotContain("<title>step 3:", File.ReadAllText(html));
                resumed.Append(3, 1, .59);
            }
            var saved = repository.Load().Journal.Entries;
            Assert.Equal(3, saved.Count);
            Assert.Equal(original[0], saved[0]); Assert.Equal(original[1], saved[1]);
            Assert.Equal(.59, saved[2].Value);
            string content = File.ReadAllText(html);
            Assert.Contains("DPO loss by step", content);
            Assert.Contains("global step", content);
            Assert.Contains("<title>step 3:", content);
            byte[] journalBytes = File.ReadAllBytes(repository.Path);
            using var fresh = new DpoMetricReporter(html, 100, 10, false, 0);
            Assert.Equal(2, fresh.ArchivedPaths.Count);
            Assert.Equal(content, File.ReadAllText(fresh.ArchivedPaths.Single(path => path.EndsWith(".html"))));
            Assert.Equal(journalBytes, File.ReadAllBytes(fresh.ArchivedPaths.Single(path => path.EndsWith(".jsonl"))));
            Assert.Empty(repository.Load().Journal.Entries);
            Assert.DoesNotContain("<title>step 3:", File.ReadAllText(html));
            fresh.Append(1, 1, .7);
            Assert.Single(repository.Load().Journal.Entries);
        }
        finally { Directory.Delete(directory, true); }
    }
    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public void NewRunPreservesPartialOldHistoryAndUsesUniqueArchives(bool hasHtml, bool hasSidecar)
    {
        string directory = Path.Combine(Path.GetTempPath(), $"NNtrain.DpoMetrics-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string html = Path.Combine(directory, "dpo.loss.html");
            string sidecar = TrainingMetricReporter.GetSidecarPath(html);
            const string old = "old history is archived byte-for-byte, not parsed";
            File.WriteAllText(hasHtml ? html : sidecar, old);
            string archive;
            using (var fresh = new DpoMetricReporter(html, 100, 10, false, 0))
            {
                archive = Assert.Single(fresh.ArchivedPaths);
                Assert.Equal(old, File.ReadAllText(archive));
                Assert.Equal(hasSidecar, archive.EndsWith(".jsonl"));
                fresh.Append(1, 1, .7);
            }
            using var another = new DpoMetricReporter(html, 100, 10, false, 0);
            Assert.Equal(2, another.ArchivedPaths.Count);
            Assert.DoesNotContain(archive, another.ArchivedPaths);
            Assert.Equal(old, File.ReadAllText(archive));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void FailedBackupDoesNotReplaceEitherOriginal()
    {
        Assert.SkipUnless(OperatingSystem.IsWindows(), "Windows file-sharing failure regression.");
        string directory = Path.Combine(Path.GetTempPath(), $"NNtrain.DpoMetrics-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string html = Path.Combine(directory, "dpo.loss.html");
            string sidecar = TrainingMetricReporter.GetSidecarPath(html);
            File.WriteAllText(html, "old html"); File.WriteAllText(sidecar, "old metrics");
            using (var locked = new FileStream(sidecar, FileMode.Open, FileAccess.ReadWrite, FileShare.None))
                Assert.ThrowsAny<IOException>(() => new DpoMetricReporter(html, 100, 10, false, 0));
            Assert.Equal("old html", File.ReadAllText(html));
            Assert.Equal("old metrics", File.ReadAllText(sidecar));
        }
        finally { Directory.Delete(directory, true); }
    }
    [Fact]
    public void ResumeOfOldCheckpointWithoutMetricsCreatesEmptyGraph()
    {
        string directory = Path.Combine(Path.GetTempPath(), $"NNtrain.DpoMetrics-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        try
        {
            string html = Path.Combine(directory, "dpo.loss.html");
            using var reporter = new DpoMetricReporter(html, 100, 10, true, 50);
            Assert.True(File.Exists(html));
            reporter.Append(51, 1, .5);
            reporter.Flush();
            Assert.Contains("<title>step 51:", File.ReadAllText(html));
        }
        finally { Directory.Delete(directory, true); }
    }
}
