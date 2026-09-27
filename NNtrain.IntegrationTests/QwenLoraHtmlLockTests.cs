using NNtrain;
using NNtrain.Training.Metrics;
using Xunit;

public sealed class QwenLoraHtmlLockTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void LockedHtmlKeepsCommittedLossesWarnsOnceAndRetriesAfterUnlock(bool recoverWithFlush)
    {
        Assert.SkipWhen(!OperatingSystem.IsWindows(), "Windows file sharing reproduces the browser replacement lock.");
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration
        {
            MaxSteps = 8, OpenLossGraph = false, LossGraphEverySteps = 4
        };
        using var reporter = new QwenLoraMetricReporter(config, fixture.HtmlPath, 8, 0, false,
            fixture.Output, fixture.Error);
        reporter.Append(1, 6);
        reporter.Append(2, 5);
        reporter.Append(3, 4);
        string previousHtml = File.ReadAllText(fixture.HtmlPath);

        // A browser can read the file while denying FILE_SHARE_DELETE. The
        // atomic rename must fail, while the separate durable journal remains writable.
        using (var browserRead = new FileStream(fixture.HtmlPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            reporter.Append(4, 3);
            reporter.Flush();
            reporter.Flush();
            Assert.Equal([1L, 2L, 3L, 4L], fixture.Entries().Select(entry => entry.GlobalStep));
            Assert.Equal([6d, 5d, 4d, 3d], fixture.Entries().Select(entry => entry.Value));
            Assert.Equal(previousHtml, File.ReadAllText(fixture.HtmlPath));
            string warning = Assert.Single(fixture.Error.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries));
            Assert.Contains("warning", warning, StringComparison.OrdinalIgnoreCase);
            Assert.Empty(Directory.GetFiles(fixture.DirectoryPath, "*.tmp"));
        }

        if (recoverWithFlush)
        {
            reporter.Flush();
            Assert.Contains("train points 4", File.ReadAllText(fixture.HtmlPath));
        }
        else
        {
            // Pending failed rendering retries on the next update, even when
            // that update is outside the normal every-four-updates cadence.
            reporter.Append(5, 2);
            Assert.Contains("train points 5", File.ReadAllText(fixture.HtmlPath));
        }
        Assert.Single(fixture.Error.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries));
        Assert.Empty(Directory.GetFiles(fixture.DirectoryPath, "*.tmp"));
    }

    [Fact]
    public void LockedHtmlAtResumeKeepsCheckpointRecoveryAndLaterRebuilds()
    {
        Assert.SkipWhen(!OperatingSystem.IsWindows(), "Windows file sharing reproduces the browser replacement lock.");
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { MaxSteps = 8, OpenLossGraph = false };
        using (var first = new QwenLoraMetricReporter(config, fixture.HtmlPath, 8, 0, false,
            fixture.Output, fixture.Error))
        {
            first.Append(1, 5);
            first.Append(2, 4);
            first.Append(3, 99);
        }
        QwenLoraMetricReporter resumed;
        using (var browserRead = new FileStream(fixture.HtmlPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            resumed = new QwenLoraMetricReporter(config, fixture.HtmlPath, 8, 2, true,
                fixture.Output, fixture.Error);
            Assert.Equal([1L, 2L], fixture.Entries().Select(entry => entry.GlobalStep));
            resumed.Flush();
            Assert.Single(fixture.Error.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries));
            Assert.Empty(Directory.GetFiles(fixture.DirectoryPath, "*.tmp"));
        }
        using (resumed)
        {
            resumed.Append(3, 3);
            Assert.Equal([5d, 4d, 3d], fixture.Entries().Select(entry => entry.Value));
            string rendered = File.ReadAllText(fixture.HtmlPath);
            Assert.Contains("train points 3", rendered);
            Assert.DoesNotContain("99.000000", rendered);
        }
    }

    [Fact]
    public void DisposeWhileHtmlIsLockedDoesNotFailAndJournalCanRebuildNextRun()
    {
        Assert.SkipWhen(!OperatingSystem.IsWindows(), "Windows file sharing reproduces the browser replacement lock.");
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { MaxSteps = 8, OpenLossGraph = false };
        var reporter = new QwenLoraMetricReporter(config, fixture.HtmlPath, 8, 0, false,
            fixture.Output, fixture.Error);
        using (var browserRead = new FileStream(fixture.HtmlPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            reporter.Append(1, 5);
            reporter.Dispose();
            Assert.Equal(1, Assert.Single(fixture.Entries()).GlobalStep);
            Assert.Single(fixture.Error.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries));
            Assert.Empty(Directory.GetFiles(fixture.DirectoryPath, "*.tmp"));
        }
        using var resumed = new QwenLoraMetricReporter(config, fixture.HtmlPath, 8, 1, true,
            fixture.Output, fixture.Error);
        Assert.Contains("train points 1", File.ReadAllText(fixture.HtmlPath));
    }

    [Fact]
    public void JournalWriteFailureStillStopsInsteadOfLosingCommittedMetricsSilently()
    {
        Assert.SkipWhen(!OperatingSystem.IsWindows(), "Windows file sharing reproduces the journal write lock.");
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { MaxSteps = 8, OpenLossGraph = false };
        using var reporter = new QwenLoraMetricReporter(config, fixture.HtmlPath, 8, 0, false,
            fixture.Output, fixture.Error);
        using (var journalRead = new FileStream(reporter.SidecarPath, FileMode.Open, FileAccess.Read, FileShare.Read))
        {
            Assert.ThrowsAny<IOException>(() => reporter.Append(1, 5));
            Assert.Empty(fixture.Entries());
            Assert.Contains("train points 0", File.ReadAllText(fixture.HtmlPath));
            Assert.Equal(string.Empty, fixture.Error.ToString());
        }
        reporter.Append(1, 5);
        Assert.Equal(1, Assert.Single(fixture.Entries()).GlobalStep);
    }

    private sealed class Fixture : IDisposable
    {
        internal string DirectoryPath { get; } = Path.Combine(Path.GetTempPath(), "NNtrain.QwenLoraHtmlLock." + Guid.NewGuid().ToString("N"));
        internal string HtmlPath => Path.Combine(DirectoryPath, "loss.html");
        internal StringWriter Output { get; } = new();
        internal StringWriter Error { get; } = new();
        internal Fixture() => Directory.CreateDirectory(DirectoryPath);
        internal IReadOnlyList<MetricJournalEntry> Entries()
            => new MetricJournalJsonlRepository(TrainingMetricReporter.GetSidecarPath(HtmlPath)).Load().Journal.Entries;
        public void Dispose()
        {
            Output.Dispose(); Error.Dispose(); Directory.Delete(DirectoryPath, recursive: true);
        }
    }
}
