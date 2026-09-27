using NNtrain;
using NNtrain.Training.Metrics;
using Xunit;

public sealed class QwenLoraMetricReporterTests
{
    [Fact]
    public void HtmlExistsAndOpensAtStartBeforeFirstLossThenRefreshes()
    {
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { MaxSteps = 6 };
        string html = QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config);
        Assert.Equal(Path.Combine(fixture.DirectoryPath, "traning.transformer.html"), html);
        int opened = 0;
        using var reporter = new QwenLoraMetricReporter(config, html, 2, 0, false,
            fixture.Output, fixture.Error, path =>
            {
                ++opened;
                Assert.Equal(html, path);
                Assert.Contains("train points 0", File.ReadAllText(path));
                Assert.Empty(new MetricJournalJsonlRepository(TrainingMetricReporter.GetSidecarPath(path)).Load().Journal.Entries);
            });
        Assert.Equal(1, opened);
        reporter.Append(1, 4.5);
        var point = Assert.Single(new MetricJournalJsonlRepository(reporter.SidecarPath).Load().Journal.Entries);
        Assert.Equal(1, point.GlobalStep);
        Assert.Equal(0.5, point.Epoch);
        Assert.Equal(4.5, point.Value);
        string rendered = File.ReadAllText(html);
        Assert.Contains("train points 1", rendered);
        Assert.Contains("http-equiv=\"refresh\"", rendered);
        Assert.Equal(1, opened);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void HeadlessSettingsStillPersistLossWithoutOpeningBrowser(bool renderHtml)
    {
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { ShowLossGraph = renderHtml, OpenLossGraph = !renderHtml };
        string html = QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config);
        using var reporter = new QwenLoraMetricReporter(config, html, 2, 0, false,
            fixture.Output, fixture.Error, _ => Assert.Fail("Headless run must not launch a browser."));
        reporter.Append(1, 2);
        Assert.Equal(renderHtml, File.Exists(html));
        Assert.Single(new MetricJournalJsonlRepository(reporter.SidecarPath).Load().Journal.Entries);
    }

    [Fact]
    public void ResumeRetainsSavedLossesAndRemovesUnsavedUpdates()
    {
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { MaxSteps = 6, OpenLossGraph = false };
        string html = QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config);
        using (var first = new QwenLoraMetricReporter(config, html, 2, 0, false, fixture.Output, fixture.Error))
        {
            first.Append(1, 5); first.Append(2, 4); first.Append(3, 99);
        }
        using var resumed = new QwenLoraMetricReporter(config, html, 2, 2, true, fixture.Output, fixture.Error);
        Assert.Equal([1L, 2L], new MetricJournalJsonlRepository(resumed.SidecarPath).Load().Journal.Entries.Select(x => x.GlobalStep));
        Assert.DoesNotContain("99.000000", File.ReadAllText(html));
        resumed.Append(3, 3);
        Assert.Equal([5d, 4d, 3d], new MetricJournalJsonlRepository(resumed.SidecarPath).Load().Journal.Entries.Select(x => x.Value));
        Assert.Empty(resumed.ArchivedPaths);
    }

    [Fact]
    public void FreshRunArchivesBothArtifactsBeforeReplacingAndFlushesFinalPartialInterval()
    {
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { OpenLossGraph = false, LossGraphEverySteps = 3 };
        string html = QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config);
        string sidecar = TrainingMetricReporter.GetSidecarPath(html);
        File.WriteAllText(html, "previous HTML");
        File.WriteAllText(sidecar, "previous journal");
        using (var reporter = new QwenLoraMetricReporter(config, html, 4, 0, false, fixture.Output, fixture.Error))
        {
            Assert.Equal(2, reporter.ArchivedPaths.Count);
            Assert.Equal("previous HTML", File.ReadAllText(reporter.ArchivedPaths[0]));
            Assert.Equal("previous journal", File.ReadAllText(reporter.ArchivedPaths[1]));
            reporter.Append(1, 5); reporter.Append(2, 4);
            Assert.Equal(2, new MetricJournalJsonlRepository(sidecar).Load().Journal.Count);
            Assert.Contains("train points 1", File.ReadAllText(html));
        }
        Assert.Contains("train points 2", File.ReadAllText(html));
    }

    [Fact]
    public void MissingResumeJournalPreservesHtmlAndNewHistoryCanStartAtSavedStep()
    {
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { MaxSteps = 8, OpenLossGraph = false };
        string html = QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config);
        File.WriteAllText(html, "history without exact steps");
        Assert.Throws<IOException>(() => new QwenLoraMetricReporter(config, html, 2, 3, true, fixture.Output, fixture.Error));
        Assert.Equal("history without exact steps", File.ReadAllText(html));
        string newHtml = Path.Combine(fixture.DirectoryPath, "new-history.html");
        using var reporter = new QwenLoraMetricReporter(config, newHtml, 2, 3, true, fixture.Output, fixture.Error);
        reporter.Append(4, 2);
        Assert.Equal(4, Assert.Single(new MetricJournalJsonlRepository(reporter.SidecarPath).Load().Journal.Entries).GlobalStep);
    }

    [Fact]
    public void PathsResolveAgainstConfigurationAndCannotOverwriteTrainingInputs()
    {
        using var fixture = new Fixture();
        var config = new QwenLoraTrainingConfiguration { LossGraphPath = "plots/loss.html" };
        Assert.Equal(Path.Combine(fixture.DirectoryPath, "plots", "loss.html"),
            QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config));
        Assert.Throws<ArgumentException>(() => (config with { LossGraphEverySteps = 0 }).Validate());
        Assert.Throws<ArgumentException>(() => (config with { LossGraphPath = " " }).Validate());
        Assert.Throws<ArgumentException>(() => QwenLoraMetricReporter.ResolveHtmlPath(fixture.ConfigPath, config with { LossGraphPath = "base.gguf" }));
        File.WriteAllText(fixture.ConfigPath, "{\"adapterPath\":\"traning.transformer.html\"}");
        Assert.Equal(2, Program.Run(["qwen-lora", "--model", "absent.gguf", "--config", fixture.ConfigPath], fixture.Output, fixture.Error));
        Assert.Contains("distinct", fixture.Error.ToString());
    }

    [Fact]
    public void ExistingLoraCommandRoutesGgufAndLeavesJsonModelsOnOriginalPath()
    {
        Assert.True(LoraCommand.UsesGgufModel(["lora", "--resume", "--config", "train.json", "--model", "MODEL.GGUF"]));
        Assert.False(LoraCommand.UsesGgufModel(["lora", "--model", "model.json"]));
        Assert.False(LoraCommand.UsesGgufModel(["lora", "--config", "--model", "model.gguf"]));
        using var output = new StringWriter(); using var error = new StringWriter();
        Assert.Equal(2, Program.Run(["lora", "--model", "absent.gguf", "--generate", "a"], output, error));
        Assert.Contains("Unknown qwen-lora option: --generate", error.ToString());
    }

    private sealed class Fixture : IDisposable
    {
        internal string DirectoryPath { get; } = Path.Combine(Path.GetTempPath(), "NNtrain.QwenLoraMetrics." + Guid.NewGuid().ToString("N"));
        internal string ConfigPath => Path.Combine(DirectoryPath, "traning.transformer.json");
        internal StringWriter Output { get; } = new();
        internal StringWriter Error { get; } = new();
        internal Fixture() => Directory.CreateDirectory(DirectoryPath);
        public void Dispose() { Output.Dispose(); Error.Dispose(); Directory.Delete(DirectoryPath, recursive: true); }
    }
}
