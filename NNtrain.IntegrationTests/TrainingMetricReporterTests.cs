using NNtrain;
using NNtrain.Training.Metrics;
using Xunit;

public sealed class TrainingMetricReporterTests
{
    [Fact]
    public void FreshRunClearsArtifactsInsteadOfImportingLegacyHtml()
    {
        using var directory = new TemporaryDirectory();
        string htmlPath = Path.Combine(directory.Root, "training.loss.html");
        var legacy = new LossGraph(htmlPath, totalEpochs: 2);
        legacy.AddPoint(0.5f, 8f);
        legacy.Write();
        string sidecarPath = TrainingMetricReporter.GetSidecarPath(htmlPath);
        var oldRepository = new MetricJournalJsonlRepository(sidecarPath);
        oldRepository.AppendAndFlush(Entry(9, 0.9, 7));

        TrainingMetricReporter reporter = TrainingMetricReporter.Open(
            htmlPath,
            totalEpochs: 2,
            resume: false,
            checkpointGlobalStep: -1,
            checkpointEpoch: 0,
            renderHtml: true);

        Assert.Empty(
            new MetricJournalJsonlRepository(sidecarPath)
                .Load()
                .Journal
                .Entries);
        Assert.Contains("train points 0", File.ReadAllText(htmlPath));

        reporter.AppendCommittedLoss(
            globalStep: 1,
            epoch: 0.25,
            kind: MetricKinds.TrainLoss,
            value: 6,
            timestamp: DateTimeOffset.UnixEpoch);

        MetricJournalEntry persisted = Assert.Single(
            new MetricJournalJsonlRepository(sidecarPath)
                .Load()
                .Journal
                .Entries);
        Assert.Equal(1, persisted.GlobalStep);
        Assert.Contains("train points 1", File.ReadAllText(htmlPath));
    }

    [Fact]
    public void ResumeTruncatesAfterCheckpointAndNeverReimportsHtml()
    {
        using var directory = new TemporaryDirectory();
        string htmlPath = Path.Combine(directory.Root, "training.loss.html");
        string sidecarPath = TrainingMetricReporter.GetSidecarPath(htmlPath);
        var repository = new MetricJournalJsonlRepository(sidecarPath);
        repository.AppendAndFlush(Entry(10, 0.1, 5));
        repository.AppendAndFlush(Entry(20, 0.2, 4));
        repository.AppendAndFlush(Entry(30, 0.3, 3));
        var unrelatedLegacy = new LossGraph(htmlPath, totalEpochs: 2);
        unrelatedLegacy.AddPoint(1f, 99f);
        unrelatedLegacy.Write();

        TrainingMetricReporter reporter = TrainingMetricReporter.Open(
            htmlPath,
            totalEpochs: 2,
            resume: true,
            checkpointGlobalStep: 20,
            checkpointEpoch: 0.2,
            renderHtml: true);

        Assert.Equal(
            [10L, 20L],
            repository.Load().Journal.Entries
                .Select(entry => entry.GlobalStep)
                .ToArray());
        string recoveredHtml = File.ReadAllText(htmlPath);
        Assert.DoesNotContain("99.000000", recoveredHtml);
        Assert.DoesNotContain("3.000000", recoveredHtml);

        reporter.AppendCommittedLoss(
            globalStep: 21,
            epoch: 0.21,
            kind: MetricKinds.TrainLoss,
            value: 3.5,
            timestamp: DateTimeOffset.UnixEpoch);
        Assert.Equal(
            [10L, 20L, 21L],
            repository.Load().Journal.Entries
                .Select(entry => entry.GlobalStep)
                .ToArray());

        reporter.AppendCommittedLoss(
            globalStep: 21,
            epoch: 0.21,
            kind: MetricKinds.TrainLoss,
            value: 3.5,
            timestamp: DateTimeOffset.UtcNow);
        Assert.Equal(3, repository.Load().Journal.Count);
        Assert.Throws<InvalidDataException>(() =>
            reporter.AppendCommittedLoss(
                globalStep: 21,
                epoch: 0.21,
                kind: MetricKinds.TrainLoss,
                value: 3.6,
                timestamp: DateTimeOffset.UtcNow));
    }

    [Fact]
    public void DisabledHtmlStillPersistsAuthoritativeSidecar()
    {
        using var directory = new TemporaryDirectory();
        string htmlPath = Path.Combine(directory.Root, "training.loss.html");
        TrainingMetricReporter reporter = TrainingMetricReporter.Open(
            htmlPath,
            totalEpochs: 1,
            resume: false,
            checkpointGlobalStep: -1,
            checkpointEpoch: 0,
            renderHtml: false);

        reporter.AppendCommittedLoss(
            globalStep: 1,
            epoch: 1,
            kind: MetricKinds.TrainLoss,
            value: 2,
            timestamp: DateTimeOffset.UnixEpoch);

        Assert.False(File.Exists(htmlPath));
        Assert.Single(
            new MetricJournalJsonlRepository(reporter.SidecarPath)
                .Load()
                .Journal
                .Entries);
    }

    [Fact]
    public void DatasetContinuationDefaultsToEveryTwoThousandCommittedSteps()
    {
        var config = new WikiTrainingConfiguration();
        var generatedAt = new List<long>();
        using var warning = new StringWriter();

        Assert.Equal(2000, config.DatasetSampleEverySteps);
        Assert.False(
            WikiLanguageModelCommand.ShouldGenerateDatasetContinuation(
                1999,
                config.DatasetSampleEverySteps,
                retainedDocumentCount: 1));
        Assert.True(
            WikiLanguageModelCommand.ShouldGenerateDatasetContinuation(
                2000,
                config.DatasetSampleEverySteps,
                retainedDocumentCount: 1));
        Assert.False(
            WikiLanguageModelCommand.ShouldGenerateDatasetContinuation(
                2000,
                config.DatasetSampleEverySteps,
                retainedDocumentCount: 0));

        foreach (long step in new long[] { 1999, 2000, 2001 })
        {
            long capturedStep = step;
            WikiLanguageModelCommand.RunDatasetContinuationAfterCommittedStep(
                committedGlobalStep: step,
                everySteps: config.DatasetSampleEverySteps,
                retainedDocumentCount: 1,
                generate: () => generatedAt.Add(capturedStep),
                warning: warning);
        }
        Assert.Equal([2000L], generatedAt);
        Assert.Equal(string.Empty, warning.ToString());
    }

    [Fact]
    public void GenerationFailureAfterMetricAndCheckpointCommitWarnsAndContinues()
    {
        var events = new List<string>();
        using var warning = new StringWriter();
        events.Add("metric commit");
        events.Add("checkpoint commit");

        WikiLanguageModelCommand.RunDatasetContinuationAfterCommittedStep(
            committedGlobalStep: 2000,
            everySteps: 2000,
            retainedDocumentCount: 1,
            generate: () =>
            {
                Assert.Equal(
                    ["metric commit", "checkpoint commit"],
                    events);
                events.Add("generation");
                throw new InvalidOperationException("sample failed");
            },
            warning: warning);
        events.Add("next step");

        Assert.Equal(
            [
                "metric commit",
                "checkpoint commit",
                "generation",
                "next step",
            ],
            events);
        Assert.Contains(
            "Warning: dataset continuation generation at step 2,000 failed",
            warning.ToString());
    }

    [Theory]
    [InlineData(600)]
    [InlineData(700)]
    [InlineData(719)]
    public void FatalCudaGenerationFailureIsNotDowngradedToWarning(
        int status)
    {
        using var warning = new StringWriter();

        InvalidOperationException exception =
            Assert.Throws<InvalidOperationException>(() =>
                WikiLanguageModelCommand
                    .RunDatasetContinuationAfterCommittedStep(
                        committedGlobalStep: 2000,
                        everySteps: 2000,
                        retainedDocumentCount: 1,
                        generate: () => throw new InvalidOperationException(
                            $"generation failed with CUDA error {status}"),
                        warning: warning));

        Assert.Contains($"CUDA error {status}", exception.Message);
        Assert.Equal(string.Empty, warning.ToString());
    }

    private static MetricJournalEntry Entry(
        long globalStep,
        double epoch,
        double value)
        => new(
            globalStep,
            epoch,
            epoch / 2d,
            MetricKinds.TrainLoss,
            value,
            DateTimeOffset.UnixEpoch);

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.TrainingMetricReporterTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
