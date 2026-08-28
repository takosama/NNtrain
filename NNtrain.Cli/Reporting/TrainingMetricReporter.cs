using NNtrain.Training.Metrics;

namespace NNtrain;

/// <summary>
/// Owns the durable loss journal for one training run. The JSONL sidecar is
/// authoritative; the existing HTML graph is a projection that can always be
/// rebuilt from it.
/// </summary>
internal sealed class TrainingMetricReporter
{
    private readonly MetricJournalJsonlRepository _repository;
    private readonly string _htmlPath;
    private readonly int _totalEpochs;
    private readonly bool _renderHtml;
    private readonly MetricJournal _journal;

    private TrainingMetricReporter(
        MetricJournalJsonlRepository repository,
        string htmlPath,
        int totalEpochs,
        bool renderHtml,
        MetricJournal journal)
    {
        _repository = repository;
        _htmlPath = htmlPath;
        _totalEpochs = totalEpochs;
        _renderHtml = renderHtml;
        _journal = journal;
    }

    internal string SidecarPath => _repository.Path;

    internal string HtmlPath => _htmlPath;

    internal void TryOpenHtml(TextWriter error)
    {
        if (_renderHtml)
            new LossGraph(_htmlPath, _totalEpochs).TryOpen(error);
    }

    internal static string GetSidecarPath(string htmlPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(htmlPath);
        return Path.ChangeExtension(
            Path.GetFullPath(htmlPath),
            ".metrics.jsonl");
    }

    internal static TrainingMetricReporter Open(
        string htmlPath,
        int totalEpochs,
        bool resume,
        long checkpointGlobalStep,
        double checkpointEpoch,
        bool renderHtml)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(htmlPath);
        if (totalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalEpochs));
        if (checkpointGlobalStep < -1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(checkpointGlobalStep));
        }
        if (!double.IsFinite(checkpointEpoch)
            || checkpointEpoch < 0d
            || checkpointEpoch > totalEpochs)
        {
            throw new ArgumentOutOfRangeException(nameof(checkpointEpoch));
        }

        string fullHtmlPath = Path.GetFullPath(htmlPath);
        var repository = new MetricJournalJsonlRepository(
            GetSidecarPath(fullHtmlPath));
        MetricJournal journal;
        if (resume)
        {
            journal = LossGraphMetricAdapter.LoadSidecarOrImportLegacy(
                repository.Path,
                fullHtmlPath,
                totalEpochs,
                checkpointGlobalStep,
                (float)checkpointEpoch);
        }
        else
        {
            // A non-resume invocation is a new run even if artifacts from an
            // older run still exist beside the configuration file.
            repository.ReplaceAtomically([]);
            journal = new MetricJournal();
        }

        var reporter = new TrainingMetricReporter(
            repository,
            fullHtmlPath,
            totalEpochs,
            renderHtml,
            journal);
        reporter.RenderHtml();
        return reporter;
    }

    internal void AppendCommittedLoss(
        long globalStep,
        double epoch,
        string kind,
        double value,
        DateTimeOffset? timestamp = null)
    {
        var entry = new MetricJournalEntry(
            globalStep,
            epoch,
            Math.Clamp(epoch / _totalEpochs, 0d, 1d),
            kind,
            value,
            timestamp ?? DateTimeOffset.UtcNow);
        entry.Validate();

        MetricJournalEntry? existing = _journal.Entries.LastOrDefault(
            candidate => candidate.GlobalStep == globalStep
                && candidate.Epoch == epoch
                && string.Equals(
                    candidate.Kind,
                    kind,
                    StringComparison.Ordinal));
        if (existing is not null)
        {
            if (existing.Value != value)
            {
                throw new InvalidDataException(
                    $"Committed metric '{kind}' at global step " +
                    $"{globalStep} already has value {existing.Value}, " +
                    $"not {value}.");
            }
            return;
        }

        // Persist first. The HTML is only a projection and must never get
        // ahead of its authoritative sidecar.
        _repository.AppendAndFlush(entry);
        _journal.Append(entry);
        RenderHtml();
    }

    internal void AppendCommittedEpochLosses(
        long globalStep,
        double epoch,
        double trainingLoss,
        double evaluationLoss)
    {
        DateTimeOffset timestamp = DateTimeOffset.UtcNow;
        AppendCommittedLoss(
            globalStep,
            epoch,
            MetricKinds.TrainLoss,
            trainingLoss,
            timestamp);
        AppendCommittedLoss(
            globalStep,
            epoch,
            MetricKinds.EvaluationLoss,
            evaluationLoss,
            timestamp);
    }

    private void RenderHtml()
    {
        if (_renderHtml)
        {
            LossGraphMetricAdapter.RenderFromJournal(
                _journal,
                _htmlPath,
                _totalEpochs);
        }
    }
}
