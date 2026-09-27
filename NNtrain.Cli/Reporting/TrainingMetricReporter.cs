using NNtrain.Training.Metrics;

namespace NNtrain;

/// <summary>
/// Owns the durable loss journal for one training run. The JSONL sidecar is
/// authoritative; the existing HTML graph is a projection that can always be
/// rebuilt from it.
/// </summary>
internal sealed class TrainingMetricReporter : IDisposable
{
    private readonly MetricJournalJsonlRepository _repository;
    private readonly string _htmlPath;
    private readonly int _totalEpochs;
    private readonly bool _renderHtml;
    private readonly MetricJournal _journal;
    private readonly int _renderEverySteps;
    private readonly TextWriter _renderWarning;
    private bool _dirty;
    private bool _renderFailed;

    private TrainingMetricReporter(
        MetricJournalJsonlRepository repository,
        string htmlPath,
        int totalEpochs,
        bool renderHtml,
        MetricJournal journal,
        int renderEverySteps,
        TextWriter renderWarning)
    {
        _repository = repository;
        _htmlPath = htmlPath;
        _totalEpochs = totalEpochs;
        _renderHtml = renderHtml;
        _journal = journal;
        _renderEverySteps = renderEverySteps;
        _renderWarning = renderWarning;
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
        bool renderHtml,
        int renderEverySteps = 1,
        TextWriter? renderWarning = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(htmlPath);
        if (totalEpochs <= 0)
            throw new ArgumentOutOfRangeException(nameof(totalEpochs));
        if (renderEverySteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(renderEverySteps));
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
            journal,
            renderEverySteps,
            renderWarning ?? Console.Error);
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
        _dirty = true;
        if (_renderFailed || _journal.Count == 1 || globalStep % _renderEverySteps == 0)
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
            try
            {
                LossGraphMetricAdapter.RenderFromJournal(
                    _journal,
                    _htmlPath,
                    _totalEpochs);
            }
            catch (Exception exception) when (exception is UnauthorizedAccessException
                || exception is IOException)
            {
                // Browsers and file scanners may temporarily deny replacement on
                // Windows. The durable journal is already committed; a derived
                // HTML failure must not discard the in-memory optimizer update.
                _dirty = true;
                if (!_renderFailed)
                    _renderWarning.WriteLine($"Warning: loss graph update deferred for '{_htmlPath}': "
                        + exception.Message + " Loss history is saved; retrying on the next update or flush.");
                _renderFailed = true;
                return;
            }
        }
        _renderFailed = false;
        _dirty = false;
    }

    internal void Flush()
    {
        if (_dirty) RenderHtml();
    }

    public void Dispose() => Flush();
}
