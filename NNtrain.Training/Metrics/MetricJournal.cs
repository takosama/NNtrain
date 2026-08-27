namespace NNtrain.Training.Metrics;

/// <summary>Canonical metric names understood by training reporters.</summary>
public static class MetricKinds
{
    public const string TrainLoss = "train_loss";
    public const string EvaluationLoss = "eval_loss";
    public const string LearningRate = "learning_rate";
    public const string GradientNorm = "gradient_norm";
    public const string StepMilliseconds = "step_milliseconds";
}

/// <summary>A single append-only metric observation.</summary>
public sealed record MetricJournalEntry(
    long GlobalStep,
    double Epoch,
    double Progress,
    string Kind,
    double Value,
    DateTimeOffset Timestamp)
{
    public MetricJournalEntry Validate()
    {
        if (GlobalStep < 0)
            throw new ArgumentOutOfRangeException(nameof(GlobalStep));
        if (!double.IsFinite(Epoch) || Epoch < 0d)
            throw new ArgumentOutOfRangeException(nameof(Epoch));
        if (!double.IsFinite(Progress) || Progress is < 0d or > 1d)
            throw new ArgumentOutOfRangeException(nameof(Progress));
        ArgumentException.ThrowIfNullOrWhiteSpace(Kind);
        if (Kind.IndexOfAny(['\r', '\n']) >= 0)
            throw new ArgumentException(
                "Metric kinds cannot contain line breaks.",
                nameof(Kind));
        if (!double.IsFinite(Value))
            throw new ArgumentOutOfRangeException(nameof(Value));
        if (Timestamp == default)
            throw new ArgumentOutOfRangeException(nameof(Timestamp));
        return this;
    }
}

/// <summary>
/// In-memory ordered metric journal. Multiple metric kinds may share a step,
/// epoch and progress position.
/// </summary>
public sealed class MetricJournal
{
    private readonly object _sync = new();
    private readonly List<MetricJournalEntry> _entries = [];

    public IReadOnlyList<MetricJournalEntry> Entries
    {
        get
        {
            lock (_sync)
                return _entries.ToArray();
        }
    }

    public int Count
    {
        get
        {
            lock (_sync)
                return _entries.Count;
        }
    }

    public void Append(MetricJournalEntry entry)
    {
        ArgumentNullException.ThrowIfNull(entry);
        entry.Validate();
        lock (_sync)
        {
            if (_entries.Count > 0
                && ComparePosition(entry, _entries[^1]) < 0)
            {
                throw new ArgumentException(
                    "Metric journal positions must be appended in nondecreasing order.",
                    nameof(entry));
            }
            _entries.Add(entry);
        }
    }

    /// <summary>
    /// Removes observations written after a committed checkpoint step.
    /// Returns the number of removed observations.
    /// </summary>
    public int TruncateAfter(long checkpointGlobalStep)
    {
        if (checkpointGlobalStep < -1)
            throw new ArgumentOutOfRangeException(nameof(checkpointGlobalStep));
        lock (_sync)
        {
            int firstRemoved = _entries.FindIndex(
                entry => entry.GlobalStep > checkpointGlobalStep);
            if (firstRemoved < 0)
                return 0;
            int removed = _entries.Count - firstRemoved;
            _entries.RemoveRange(firstRemoved, removed);
            return removed;
        }
    }

    internal static int ComparePosition(
        MetricJournalEntry left,
        MetricJournalEntry right)
    {
        int comparison = left.GlobalStep.CompareTo(right.GlobalStep);
        if (comparison != 0)
            return comparison;
        comparison = left.Epoch.CompareTo(right.Epoch);
        return comparison != 0
            ? comparison
            : left.Progress.CompareTo(right.Progress);
    }
}
