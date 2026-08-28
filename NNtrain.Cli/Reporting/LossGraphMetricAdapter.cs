using NNtrain.Training.Metrics;

namespace NNtrain;

/// <summary>
/// Converts the durable metric journal to the existing LossGraph HTML format
/// and imports that format once when a legacy run has no sidecar.
/// </summary>
internal static class LossGraphMetricAdapter
{
    internal static void RenderFromJournal(
        MetricJournal journal,
        string htmlPath,
        int totalEpochs)
    {
        ArgumentNullException.ThrowIfNull(journal);
        var graph = new LossGraph(htmlPath, totalEpochs);
        var lastTrainingLoss = new Dictionary<double, float>();
        var pendingEvaluationLoss = new Dictionary<double, float>();

        foreach (MetricJournalEntry entry in journal.Entries)
        {
            if (entry.Epoch <= 0d)
                continue;
            bool isTraining = string.Equals(
                entry.Kind,
                MetricKinds.TrainLoss,
                StringComparison.OrdinalIgnoreCase);
            bool isEvaluation = string.Equals(
                entry.Kind,
                MetricKinds.EvaluationLoss,
                StringComparison.OrdinalIgnoreCase);
            if (!isTraining && !isEvaluation)
                continue;
            float epoch = ToGraphNumber(entry.Epoch, nameof(entry.Epoch));
            float value = ToGraphNumber(entry.Value, nameof(entry.Value));

            if (isTraining)
            {
                graph.AddPoint(epoch, value);
                lastTrainingLoss[entry.Epoch] = value;
                if (pendingEvaluationLoss.Remove(
                    entry.Epoch,
                    out float evaluation))
                {
                    graph.AddPoint(epoch, value, evaluation);
                }
            }
            else
            {
                if (lastTrainingLoss.TryGetValue(
                    entry.Epoch,
                    out float training))
                {
                    graph.AddPoint(epoch, training, value);
                }
                else
                {
                    pendingEvaluationLoss[entry.Epoch] = value;
                }
            }
        }

        graph.Write();
    }

    internal static MetricJournal LoadSidecarOrImportLegacy(
        string sidecarPath,
        string htmlPath,
        int totalEpochs,
        long checkpointGlobalStep,
        float checkpointEpoch,
        DateTimeOffset? importTimestamp = null)
    {
        if (checkpointGlobalStep < -1)
        {
            throw new ArgumentOutOfRangeException(
                nameof(checkpointGlobalStep));
        }
        var repository = new MetricJournalJsonlRepository(sidecarPath);
        if (repository.Exists)
            return repository.RecoverThrough(checkpointGlobalStep).Journal;

        var graph = new LossGraph(htmlPath, totalEpochs);
        IReadOnlyList<LossGraph.LossPoint> points =
            checkpointGlobalStep < 0
                ? []
                : graph.ImportExisting(checkpointEpoch);
        var journal = new MetricJournal();
        long firstStep = Math.Max(
            0L,
            checkpointGlobalStep - Math.Max(0, points.Count - 1));
        DateTimeOffset timestamp = importTimestamp ?? DateTimeOffset.UtcNow;

        for (int index = 0; index < points.Count; index++)
        {
            LossGraph.LossPoint point = points[index];
            long globalStep = Math.Min(
                checkpointGlobalStep,
                checked(firstStep + index));
            double progress = Math.Clamp(
                point.Epoch / totalEpochs,
                0d,
                1d);
            journal.Append(new MetricJournalEntry(
                globalStep,
                point.Epoch,
                progress,
                MetricKinds.TrainLoss,
                point.Training,
                timestamp));
            if (point.Evaluation.HasValue)
            {
                journal.Append(new MetricJournalEntry(
                    globalStep,
                    point.Epoch,
                    progress,
                    MetricKinds.EvaluationLoss,
                    point.Evaluation.Value,
                    timestamp));
            }
        }

        repository.ReplaceAtomically(journal.Entries);
        return journal;
    }

    private static float ToGraphNumber(double value, string name)
    {
        float converted = (float)value;
        if (!float.IsFinite(converted))
            throw new InvalidDataException(
                $"Metric {name} value {value} cannot be rendered by LossGraph.");
        return converted;
    }
}
