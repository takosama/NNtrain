using NNtrain.Training.Metrics;

namespace NNtrain;

internal sealed class DpoMetricReporter : IDisposable
{
    private const int MaximumGraphPoints = 2000;
    private readonly MetricJournalJsonlRepository repository;
    private readonly List<MetricJournalEntry> losses;
    private readonly int maxSteps, interval;
    private bool dirty;
    internal string HtmlPath { get; }
    internal string SidecarPath => repository.Path;
    internal IReadOnlyList<string> ArchivedPaths { get; }
    internal DpoMetricReporter(string htmlPath, int maxSteps, int interval, bool resume, int checkpointStep)
    {
        if (maxSteps <= 0 || interval <= 0 || checkpointStep < 0) throw new ArgumentOutOfRangeException(nameof(maxSteps));
        HtmlPath = Path.GetFullPath(htmlPath);
        this.maxSteps = maxSteps; this.interval = interval;
        repository = new(TrainingMetricReporter.GetSidecarPath(HtmlPath));
        // A missing adapter means a new run, not a continuation of its old
        // metrics. Copy both old artifacts before replacing either original.
        // If a backup fails, the old history is left untouched.
        ArchivedPaths = !resume ? ArchiveExistingHistory(HtmlPath, repository.Path) : [];
        if (resume && !repository.Exists && File.Exists(HtmlPath))
            throw new IOException("DPO metrics sidecar is missing. Restore it or choose a new lossGraphPath; existing HTML was not overwritten.");
        var journal = resume ? repository.RecoverThrough(checkpointStep).Journal : new MetricJournal();
        if (!resume || !repository.Exists) repository.ReplaceAtomically([]);
        losses = journal.Entries.Where(e => e.Kind == MetricKinds.TrainLoss).ToList();
        Render();
    }
    private static IReadOnlyList<string> ArchiveExistingHistory(string htmlPath, string sidecarPath)
    {
        bool hasHtml = File.Exists(htmlPath), hasSidecar = File.Exists(sidecarPath);
        if (!hasHtml && !hasSidecar) return [];
        string archiveHtml = Path.Combine(Path.GetDirectoryName(htmlPath)!,
            Path.GetFileNameWithoutExtension(htmlPath) + ".previous-" +
            DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff", System.Globalization.CultureInfo.InvariantCulture) +
            "-" + Guid.NewGuid().ToString("N") + ".html");
        var archived = new List<string>();
        if (hasHtml)
        {
            File.Copy(htmlPath, archiveHtml, overwrite: false);
            archived.Add(archiveHtml);
        }
        if (hasSidecar)
        {
            string archiveSidecar = TrainingMetricReporter.GetSidecarPath(archiveHtml);
            File.Copy(sidecarPath, archiveSidecar, overwrite: false);
            archived.Add(archiveSidecar);
        }
        return archived;
    }
    internal void Append(int step, int epoch, double loss)
    {
        if (losses.Count > 0 && losses[^1].GlobalStep >= step)
            throw new InvalidOperationException("DPO metrics must advance the committed global step.");
        var entry = new MetricJournalEntry(step, epoch, Math.Clamp((double)step / maxSteps, 0, 1),
            MetricKinds.TrainLoss, loss, DateTimeOffset.UtcNow).Validate();
        repository.AppendAndFlush(entry);
        losses.Add(entry);
        dirty = true;
        if (step == 1 || step % interval == 0) Render();
    }
    internal void Flush() { if (dirty) Render(); }
    private void Render()
    {
        int upper = Math.Max(maxSteps, losses.Count == 0 ? 1 : checked((int)losses[^1].GlobalStep));
        var graph = new LossGraph(HtmlPath, upper, steps: true);
        int count = Math.Min(MaximumGraphPoints, losses.Count);
        for (int i = 0; i < count; i++)
        {
            int index = count == 1 ? 0 : (int)((long)i * (losses.Count - 1) / (count - 1));
            var point = losses[index];
            graph.AddPoint(point.GlobalStep, (float)point.Value);
        }
        graph.Write(atomically: true);
        dirty = false;
    }
    public void Dispose() => Flush();
}
