using NNtrain.Training.Metrics;

namespace NNtrain;

/// <summary>Projects Qwen's committed update loss onto the existing epoch loss graph.</summary>
internal sealed class QwenLoraMetricReporter : IDisposable
{
    private readonly TrainingMetricReporter reporter;
    private readonly int exampleCount;
    internal IReadOnlyList<string> ArchivedPaths { get; }
    internal string HtmlPath => reporter.HtmlPath;
    internal string SidecarPath => reporter.SidecarPath;

    internal static string ResolveHtmlPath(string configPath, QwenLoraTrainingConfiguration config)
    {
        string fullConfigPath = Path.GetFullPath(configPath);
        string path = config.LossGraphPath is null
            ? Path.ChangeExtension(fullConfigPath, ".html")
            : Path.GetFullPath(config.LossGraphPath, Path.GetDirectoryName(fullConfigPath)!);
        if (!string.Equals(Path.GetExtension(path), ".html", StringComparison.OrdinalIgnoreCase)
            && !string.Equals(Path.GetExtension(path), ".htm", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("lossGraphPath must have an .html or .htm extension.");
        return path;
    }

    internal QwenLoraMetricReporter(QwenLoraTrainingConfiguration config, string htmlPath,
        int exampleCount, int checkpointStep, bool resume, TextWriter output, TextWriter error,
        Action<string>? openHtml = null)
    {
        if (exampleCount <= 0 || checkpointStep < 0)
            throw new ArgumentOutOfRangeException(nameof(exampleCount));
        this.exampleCount = exampleCount;
        string sidecar = TrainingMetricReporter.GetSidecarPath(htmlPath);
        // New training must not silently discard an older graph. Resume uses
        // the sidecar's exact steps; HTML alone cannot establish those steps.
        ArchivedPaths = !resume ? ArchiveExistingHistory(htmlPath, sidecar) : [];
        if (resume && !File.Exists(sidecar) && File.Exists(htmlPath))
            throw new IOException("Qwen LoRA metrics sidecar is missing. Restore it or choose a new lossGraphPath; existing HTML was not overwritten.");
        int totalEpochs = checked((int)((Math.Max((long)config.MaxSteps, checkpointStep) + exampleCount - 1) / exampleCount));
        reporter = TrainingMetricReporter.Open(htmlPath, totalEpochs, resume,
            checkpointStep, (double)checkpointStep / exampleCount, config.ShowLossGraph,
            config.LossGraphEverySteps);
        foreach (string archive in ArchivedPaths) output.WriteLine($"Previous loss history = {archive}");
        output.WriteLine($"metrics = {reporter.SidecarPath}");
        if (config.ShowLossGraph)
        {
            output.WriteLine($"loss graph = {reporter.HtmlPath}, every {config.LossGraphEverySteps} step(s)");
            // The initial (possibly resumed) graph exists before opening it,
            // and this runs before the first optimizer update.
            if (config.OpenLossGraph)
            {
                if (openHtml is null) reporter.TryOpenHtml(error);
                else openHtml(reporter.HtmlPath);
            }
        }
        output.Flush();
    }

    private static IReadOnlyList<string> ArchiveExistingHistory(string htmlPath, string sidecar)
    {
        if (!File.Exists(htmlPath) && !File.Exists(sidecar)) return [];
        string archiveHtml = Path.Combine(Path.GetDirectoryName(htmlPath)!,
            Path.GetFileNameWithoutExtension(htmlPath) + ".previous-" +
            DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff", System.Globalization.CultureInfo.InvariantCulture) +
            "-" + Guid.NewGuid().ToString("N") + ".html");
        var paths = new List<string>();
        if (File.Exists(htmlPath))
        {
            File.Copy(htmlPath, archiveHtml, overwrite: false);
            paths.Add(archiveHtml);
        }
        if (File.Exists(sidecar))
        {
            string archiveSidecar = TrainingMetricReporter.GetSidecarPath(archiveHtml);
            File.Copy(sidecar, archiveSidecar, overwrite: false);
            paths.Add(archiveSidecar);
        }
        return paths;
    }

    internal void Append(int step, double loss)
        => reporter.AppendCommittedLoss(step, (double)step / exampleCount, MetricKinds.TrainLoss, loss);
    internal void Flush() => reporter.Flush();
    public void Dispose() => reporter.Dispose();
}
