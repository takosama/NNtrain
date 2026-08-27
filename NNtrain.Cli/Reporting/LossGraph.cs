using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.RegularExpressions;

namespace NNtrain;

internal sealed class LossGraph
{
    private const int Width = 1000;
    private const int Height = 600;
    private const int Left = 86;
    private const int Right = 32;
    private const int Top = 72;
    private const int Bottom = 72;

    private readonly List<LossPoint> _losses = [];
    private readonly int _totalEpochs;

    private static readonly Regex PersistedPointPattern = new(
        "<circle class=\"(?<series>train|eval)-point\"[^>]*>\\s*" +
        "<title>epoch (?<epoch>[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?): " +
        "(?<loss>[+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][+-]?\\d+)?)" +
        "</title>\\s*</circle>",
        RegexOptions.CultureInvariant);

    internal LossGraph(string path, int totalEpochs)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (totalEpochs <= 0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(totalEpochs),
                totalEpochs,
                "Total epochs must be positive.");
        }

        Path = System.IO.Path.GetFullPath(path);
        _totalEpochs = totalEpochs;
    }

    internal string Path { get; }

    internal int TotalEpochs => _totalEpochs;

    internal void RestoreExisting(float resumeEpoch)
    {
        IReadOnlyList<LossPoint> restored = ImportExisting(resumeEpoch);
        _losses.Clear();
        _losses.AddRange(restored);
    }

    internal IReadOnlyList<LossPoint> ImportExisting(float resumeEpoch)
    {
        ValidateResumeEpoch(resumeEpoch);
        var losses = new List<LossPoint>();
        if (!File.Exists(Path))
            return losses;

        string html = File.ReadAllText(Path);
        MatchCollection matches = PersistedPointPattern.Matches(html);
        foreach (Match match in matches)
        {
            if (!string.Equals(
                match.Groups["series"].Value,
                "train",
                StringComparison.Ordinal))
            {
                continue;
            }

            if (!TryReadPersistedPoint(match, out float epoch, out float loss)
                || epoch <= 0f
                || epoch > resumeEpoch
                || epoch > _totalEpochs
                || (losses.Count > 0 && epoch < losses[^1].Epoch))
            {
                continue;
            }

            losses.Add(new LossPoint(epoch, loss, null));
        }

        foreach (Match match in matches)
        {
            if (!string.Equals(
                match.Groups["series"].Value,
                "eval",
                StringComparison.Ordinal)
                || !TryReadPersistedPoint(
                    match,
                    out float epoch,
                    out float loss)
                || epoch > resumeEpoch)
            {
                continue;
            }

            for (int index = losses.Count - 1; index >= 0; index--)
            {
                if (losses[index].Epoch != epoch)
                    continue;

                losses[index] = losses[index] with { Evaluation = loss };
                break;
            }
        }

        return losses;
    }

    internal void AddEpoch(int epoch, float trainingLoss, float evaluationLoss)
        => AddPoint(epoch, trainingLoss, evaluationLoss);

    internal void AddPoint(
        float epoch,
        float trainingLoss,
        float? evaluationLoss = null)
    {
        if (!float.IsFinite(epoch) || epoch <= 0f || epoch > _totalEpochs)
        {
            throw new ArgumentOutOfRangeException(
                nameof(epoch),
                epoch,
                $"Epoch must be greater than 0 and at most {_totalEpochs}.");
        }

        if (_losses.Count > 0 && epoch < _losses[^1].Epoch)
        {
            throw new ArgumentException(
                "Loss graph epochs must be added in increasing order.",
                nameof(epoch));
        }

        ValidateLoss(trainingLoss, nameof(trainingLoss));
        if (evaluationLoss.HasValue)
            ValidateLoss(evaluationLoss.Value, nameof(evaluationLoss));
        var point = new LossPoint(epoch, trainingLoss, evaluationLoss);
        if (_losses.Count > 0
            && epoch == _losses[^1].Epoch
            && evaluationLoss.HasValue
            && !_losses[^1].Evaluation.HasValue)
        {
            _losses[^1] = point;
        }
        else
        {
            _losses.Add(point);
        }
    }

    internal void Write()
    {
        string? directory = System.IO.Path.GetDirectoryName(Path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);

        File.WriteAllText(Path, BuildHtml(), new UTF8Encoding(false));
    }

    internal void TryOpen(TextWriter error)
    {
        ArgumentNullException.ThrowIfNull(error);
        try
        {
            Process.Start(new ProcessStartInfo
            {
                FileName = Path,
                UseShellExecute = true,
            });
        }
        catch (Exception exception) when (
            exception is Win32Exception
            or InvalidOperationException
            or PlatformNotSupportedException)
        {
            error.WriteLine(
                $"Warning: loss graph could not be opened automatically: " +
                exception.Message);
        }
    }

    private string BuildHtml()
    {
        float currentEpoch = _losses.Count == 0 ? 0f : _losses[^1].Epoch;
        int activeEpoch = currentEpoch == 0f
            ? 0
            : Math.Min(_totalEpochs, (int)MathF.Ceiling(currentEpoch));
        float activeEpochProgress = activeEpoch == 0
            ? 0f
            : Math.Clamp(currentEpoch - activeEpoch + 1f, 0f, 1f);
        float xMaximum = currentEpoch > 0f ? currentEpoch : 1f;
        GetYRange(out float yMinimum, out float yMaximum);
        int plotWidth = Width - Left - Right;
        int plotHeight = Height - Top - Bottom;

        var html = new StringBuilder(16_384);
        html.AppendLine("<!doctype html>");
        html.AppendLine("<html lang=\"ja\">");
        html.AppendLine("<head>");
        html.AppendLine("<meta charset=\"utf-8\">");
        html.AppendLine("<meta http-equiv=\"refresh\" content=\"1\">");
        html.AppendLine("<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
        html.AppendLine("<title>NNtrain loss</title>");
        html.AppendLine("<style>");
        html.AppendLine("body{margin:0;background:#111827;color:#e5e7eb;font-family:Segoe UI,sans-serif;display:grid;place-items:center;min-height:100vh}");
        html.AppendLine("main{width:min(1100px,96vw)}");
        html.AppendLine("h1{font-size:22px;margin:0 0 8px}p{color:#9ca3af;margin:0 0 14px}svg{width:100%;height:auto;background:#0b1220;border:1px solid #263244;border-radius:12px}text{fill:#9ca3af;font-size:13px}.grid{stroke:#263244;stroke-width:1}.axis{stroke:#64748b;stroke-width:1.5}.train{stroke:#38bdf8;fill:none;stroke-width:3}.eval{stroke:#fb7185;fill:none;stroke-width:3}.train-point{fill:#38bdf8}.eval-point{fill:#fb7185}.legend{font-size:14px;fill:#e5e7eb}</style>");
        html.AppendLine("</head><body><main>");
        html.AppendLine("<h1>Loss by epoch</h1>");
        html.Append("<p>epoch ")
            .Append(activeEpoch)
            .Append(" / ")
            .Append(_totalEpochs)
            .Append(" · progress ")
            .Append(activeEpochProgress.ToString("0.0%", CultureInfo.InvariantCulture))
            .Append(" · train points ")
            .Append(_losses.Count)
            .Append(" · eval points ")
            .Append(_losses.Count(loss => loss.Evaluation.HasValue))
            .AppendLine(" · 1秒ごとに自動更新</p>");
        html.Append("<svg viewBox=\"0 0 ")
            .Append(Width)
            .Append(' ')
            .Append(Height)
            .AppendLine("\" role=\"img\" aria-label=\"training and evaluation loss graph\">");

        AppendGrid(html, yMinimum, yMaximum, plotWidth, plotHeight);
        AppendXAxisLabels(html, xMaximum, plotWidth, plotHeight);
        html.Append("<line class=\"axis\" x1=\"").Append(Left)
            .Append("\" y1=\"").Append(Top + plotHeight)
            .Append("\" x2=\"").Append(Left + plotWidth)
            .Append("\" y2=\"").Append(Top + plotHeight)
            .AppendLine("\"/>");
        html.Append("<line class=\"axis\" x1=\"").Append(Left)
            .Append("\" y1=\"").Append(Top)
            .Append("\" x2=\"").Append(Left)
            .Append("\" y2=\"").Append(Top + plotHeight)
            .AppendLine("\"/>");

        AppendSeries(
            html,
            "train",
            "train-point",
            loss => loss.Training,
            xMaximum,
            yMinimum,
            yMaximum,
            plotWidth,
            plotHeight);
        AppendSeries(
            html,
            "eval",
            "eval-point",
            loss => loss.Evaluation,
            xMaximum,
            yMinimum,
            yMaximum,
            plotWidth,
            plotHeight);

        bool hasEvaluation = _losses.Any(loss => loss.Evaluation.HasValue);
        if (hasEvaluation)
        {
            html.AppendLine("<line x1=\"720\" y1=\"34\" x2=\"750\" y2=\"34\" class=\"train\"/><text x=\"760\" y=\"39\" class=\"legend\">train loss</text>");
            html.AppendLine("<line x1=\"850\" y1=\"34\" x2=\"880\" y2=\"34\" class=\"eval\"/><text x=\"890\" y=\"39\" class=\"legend\">eval loss</text>");
        }
        else
        {
            html.AppendLine("<line x1=\"785\" y1=\"34\" x2=\"815\" y2=\"34\" class=\"train\"/><text x=\"825\" y=\"39\" class=\"legend\">train loss</text>");
        }
        html.Append("<text x=\"").Append(Left + plotWidth / 2)
            .Append("\" y=\"").Append(Height - 20)
            .AppendLine("\" text-anchor=\"middle\">epoch</text>");
        html.Append("<text x=\"22\" y=\"").Append(Top + plotHeight / 2)
            .AppendLine("\" text-anchor=\"middle\" transform=\"rotate(-90 22 300)\">loss</text>");
        html.AppendLine("</svg></main></body></html>");
        return html.ToString();
    }

    private void GetYRange(out float minimum, out float maximum)
    {
        if (_losses.Count == 0)
        {
            minimum = 0f;
            maximum = 1f;
            return;
        }

        IEnumerable<float> values = _losses.Select(loss => loss.Training)
            .Concat(
                _losses
                    .Where(loss => loss.Evaluation.HasValue)
                    .Select(loss => loss.Evaluation!.Value));
        float dataMinimum = values.Min();
        float dataMaximum = values.Max();
        minimum = MathF.Min(0f, dataMinimum);
        maximum = MathF.Max(1e-6f, dataMaximum);
        float range = maximum - minimum;
        maximum += MathF.Max(0.05f, range * 0.1f);
        if (minimum < 0f)
            minimum -= MathF.Max(0.05f, range * 0.1f);
    }

    private static void AppendGrid(
        StringBuilder html,
        float yMinimum,
        float yMaximum,
        int plotWidth,
        int plotHeight)
    {
        const int intervals = 5;
        for (int tick = 0; tick <= intervals; tick++)
        {
            float fraction = (float)tick / intervals;
            float y = Top + plotHeight * fraction;
            float value = yMaximum - (yMaximum - yMinimum) * fraction;
            html.Append("<line class=\"grid\" x1=\"").Append(Left)
                .Append("\" y1=\"").Append(Number(y))
                .Append("\" x2=\"").Append(Left + plotWidth)
                .Append("\" y2=\"").Append(Number(y))
                .AppendLine("\"/>");
            html.Append("<text x=\"").Append(Left - 12)
                .Append("\" y=\"").Append(Number(y + 4))
                .Append("\" text-anchor=\"end\">")
                .Append(value.ToString("0.000", CultureInfo.InvariantCulture))
                .AppendLine("</text>");
        }
    }

    private static void AppendXAxisLabels(
        StringBuilder html,
        float xMaximum,
        int plotWidth,
        int plotHeight)
    {
        const int tickCount = 5;
        var epochs = new SortedSet<float>();
        for (int tick = 0; tick <= tickCount; tick++)
        {
            float epoch = xMaximum * tick / tickCount;
            epochs.Add(epoch);
        }

        foreach (float epoch in epochs)
        {
            float x = XCoordinate(epoch, xMaximum, plotWidth);
            html.Append("<text x=\"").Append(Number(x))
                .Append("\" y=\"").Append(Top + plotHeight + 26)
                .Append("\" text-anchor=\"middle\">")
                .Append(EpochNumber(epoch))
                .AppendLine("</text>");
        }
    }

    private void AppendSeries(
        StringBuilder html,
        string lineClass,
        string pointClass,
        Func<LossPoint, float?> selector,
        float xMaximum,
        float yMinimum,
        float yMaximum,
        int plotWidth,
        int plotHeight)
    {
        LossPoint[] points = _losses
            .Where(loss => selector(loss).HasValue)
            .ToArray();
        if (points.Length == 0)
            return;

        html.Append("<polyline class=\"").Append(lineClass)
            .Append("\" points=\"");
        foreach (LossPoint loss in points)
        {
            float x = XCoordinate(loss.Epoch, xMaximum, plotWidth);
            float y = YCoordinate(
                selector(loss)!.Value,
                yMinimum,
                yMaximum,
                plotHeight);
            html.Append(Number(x)).Append(',').Append(Number(y)).Append(' ');
        }
        html.AppendLine("\"/>");

        foreach (LossPoint loss in points)
        {
            float value = selector(loss)!.Value;
            float x = XCoordinate(loss.Epoch, xMaximum, plotWidth);
            float y = YCoordinate(value, yMinimum, yMaximum, plotHeight);
            html.Append("<circle class=\"").Append(pointClass)
                .Append("\" cx=\"").Append(Number(x))
                .Append("\" cy=\"").Append(Number(y))
                .AppendLine("\" r=\"5\">");
            html.Append("<title>epoch ").Append(EpochNumber(loss.Epoch))
                .Append(": ")
                .Append(value.ToString("0.000000", CultureInfo.InvariantCulture))
                .AppendLine("</title></circle>");
        }
    }

    private static float XCoordinate(
        float epoch,
        float xMaximum,
        int plotWidth)
        => xMaximum == 0f
            ? Left
            : Left + epoch / xMaximum * plotWidth;

    private static float YCoordinate(
        float value,
        float minimum,
        float maximum,
        int plotHeight)
        => Top + (maximum - value) / (maximum - minimum) * plotHeight;

    private static string Number(float value)
        => value.ToString("0.###", CultureInfo.InvariantCulture);

    private static string EpochNumber(float value)
    {
        float magnitude = MathF.Abs(value);
        string format = magnitude switch
        {
            >= 1f => "0.###",
            >= 0.01f => "0.####",
            >= 0.001f => "0.#####",
            _ => "0.######",
        };
        return value.ToString(format, CultureInfo.InvariantCulture);
    }

    private static void ValidateLoss(float loss, string parameterName)
    {
        if (!float.IsFinite(loss))
        {
            throw new ArgumentOutOfRangeException(
                parameterName,
                loss,
                "Loss graph values must be finite.");
        }
    }

    private void ValidateResumeEpoch(float resumeEpoch)
    {
        if (!float.IsFinite(resumeEpoch)
            || resumeEpoch < 0f
            || resumeEpoch > _totalEpochs)
        {
            throw new ArgumentOutOfRangeException(
                nameof(resumeEpoch),
                resumeEpoch,
                $"Resume epoch must be from 0 through {_totalEpochs}.");
        }
    }

    private static bool TryReadPersistedPoint(
        Match match,
        out float epoch,
        out float loss)
    {
        epoch = 0f;
        loss = 0f;
        return float.TryParse(
            match.Groups["epoch"].Value,
            NumberStyles.Float,
            CultureInfo.InvariantCulture,
            out epoch)
            && float.IsFinite(epoch)
            && float.TryParse(
                match.Groups["loss"].Value,
                NumberStyles.Float,
                CultureInfo.InvariantCulture,
                out loss)
            && float.IsFinite(loss);
    }

    internal readonly record struct LossPoint(
        float Epoch,
        float Training,
        float? Evaluation);
}
