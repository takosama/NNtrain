using System.ComponentModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace NNtrain;

internal sealed class LossGraph
{
    private const int Width = 1000;
    private const int Height = 600;
    private const int Left = 86;
    private const int Right = 32;
    private const int Top = 72;
    private const int Bottom = 72;

    private readonly List<EpochLoss> _losses = [];
    private readonly int _totalEpochs;

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

    internal void AddEpoch(int epoch, float trainingLoss, float evaluationLoss)
    {
        if (epoch <= 0 || epoch > _totalEpochs)
        {
            throw new ArgumentOutOfRangeException(
                nameof(epoch),
                epoch,
                $"Epoch must be between 1 and {_totalEpochs}.");
        }

        if (_losses.Count > 0 && epoch <= _losses[^1].Epoch)
        {
            throw new ArgumentException(
                "Loss graph epochs must be added in increasing order.",
                nameof(epoch));
        }

        ValidateLoss(trainingLoss, nameof(trainingLoss));
        ValidateLoss(evaluationLoss, nameof(evaluationLoss));
        _losses.Add(new EpochLoss(epoch, trainingLoss, evaluationLoss));
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
        int currentEpoch = _losses.Count == 0 ? 0 : _losses[^1].Epoch;
        int xMaximum = Math.Max(1, currentEpoch);
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
            .Append(currentEpoch)
            .Append(" / ")
            .Append(_totalEpochs)
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

        html.AppendLine("<line x1=\"720\" y1=\"34\" x2=\"750\" y2=\"34\" class=\"train\"/><text x=\"760\" y=\"39\" class=\"legend\">train loss</text>");
        html.AppendLine("<line x1=\"850\" y1=\"34\" x2=\"880\" y2=\"34\" class=\"eval\"/><text x=\"890\" y=\"39\" class=\"legend\">eval loss</text>");
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

        float dataMinimum = _losses.Min(loss =>
            MathF.Min(loss.Training, loss.Evaluation));
        float dataMaximum = _losses.Max(loss =>
            MathF.Max(loss.Training, loss.Evaluation));
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
        int xMaximum,
        int plotWidth,
        int plotHeight)
    {
        int tickCount = Math.Min(5, xMaximum);
        var epochs = new SortedSet<int>();
        for (int tick = 0; tick <= tickCount; tick++)
        {
            int epoch = 1 + (int)Math.Round(
                (xMaximum - 1) * (double)tick / Math.Max(1, tickCount));
            epochs.Add(epoch);
        }

        foreach (int epoch in epochs)
        {
            float x = XCoordinate(epoch, xMaximum, plotWidth);
            html.Append("<text x=\"").Append(Number(x))
                .Append("\" y=\"").Append(Top + plotHeight + 26)
                .Append("\" text-anchor=\"middle\">")
                .Append(epoch)
                .AppendLine("</text>");
        }
    }

    private void AppendSeries(
        StringBuilder html,
        string lineClass,
        string pointClass,
        Func<EpochLoss, float> selector,
        int xMaximum,
        float yMinimum,
        float yMaximum,
        int plotWidth,
        int plotHeight)
    {
        if (_losses.Count == 0)
            return;

        html.Append("<polyline class=\"").Append(lineClass)
            .Append("\" points=\"");
        foreach (EpochLoss loss in _losses)
        {
            float x = XCoordinate(loss.Epoch, xMaximum, plotWidth);
            float y = YCoordinate(
                selector(loss),
                yMinimum,
                yMaximum,
                plotHeight);
            html.Append(Number(x)).Append(',').Append(Number(y)).Append(' ');
        }
        html.AppendLine("\"/>");

        foreach (EpochLoss loss in _losses)
        {
            float value = selector(loss);
            float x = XCoordinate(loss.Epoch, xMaximum, plotWidth);
            float y = YCoordinate(value, yMinimum, yMaximum, plotHeight);
            html.Append("<circle class=\"").Append(pointClass)
                .Append("\" cx=\"").Append(Number(x))
                .Append("\" cy=\"").Append(Number(y))
                .AppendLine("\" r=\"5\">");
            html.Append("<title>epoch ").Append(loss.Epoch)
                .Append(": ")
                .Append(value.ToString("0.000000", CultureInfo.InvariantCulture))
                .AppendLine("</title></circle>");
        }
    }

    private static float XCoordinate(int epoch, int xMaximum, int plotWidth)
        => xMaximum == 1
            ? Left
            : Left + (epoch - 1f) / (xMaximum - 1f) * plotWidth;

    private static float YCoordinate(
        float value,
        float minimum,
        float maximum,
        int plotHeight)
        => Top + (maximum - value) / (maximum - minimum) * plotHeight;

    private static string Number(float value)
        => value.ToString("0.###", CultureInfo.InvariantCulture);

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

    private readonly record struct EpochLoss(
        int Epoch,
        float Training,
        float Evaluation);
}
