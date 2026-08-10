using System.Globalization;
using System.Text.RegularExpressions;
using NNtrain;
using Xunit;

public sealed class LossGraphTests
{
    [Fact]
    public void WritesConnectedTrainingAndEvaluationPointsForEveryEpoch()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Root, "loss.html");
        var graph = new LossGraph(path, totalEpochs: 3);
        graph.AddEpoch(1, trainingLoss: 4.5f, evaluationLoss: 4.6f);
        graph.AddEpoch(2, trainingLoss: 3.25f, evaluationLoss: 3.5f);

        graph.Write();

        string html = File.ReadAllText(path);
        Assert.Contains("http-equiv=\"refresh\" content=\"1\"", html);
        Assert.Contains("<polyline class=\"train\"", html);
        Assert.Contains("<polyline class=\"eval\"", html);
        Assert.Equal(2, Count(html, "class=\"train-point\""));
        Assert.Equal(2, Count(html, "class=\"eval-point\""));
        Assert.Contains("epoch 1: 4.500000", html);
        Assert.Contains("epoch 2: 3.250000", html);
        Assert.Contains("epoch 2: 3.500000", html);
    }

    [Fact]
    public void WritesFractionalEpochTrainingPointsAndEpochValidationPoints()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Root, "loss.html");
        var graph = new LossGraph(path, totalEpochs: 2);
        graph.AddPoint(0.5f, trainingLoss: 5f);
        graph.AddPoint(1f, trainingLoss: 4f, evaluationLoss: 4.5f);

        graph.Write();

        string html = File.ReadAllText(path);
        Assert.Equal(2, Count(html, "class=\"train-point\""));
        Assert.Equal(1, Count(html, "class=\"eval-point\""));
        Assert.Contains("epoch 0.5: 5.000000", html);
        Assert.Contains("epoch 1: 4.500000", html);
        Assert.Contains(">epoch</text>", html);
    }

    [Fact]
    public void ExpandsSubEpochProgressAcrossPlotWidthWithoutRoundingToZero()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Root, "loss.html");
        var graph = new LossGraph(path, totalEpochs: 1);
        graph.AddPoint(0.0001f, trainingLoss: 5f);
        graph.AddPoint(0.0002f, trainingLoss: 4f);

        graph.Write();

        string html = File.ReadAllText(path);
        Assert.Contains("epoch 0.0001: 5.000000", html);
        Assert.Contains("epoch 0.0002: 4.000000", html);
        Assert.Contains("train points 2", html);
        MatchCollection matches = Regex.Matches(
            html,
            "class=\"train-point\" cx=\"([^\"]+)\"");
        Assert.Equal(2, matches.Count);
        float firstX = float.Parse(
            matches[0].Groups[1].Value,
            CultureInfo.InvariantCulture);
        float lastX = float.Parse(
            matches[1].Groups[1].Value,
            CultureInfo.InvariantCulture);
        Assert.True(firstX > 400f);
        Assert.True(lastX > 900f);
    }

    [Fact]
    public void KeepsRepeatedProgressPointsAndOnlyReplacesEpochEndPoint()
    {
        using var directory = new TemporaryDirectory();
        string path = Path.Combine(directory.Root, "loss.html");
        var graph = new LossGraph(path, totalEpochs: 1);
        graph.AddPoint(0.1f, trainingLoss: 5f);
        graph.AddPoint(0.1f, trainingLoss: 4.5f);
        graph.AddPoint(0.1f, trainingLoss: 4f, evaluationLoss: 4.25f);

        graph.Write();

        string html = File.ReadAllText(path);
        Assert.Equal(2, Count(html, "class=\"train-point\""));
        Assert.Equal(1, Count(html, "class=\"eval-point\""));
        Assert.Contains("epoch 0.1: 5.000000", html);
        Assert.Contains("epoch 0.1: 4.000000", html);
        Assert.DoesNotContain("epoch 0.1: 4.500000", html);
    }

    private static int Count(string text, string value)
    {
        int count = 0;
        int start = 0;
        while ((start = text.IndexOf(
            value,
            start,
            StringComparison.Ordinal)) >= 0)
        {
            count++;
            start += value.Length;
        }

        return count;
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.LossGraphTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose()
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
