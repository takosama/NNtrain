using NNtrain;
using Xunit;

public sealed class CanonicalExecutionEntryTests
{
    [Fact]
    public void ProgramPassesOneWikiSpecWithoutReReadingConfigurationPath()
    {
        using var directory = new TemporaryDirectory();
        string canonicalPath = Path.Combine(directory.Root, "canonical.json");
        var config = new WikiTrainingConfiguration
        {
            DataPath = Path.Combine(directory.Root, "missing-wiki"),
            TokenizerPath = Path.Combine(directory.Root, "tokenizer.json"),
            CheckpointPath = Path.Combine(directory.Root, "checkpoint.json"),
            Device = WikiTrainingConfiguration.CpuDevice,
        };
        var canonical = new CanonicalWikiTrainingSpec(
            canonicalPath,
            SourceSchemaVersion: 2,
            config);
        int loadCount = 0;
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = Program.Run(
            ["--config", "must-not-be-read.json"],
            output,
            error,
            openLossGraph: false,
            path =>
            {
                Assert.Equal("must-not-be-read.json", path);
                loadCount++;
                return canonical;
            });

        Assert.Equal(1, loadCount);
        Assert.Equal(2, exitCode);
        Assert.Contains(
            $"configuration = {canonicalPath}",
            output.ToString());
        Assert.Contains(
            "Wikipedia data directory was not found",
            error.ToString());
    }

    [Fact]
    public void ProgramPassesOneClassificationSpecWithoutReReadingPath()
    {
        using var directory = new TemporaryDirectory();
        string canonicalPath = Path.Combine(directory.Root, "canonical.json");
        var config = new TrainingConfiguration
        {
            TrainingData = new DatasetConfiguration
            {
                ImagePath = Path.Combine(directory.Root, "missing-images"),
                LabelPath = Path.Combine(directory.Root, "missing-labels"),
            },
            EvaluationData = new DatasetConfiguration
            {
                ImagePath = Path.Combine(directory.Root, "missing-eval-images"),
                LabelPath = Path.Combine(directory.Root, "missing-eval-labels"),
            },
            CheckpointPath = Path.Combine(directory.Root, "checkpoint.json"),
        };
        var canonical = new CanonicalClassificationTrainingSpec(
            canonicalPath,
            SourceSchemaVersion: null,
            config);
        int loadCount = 0;
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = Program.Run(
            ["--config", "must-not-be-read.json"],
            output,
            error,
            openLossGraph: false,
            _ =>
            {
                loadCount++;
                return canonical;
            });

        Assert.Equal(1, loadCount);
        Assert.Equal(2, exitCode);
        Assert.Contains(
            "Training image data file was not found",
            error.ToString());
    }

    [Fact]
    public void GenerationConfigurationBypassesTrainingConfigLoader()
    {
        int loadCount = 0;
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = Program.Run(
            ["--generate-config", "missing-generation.json"],
            output,
            error,
            openLossGraph: false,
            _ =>
            {
                loadCount++;
                throw new InvalidOperationException(
                    "Training config loader must not run.");
            });

        Assert.Equal(0, loadCount);
        Assert.Equal(2, exitCode);
        Assert.Contains("Could not find file", error.ToString());
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                "nntrain-canonical-entry-tests",
                Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
