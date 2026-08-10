using NNtrain;
using Xunit;

public sealed class WikiTrainingConfigurationTests
{
    [Fact]
    public void LoadReadsWikiSettingsAndResolvesPaths()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "dataPath": "data/wiki",
              "textColumn": "body",
              "tokenizerPath": "artifacts/tokenizer.json",
              "checkpointPath": "artifacts/model.json",
              "vocabularySize": 300,
              "tokenizerTrainingDocuments": 3,
              "tokenizerTrainingBytes": 100,
              "maxTrainingDocuments": 4,
              "maxTrainingTokens": 20,
              "validationFraction": 0.1,
              "epochs": 2,
              "batchSize": 1,
              "contextLength": 4,
              "modelWidth": 8,
              "heads": 2,
              "hiddenSize": 16,
              "layers": 1,
              "dropout": 0.2,
              "initializationScale": 0.03,
              "optimizer": "nekomuon",
              "learningRate": 0.001,
              "auxiliaryLearningRate": 0.002,
              "weightDecay": 0.02,
              "seed": 9,
              "logEveryBatches": 2,
              "showLossGraph": true,
              "graphUpdateSteps": 100,
              "datasetSampleEverySteps": 1000,
              "datasetSamplePoolSize": 16,
              "maxNewTokens": 3,
              "temperature": 0.7,
              "topK": 5,
              "useSimd": false,
              "maxDegreeOfParallelism": 3
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.True(WikiTrainingConfiguration.IsWikiConfiguration(path));
        Assert.Equal(
            Path.Combine(directory.Root, "data", "wiki"),
            configuration.DataPath);
        Assert.Equal(
            Path.Combine(directory.Root, "artifacts", "tokenizer.json"),
            configuration.TokenizerPath);
        Assert.Equal(
            Path.Combine(directory.Root, "artifacts", "model.json"),
            configuration.CheckpointPath);
        Assert.Equal("body", configuration.TextColumn);
        Assert.Equal(300, configuration.VocabularySize);
        Assert.Equal(1, configuration.BatchSize);
        Assert.Equal(4, configuration.ContextLength);
        Assert.Equal(8, configuration.ModelWidth);
        Assert.Equal(2, configuration.Heads);
        Assert.Equal(16, configuration.HiddenSize);
        Assert.Equal(1, configuration.Layers);
        Assert.Equal("nekomuon", configuration.Optimizer);
        Assert.Equal(0.001f, configuration.LearningRate);
        Assert.Equal(0.002f, configuration.AuxiliaryLearningRate);
        Assert.True(configuration.ShowLossGraph);
        Assert.Equal(100, configuration.GraphUpdateSteps);
        Assert.Equal(1000, configuration.DatasetSampleEverySteps);
        Assert.Equal(16, configuration.DatasetSamplePoolSize);
        Assert.False(configuration.UseSimd);
        Assert.Equal(3, configuration.MaxDegreeOfParallelism);
    }

    [Fact]
    public void DetectsTaskPropertyCaseInsensitively()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            { "TASK": "GPT_RIN_WIKI_JP" }
            """);

        Assert.True(WikiTrainingConfiguration.IsWikiConfiguration(path));
    }

    [Fact]
    public void AllowsUnlimitedStreamingAllDataConfiguration()
    {
        var configuration = new WikiTrainingConfiguration
        {
            MaxTrainingDocuments = 0,
            MaxTrainingTokens = 0,
            MaxDocumentTokens = 4096,
            ValidationFraction = 0f,
        };

        configuration.Validate();
    }

    [Fact]
    public void StreamingAllDataRejectsHeldOutValidationFraction()
    {
        var configuration = new WikiTrainingConfiguration
        {
            MaxTrainingTokens = 0,
            ValidationFraction = 0.05f,
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("ValidationFraction", exception.ParamName);
    }

    [Fact]
    public void RejectsUnsupportedWikiOptimizer()
    {
        var configuration = new WikiTrainingConfiguration
        {
            Optimizer = "lion",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("Optimizer", exception.ParamName);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.WikiConfigTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        internal string Write(string json)
        {
            string path = Path.Combine(Root, "training.json");
            File.WriteAllText(path, json);
            return path;
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
