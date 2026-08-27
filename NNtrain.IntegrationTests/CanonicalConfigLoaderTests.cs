using NNtrain;
using Xunit;

public sealed class CanonicalConfigLoaderTests
{
    [Fact]
    public void VersionTwoClassificationMatchesExistingFacade()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "image-classification" },
              "data": {
                "trainingData": {
                  "imagePath": "data/train-images",
                  "labelPath": "data/train-labels"
                },
                "evaluationData": {
                  "imagePath": "data/eval-images",
                  "labelPath": "data/eval-labels"
                }
              },
              "training": {
                "epochs": 3,
                "batchSize": 8,
                "microBatchCount": 2
              },
              "runtime": { "seed": 17, "useSimd": false },
              "model": { "heads": 2, "hiddenSize": 16, "layers": 1 }
            }
            """);

        TrainingConfiguration existing = TrainingConfiguration.Load(path);
        CanonicalTrainingSpec loaded = ConfigLoader.Load(path);
        CanonicalClassificationTrainingSpec canonical =
            Assert.IsType<CanonicalClassificationTrainingSpec>(loaded);

        Assert.Equal(
            CanonicalTrainingTaskKind.ImageClassification,
            canonical.TaskKind);
        Assert.Equal(2, canonical.SourceSchemaVersion);
        Assert.False(canonical.UsesLegacySchema);
        Assert.Equal(Path.GetFullPath(path), canonical.ConfigurationPath);
        Assert.Equal(existing.Epochs, canonical.Configuration.Epochs);
        Assert.Equal(
            existing.EffectiveBatchSize,
            canonical.Configuration.EffectiveBatchSize);
        Assert.Equal(existing.Seed, canonical.Configuration.Seed);
        Assert.Equal(
            existing.TrainingData.ImagePath,
            canonical.Configuration.TrainingData.ImagePath);
        Assert.Equal(
            existing.EvaluationData.LabelPath,
            canonical.Configuration.EvaluationData.LabelPath);
    }

    [Fact]
    public void LegacyWikiMatchesExistingFacade()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "dataPath": "data/wiki",
              "tokenizerPath": "artifacts/tokenizer.json",
              "checkpointPath": "artifacts/model.json",
              "validationFraction": 0.0,
              "epochs": 2,
              "batchSize": 1,
              "contextLength": 4,
              "vocabularySize": 300,
              "modelWidth": 8,
              "heads": 2,
              "hiddenSize": 16,
              "layers": 1,
              "modelArchitecture": "transformer",
              "precisionMode": "float32"
            }
            """);

        WikiTrainingConfiguration existing =
            WikiTrainingConfiguration.Load(path);
        CanonicalTrainingSpec loaded = ConfigLoader.Load(path);
        CanonicalWikiTrainingSpec canonical =
            Assert.IsType<CanonicalWikiTrainingSpec>(loaded);

        Assert.Equal(
            CanonicalTrainingTaskKind.WikiLanguageModel,
            canonical.TaskKind);
        Assert.Null(canonical.SourceSchemaVersion);
        Assert.True(canonical.UsesLegacySchema);
        Assert.Equal(Path.GetFullPath(path), canonical.ConfigurationPath);
        Assert.Equal(existing.Epochs, canonical.Configuration.Epochs);
        Assert.Equal(existing.DataPath, canonical.Configuration.DataPath);
        Assert.Equal(
            existing.TokenizerPath,
            canonical.Configuration.TokenizerPath);
        Assert.Equal(
            existing.CheckpointPath,
            canonical.Configuration.CheckpointPath);
        Assert.Equal(
            existing.GetPrecisionMode(),
            canonical.Configuration.GetPrecisionMode());
    }

    [Fact]
    public void VersionTwoWikiIsDispatchedWithoutLegacyTaskProbe()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "wiki-language-model" },
              "data": { "dataPath": "data/wiki" },
              "training": {
                "epochs": 2,
                "batchSize": 1,
                "validationFraction": 0.0
              },
              "runtime": { "device": "cpu" },
              "model": {
                "vocabularySize": 300,
                "contextLength": 4,
                "modelWidth": 8,
                "heads": 2,
                "hiddenSize": 16,
                "layers": 1,
                "modelArchitecture": "transformer"
              }
            }
            """);

        CanonicalWikiTrainingSpec canonical =
            Assert.IsType<CanonicalWikiTrainingSpec>(ConfigLoader.Load(path));

        Assert.Equal(2, canonical.SourceSchemaVersion);
        Assert.Equal(
            Path.Combine(directory.Root, "data", "wiki"),
            canonical.Configuration.DataPath);
    }

    [Fact]
    public void CanonicalBoundaryPreservesConfigurationValidation()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "image-classification" },
              "runtime": { "seed": 17 },
              "training": { "seed": 18 }
            }
            """);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => ConfigLoader.Load(path));

        Assert.Contains("specified in more than one v2 section", exception.Message);
    }

    [Fact]
    public void CanonicalWikiAcceptsLegacyMix8Settings()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "validationFraction": 0.0,
              "epochs": 1,
              "batchSize": 1,
              "contextLength": 4,
              "vocabularySize": 300,
              "modelWidth": 8,
              "heads": 2,
              "hiddenSize": 16,
              "layers": 1,
              "modelArchitecture": "transformer",
              "precision": "mix8_32",
              "bfp8_block_size": 32
            }
            """);

        CanonicalWikiTrainingSpec canonical =
            Assert.IsType<CanonicalWikiTrainingSpec>(ConfigLoader.Load(path));
        Assert.Equal(TensorPrecisionMode.Mix8_32, canonical.PrecisionMode);
        Assert.Equal(32, canonical.Bfp8BlockSize);
    }

    [Fact]
    public void CanonicalVersionTwoModelsAcceptBfp8Settings()
    {
        using var directory = new TemporaryDirectory();
        string wikiPath = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "wiki-language-model" },
              "data": { "dataPath": "data/wiki" },
              "training": { "epochs": 1, "batchSize": 1, "validationFraction": 0.0 },
              "model": {
                "vocabularySize": 300,
                "contextLength": 4,
                "modelWidth": 8,
                "heads": 2,
                "hiddenSize": 16,
                "layers": 1,
                "modelArchitecture": "transformer",
                "precision": "bfp8",
                "bfp8_block_size": 64
              }
            }
            """);
        CanonicalWikiTrainingSpec wiki =
            Assert.IsType<CanonicalWikiTrainingSpec>(ConfigLoader.Load(wikiPath));
        Assert.Equal(TensorPrecisionMode.Bfp8, wiki.PrecisionMode);
        Assert.Equal(64, wiki.Bfp8BlockSize);

        string classificationPath = Path.Combine(
            directory.Root,
            "classification.json");
        File.WriteAllText(
            classificationPath,
            """
            {
              "schemaVersion": 2,
              "task": { "type": "image-classification" },
              "data": {
                "trainingData": { "imagePath": "data/train", "labelPath": "data/train-labels" },
                "evaluationData": { "imagePath": "data/eval", "labelPath": "data/eval-labels" }
              },
              "training": { "epochs": 1, "batchSize": 1 },
              "model": {
                "heads": 1,
                "hiddenSize": 8,
                "layers": 1,
                "precision": "mix8_32",
                "bfp8_block_size": 16
              }
            }
            """);
        CanonicalClassificationTrainingSpec classification =
            Assert.IsType<CanonicalClassificationTrainingSpec>(
                ConfigLoader.Load(classificationPath));
        Assert.Equal(
            TensorPrecisionMode.Mix8_32,
            classification.PrecisionMode);
        Assert.Equal(16, classification.Bfp8BlockSize);
    }

    [Fact]
    public void Bfp8BlockSizeDefaultsTo128AndRejectsNonPositiveValues()
    {
        var defaults = new CanonicalClassificationTrainingSpec(
            "unused.json",
            null,
            new TrainingConfiguration());
        Assert.Equal(128, defaults.Bfp8BlockSize);
        Assert.Equal(TensorPrecisionMode.Float32, defaults.PrecisionMode);

        var invalid = new ModelConfiguration
        {
            Precision = TensorPrecisionModeNames.Mix8_32,
            Bfp8BlockSize = 0,
        };
        Assert.Throws<ArgumentOutOfRangeException>(invalid.Validate);
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                "nntrain-canonical-config-tests",
                Guid.NewGuid().ToString("N"));
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
