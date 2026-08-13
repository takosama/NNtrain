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
              "resumeFromCheckpoint": true,
              "autoResume": true,
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
              "modelArchitecture": "hyena",
              "modelDType": "float32",
              "forgetMemoryKeyWidth": 6,
              "forgetMemoryValueWidth": 7,
              "forgetMemoryRetentionMinimum": 0.25,
              "forgetMemoryRetentionMaximum": 0.9,
              "hyenaFilterWidth": 12,
              "hyenaConvolutionAlgorithm": "fft",
              "dropout": 0.2,
              "initializationScale": 0.03,
              "optimizer": "nekomuon",
              "learningRate": 0.001,
              "auxiliaryLearningRate": 0.002,
              "nekoMuonNewtonSchulzInterval": 7,
              "warmupPercent": 20,
              "weightDecay": 0.02,
              "adamWUseBFloat16FirstMoment": true,
              "adamWUseBFloat16SecondMoment": true,
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
        Assert.True(configuration.ResumeFromCheckpoint);
        Assert.True(configuration.AutoResume);
        Assert.Equal("body", configuration.TextColumn);
        Assert.Equal(300, configuration.VocabularySize);
        Assert.Equal(1, configuration.BatchSize);
        Assert.Equal(4, configuration.ContextLength);
        Assert.Equal(8, configuration.ModelWidth);
        Assert.Equal(2, configuration.Heads);
        Assert.Equal(16, configuration.HiddenSize);
        Assert.Equal(1, configuration.Layers);
        Assert.Equal("hyena", configuration.ModelArchitecture);
        Assert.Equal("float32", configuration.ModelDType);
        Assert.Equal(TensorDType.Float32, configuration.GetModelDType());
        Assert.Equal(6, configuration.ForgetMemoryKeyWidth);
        Assert.Equal(7, configuration.ForgetMemoryValueWidth);
        Assert.Equal(0.25f, configuration.ForgetMemoryRetentionMinimum);
        Assert.Equal(0.9f, configuration.ForgetMemoryRetentionMaximum);
        Assert.Equal(12, configuration.HyenaFilterWidth);
        Assert.Equal("fft", configuration.HyenaConvolutionAlgorithm);
        Assert.Equal(
            HyenaConvolutionAlgorithm.Fft,
            configuration.GetHyenaConvolutionAlgorithm());
        Assert.Equal("nekomuon", configuration.Optimizer);
        Assert.Equal(0.001f, configuration.LearningRate);
        Assert.Equal(0.002f, configuration.AuxiliaryLearningRate);
        Assert.Equal(7, configuration.NekoMuonNewtonSchulzInterval);
        Assert.Equal(20f, configuration.WarmupPercent);
        Assert.True(configuration.AdamWUseBFloat16FirstMoment);
        Assert.True(configuration.AdamWUseBFloat16SecondMoment);
        Assert.True(configuration.ShowLossGraph);
        Assert.Equal(100, configuration.GraphUpdateSteps);
        Assert.Equal(1000, configuration.DatasetSampleEverySteps);
        Assert.Equal(16, configuration.DatasetSamplePoolSize);
        Assert.False(configuration.UseSimd);
        Assert.Equal(3, configuration.MaxDegreeOfParallelism);
    }

    [Fact]
    public void LoadReadsGroupedCheckpointOptimizerAndSchedulerSettings()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "checkpoint": {
                "directory": "artifacts/checkpoints",
                "fileName": "latest.json",
                "resume": true,
                "autoResume": true
              },
              "optimization": {
                "optimizer": {
                  "type": "nekomuon",
                  "learningRate": 0.001,
                  "auxiliaryLearningRate": 0.002,
                  "weightDecay": 0.02,
                  "nekoMuonNewtonSchulzInterval": 7,
                  "adamWUseBFloat16FirstMoment": true,
                  "adamWUseBFloat16SecondMoment": true
                },
                "scheduler": {
                  "type": "warmupCosineProgress",
                  "warmupPercent": 25
                }
              }
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal(
            Path.Combine(
                directory.Root,
                "artifacts",
                "checkpoints",
                "latest.json"),
            configuration.CheckpointPath);
        Assert.True(configuration.ResumeFromCheckpoint);
        Assert.True(configuration.AutoResume);
        Assert.Equal("nekomuon", configuration.Optimizer);
        Assert.Equal(0.001f, configuration.LearningRate);
        Assert.Equal(0.002f, configuration.AuxiliaryLearningRate);
        Assert.Equal(0.02f, configuration.WeightDecay);
        Assert.Equal(7, configuration.NekoMuonNewtonSchulzInterval);
        Assert.True(configuration.AdamWUseBFloat16FirstMoment);
        Assert.True(configuration.AdamWUseBFloat16SecondMoment);
        Assert.Equal(25f, configuration.WarmupPercent);
    }

    [Fact]
    public void GroupedCheckpointUsesConfigurationFileNameByDefault()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "checkpoint": {
                "directory": "checkpoints"
              }
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal(
            Path.Combine(
                directory.Root,
                "checkpoints",
                "training.wiki-model.json"),
            configuration.CheckpointPath);
    }

    [Fact]
    public void RejectsMixedGroupedAndLegacyCheckpointSettings()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "checkpointPath": "legacy.json",
              "checkpoint": { "directory": "checkpoints" }
            }
            """);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => WikiTrainingConfiguration.Load(path));

        Assert.Contains("checkpointPath", exception.Message);
        Assert.Contains("checkpoint", exception.Message);
    }

    [Fact]
    public void RejectsMixedGroupedAndLegacyOptimizationSettings()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "learningRate": 0.1,
              "optimization": {
                "optimizer": { "type": "adamw" },
                "scheduler": { "type": "warmupCosineProgress" }
              }
            }
            """);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => WikiTrainingConfiguration.Load(path));

        Assert.Contains("learningRate", exception.Message);
        Assert.Contains("optimization", exception.Message);
    }

    [Fact]
    public void RejectsUnknownGroupedScheduler()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "optimization": {
                "optimizer": { "type": "adamw" },
                "scheduler": { "type": "oneCycle" }
              }
            }
            """);

        ArgumentException exception = Assert.Throws<ArgumentException>(
            () => WikiTrainingConfiguration.Load(path));

        Assert.Contains("oneCycle", exception.Message);
    }

    [Theory]
    [InlineData("../outside.json")]
    [InlineData("nested/model.json")]
    public void RejectsCheckpointFileNameWithDirectoryComponent(
        string fileName)
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            $$"""
            {
              "task": "gpt_rin_wiki_jp",
              "checkpoint": {
                "directory": "checkpoints",
                "fileName": "{{fileName}}"
              }
            }
            """);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => WikiTrainingConfiguration.Load(path));

        Assert.Contains("checkpoint.fileName", exception.Message);
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
        Assert.Equal("forgetmemoryv2", configuration.ModelArchitecture);
        Assert.True(configuration.IsForgetMemoryV2Architecture());
        Assert.Equal(TensorDType.Float16, configuration.GetModelDType());
        Assert.Equal(
            HyenaConvolutionAlgorithm.Auto,
            configuration.GetHyenaConvolutionAlgorithm());
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

    [Fact]
    public void ExplicitFloat32OverridesTheFloat16V2Default()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelDType = "FLOAT32",
        };

        configuration.Validate();

        Assert.Equal(TensorDType.Float32, configuration.GetModelDType());
    }

    [Fact]
    public void NonV2ArchitectureDefaultsToFloat32()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
        };

        configuration.Validate();

        Assert.Null(configuration.GetExplicitModelDType());
        Assert.Equal(TensorDType.Float32, configuration.GetModelDType());
    }

    [Fact]
    public void RejectsUnknownModelDType()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelDType = "float8",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("ModelDType", exception.ParamName);
        Assert.Contains("float16", exception.Message);
        Assert.Contains("float32", exception.Message);
    }

    [Fact]
    public void RejectsFloat16ForArchitectureWithoutFloat16Parameters()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = WikiTrainingConfiguration.HyenaArchitecture,
            ModelDType = WikiTrainingConfiguration.Float16ModelDType,
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("ModelDType", exception.ParamName);
    }

    [Fact]
    public void RejectsUnsupportedModelArchitecture()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = "attention-plus",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("ModelArchitecture", exception.ParamName);
    }

    [Fact]
    public void AcceptsForgetScanArchitecture()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = "forgetscan",
        };

        configuration.Validate();

        Assert.True(configuration.IsArchitecture(
            WikiTrainingConfiguration.ForgetScanArchitecture));
    }

    [Theory]
    [InlineData("forgetmemoryv2")]
    [InlineData("frogetmemoryv2")]
    public void AcceptsForgetMemoryV2ArchitectureAndAlias(string architecture)
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = architecture,
            ForgetMemoryKeyWidth = 5,
            ForgetMemoryValueWidth = 7,
            ForgetMemoryRetentionMinimum = 0.2f,
            ForgetMemoryRetentionMaximum = 0.95f,
        };

        configuration.Validate();

        Assert.True(configuration.IsForgetMemoryV2Architecture());
    }

    [Fact]
    public void RejectsInvalidForgetMemoryRetentionRange()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ForgetMemoryRetentionMinimum = 0.9f,
            ForgetMemoryRetentionMaximum = 0.5f,
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("ForgetMemoryRetentionMinimum", exception.ParamName);
    }

    [Theory]
    [InlineData(-1f)]
    [InlineData(100f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidWarmupPercent(float warmupPercent)
    {
        var configuration = new WikiTrainingConfiguration
        {
            WarmupPercent = warmupPercent,
        };

        ArgumentOutOfRangeException exception =
            Assert.Throws<ArgumentOutOfRangeException>(
                configuration.Validate);

        Assert.Equal("WarmupPercent", exception.ParamName);
    }

    [Fact]
    public void RejectsInvalidNekoMuonNewtonSchulzInterval()
    {
        var configuration = new WikiTrainingConfiguration
        {
            NekoMuonNewtonSchulzInterval = 0,
        };

        ArgumentOutOfRangeException exception =
            Assert.Throws<ArgumentOutOfRangeException>(
                configuration.Validate);

        Assert.Equal(
            "NekoMuonNewtonSchulzInterval",
            exception.ParamName);
    }

    [Fact]
    public void RejectsUnsupportedHyenaConvolutionAlgorithm()
    {
        var configuration = new WikiTrainingConfiguration
        {
            HyenaConvolutionAlgorithm = "ntt",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("HyenaConvolutionAlgorithm", exception.ParamName);
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
