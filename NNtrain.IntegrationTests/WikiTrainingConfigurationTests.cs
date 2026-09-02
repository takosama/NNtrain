using NNtrain;
using Xunit;

public sealed class WikiTrainingConfigurationTests
{
    [Fact]
    public void FineWebSelectsFineWebDefaultsFromVersionTwoDataSection()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "wiki-language-model" },
              "data": { "dataset": "fineweb" }
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal("fineweb", configuration.Dataset);
        Assert.True(configuration.IsFineWebDataset());
        Assert.Equal(
            Path.Combine(directory.Root, "data", "fineweb"),
            configuration.DataPath);
        Assert.Equal(
            Path.Combine(directory.Root, "fineweb-bpe.json"),
            configuration.TokenizerPath);
    }

    [Fact]
    public void FineWebHonorsExplicitDataAndTokenizerPaths()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "dataset": "fineweb",
              "dataPath": "corpus/custom",
              "tokenizerPath": "tokenizers/custom.json"
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal(
            Path.Combine(directory.Root, "corpus", "custom"),
            configuration.DataPath);
        Assert.Equal(
            Path.Combine(directory.Root, "tokenizers", "custom.json"),
            configuration.TokenizerPath);
    }

    [Fact]
    public void RejectsUnsupportedDataset()
    {
        var configuration = new WikiTrainingConfiguration
        {
            Dataset = "commoncrawl",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("Dataset", exception.ParamName);
    }

    [Theory]
    [InlineData("training.example.json")]
    [InlineData("training.transformer.json")]
    [InlineData("training.forgetmemoryv2-wiki-jp.json")]
    [InlineData("training.forgetmemorydrn-wiki-jp.json")]
    [InlineData("training.hyena-wiki-jp.json")]
    [InlineData("training.forgetscan-wiki-jp.json")]
    public void CheckedInWikiProfilesUseVersionTwoSchema(string fileName)
    {
        string path = Path.GetFullPath(
            Path.Combine(
                AppContext.BaseDirectory,
                "..", "..", "..", "..",
                fileName));

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.True(WikiTrainingConfiguration.IsWikiConfiguration(path));
        Assert.True(configuration.Epochs > 0);
    }

    [Fact]
    public void LoadReadsVersionTwoWikiSections()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "wiki-language-model" },
              "data": { "dataPath": "data/wiki", "textColumn": "body" },
              "training": {
                "epochs": 2,
                "batchSize": 1,
                "gradientAccumulationSteps": 4,
                "validationFraction": 0.0
              },
              "runtime": { "device": "cpu", "seed": 29 },
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

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.True(WikiTrainingConfiguration.IsWikiConfiguration(path));
        Assert.Equal(2, configuration.Epochs);
        Assert.Equal(4, configuration.GradientAccumulationSteps);
        Assert.Equal(29, configuration.Seed);
        Assert.Equal(8, configuration.ModelWidth);
        Assert.Equal(
            Path.Combine(directory.Root, "data", "wiki"),
            configuration.DataPath);
    }

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
              "precisionMode": "float32",
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
              "nekoMuonBetaFast": 0.94,
              "nekoMuonNewtonSchulzInterval": 7,
              "warmupPercent": 20,
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
        Assert.Equal("float32", configuration.PrecisionMode);
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
        Assert.Equal(0.94f, configuration.NekoMuonBetaFast);
        Assert.True(configuration.HasNekoMuonBetaFastOverride);
        Assert.Equal(7, configuration.NekoMuonNewtonSchulzInterval);
        Assert.Equal(20f, configuration.WarmupPercent);
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
              "precisionMode": "bfloat16",
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
                  "nekoMuonBetaFast": 0.95,
                  "weightDecay": 0.02,
                  "nekoMuonNewtonSchulzInterval": 7,
                  "nekoMuonNewtonSchulzDepthMode": "minimum",
                  "nekoMuonNewtonSchulzDepth": 1.5
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
        Assert.Equal(0.95f, configuration.NekoMuonBetaFast);
        Assert.True(configuration.HasNekoMuonBetaFastOverride);
        Assert.Equal(0.02f, configuration.WeightDecay);
        Assert.Equal(7, configuration.NekoMuonNewtonSchulzInterval);
        Assert.True(
            configuration.HasNekoMuonNewtonSchulzDepthPolicyOverride);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Minimum,
            configuration.GetNekoMuonNewtonSchulzDepthMode());
        Assert.Equal(1.5f, configuration.GetNekoMuonNewtonSchulzDepth());
        Assert.Equal(
            TensorPrecisionMode.BFloat16,
            configuration.GetPrecisionMode());
        Assert.Equal(25f, configuration.WarmupPercent);
    }

    [Fact]
    public void GroupedOmittedBetaFastDoesNotCreateResumeOverride()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "optimization": {
                "optimizer": { "type": "nekomuon" },
                "scheduler": { "type": "warmupCosineProgress" }
              }
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal(0.9f, configuration.NekoMuonBetaFast);
        Assert.False(configuration.HasNekoMuonBetaFastOverride);
    }

    [Fact]
    public void VersionTwoGroupedBetaFastCreatesResumeOverride()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "wiki-language-model" },
              "optimization": {
                "optimizer": {
                  "type": "nekomuon",
                  "nekoMuonBetaFast": 0.95
                },
                "scheduler": { "type": "warmupCosineProgress" }
              }
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal(0.95f, configuration.NekoMuonBetaFast);
        Assert.True(configuration.HasNekoMuonBetaFastOverride);
    }

    [Fact]
    public void LoadAcceptsGroupedOrdinaryMuonFixedNs5Settings()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.Write(
            """
            {
              "task": "gpt_rin_wiki_jp",
              "optimization": {
                "optimizer": {
                  "type": "muon",
                  "learningRate": 0.001,
                  "auxiliaryLearningRate": 0.0003,
                  "weightDecay": 0.01,
                  "nekoMuonNewtonSchulzInterval": 1,
                  "nekoMuonNewtonSchulzDepthMode": "fixed",
                  "nekoMuonNewtonSchulzDepth": 5
                },
                "scheduler": {
                  "type": "warmupCosineProgress",
                  "warmupPercent": 0
                }
              }
            }
            """);

        WikiTrainingConfiguration configuration =
            WikiTrainingConfiguration.Load(path);

        Assert.Equal(
            WikiTrainingConfiguration.MuonOptimizer,
            configuration.Optimizer);
        Assert.Equal(1, configuration.NekoMuonNewtonSchulzInterval);
        Assert.Equal("fixed", configuration.NekoMuonNewtonSchulzDepthMode);
        Assert.Equal(5f, configuration.NekoMuonNewtonSchulzDepth);
        Assert.Equal(0f, configuration.WarmupPercent);
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
        Assert.Equal("forgetmemoryv3", configuration.ModelArchitecture);
        Assert.True(configuration.IsForgetMemoryV3Architecture());
        Assert.Equal(TensorDType.BFloat16, configuration.GetModelDType());
        Assert.Equal(
            TensorPrecisionMode.Mix16_32,
            configuration.GetPrecisionMode());
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
            Optimizer = "rmsprop",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("Optimizer", exception.ParamName);
    }

    [Fact]
    public void AcceptsOrdinaryMuonWithImplicitFixedPolicy()
    {
        var configuration = new WikiTrainingConfiguration
        {
            Optimizer = WikiTrainingConfiguration.MuonOptimizer,
        };

        configuration.Validate();

        Assert.False(
            configuration.HasNekoMuonNewtonSchulzDepthPolicyOverride);
    }

    [Fact]
    public void AcceptsOrdinaryMuonWithLegacyFieldsSpellingFixedNs5()
    {
        var configuration = new WikiTrainingConfiguration
        {
            Optimizer = WikiTrainingConfiguration.MuonOptimizer,
            NekoMuonNewtonSchulzInterval = 1,
            NekoMuonNewtonSchulzDepthMode = "fixed",
            NekoMuonNewtonSchulzDepth = 5f,
        };

        configuration.Validate();

        Assert.False(
            configuration.HasNekoMuonNewtonSchulzDepthPolicyOverride);
    }

    [Theory]
    [InlineData(2, "fixed", 5f)]
    [InlineData(1, "fixed", 4f)]
    [InlineData(1, "adaptive", null)]
    public void RejectsOrdinaryMuonPolicyVariants(
        int interval,
        string mode,
        float? depth)
    {
        var configuration = new WikiTrainingConfiguration
        {
            Optimizer = WikiTrainingConfiguration.MuonOptimizer,
            NekoMuonNewtonSchulzInterval = interval,
            NekoMuonNewtonSchulzDepthMode = mode,
            NekoMuonNewtonSchulzDepth = depth,
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("NekoMuonNewtonSchulzDepthMode", exception.ParamName);
        Assert.Contains("fixed depth 5", exception.Message);
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
    public void RejectsUnknownPrecisionMode()
    {
        var configuration = new WikiTrainingConfiguration
        {
            PrecisionMode = "float8",
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("PrecisionMode", exception.ParamName);
        Assert.Contains("mix16_32", exception.Message);
        Assert.Contains("bfloat16", exception.Message);
        Assert.Contains("float32", exception.Message);
    }

    [Fact]
    public void RejectsMix16_32ForArchitectureWithout16BitParameters()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = WikiTrainingConfiguration.HyenaArchitecture,
            PrecisionMode = WikiTrainingConfiguration.Mix16_32PrecisionMode,
        };

        ArgumentException exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("PrecisionMode", exception.ParamName);
    }

    [Theory]
    [InlineData("fp16_32", TensorDType.BFloat16)]
    [InlineData("mix16_32", TensorDType.BFloat16)]
    [InlineData("bfloat16", TensorDType.BFloat16)]
    [InlineData("float32", TensorDType.Float32)]
    [InlineData("bfp8", TensorDType.Bfp8)]
    [InlineData("mix8_32", TensorDType.Bfp8)]
    public void AllowsTransformerPrecisionModes(
        string configuredMode,
        TensorDType expectedDType)
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
            PrecisionMode = configuredMode,
        };

        configuration.Validate();

        Assert.Equal(expectedDType, configuration.GetModelDType());
    }

    [Theory]
    [InlineData("bfp8", "gainshareadamw")]
    [InlineData("bfp8", "lion")]
    [InlineData("mix8_32", "gainshareadamw")]
    [InlineData("mix8_32", "lion")]
    public void AllowsBfp8OptimizersWithResidentCudaUpdates(
        string precisionMode,
        string optimizer)
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
            PrecisionMode = precisionMode,
            Optimizer = optimizer,
        };

        configuration.Validate();

        Assert.Equal(
            TensorPrecisionModeNames.Parse(precisionMode),
            configuration.GetPrecisionMode());
    }

    [Fact]
    public void LegacyFloat16ModelDTypeMapsToMix16_32()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
            ModelDType = WikiTrainingConfiguration.LegacyFloat16ModelDType,
        };

        configuration.Validate();

        Assert.Equal(
            TensorPrecisionMode.Mix16_32,
            configuration.GetPrecisionMode());
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
    public void AcceptsForgetMemoryV3Architecture()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = "forgetmemoryv3",
        };

        configuration.Validate();

        Assert.True(configuration.IsForgetMemoryV3Architecture());
        Assert.True(configuration.IsForgetMemoryArchitecture());
    }

    [Fact]
    public void AcceptsForgetMemoryDrnArchitecture()
    {
        var configuration = new WikiTrainingConfiguration
        {
            ModelArchitecture = "forgetmemorydrn",
        };

        configuration.Validate();

        Assert.True(configuration.IsForgetMemoryDrnArchitecture());
        Assert.True(configuration.IsForgetMemoryArchitecture());
        Assert.Equal(TensorDType.BFloat16, configuration.GetModelDType());
        Assert.Equal(
            TensorPrecisionMode.Mix16_32,
            configuration.GetPrecisionMode());
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
    public void RejectsNonPositiveCudaGraphCacheBudget()
    {
        var configuration = new WikiTrainingConfiguration
        {
            CudaGraphCacheBudgetMiB = 0,
        };

        ArgumentOutOfRangeException exception =
            Assert.Throws<ArgumentOutOfRangeException>(
                configuration.Validate);

        Assert.Equal("CudaGraphCacheBudgetMiB", exception.ParamName);
    }

    [Fact]
    public void RejectsNonPositiveGradientAccumulation()
    {
        var configuration = new WikiTrainingConfiguration
        {
            GradientAccumulationSteps = 0,
        };

        ArgumentOutOfRangeException exception =
            Assert.Throws<ArgumentOutOfRangeException>(
                configuration.Validate);

        Assert.Equal("GradientAccumulationSteps", exception.ParamName);
    }

    [Fact]
    public void AcceptsExplicitAdaptiveNekoMuonDepthPolicyWithoutDepth()
    {
        var configuration = new WikiTrainingConfiguration
        {
            NekoMuonNewtonSchulzDepthMode = "adaptive",
        };

        configuration.Validate();

        Assert.True(
            configuration.HasNekoMuonNewtonSchulzDepthPolicyOverride);
        Assert.Equal(
            NekoMuonNewtonSchulzDepthMode.Adaptive,
            configuration.GetNekoMuonNewtonSchulzDepthMode());
        Assert.Equal(0f, configuration.GetNekoMuonNewtonSchulzDepth());
    }

    [Theory]
    [InlineData(null, 1f)]
    [InlineData("adaptive", 1f)]
    [InlineData("minimum", null)]
    [InlineData("fixed", -1f)]
    [InlineData("fixed", 6f)]
    [InlineData("unknown", 1f)]
    public void RejectsInvalidNekoMuonDepthPolicy(
        string? mode,
        float? depth)
    {
        var configuration = new WikiTrainingConfiguration
        {
            NekoMuonNewtonSchulzDepthMode = mode,
            NekoMuonNewtonSchulzDepth = depth,
        };

        Assert.ThrowsAny<ArgumentException>(configuration.Validate);
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
