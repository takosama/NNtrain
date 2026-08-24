using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class TrainingConfigurationTests
{
    [Fact]
    public void CheckedInClassificationProfileUsesVersionTwoSchema()
    {
        string path = Path.GetFullPath(
            Path.Combine(
                AppContext.BaseDirectory,
                "..", "..", "..", "..",
                "training.cifar100.json"));

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(path);

        Assert.Equal(200, configuration.Epochs);
    }

    [Fact]
    public void LoadReadsVersionTwoClassificationSections()
    {
        using var directory = new TemporaryDirectory();
        string path = directory.WriteConfiguration(
            """
            {
              "schemaVersion": 2,
              "task": { "type": "image-classification" },
              "data": {
                "trainingData": {
                  "imagePath": "train-images",
                  "labelPath": "train-labels"
                },
                "evaluationData": {
                  "imagePath": "eval-images",
                  "labelPath": "eval-labels"
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

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(path);

        Assert.Equal(3, configuration.Epochs);
        Assert.Equal(16, configuration.EffectiveBatchSize);
        Assert.Equal(17, configuration.Seed);
        Assert.False(configuration.UseSimd);
        Assert.Equal(2, configuration.Model.Heads);
    }

    [Fact]
    public void LoadReadsEverySettingAndResolvesDatasetPaths()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "data/train-images.idx3-ubyte",
                "labelPath": "data/train-labels.idx1-ubyte"
              },
              "evaluationData": {
                "imagePath": "data/eval-images.idx3-ubyte",
                "labelPath": "data/eval-labels.idx1-ubyte"
              },
              "epochs": 7,
              "batchSize": 11,
              "microBatchSize": 5,
              "microBatchCount": 3,
              "optimizer": "adamw",
              "learningRate": 0.025,
              "auxiliaryLearningRate": 0.0007,
              "weightDecay": 0.04,
              "gainShareBlockDepth": 2,
              "gainShareBeta1": 0.7,
              "gainShareBeta2": 0.8,
              "gainShareEpsilon": 0.00001,
              "gainShareRho": 0.8,
              "gainShareGamma": 0.75,
              "gainShareMinScale": 0.25,
              "gainShareMaxScale": 3.0,
              "labelSmoothing": 0.2,
              "warmupEpochs": 2,
              "minimumLearningRateRatio": 0.03,
              "earlyStoppingPatience": 5,
              "earlyStoppingMinimumDelta": 0.002,
              "useSimd": false,
              "showLossGraph": false,
              "resumeFromCheckpoint": true,
              "autoResume": true,
              "checkpointPath": "artifacts/resume.json",
              "seed": 42,
              "model": {
                "heads": 2,
                "hiddenSize": 64,
                "layers": 3,
                "seed": 9,
                "initializationScale": 0.03,
                "dropout": 0.15
              }
            }
            """);

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(configurationPath);

        Assert.Equal(
            Path.Combine(directory.Root, "data", "train-images.idx3-ubyte"),
            configuration.TrainingData.ImagePath);
        Assert.Equal(
            Path.Combine(directory.Root, "data", "train-labels.idx1-ubyte"),
            configuration.TrainingData.LabelPath);
        Assert.Equal(
            Path.Combine(directory.Root, "data", "eval-images.idx3-ubyte"),
            configuration.EvaluationData.ImagePath);
        Assert.Equal(
            Path.Combine(directory.Root, "data", "eval-labels.idx1-ubyte"),
            configuration.EvaluationData.LabelPath);
        Assert.Equal(7, configuration.Epochs);
        Assert.Equal(11, configuration.BatchSize);
        Assert.Equal(5, configuration.MicroBatchSize);
        Assert.Equal(5, configuration.ResolvedMicroBatchSize);
        Assert.Equal(3, configuration.MicroBatchCount);
        Assert.Equal(15, configuration.EffectiveBatchSize);
        Assert.Equal("adamw", configuration.Optimizer);
        Assert.Equal(0.025f, configuration.LearningRate);
        Assert.Equal(0.0007f, configuration.AuxiliaryLearningRate);
        Assert.Equal(0.04f, configuration.WeightDecay);
        Assert.Equal(2, configuration.GainShareBlockDepth);
        Assert.Equal(0.7f, configuration.GainShareBeta1);
        Assert.Equal(0.8f, configuration.GainShareBeta2);
        Assert.Equal(0.00001f, configuration.GainShareEpsilon);
        Assert.Equal(0.8f, configuration.GainShareRho);
        Assert.Equal(0.75f, configuration.GainShareGamma);
        Assert.Equal(0.25f, configuration.GainShareMinScale);
        Assert.Equal(3f, configuration.GainShareMaxScale);
        Assert.Equal(0.2f, configuration.LabelSmoothing);
        Assert.Equal(2, configuration.WarmupEpochs);
        Assert.Equal(0.03f, configuration.MinimumLearningRateRatio);
        Assert.Equal(5, configuration.EarlyStoppingPatience);
        Assert.Equal(0.002f, configuration.EarlyStoppingMinimumDelta);
        Assert.False(configuration.UseSimd);
        Assert.False(configuration.ShowLossGraph);
        Assert.True(configuration.ResumeFromCheckpoint);
        Assert.True(configuration.AutoResume);
        Assert.Equal(
            Path.Combine(directory.Root, "artifacts", "resume.json"),
            configuration.CheckpointPath);
        Assert.Equal(42, configuration.Seed);
        Assert.Equal(2, configuration.Model.Heads);
        Assert.Equal(64, configuration.Model.HiddenSize);
        Assert.Equal(3, configuration.Model.Layers);
        Assert.Equal(9, configuration.Model.Seed);
        Assert.Equal(0.03f, configuration.Model.InitializationScale);
        Assert.Equal(0.15f, configuration.Model.Dropout);
    }

    [Fact]
    public void LoadNormalizesGroupedCheckpointAndOptimizationSettings()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "epochs": 8,
              "optimization": {
                "optimizer": {
                  "type": "gainshareadamw",
                  "learningRate": 0.012,
                  "auxiliaryLearningRate": 0.003,
                  "weightDecay": 0.04,
                  "gainShareBlockDepth": 3,
                  "gainShareBeta1": 0.71,
                  "gainShareBeta2": 0.82,
                  "gainShareEpsilon": 0.00002,
                  "gainShareRho": 0.83,
                  "gainShareGamma": 0.76,
                  "gainShareMinScale": 0.27,
                  "gainShareMaxScale": 3.2
                },
                "scheduler": {
                  "type": "linearWarmupCosineAnnealing",
                  "warmupEpochs": 2,
                  "minimumLearningRateRatio": 0.05
                }
              },
              "checkpoint": {
                "directory": "artifacts/checkpoints",
                "fileName": "classifier.state.json",
                "resume": true,
                "autoResume": true
              }
            }
            """);

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(configurationPath);

        Assert.Equal("gainshareadamw", configuration.Optimizer);
        Assert.Equal(0.012f, configuration.LearningRate);
        Assert.Equal(0.003f, configuration.AuxiliaryLearningRate);
        Assert.Equal(0.04f, configuration.WeightDecay);
        Assert.Equal(3, configuration.GainShareBlockDepth);
        Assert.Equal(0.71f, configuration.GainShareBeta1);
        Assert.Equal(0.82f, configuration.GainShareBeta2);
        Assert.Equal(0.00002f, configuration.GainShareEpsilon);
        Assert.Equal(0.83f, configuration.GainShareRho);
        Assert.Equal(0.76f, configuration.GainShareGamma);
        Assert.Equal(0.27f, configuration.GainShareMinScale);
        Assert.Equal(3.2f, configuration.GainShareMaxScale);
        Assert.Equal(2, configuration.WarmupEpochs);
        Assert.Equal(0.05f, configuration.MinimumLearningRateRatio);
        Assert.True(configuration.ResumeFromCheckpoint);
        Assert.True(configuration.AutoResume);
        Assert.Equal(
            Path.Combine(
                directory.Root,
                "artifacts",
                "checkpoints",
                "classifier.state.json"),
            configuration.CheckpointPath);
        Assert.False(
            Directory.Exists(
                Path.Combine(
                    directory.Root,
                    "artifacts",
                    "checkpoints")));
    }

    [Fact]
    public void GroupedCheckpointUsesLegacyDefaultFileName()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "checkpoint": {
                "directory": "checkpoints"
              }
            }
            """);

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(configurationPath);

        Assert.Equal(
            Path.Combine(
                directory.Root,
                "checkpoints",
                "training.checkpoint.json"),
            configuration.CheckpointPath);
    }

    [Theory]
    [InlineData(
        "\"optimization\": {}, \"optimizer\": \"adamw\"",
        "optimization")]
    [InlineData(
        "\"checkpoint\": {}, \"checkpointPath\": \"state.json\"",
        "checkpoint")]
    public void LoadRejectsMixedGroupedAndLegacySettings(
        string settings,
        string sectionName)
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            $$"""
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              {{settings}}
            }
            """);

        var exception = Assert.Throws<InvalidDataException>(
            () => TrainingConfiguration.Load(configurationPath));

        Assert.Contains(sectionName, exception.Message);
    }

    [Fact]
    public void LoadRejectsUnknownGroupedSchedulerType()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "optimization": {
                "scheduler": {
                  "type": "step"
                }
              }
            }
            """);

        var exception = Assert.Throws<ArgumentException>(
            () => TrainingConfiguration.Load(configurationPath));

        Assert.Contains("step", exception.Message);
    }

    [Fact]
    public void LoadRejectsUnknownGroupedOptimizerType()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "optimization": {
                "optimizer": {
                  "type": "sgd"
                }
              }
            }
            """);

        var exception = Assert.Throws<ArgumentException>(
            () => TrainingConfiguration.Load(configurationPath));

        Assert.Contains("sgd", exception.Message);
        Assert.Contains("adamw", exception.Message);
    }

    [Fact]
    public void LoadRejectsCheckpointFileNameWithDirectoryComponent()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "checkpoint": {
                "directory": "checkpoints",
                "fileName": "nested/state.json"
              }
            }
            """);

        var exception = Assert.Throws<InvalidDataException>(
            () => TrainingConfiguration.Load(configurationPath));

        Assert.Contains("checkpoint.fileName", exception.Message);
    }

    [Fact]
    public void LoadRejectsGroupedCheckpointWithoutDirectory()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "checkpoint": {}
            }
            """);

        InvalidDataException exception = Assert.Throws<InvalidDataException>(
            () => TrainingConfiguration.Load(configurationPath));

        Assert.Contains("checkpoint.directory", exception.Message);
    }

    [Fact]
    public void LoadAppliesDocumentedDefaults()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              }
            }
            """);

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(configurationPath);

        Assert.Equal(200, configuration.Epochs);
        Assert.Equal(32, configuration.BatchSize);
        Assert.Null(configuration.MicroBatchSize);
        Assert.Equal(32, configuration.ResolvedMicroBatchSize);
        Assert.Equal(1, configuration.MicroBatchCount);
        Assert.Equal(32, configuration.EffectiveBatchSize);
        Assert.Equal(
            TrainingConfiguration.GainShareAdamWOptimizer,
            configuration.Optimizer);
        Assert.Equal(3e-4f, configuration.LearningRate);
        Assert.Equal(3e-4f, configuration.AuxiliaryLearningRate);
        Assert.Equal(5e-4f, configuration.WeightDecay);
        Assert.Equal(1, configuration.GainShareBlockDepth);
        Assert.Equal(0.9f, configuration.GainShareBeta1);
        Assert.Equal(0.999f, configuration.GainShareBeta2);
        Assert.Equal(1e-8f, configuration.GainShareEpsilon);
        Assert.Equal(0.95f, configuration.GainShareRho);
        Assert.Equal(1f, configuration.GainShareGamma);
        Assert.Equal(0.5f, configuration.GainShareMinScale);
        Assert.Equal(2f, configuration.GainShareMaxScale);
        Assert.Equal(0.1f, configuration.LabelSmoothing);
        Assert.Equal(0, configuration.WarmupEpochs);
        Assert.Equal(0.01f, configuration.MinimumLearningRateRatio);
        Assert.Equal(0, configuration.EarlyStoppingPatience);
        Assert.Equal(1e-4f, configuration.EarlyStoppingMinimumDelta);
        Assert.True(configuration.UseSimd);
        Assert.True(configuration.ShowLossGraph);
        Assert.Equal(1234, configuration.Seed);
        Assert.Equal(1, configuration.Model.Heads);
        Assert.Equal(128, configuration.Model.HiddenSize);
        Assert.Equal(32, configuration.Model.Layers);
        Assert.Equal(0, configuration.Model.Seed);
        Assert.Equal(0.02f, configuration.Model.InitializationScale);
        Assert.Equal(0f, configuration.Model.Dropout);
        Assert.Equal(
            DatasetConfiguration.MnistType,
            configuration.TrainingData.Type);
        Assert.Equal(
            DatasetConfiguration.MnistType,
            configuration.EvaluationData.Type);
    }

    [Fact]
    public void LoadReadsCifar100DataPathsWithoutChangingMnistFields()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "type": "cifar100",
                "dataPath": "data/cifar-100-binary/train.bin",
                "patchSize": 4,
                "normalize": true,
                "augmentation": {
                  "randomCropPadding": 4,
                  "horizontalFlip": true,
                  "verticalFlip": false
                }
              },
              "evaluationData": {
                "type": "cifar100",
                "dataPath": "data/cifar-100-binary/test.bin",
                "patchSize": 4,
                "normalize": true,
                "augmentation": {
                  "randomCropPadding": 0,
                  "horizontalFlip": false,
                  "verticalFlip": false
                }
              }
            }
            """);

        TrainingConfiguration configuration =
            TrainingConfiguration.Load(configurationPath);

        Assert.Equal(
            Path.Combine(
                directory.Root,
                "data",
                "cifar-100-binary",
                "train.bin"),
            configuration.TrainingData.DataPath);
        Assert.Equal(
            Path.Combine(
                directory.Root,
                "data",
                "cifar-100-binary",
                "test.bin"),
            configuration.EvaluationData.DataPath);
        Assert.Equal(string.Empty, configuration.TrainingData.ImagePath);
        Assert.Equal(string.Empty, configuration.TrainingData.LabelPath);
        Assert.Equal(4, configuration.TrainingData.PatchSize);
        Assert.Equal(4, configuration.EvaluationData.PatchSize);
        Assert.True(configuration.TrainingData.Normalize);
        Assert.Equal(
            4,
            configuration.TrainingData.Augmentation.RandomCropPadding);
        Assert.True(
            configuration.TrainingData.Augmentation.HorizontalFlip);
        Assert.False(
            configuration.TrainingData.Augmentation.VerticalFlip);
        Assert.True(configuration.EvaluationData.Normalize);
        Assert.Equal(
            0,
            configuration.EvaluationData.Augmentation.RandomCropPadding);
        Assert.False(
            configuration.EvaluationData.Augmentation.HorizontalFlip);
        Assert.False(
            configuration.EvaluationData.Augmentation.VerticalFlip);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(33)]
    public void Cifar100ConfigurationRejectsInvalidCropPadding(int padding)
    {
        var configuration = new DatasetConfiguration
        {
            Type = DatasetConfiguration.Cifar100Type,
            DataPath = "train.bin",
            Augmentation = new Cifar100AugmentationConfiguration
            {
                RandomCropPadding = padding,
            },
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => configuration.Validate("Training"));

        Assert.Equal("RandomCropPadding", exception.ParamName);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(3)]
    [InlineData(33)]
    public void Cifar100ConfigurationRejectsInvalidPatchSize(int patchSize)
    {
        var configuration = new DatasetConfiguration
        {
            Type = DatasetConfiguration.Cifar100Type,
            DataPath = "train.bin",
            PatchSize = patchSize,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            () => configuration.Validate("Training"));

        Assert.Equal("PatchSize", exception.ParamName);
    }

    [Fact]
    public void Cifar100ConfigurationRequiresADataPath()
    {
        var configuration = new DatasetConfiguration
        {
            Type = DatasetConfiguration.Cifar100Type,
        };

        var exception = Assert.Throws<ArgumentException>(
            () => configuration.Validate("Training"));

        Assert.Equal("DataPath", exception.ParamName);
    }

    [Fact]
    public void RejectsUnknownDatasetTypes()
    {
        var configuration = new DatasetConfiguration
        {
            Type = "unknown",
        };

        var exception = Assert.Throws<ArgumentException>(
            () => configuration.Validate("Training"));

        Assert.Equal("Type", exception.ParamName);
        Assert.Contains("Unsupported", exception.Message);
    }

    [Fact]
    public void RejectsUnknownOptimizers()
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            Optimizer = "sgd",
        };

        var exception = Assert.Throws<ArgumentException>(
            configuration.Validate);

        Assert.Equal("Optimizer", exception.ParamName);
        Assert.Contains("gainshareadamw", exception.Message);
        Assert.Contains("lion", exception.Message);
        Assert.Contains("nekomuon", exception.Message);
        Assert.Contains("adamw", exception.Message);
    }

    [Theory]
    [InlineData("epochs", "Epochs")]
    [InlineData("batchSize", "BatchSize")]
    [InlineData("microBatchSize", "MicroBatchSize")]
    [InlineData("microBatchCount", "MicroBatchCount")]
    [InlineData("learningRate", "LearningRate")]
    [InlineData("auxiliaryLearningRate", "AuxiliaryLearningRate")]
    [InlineData("heads", "Heads")]
    [InlineData("hiddenSize", "HiddenSize")]
    [InlineData("layers", "Layers")]
    [InlineData("initializationScale", "InitializationScale")]
    public void RejectsNonPositiveNumericSettings(
        string setting,
        string parameterName)
    {
        var configuration = new TrainingConfiguration
        {
            TrainingData = new DatasetConfiguration
            {
                ImagePath = "train-images",
                LabelPath = "train-labels",
            },
            EvaluationData = new DatasetConfiguration
            {
                ImagePath = "eval-images",
                LabelPath = "eval-labels",
            },
            Epochs = setting == "epochs" ? 0 : 1,
            BatchSize = setting == "batchSize" ? 0 : 1,
            MicroBatchSize = setting == "microBatchSize" ? 0 : 1,
            MicroBatchCount = setting == "microBatchCount" ? 0 : 1,
            LearningRate = setting == "learningRate" ? 0f : 0.1f,
            AuxiliaryLearningRate =
                setting == "auxiliaryLearningRate" ? 0f : 0.01f,
            Model = new ModelConfiguration
            {
                Heads = setting == "heads" ? 0 : 1,
                HiddenSize = setting == "hiddenSize" ? 0 : 1,
                Layers = setting == "layers" ? 0 : 1,
                InitializationScale =
                    setting == "initializationScale" ? 0f : 0.1f,
            },
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal(parameterName, exception.ParamName);
    }

    [Fact]
    public void RejectsAnOverflowingEffectiveBatchSize()
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            MicroBatchSize = int.MaxValue,
            MicroBatchCount = 2,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("MicroBatchCount", exception.ParamName);
        Assert.Contains("Effective batch size", exception.Message);
    }

    [Fact]
    public void LoadRejectsUnknownJsonProperties()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = directory.WriteConfiguration(
            """
            {
              "trainingData": {
                "imagePath": "train-images",
                "labelPath": "train-labels"
              },
              "evaluationData": {
                "imagePath": "eval-images",
                "labelPath": "eval-labels"
              },
              "epohcs": 3
            }
            """);

        Assert.Throws<JsonException>(
            () => TrainingConfiguration.Load(configurationPath));
    }

    [Theory]
    [InlineData(-0.01f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void RejectsInvalidLabelSmoothing(float smoothing)
    {
        var configuration = new TrainingConfiguration
        {
            TrainingData = new DatasetConfiguration
            {
                ImagePath = "train-images",
                LabelPath = "train-labels",
            },
            EvaluationData = new DatasetConfiguration
            {
                ImagePath = "eval-images",
                LabelPath = "eval-labels",
            },
            LabelSmoothing = smoothing,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("LabelSmoothing", exception.ParamName);
    }

    [Theory]
    [InlineData(-0.01f)]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    public void RejectsInvalidWeightDecay(float weightDecay)
    {
        var configuration = new TrainingConfiguration
        {
            TrainingData = new DatasetConfiguration
            {
                ImagePath = "train-images",
                LabelPath = "train-labels",
            },
            EvaluationData = new DatasetConfiguration
            {
                ImagePath = "eval-images",
                LabelPath = "eval-labels",
            },
            WeightDecay = weightDecay,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("WeightDecay", exception.ParamName);
    }

    [Fact]
    public void RejectsNegativeGainShareBlockDepth()
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            GainShareBlockDepth = -1,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("GainShareBlockDepth", exception.ParamName);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidGainShareRho(float rho)
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            GainShareRho = rho,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("GainShareRho", exception.ParamName);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidGainShareBeta1(float beta1)
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            GainShareBeta1 = beta1,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("GainShareBeta1", exception.ParamName);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(-1f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidGainShareEpsilon(float epsilon)
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            GainShareEpsilon = epsilon,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("GainShareEpsilon", exception.ParamName);
    }

    [Fact]
    public void RejectsGainShareMaximumBelowMinimum()
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            GainShareMinScale = 1f,
            GainShareMaxScale = 0.5f,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("GainShareMaxScale", exception.ParamName);
    }

    [Theory]
    [InlineData(-0.1f)]
    [InlineData(1f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidDropout(float dropout)
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            Model = new ModelConfiguration { Dropout = dropout },
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("Dropout", exception.ParamName);
    }

    [Fact]
    public void RejectsWarmupThatConsumesEveryEpoch()
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            Epochs = 5,
            WarmupEpochs = 5,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("WarmupEpochs", exception.ParamName);
    }

    [Theory]
    [InlineData(0f)]
    [InlineData(1.01f)]
    [InlineData(float.NaN)]
    public void RejectsInvalidMinimumLearningRateRatio(float ratio)
    {
        TrainingConfiguration configuration = CreateValidConfiguration() with
        {
            MinimumLearningRateRatio = ratio,
        };

        var exception = Assert.Throws<ArgumentOutOfRangeException>(
            configuration.Validate);

        Assert.Equal("MinimumLearningRateRatio", exception.ParamName);
    }

    [Fact]
    public void ModelHeadsMustEvenlyDivideTheModelWidth()
    {
        var model = new ModelConfiguration { Heads = 3 };

        var exception = Assert.Throws<ArgumentException>(
            () => model.ValidateForModelWidth(28));

        Assert.Equal("Heads", exception.ParamName);
        Assert.Contains("evenly divide", exception.Message);
    }

    private static TrainingConfiguration CreateValidConfiguration()
    {
        return new TrainingConfiguration
        {
            TrainingData = new DatasetConfiguration
            {
                ImagePath = "train-images",
                LabelPath = "train-labels",
            },
            EvaluationData = new DatasetConfiguration
            {
                ImagePath = "eval-images",
                LabelPath = "eval-labels",
            },
        };
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.ConfigTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        internal string WriteConfiguration(string json)
        {
            string path = Path.Combine(Root, "training.json");
            File.WriteAllText(path, json);
            return path;
        }

        public void Dispose()
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
