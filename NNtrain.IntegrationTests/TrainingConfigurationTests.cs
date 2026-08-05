using System.Text.Json;
using NNtrain;
using Xunit;

public sealed class TrainingConfigurationTests
{
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
              "learningRate": 0.025,
              "weightDecay": 0.04,
              "labelSmoothing": 0.2,
              "useSimd": false,
              "seed": 42,
              "model": {
                "heads": 2,
                "hiddenSize": 64,
                "layers": 3,
                "seed": 9,
                "initializationScale": 0.03
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
        Assert.Equal(0.025f, configuration.LearningRate);
        Assert.Equal(0.04f, configuration.WeightDecay);
        Assert.Equal(0.2f, configuration.LabelSmoothing);
        Assert.False(configuration.UseSimd);
        Assert.Equal(42, configuration.Seed);
        Assert.Equal(2, configuration.Model.Heads);
        Assert.Equal(64, configuration.Model.HiddenSize);
        Assert.Equal(3, configuration.Model.Layers);
        Assert.Equal(9, configuration.Model.Seed);
        Assert.Equal(0.03f, configuration.Model.InitializationScale);
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
        Assert.Equal(1e-4f, configuration.LearningRate);
        Assert.Equal(0.05f, configuration.WeightDecay);
        Assert.Equal(0.1f, configuration.LabelSmoothing);
        Assert.True(configuration.UseSimd);
        Assert.Equal(1234, configuration.Seed);
        Assert.Equal(1, configuration.Model.Heads);
        Assert.Equal(128, configuration.Model.HiddenSize);
        Assert.Equal(32, configuration.Model.Layers);
        Assert.Equal(0, configuration.Model.Seed);
        Assert.Equal(0.02f, configuration.Model.InitializationScale);
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
                "dataPath": "data/cifar-100-binary/train.bin"
              },
              "evaluationData": {
                "type": "cifar100",
                "dataPath": "data/cifar-100-binary/test.bin"
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

    [Theory]
    [InlineData("epochs", "Epochs")]
    [InlineData("batchSize", "BatchSize")]
    [InlineData("learningRate", "LearningRate")]
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
            LearningRate = setting == "learningRate" ? 0f : 0.1f,
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
    public void ModelHeadsMustEvenlyDivideTheModelWidth()
    {
        var model = new ModelConfiguration { Heads = 3 };

        var exception = Assert.Throws<ArgumentException>(
            () => model.ValidateForModelWidth(28));

        Assert.Equal("Heads", exception.ParamName);
        Assert.Contains("evenly divide", exception.Message);
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
