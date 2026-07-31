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
              "stepsPerEpoch": 11,
              "learningRate": 0.025,
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
        Assert.Equal(11, configuration.StepsPerEpoch);
        Assert.Equal(0.025f, configuration.LearningRate);
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
        Assert.Equal(256, configuration.StepsPerEpoch);
        Assert.Equal(1e-4f, configuration.LearningRate);
        Assert.Equal(1234, configuration.Seed);
        Assert.Equal(1, configuration.Model.Heads);
        Assert.Equal(128, configuration.Model.HiddenSize);
        Assert.Equal(32, configuration.Model.Layers);
        Assert.Equal(0, configuration.Model.Seed);
        Assert.Equal(0.02f, configuration.Model.InitializationScale);
    }

    [Theory]
    [InlineData("epochs", "Epochs")]
    [InlineData("steps", "StepsPerEpoch")]
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
            StepsPerEpoch = setting == "steps" ? 0 : 1,
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
