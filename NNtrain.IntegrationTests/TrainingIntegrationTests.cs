using System.Buffers.Binary;
using NNtrain;
using Xunit;

public sealed class TrainingIntegrationTests
{
    [Fact]
    public void TwoStepRunIntegratesMnistModelAutogradAndAdamW()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles training = WriteDataset(
            directory.Root,
            "train",
            [0, 1]);
        DatasetFiles evaluation = WriteDataset(
            directory.Root,
            "eval",
            [2, 3]);
        var configuration = new TrainingConfiguration
        {
            TrainingData = new DatasetConfiguration
            {
                ImagePath = training.ImagePath,
                LabelPath = training.LabelPath,
            },
            EvaluationData = new DatasetConfiguration
            {
                ImagePath = evaluation.ImagePath,
                LabelPath = evaluation.LabelPath,
            },
            Epochs = 1,
            StepsPerEpoch = 2,
            LearningRate = 0.001f,
            Seed = 7,
            Model = new ModelConfiguration
            {
                Heads = 1,
                HiddenSize = 4,
                Layers = 1,
                Seed = 3,
                InitializationScale = 0.01f,
            },
        };
        TrainingComponents components =
            TrainingComposition.Create(configuration);

        TrainingEpochResult result =
            Assert.Single(components.Trainer.Run());

        Assert.Equal(1, result.Epoch);
        Assert.Equal(2, result.TrainingSteps);
        Assert.Equal(2, result.EvaluationSamples);
        Assert.True(float.IsFinite(result.Training.Loss));
        Assert.InRange(result.Training.Accuracy, 0f, 1f);
        Assert.True(float.IsFinite(result.Evaluation.Loss));
        Assert.InRange(result.Evaluation.Accuracy, 0f, 1f);
        Assert.Equal(2, components.Optimizer.CaptureState().Step);
    }

    [Fact]
    public void ProgramDisplaysAnUnderstandableMissingDataFileError()
    {
        using var directory = new TemporaryDirectory();
        string configurationPath = Path.Combine(
            directory.Root,
            "training.json");
        File.WriteAllText(
            configurationPath,
            """
            {
              "trainingData": {
                "imagePath": "missing/train-images.idx3-ubyte",
                "labelPath": "missing/train-labels.idx1-ubyte"
              },
              "evaluationData": {
                "imagePath": "missing/eval-images.idx3-ubyte",
                "labelPath": "missing/eval-labels.idx1-ubyte"
              },
              "epochs": 1,
              "stepsPerEpoch": 2,
              "model": {
                "layers": 1
              }
            }
            """);
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = Program.Run(
            ["--config", configurationPath],
            output,
            error);

        Assert.Equal(2, exitCode);
        Assert.Equal(string.Empty, output.ToString());
        Assert.Contains(
            "Training image data file was not found",
            error.ToString());
        Assert.Contains(
            Path.Combine(
                directory.Root,
                "missing",
                "train-images.idx3-ubyte"),
            error.ToString());
        Assert.Contains("training configuration", error.ToString());
    }

    private static DatasetFiles WriteDataset(
        string root,
        string prefix,
        byte[] labels)
    {
        var imageData = new byte[16 + labels.Length * 28 * 28];
        var labelData = new byte[8 + labels.Length];
        WriteInt32(imageData, 0, 2051);
        WriteInt32(imageData, 4, labels.Length);
        WriteInt32(imageData, 8, 28);
        WriteInt32(imageData, 12, 28);
        WriteInt32(labelData, 0, 2049);
        WriteInt32(labelData, 4, labels.Length);

        for (int index = 0; index < labels.Length; index++)
        {
            imageData[16 + index * 28 * 28] = (byte)(64 + index * 64);
            labelData[8 + index] = labels[index];
        }

        string imagePath = Path.Combine(root, $"{prefix}-images.idx3-ubyte");
        string labelPath = Path.Combine(root, $"{prefix}-labels.idx1-ubyte");
        File.WriteAllBytes(imagePath, imageData);
        File.WriteAllBytes(labelPath, labelData);
        return new DatasetFiles(imagePath, labelPath);
    }

    private static void WriteInt32(byte[] destination, int offset, int value)
    {
        BinaryPrimitives.WriteInt32BigEndian(
            destination.AsSpan(offset, sizeof(int)),
            value);
    }

    private readonly record struct DatasetFiles(
        string ImagePath,
        string LabelPath);

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.IntegrationTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose()
        {
            Directory.Delete(Root, recursive: true);
        }
    }
}
