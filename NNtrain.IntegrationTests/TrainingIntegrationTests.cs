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
            BatchSize = 2,
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
        var trainingDataset = new Mnist(
            configuration.TrainingData.ImagePath,
            configuration.TrainingData.LabelPath);
        var evaluationDataset = new Mnist(
            configuration.EvaluationData.ImagePath,
            configuration.EvaluationData.LabelPath);
        var model = new TransformerClassifier(
            trainingDataset.Rows,
            trainingDataset.Columns,
            configuration.Model.Heads,
            configuration.Model.HiddenSize,
            configuration.Model.Layers,
            trainingDataset.ClassCount,
            new Random(configuration.Model.Seed),
            configuration.Model.InitializationScale);
        var optimizer = new AdamW(
            model.Parameters(),
            new AdamWOptions
            {
                LearningRate = configuration.LearningRate,
            });
        var trainer = new Trainer(
            model,
            trainingDataset,
            evaluationDataset,
            optimizer,
            new TrainerOptions
            {
                Epochs = configuration.Epochs,
                StepsPerEpoch = 2,
                RandomSeed = configuration.Seed,
            });

        TrainingEpochResult result =
            Assert.Single(trainer.Run());

        Assert.Equal(1, result.Epoch);
        Assert.Equal(2, result.TrainingSteps);
        Assert.Equal(2, result.EvaluationSamples);
        Assert.True(float.IsFinite(result.Training.Loss));
        Assert.InRange(result.Training.Accuracy, 0f, 1f);
        Assert.True(float.IsFinite(result.Evaluation.Loss));
        Assert.InRange(result.Evaluation.Accuracy, 0f, 1f);
        Assert.Equal(2, optimizer.CaptureState().Step);
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
              "batchSize": 2,
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

    [Fact]
    public void ProgramDisplaysLossForEveryTrainingBatch()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles training = WriteDataset(
            directory.Root,
            "train",
            [0, 1, 2]);
        DatasetFiles evaluation = WriteDataset(directory.Root, "eval", [0]);
        string configurationPath = Path.Combine(directory.Root, "training.json");
        File.WriteAllText(
            configurationPath,
            $$"""
            {
              "trainingData": {
                "imagePath": "{{training.ImagePath.Replace("\\", "\\\\")}}",
                "labelPath": "{{training.LabelPath.Replace("\\", "\\\\")}}"
              },
              "evaluationData": {
                "imagePath": "{{evaluation.ImagePath.Replace("\\", "\\\\")}}",
                "labelPath": "{{evaluation.LabelPath.Replace("\\", "\\\\")}}"
              },
              "epochs": 1,
              "batchSize": 2,
              "model": { "heads": 1, "hiddenSize": 2, "layers": 1 }
            }
            """);
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = Program.Run(
            ["--config", configurationPath],
            output,
            error);

        Assert.Equal(0, exitCode);
        Assert.Equal(string.Empty, error.ToString());
        Assert.Contains(
            $"workers = {Environment.ProcessorCount}",
            output.ToString());
        Assert.Contains("epoch 1, batch 1/2, loss = ", output.ToString());
        Assert.Contains("epoch 1, batch 2/2, loss = ", output.ToString());
        Assert.Contains("epoch 1, train loss = ", output.ToString());
        Assert.Contains("epoch 1, eval 100%", output.ToString());
    }

    [Fact]
    public void ProgramTrainsAndEvaluatesCifar100WithoutReplacingMnist()
    {
        using var directory = new TemporaryDirectory();
        string trainingPath = WriteCifar100Dataset(
            directory.Root,
            "train.bin",
            [42]);
        string evaluationPath = WriteCifar100Dataset(
            directory.Root,
            "test.bin",
            [42]);
        string configurationPath = Path.Combine(directory.Root, "cifar.json");
        File.WriteAllText(
            configurationPath,
            $$"""
            {
              "trainingData": {
                "type": "cifar100",
                "dataPath": "{{trainingPath.Replace("\\", "\\\\")}}"
              },
              "evaluationData": {
                "type": "cifar100",
                "dataPath": "{{evaluationPath.Replace("\\", "\\\\")}}"
              },
              "epochs": 1,
              "batchSize": 1,
              "model": { "heads": 1, "hiddenSize": 2, "layers": 1 }
            }
            """);
        using var output = new StringWriter();
        using var error = new StringWriter();

        int exitCode = Program.Run(
            ["--config", configurationPath],
            output,
            error);

        Assert.Equal(0, exitCode);
        Assert.Equal(string.Empty, error.ToString());
        Assert.Contains("epoch 1, batch 1/1, loss = ", output.ToString());
        Assert.Contains("epoch 1, eval 100%", output.ToString());
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

    private static string WriteCifar100Dataset(
        string root,
        string fileName,
        byte[] fineLabels)
    {
        const int recordSize = 3074;
        var data = new byte[fineLabels.Length * recordSize];

        for (int index = 0; index < fineLabels.Length; index++)
        {
            int recordOffset = index * recordSize;
            data[recordOffset] = 0;
            data[recordOffset + 1] = fineLabels[index];
            data[recordOffset + 2] = (byte)(64 + index);
        }

        string path = Path.Combine(root, fileName);
        File.WriteAllBytes(path, data);
        return path;
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
