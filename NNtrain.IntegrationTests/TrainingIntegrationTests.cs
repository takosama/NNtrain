using System.Buffers.Binary;
using NNtrain;
using Xunit;

public sealed class TrainingIntegrationTests
{
    [Fact]
    public void TwoStepRunIntegratesMnistModelAutogradAndNekoMuon()
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
        var optimizer = new NekoMuon(
            model.Parameters(),
            new NekoMuonOptions
            {
                LearningRate = configuration.LearningRate,
                WeightDecay = configuration.WeightDecay,
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
                LabelSmoothing = configuration.LabelSmoothing,
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
        Assert.Contains("optimizer = gainshareadamw", output.ToString());
        Assert.Contains("learning rate = 0.000300", output.ToString());
        Assert.Contains(
            "gainshare = block depth 1, betas (0.900, 0.999), " +
            "eps 1.0E-008, rho 0.950, gamma 1.000, " +
            "scale [0.500, 2.000]",
            output.ToString());
        Assert.Contains("weight decay = 0.000500", output.ToString());
        Assert.Contains("epoch 1, batch 1/2, loss = ", output.ToString());
        Assert.Contains("epoch 1, batch 2/2, loss = ", output.ToString());
        Assert.Contains("epoch 1, train loss = ", output.ToString());
        Assert.Contains("epoch 1, eval 100%", output.ToString());
        string lossGraphPath = Path.ChangeExtension(
            configurationPath,
            ".loss.html");
        Assert.Contains($"loss graph = {lossGraphPath}", output.ToString());
        Assert.True(File.Exists(lossGraphPath));
        string lossGraph = File.ReadAllText(lossGraphPath);
        Assert.Contains("<polyline class=\"train\"", lossGraph);
        Assert.Contains("<polyline class=\"eval\"", lossGraph);
        Assert.Contains("class=\"train-point\"", lossGraph);
        Assert.Contains("class=\"eval-point\"", lossGraph);
        string checkpointPath = Path.ChangeExtension(
            configurationPath,
            ".best-model.json");
        Assert.True(File.Exists(checkpointPath));
        Assert.Contains(
            $"checkpoint = {checkpointPath}",
            output.ToString());
    }

    [Fact]
    public void ProgramStopsAfterConfiguredEvaluationPatience()
    {
        using var directory = new TemporaryDirectory();
        DatasetFiles training = WriteDataset(directory.Root, "train", [0]);
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
              "epochs": 4,
              "batchSize": 1,
              "earlyStoppingPatience": 2,
              "earlyStoppingMinimumDelta": 100,
              "showLossGraph": false,
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
        Assert.Contains("early stopping at epoch 3", output.ToString());
        Assert.DoesNotContain("epoch 4, batch", output.ToString());
        Assert.True(
            File.Exists(
                Path.ChangeExtension(
                    configurationPath,
                    ".best-model.json")));
    }

    [Fact]
    public void NekoMuonConfigurationBuildsHiddenAndAuxiliaryOptimizerGroups()
    {
        var model = new TransformerClassifier(
            seqLen: 2,
            dModel: 4,
            numHeads: 2,
            dHidden: 8,
            numLayers: 1,
            numClasses: 3,
            rng: new Random(13));
        var configuration = new TrainingConfiguration
        {
            Optimizer = TrainingConfiguration.NekoMuonOptimizer,
            LearningRate = 0.003f,
            AuxiliaryLearningRate = 0.0003f,
            WeightDecay = 0.01f,
        };

        var composite = Assert.IsType<CompositeOptimizer>(
            Program.CreateOptimizer(model, configuration));
        var nekoMuon = Assert.IsType<NekoMuon>(composite.Optimizers[0]);
        var adamW = Assert.IsType<AdamW>(composite.Optimizers[1]);
        NekoMuonState nekoState = nekoMuon.CaptureState();
        AdamWState adamState = adamW.CaptureState();

        Assert.Equal(4, nekoState.ParameterStates.Length);
        Assert.Equal(11, adamState.ParameterStates.Length);
        Assert.Equal(0.003f, nekoState.Options.LearningRate);
        Assert.Equal(0.0003f, adamState.Options.LearningRate);
        Assert.Equal(0.9f, adamState.Options.Beta1);
        Assert.Equal(0.95f, adamState.Options.Beta2);
        Assert.Equal(1e-8f, adamState.Options.Epsilon);
        Assert.Equal(0.01f, adamState.Options.WeightDecay);
    }

    [Fact]
    public void DefaultConfigurationBuildsRequestedGainShareAdamW()
    {
        var model = new TransformerClassifier(
            seqLen: 2,
            dModel: 4,
            numHeads: 2,
            dHidden: 8,
            numLayers: 1,
            numClasses: 3,
            rng: new Random(17));
        var configuration = new TrainingConfiguration();

        var optimizer = Assert.IsType<GainShareAdamW>(
            Program.CreateOptimizer(model, configuration));
        GainShareAdamWState state = optimizer.CaptureState();

        Assert.Equal(3, state.GroupStates.Length);
        Assert.Equal([1, 12, 2], state.GroupStates
            .Select(group => group.ParameterIndices.Length));
        Assert.Equal(3e-4f, state.Options.LearningRate);
        Assert.Equal(0.9f, state.Options.Beta1);
        Assert.Equal(0.999f, state.Options.Beta2);
        Assert.Equal(1e-8f, state.Options.Epsilon);
        Assert.Equal(0.95f, state.Options.Rho);
        Assert.Equal(1f, state.Options.Gamma);
        Assert.Equal(0.5f, state.Options.MinScale);
        Assert.Equal(2f, state.Options.MaxScale);
        Assert.Equal(5e-4f, state.Options.WeightDecay);
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
                "dataPath": "{{trainingPath.Replace("\\", "\\\\")}}",
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
                "dataPath": "{{evaluationPath.Replace("\\", "\\\\")}}",
                "patchSize": 4,
                "normalize": true,
                "augmentation": {
                  "randomCropPadding": 0,
                  "horizontalFlip": false,
                  "verticalFlip": false
                }
              },
              "epochs": 1,
              "batchSize": 1,
              "optimizer": "nekomuon",
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
        Assert.Contains("optimizer = nekomuon", output.ToString());
        Assert.Contains(
            "tokenization = 4x4 patches, 64 tokens x 48 features",
            output.ToString());
        Assert.Contains("normalization = cifar100", output.ToString());
        Assert.Contains(
            "augmentation = random crop (padding 4), horizontal flip",
            output.ToString());
        Assert.DoesNotContain("vertical flip", output.ToString());
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
