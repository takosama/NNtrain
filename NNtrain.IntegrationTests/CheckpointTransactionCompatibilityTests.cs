using System.Text.Json;
using NNtrain;
using NNtrain.Training.Persistence;
using Xunit;

public sealed class CheckpointTransactionCompatibilityTests
{
    [Fact]
    public void ClassificationStreamingWriterPreservesBytesAndLoadsIntoModel()
    {
        using var directory = new TemporaryDirectory();
        string legacyPath = Path.Combine(directory.Root, "legacy.json");
        string streamingPath = Path.Combine(directory.Root, "streaming.json");
        var source = new TinyCheckpointModule(0.75f);
        ClassificationTrainingCheckpoint checkpoint =
            CreateClassificationCheckpoint(2, 0.75f);

        ClassificationCheckpoint.Save(legacyPath, checkpoint);
        ClassificationCheckpoint.Save(streamingPath, source, checkpoint);

        Assert.Equal(
            File.ReadAllBytes(legacyPath),
            File.ReadAllBytes(streamingPath));
        Assert.Equal(
            File.ReadAllBytes(
                ClassificationCheckpoint.GetSafeTensorsPath(legacyPath)),
            File.ReadAllBytes(
                ClassificationCheckpoint.GetSafeTensorsPath(streamingPath)));

        var destination = new TinyCheckpointModule(-10f);
        ClassificationTrainingCheckpoint resumed =
            ClassificationCheckpoint.LoadIntoModel(
                streamingPath,
                destination);
        Assert.Empty(resumed.Model.Parameters);
        Assert.Equal(
            [0.75f],
            destination.state_dict().Parameters.Single().Values);
        Assert.Equal(checkpoint.CompletedEpoch, resumed.CompletedEpoch);
        Assert.Equal(
            checkpoint.Optimizer.StateJsonText,
            resumed.Optimizer.StateJsonText);
    }

    [Fact]
    public void ClassificationArtifactCheckpointStreamsModelOptimizerAndBest()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "training.json");
        var source = new VectorCheckpointModule(
            Enumerable.Range(0, 4097)
                .Select(index => (index - 2000) / 4097f)
                .ToArray());
        var sourceOptimizer = new AdamW(
            source.parameters(),
            new AdamWOptions { LearningRate = 0.01f, WeightDecay = 0f });
        source.parameters().Single().T.Sum().Backward();
        sourceOptimizer.step();
        ClassificationTrainingCheckpoint checkpoint =
            CreateClassificationCheckpoint(1, 0f);

        ClassificationCheckpoint.Save(
            checkpointPath,
            source,
            sourceOptimizer,
            checkpoint,
            publishCurrentAsBest: true);

        ClassificationTrainingCheckpoint manifest =
            torch.load<ClassificationTrainingCheckpoint>(checkpointPath);
        Assert.Empty(manifest.Model.Parameters);
        Assert.Null(manifest.BestModel);
        Assert.Equal("ArtifactStream", manifest.Optimizer.OptimizerType);
        Assert.NotNull(manifest.OptimizerStateTypes);
        Assert.Equal(["AdamW"], manifest.OptimizerStateTypes);
        Assert.NotNull(manifest.OptimizerLeafNames);
        Assert.Equal(["default/0000"], manifest.OptimizerLeafNames);
        Assert.Equal(0, manifest.ArtifactSlot);
        Assert.Equal(0, manifest.BestArtifactSlot);
        string currentArtifact =
            ClassificationCheckpoint.GetCurrentModelArtifactPath(
                checkpointPath,
                manifest.ArtifactSlot);
        string bestArtifact = ClassificationCheckpoint.GetBestModelArtifactPath(
            checkpointPath,
            manifest.BestArtifactSlot);
        string optimizerArtifact = ClassificationCheckpoint.GetOptimizerArtifactPath(
            checkpointPath,
            manifest.ArtifactSlot,
            0);
        Assert.True(File.Exists(currentArtifact));
        Assert.Equal(
            File.ReadAllBytes(currentArtifact),
            File.ReadAllBytes(bestArtifact));
        Assert.True(File.Exists(optimizerArtifact));

        var destination = new VectorCheckpointModule(new float[4097]);
        var destinationOptimizer = new AdamW(
            destination.parameters(),
            new AdamWOptions { LearningRate = 0.01f, WeightDecay = 0f });
        ClassificationTrainingCheckpoint resumed =
            ClassificationCheckpoint.LoadIntoModel(
                checkpointPath,
                destination,
                destinationOptimizer);

        Assert.Empty(resumed.Model.Parameters);
        Assert.Equal(
            source.parameters().Single().T.Data,
            destination.parameters().Single().T.Data);
        Assert.Equal(
            sourceOptimizer.state_dict().StateJsonText,
            destinationOptimizer.state_dict().StateJsonText);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void ClassificationLegacyResumeStreamsAndMigratesBestSidecar(
        int formatVersion)
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "legacy-v2.json");
        var source = new TinyCheckpointModule(0.625f);
        var sourceOptimizer = new AdamW(
            source.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        source.parameters().Single().T.Sum().Backward();
        sourceOptimizer.step();
        ModuleState sourceState = source.state_dict();
        ClassificationTrainingCheckpoint legacy =
            CreateClassificationCheckpoint(1, 0.625f) with
            {
                FormatVersion = formatVersion,
                Model = sourceState,
                Optimizer = sourceOptimizer.state_dict(),
                BestModel = sourceState,
            };
        ClassificationCheckpoint.Save(checkpointPath, legacy);

        var destination = new TinyCheckpointModule(-10f);
        var destinationOptimizer = new AdamW(
            destination.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        ClassificationTrainingCheckpoint resumed =
            ClassificationCheckpoint.LoadIntoModel(
                checkpointPath,
                destination,
                destinationOptimizer);

        Assert.Equal(
            sourceState.Parameters.Single().Values,
            destination.state_dict().Parameters.Single().Values);
        Assert.Equal(
            sourceOptimizer.state_dict().StateJsonText,
            destinationOptimizer.state_dict().StateJsonText);
        Assert.Equal(0, resumed.BestArtifactSlot);
        Assert.True(
            File.Exists(
                ClassificationCheckpoint.GetBestModelArtifactPath(
                    checkpointPath,
                    resumed.BestArtifactSlot)));
    }

    [Fact]
    public void ClassificationArtifactSlotsRetainBestAcrossCurrentUpdates()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "slots.json");
        var model = new TinyCheckpointModule(1f);
        var optimizer = new AdamW(
            model.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        ClassificationTrainingCheckpoint first =
            CreateClassificationCheckpoint(1, 1f);
        ClassificationCheckpoint.Save(
            checkpointPath,
            model,
            optimizer,
            first,
            publishCurrentAsBest: true);

        model.load_state_dict(CreateSingleValueState(2f));
        ClassificationCheckpoint.Save(
            checkpointPath,
            model,
            optimizer,
            first with { CompletedEpoch = 2 },
            publishCurrentAsBest: false);
        ClassificationTrainingCheckpoint secondManifest =
            torch.load<ClassificationTrainingCheckpoint>(checkpointPath);
        Assert.Equal(1, secondManifest.ArtifactSlot);
        Assert.Equal(0, secondManifest.BestArtifactSlot);

        ClassificationCheckpoint.LoadBestModel(checkpointPath, model);
        Assert.Equal(
            [1f],
            model.state_dict().Parameters.Single().Values);

        model.load_state_dict(CreateSingleValueState(3f));
        ClassificationCheckpoint.Save(
            checkpointPath,
            model,
            optimizer,
            first with
            {
                CompletedEpoch = 3,
                BestEpoch = 3,
                BestEvaluationLoss = 0.5f,
            },
            publishCurrentAsBest: true);
        ClassificationTrainingCheckpoint thirdManifest =
            torch.load<ClassificationTrainingCheckpoint>(checkpointPath);
        Assert.Equal(0, thirdManifest.ArtifactSlot);
        Assert.Equal(1, thirdManifest.BestArtifactSlot);
        ClassificationCheckpoint.LoadBestModel(checkpointPath, model);
        Assert.Equal(
            [3f],
            model.state_dict().Parameters.Single().Values);
    }

    [Fact]
    public void ClassificationArtifactFailureBeforeManifestKeepsCommittedSlot()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "slot-fault.json");
        var model = new TinyCheckpointModule(1f);
        var optimizer = new AdamW(
            model.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        ClassificationTrainingCheckpoint checkpoint =
            CreateClassificationCheckpoint(1, 1f);
        ClassificationCheckpoint.Save(
            checkpointPath,
            model,
            optimizer,
            checkpoint,
            publishCurrentAsBest: true);
        model.load_state_dict(CreateSingleValueState(2f));
        ClassificationCheckpoint.Save(
            checkpointPath,
            model,
            optimizer,
            checkpoint with { CompletedEpoch = 2 },
            publishCurrentAsBest: false);
        byte[] committedManifest = File.ReadAllBytes(checkpointPath);

        model.load_state_dict(CreateSingleValueState(3f));
        Assert.Throws<InjectedCheckpointFailure>(() =>
            ClassificationCheckpoint.Save(
                checkpointPath,
                model,
                optimizer,
                checkpoint with { CompletedEpoch = 3 },
                publishCurrentAsBest: false,
                faultInjector: new ThrowAtFaultPoint(
                    CheckpointFaultPoint.BeforeManifestPublish)));

        Assert.Equal(committedManifest, File.ReadAllBytes(checkpointPath));
        var destination = new TinyCheckpointModule(-1f);
        var destinationOptimizer = new AdamW(
            destination.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        ClassificationTrainingCheckpoint resumed =
            ClassificationCheckpoint.LoadIntoModel(
                checkpointPath,
                destination,
                destinationOptimizer);
        Assert.Equal(2, resumed.CompletedEpoch);
        Assert.Equal(
            [2f],
            destination.state_dict().Parameters.Single().Values);
    }

    [Fact]
    public void ClassificationLegacyPureBfp8BestMigrationUsesTwoPassRestore()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "legacy-bfp8.json");
        float[] values = Enumerable.Range(0, 257)
            .Select(index => (index - 128) / 31f)
            .ToArray();
        var source = new VectorCheckpointModule(values);
        source.to(TensorPrecisionMode.Bfp8, 128);
        var sourceOptimizer = new AdamW(
            source.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        ModuleState state = source.state_dict();
        ClassificationTrainingCheckpoint legacy =
            CreateClassificationCheckpoint(1, 0f) with
            {
                Model = state,
                BestModel = state,
                Optimizer = sourceOptimizer.state_dict(),
            };
        ClassificationCheckpoint.Save(checkpointPath, legacy);

        var destination = new VectorCheckpointModule(new float[values.Length]);
        destination.to(TensorPrecisionMode.Bfp8, 128);
        var destinationOptimizer = new AdamW(
            destination.parameters(),
            new AdamWOptions { WeightDecay = 0f });
        ClassificationTrainingCheckpoint resumed =
            ClassificationCheckpoint.LoadIntoModel(
                checkpointPath,
                destination,
                destinationOptimizer);
        ModuleState expected = state;

        Assert.Equal(
            expected.Parameters.Single().Values,
            destination.state_dict().Parameters.Single().Values);
        ClassificationCheckpoint.LoadBestModel(checkpointPath, destination);
        Assert.Equal(
            expected.Parameters.Single().Values,
            destination.state_dict().Parameters.Single().Values);
        Assert.Equal(0, resumed.BestArtifactSlot);
    }

    [Fact]
    public void ClassificationStreamingResumePreservesBFloat16MasterValue()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "mixed.json");
        const float MasterValue = 0.1f;
        var source = new TinyCheckpointModule(
            MasterValue,
            TensorDType.BFloat16);
        ClassificationTrainingCheckpoint checkpoint =
            CreateClassificationCheckpoint(1, MasterValue) with
            {
                Model = source.state_dict(),
            };

        ClassificationCheckpoint.Save(checkpointPath, source, checkpoint);
        var destination = new TinyCheckpointModule(
            -10f,
            TensorDType.BFloat16);
        _ = ClassificationCheckpoint.LoadIntoModel(
            checkpointPath,
            destination);

        float restored = destination.state_dict()
            .Parameters.Single().Values.Single();
        Assert.Equal(
            BitConverter.SingleToInt32Bits(MasterValue),
            BitConverter.SingleToInt32Bits(restored));
    }

    [Fact]
    public void ClassificationStreamingResumeCrossesJsonReaderBuffers()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "large.json");
        float[] values = Enumerable.Range(0, 100_000)
            .Select(index => index * 0.0001f - 3f)
            .ToArray();
        var source = new VectorCheckpointModule(values);
        ClassificationTrainingCheckpoint checkpoint =
            CreateClassificationCheckpoint(1, 0f);
        ClassificationCheckpoint.Save(checkpointPath, source, checkpoint);

        var destination = new VectorCheckpointModule(
            new float[values.Length]);
        _ = ClassificationCheckpoint.LoadIntoModel(
            checkpointPath,
            destination);

        IReadOnlyList<float> restored = destination.parameters()
            .Single().T.Data;
        Assert.Equal(values.Length, restored.Count);
        foreach (int index in new[] { 0, 1, 32_767, 65_536, values.Length - 1 })
        {
            Assert.Equal(
                BitConverter.SingleToInt32Bits(values[index]),
                BitConverter.SingleToInt32Bits(restored[index]));
        }
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8, 128)]
    [InlineData(TensorPrecisionMode.Mix8_32, 8)]
    public void ClassificationBfp8CheckpointUsesFloat32ArtifactAndResumes(
        TensorPrecisionMode mode,
        int blockSize)
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "bfp8.json");
        var source = new TinyCheckpointModule(0.1f);
        source.to(mode, blockSize);
        ModuleState expected = source.state_dict();
        ClassificationTrainingCheckpoint checkpoint =
            CreateClassificationCheckpoint(1, 0.1f) with
            {
                Model = expected,
            };

        ClassificationCheckpoint.Save(checkpointPath, source, checkpoint);
        ModuleState artifact = safetensors.torch.load_file(
            ClassificationCheckpoint.GetSafeTensorsPath(checkpointPath));
        Assert.All(
            artifact.Parameters,
            parameter => Assert.Equal(TensorDType.Float32, parameter.DType));

        var destination = new TinyCheckpointModule(-10f);
        destination.to(mode, blockSize);
        ClassificationTrainingCheckpoint resumed =
            ClassificationCheckpoint.LoadIntoModel(
                checkpointPath,
                destination);
        Assert.Empty(resumed.Model.Parameters);
        Assert.Equal(
            expected.Parameters.Single().Values,
            destination.state_dict().Parameters.Single().Values);
        Bfp8QuantizationDescriptor descriptor =
            destination.parameters().Single().T.Bfp8Quantization!;
        Assert.Equal(
            mode == TensorPrecisionMode.Bfp8
                ? Bfp8ScaleGranularity.Tensor
                : Bfp8ScaleGranularity.Block,
            descriptor.Granularity);
        if (mode == TensorPrecisionMode.Mix8_32)
            Assert.Equal(blockSize, descriptor.BlockSize);
    }

    [Fact]
    public void ClassificationFailureBeforeManifestRestoresPreviousGeneration()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "training.json");
        ClassificationTrainingCheckpoint oldCheckpoint =
            CreateClassificationCheckpoint(1, 0.25f);
        ClassificationTrainingCheckpoint newCheckpoint =
            CreateClassificationCheckpoint(2, 0.75f);
        ClassificationCheckpoint.Save(checkpointPath, oldCheckpoint);
        string artifactPath =
            ClassificationCheckpoint.GetSafeTensorsPath(checkpointPath);
        byte[] oldManifest = File.ReadAllBytes(checkpointPath);
        byte[] oldArtifact = File.ReadAllBytes(artifactPath);

        Assert.Throws<InjectedCheckpointFailure>(() =>
            ClassificationCheckpoint.Save(
                checkpointPath,
                newCheckpoint,
                new ThrowAtFaultPoint(
                    CheckpointFaultPoint.BeforeManifestPublish)));

        Assert.Equal(oldManifest, File.ReadAllBytes(checkpointPath));
        Assert.Equal(oldArtifact, File.ReadAllBytes(artifactPath));
        ClassificationTrainingCheckpoint restored =
            ClassificationCheckpoint.Load(checkpointPath);
        Assert.Equal(1, restored.CompletedEpoch);
        Assert.Equal(
            [0.25f],
            restored.Model.Parameters.Single().Values);
    }

    [Fact]
    public void ClassificationAfterManifestFaultKeepsNewGeneration()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "training.json");
        ClassificationCheckpoint.Save(
            checkpointPath,
            CreateClassificationCheckpoint(1, 0.25f));

        Assert.Throws<InjectedCheckpointFailure>(() =>
            ClassificationCheckpoint.Save(
                checkpointPath,
                CreateClassificationCheckpoint(2, 0.75f),
                new ThrowAtFaultPoint(
                    CheckpointFaultPoint.AfterManifestPublish)));

        ClassificationTrainingCheckpoint restored =
            ClassificationCheckpoint.Load(checkpointPath);
        Assert.Equal(2, restored.CompletedEpoch);
        Assert.Equal(
            [0.75f],
            restored.Model.Parameters.Single().Values);
    }

    [Fact]
    public void WikiFailureBeforeManifestKeepsPreviousSlotRecoverable()
    {
        using var directory = new TemporaryDirectory();
        string checkpointPath = Path.Combine(directory.Root, "wiki.json");
        var config = new WikiTrainingConfiguration
        {
            CheckpointPath = checkpointPath,
            Epochs = 3,
            ContextLength = 2,
            ModelWidth = 4,
            Heads = 1,
            HiddenSize = 8,
            Layers = 1,
            VocabularySize = BpeTokenizer.BaseVocabularySize,
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
            Optimizer = WikiTrainingConfiguration.AdamWOptimizer,
            Dropout = 0f,
        };
        LanguageModel model = WikiLanguageModelCommand.CreateModel(
            config,
            config.VocabularySize);
        IOptimizer optimizer =
            WikiLanguageModelCommand.CreateOptimizer(model, config);
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
        ModuleState firstBest = model.state_dict();
        WikiLanguageModelCommand.SaveTrainingCheckpoint(
            config,
            config.VocabularySize,
            completedEpoch: 1,
            firstBest,
            bestLoss: 2f,
            bestEpoch: 1,
            model,
            optimizer,
            scheduler,
            globalStep: 10);
        byte[] firstManifest = File.ReadAllBytes(checkpointPath);
        WikiLanguageModelCommand.WikiModelCheckpoint firstMetadata =
            torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                checkpointPath);
        string firstBestPath =
            WikiLanguageModelCommand.GetBestModelArtifactPath(
                checkpointPath,
                firstMetadata.BestArtifactSlot);
        byte[] firstBestArtifact = File.ReadAllBytes(firstBestPath);
        ModuleState secondBest = ChangeFirstValue(firstBest, 0.5f);

        Assert.Throws<InjectedCheckpointFailure>(() =>
            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                config,
                config.VocabularySize,
                completedEpoch: 2,
                secondBest,
                bestLoss: 1f,
                bestEpoch: 2,
                model,
                optimizer,
                scheduler,
                globalStep: 20,
                checkpointFaultInjector: new ThrowAtFaultPoint(
                    CheckpointFaultPoint.BeforeManifestPublish)));

        Assert.Equal(firstManifest, File.ReadAllBytes(checkpointPath));
        Assert.Equal(firstBestArtifact, File.ReadAllBytes(firstBestPath));
        WikiLanguageModelCommand.WikiModelCheckpoint restoredMetadata =
            torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                checkpointPath);
        Assert.Equal(firstMetadata.ArtifactSlot, restoredMetadata.ArtifactSlot);
        Assert.Equal(10, restoredMetadata.GlobalStep);
        ModuleState restoredBest =
            WikiLanguageModelCommand.LoadBestTrainingModelState(
                checkpointPath);
        Assert.Equal(
            firstBest.Parameters[0].Values,
            restoredBest.Parameters[0].Values);
    }

    private static ClassificationTrainingCheckpoint
        CreateClassificationCheckpoint(int completedEpoch, float value)
        => new(
            ClassificationTrainingCheckpoint.CurrentFormatVersion,
            completedEpoch,
            new ModuleState(
                ModuleState.CurrentFormatVersion,
                [
                    new ModuleParameterState(
                        0,
                        "weight",
                        [1],
                        [value]),
                ]),
            new OptimizerStateDictionary(
                "AdamW",
                JsonSerializer.SerializeToElement(new { }),
                []),
            new LRSchedulerStateDictionary("CosineAnnealingLR", completedEpoch),
            BestModel: null,
            BestEpoch: completedEpoch,
            BestEvaluationLoss: 1f,
            EarlyStoppingReferenceLoss: 1f,
            EpochsWithoutImprovement: 0);

    private static ModuleState ChangeFirstValue(
        ModuleState state,
        float delta)
        => state with
        {
            Parameters = state.Parameters
                .Select((parameter, parameterIndex) => parameterIndex == 0
                    ? parameter with
                    {
                        Values = parameter.Values
                            .Select((value, valueIndex) => valueIndex == 0
                                ? value + delta
                                : value)
                            .ToArray(),
                    }
                    : parameter)
                .ToArray(),
        };

    private static ModuleState CreateSingleValueState(float value)
        => new(
            ModuleState.CurrentFormatVersion,
            [
                new ModuleParameterState(
                    0,
                    "weight",
                    [1],
                    [value]),
            ]);

    private sealed class ThrowAtFaultPoint(CheckpointFaultPoint point)
        : ICheckpointFaultInjector
    {
        public void OnCheckpointFaultPoint(CheckpointFaultContext context)
        {
            if (context.Point == point)
                throw new InjectedCheckpointFailure(context.Point);
        }
    }

    private sealed class InjectedCheckpointFailure(CheckpointFaultPoint point)
        : Exception(point.ToString());

    private sealed class TinyCheckpointModule : Module
    {
        internal TinyCheckpointModule(
            float value,
            TensorDType dtype = TensorDType.Float32)
            : base(dtype)
        {
            RegisterParameter(
                new Parameter(
                    [value],
                    [1],
                    "weight",
                    WeightDecayPolicy.Apply,
                    dtype));
        }
    }

    private sealed class VectorCheckpointModule : Module
    {
        internal VectorCheckpointModule(float[] values)
        {
            RegisterParameter(
                new Parameter(
                    values,
                    [values.Length],
                    "weight",
                    WeightDecayPolicy.Apply));
        }
    }

    private sealed class TemporaryDirectory : IDisposable
    {
        internal TemporaryDirectory()
        {
            Root = Path.Combine(
                Path.GetTempPath(),
                $"NNtrain.CheckpointTransactionTests-{Guid.NewGuid():N}");
            Directory.CreateDirectory(Root);
        }

        internal string Root { get; }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}
