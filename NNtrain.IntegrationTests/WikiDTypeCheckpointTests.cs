using NNtrain;
using Xunit;

public sealed class WikiDTypeCheckpointTests
{
    [Fact]
    public void V2FactorySupportsMixedDefaultAndExplicitFloat32()
    {
        WikiTrainingConfiguration defaultConfig = CreateConfiguration(
            checkpointPath: "unused.json",
            precisionMode: null,
            resume: false);

        LanguageModel defaultModel =
            WikiLanguageModelCommand.CreateModel(
                defaultConfig,
                defaultConfig.VocabularySize);
        Module defaultModule = Assert.IsAssignableFrom<Module>(defaultModel);
        Assert.Equal(TensorDType.BFloat16, defaultModule.DType);
        Assert.Equal(
            TensorPrecisionMode.Mix16_32,
            defaultModule.PrecisionMode);
        Assert.All(
            defaultModel.parameters(),
            parameter => Assert.Equal(
                TensorDType.BFloat16,
                parameter.T.DType));

        WikiTrainingConfiguration float32Config = defaultConfig with
        {
            PrecisionMode = WikiTrainingConfiguration.Float32PrecisionMode,
        };
        LanguageModel float32Model =
            WikiLanguageModelCommand.CreateModel(
                float32Config,
                float32Config.VocabularySize);
        Module float32Module = Assert.IsAssignableFrom<Module>(float32Model);
        Assert.Equal(TensorDType.Float32, float32Module.DType);
        Assert.All(
            float32Model.parameters(),
            parameter => Assert.Equal(
                TensorDType.Float32,
                parameter.T.DType));
    }

    [Theory]
    [InlineData(TensorPrecisionMode.Bfp8, 128)]
    [InlineData(TensorPrecisionMode.Mix8_32, 4)]
    public void V8Bfp8CheckpointUsesFloat32ArtifactsAndResumes(
        TensorPrecisionMode mode,
        int blockSize)
    {
        string checkpointPath = CreateCheckpointPath(
            $"{TensorPrecisionModeNames.Format(mode)}-resume");
        try
        {
            WikiTrainingConfiguration sourceConfig = CreateConfiguration(
                checkpointPath,
                precisionMode: null,
                resume: false) with
            {
                Precision = TensorPrecisionModeNames.Format(mode),
                Bfp8BlockSize = blockSize,
                Optimizer = WikiTrainingConfiguration.AdamWOptimizer,
            };
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                sourceConfig,
                sourceConfig.VocabularySize);
            Assert.All(
                source.parameters(),
                parameter =>
                {
                    Assert.Equal(TensorDType.Bfp8, parameter.T.DType);
                    Bfp8QuantizationDescriptor descriptor =
                        parameter.T.Bfp8Quantization!;
                    Assert.Equal(
                        mode == TensorPrecisionMode.Bfp8
                            ? Bfp8ScaleGranularity.Tensor
                            : Bfp8ScaleGranularity.Block,
                        descriptor.Granularity);
                    if (mode == TensorPrecisionMode.Mix8_32)
                        Assert.Equal(blockSize, descriptor.BlockSize);
                });
            ModuleState expected = source.state_dict();
            IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
                source,
                sourceConfig);
            WarmupCosineProgressLRScheduler scheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    optimizer,
                    sourceConfig.WarmupPercent);
            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                sourceConfig,
                sourceConfig.VocabularySize,
                completedEpoch: 1,
                expected,
                bestLoss: 1.25f,
                bestEpoch: 1,
                source,
                optimizer,
                scheduler,
                globalStep: 7);

            WikiLanguageModelCommand.WikiModelCheckpoint serialized =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                    checkpointPath);
            Assert.Equal(TensorDType.Bfp8, serialized.ModelDType);
            Assert.Equal(mode, serialized.PrecisionMode);
            ModuleState currentArtifact = safetensors.torch.load_file(
                WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                    checkpointPath,
                    serialized.ArtifactSlot));
            Assert.All(
                currentArtifact.Parameters,
                parameter => Assert.Equal(
                    TensorDType.Float32,
                    parameter.DType));

            WikiTrainingConfiguration resumeConfig = sourceConfig with
            {
                ResumeFromCheckpoint = true,
            };
            LanguageModel restored = WikiLanguageModelCommand.CreateModel(
                resumeConfig,
                resumeConfig.VocabularySize,
                WikiLanguageModelCommand.ResolvePrecisionModeForTraining(
                    resumeConfig));
            IOptimizer restoredOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(
                    restored,
                    resumeConfig);
            WarmupCosineProgressLRScheduler restoredScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    restoredOptimizer,
                    resumeConfig.WarmupPercent);
            ModuleState? bestState = null;
            float bestLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            long globalStep = 0;
            using var output = new StringWriter();
            _ = WikiLanguageModelCommand.RestoreTrainingCheckpoint(
                resumeConfig,
                restored,
                restoredOptimizer,
                restoredScheduler,
                ref bestState,
                ref bestLoss,
                ref bestEpoch,
                ref globalStep,
                output);

            Assert.Null(bestState);
            AssertStatesBitwiseEqual(expected, restored.state_dict());
            Assert.Equal(7, globalStep);

            LanguageModel generation = WikiLanguageModelCommand.CreateModel(
                serialized,
                sourceConfig.Seed,
                blockSize);
            WikiLanguageModelCommand.LoadGenerationModelInto(
                serialized,
                checkpointPath,
                generation);
            AssertStatesBitwiseEqual(expected, generation.state_dict());
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
        }
    }

    [Fact]
    public void LegacyV4V2CheckpointResumesAndGeneratesAsFloat32()
    {
        string checkpointPath = CreateCheckpointPath("legacy-v4");
        try
        {
            WikiTrainingConfiguration config = CreateConfiguration(
                checkpointPath,
                precisionMode: null,
                resume: true);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize,
                TensorDType.Float32);
            ModuleState state = source.state_dict();
            WikiLanguageModelCommand.WikiModelCheckpoint legacy =
                CreateCheckpoint(
                    config,
                    state,
                    formatVersion: 4,
                    modelDType: null);
            torch.save(legacy, checkpointPath);

            Assert.Equal(
                TensorDType.Float32,
                WikiLanguageModelCommand.ResolveModelDTypeForTraining(
                    config));

            WikiLanguageModelCommand.WikiModelCheckpoint loaded =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                    checkpointPath);
            LanguageModel generationModel =
                WikiLanguageModelCommand.CreateModel(loaded, config.Seed);
            Module module = Assert.IsAssignableFrom<Module>(generationModel);
            Assert.Equal(TensorDType.Float32, module.DType);
            Assert.All(
                generationModel.parameters(),
                parameter => Assert.Equal(
                    TensorDType.Float32,
                    parameter.T.DType));
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
        }
    }

    [Fact]
    public void AutoResumeV8MixedPreservesExactMastersAndOptimizerContinuity()
    {
        string checkpointPath = CreateCheckpointPath("f16-exact");
        try
        {
            WikiTrainingConfiguration sourceConfig = CreateConfiguration(
                checkpointPath,
                WikiTrainingConfiguration.Mix16_32PrecisionMode,
                resume: false) with
            {
                Optimizer = WikiTrainingConfiguration.NekoMuonOptimizer,
                AuxiliaryLearningRate = 0.003f,
            };
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                sourceConfig,
                sourceConfig.VocabularySize);
            IOptimizer sourceOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(
                    source,
                    sourceConfig);
            WarmupCosineProgressLRScheduler sourceScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    sourceOptimizer,
                    sourceConfig.WarmupPercent);

            TrainOneStep(
                source,
                sourceOptimizer,
                sourceScheduler,
                progress: 0.25d);
            ModuleState exactSourceState = source.state_dict();
            Assert.Contains(
                exactSourceState.Parameters,
                parameter => parameter.Values.Any(
                    value => BitConverter.SingleToInt32Bits(value)
                        != BitConverter.SingleToInt32Bits(
                            QuantizeBFloat16(value))));

            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                sourceConfig,
                sourceConfig.VocabularySize,
                completedEpoch: 1,
                exactSourceState,
                bestLoss: 1.5f,
                bestEpoch: 1,
                source,
                sourceOptimizer,
                sourceScheduler,
                globalStep: 11);

            WikiLanguageModelCommand.WikiModelCheckpoint serialized =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                    checkpointPath);
            Assert.Equal(8, serialized.FormatVersion);
            Assert.Equal(TensorDType.BFloat16, serialized.ModelDType);
            Assert.Equal(
                TensorPrecisionMode.Mix16_32,
                serialized.PrecisionMode);
            Assert.Null(serialized.CurrentModel);
            Assert.Empty(serialized.Model.Parameters);
            Assert.NotNull(serialized.OptimizerStateTypes);
            Assert.Equal(
                ["NekoMuon", "AdamW"],
                serialized.OptimizerStateTypes!);
            for (int optimizerIndex = 0;
                optimizerIndex < serialized.OptimizerStateTypes.Length;
                optimizerIndex++)
            {
                Assert.True(File.Exists(
                    WikiLanguageModelCommand.GetOptimizerBinaryArtifactPath(
                        checkpointPath,
                        serialized.ArtifactSlot,
                        optimizerIndex)));
                Assert.False(File.Exists(
                    WikiLanguageModelCommand.GetOptimizerArtifactPath(
                        checkpointPath,
                        serialized.ArtifactSlot,
                        optimizerIndex)));
            }
            Assert.True(new FileInfo(checkpointPath).Length < 128 * 1024);
            Assert.DoesNotContain(
                "\"Values\"",
                File.ReadAllText(checkpointPath));

            ModuleState safeState = safetensors.torch.load_file(
                WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                    checkpointPath,
                    serialized.ArtifactSlot));
            AssertSafeStateEqualsExactMaster(
                safeState,
                exactSourceState);
            ModuleState generationState =
                WikiLanguageModelCommand.LoadGenerationModelState(
                    serialized,
                    checkpointPath);
            AssertSafeStateEqualsQuantizedMaster(
                generationState,
                exactSourceState);

            string markerPath = TrainingRunGuard.GetMarkerPath(
                checkpointPath);
            File.WriteAllText(markerPath, "{\"interrupted\":true}");
            using TrainingRunGuard run = TrainingRunGuard.Begin(
                checkpointPath + ".config.json",
                checkpointPath);
            using var autoResumeOutput = new StringWriter();
            bool autoResume = Program.ResolveAutomaticResume(
                explicitResume: false,
                autoResume: true,
                run,
                checkpointPath,
                autoResumeOutput);
            Assert.True(autoResume);
            Assert.Contains(
                "auto-resume = interrupted training detected",
                autoResumeOutput.ToString());
            WikiTrainingConfiguration resumeConfig = sourceConfig with
            {
                ResumeFromCheckpoint = autoResume,
            };

            TensorPrecisionMode resumeMode =
                WikiLanguageModelCommand.ResolvePrecisionModeForTraining(
                    resumeConfig);
            Assert.Equal(TensorPrecisionMode.Mix16_32, resumeMode);
            LanguageModel restored =
                WikiLanguageModelCommand.CreateModel(
                    resumeConfig,
                    resumeConfig.VocabularySize,
                    resumeMode);
            IOptimizer restoredOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(
                    restored,
                    resumeConfig);
            WarmupCosineProgressLRScheduler restoredScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    restoredOptimizer,
                    resumeConfig.WarmupPercent);
            ModuleState? bestState = null;
            float bestLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            long globalStep = 0;
            using var output = new StringWriter();

            WikiLanguageModelCommand.RestoreTrainingCheckpoint(
                resumeConfig,
                restored,
                restoredOptimizer,
                restoredScheduler,
                ref bestState,
                ref bestLoss,
                ref bestEpoch,
                ref globalStep,
                output);

            Assert.Null(bestState);
            AssertStatesBitwiseEqual(
                exactSourceState,
                restored.state_dict());
            AssertOptimizerStatesEqual(
                sourceOptimizer.state_dict(),
                restoredOptimizer.state_dict());
            Assert.Equal(
                sourceScheduler.state_dict(),
                restoredScheduler.state_dict());

            TrainOneStep(
                source,
                sourceOptimizer,
                sourceScheduler,
                progress: 0.5d);
            TrainOneStep(
                restored,
                restoredOptimizer,
                restoredScheduler,
                progress: 0.5d);

            AssertStatesBitwiseEqual(
                source.state_dict(),
                restored.state_dict());
            AssertOptimizerStatesEqual(
                sourceOptimizer.state_dict(),
                restoredOptimizer.state_dict());
            Assert.Equal(
                sourceScheduler.state_dict(),
                restoredScheduler.state_dict());

            int firstArtifactSlot = serialized.ArtifactSlot;
            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                sourceConfig,
                sourceConfig.VocabularySize,
                completedEpoch: 1,
                new ModuleState(ModuleState.CurrentFormatVersion, []),
                bestLoss: 1.5f,
                bestEpoch: 1,
                source,
                sourceOptimizer,
                sourceScheduler,
                globalStep: 12);
            WikiLanguageModelCommand.WikiModelCheckpoint secondManifest =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                    checkpointPath);
            Assert.NotEqual(firstArtifactSlot, secondManifest.ArtifactSlot);
            Assert.Equal(
                serialized.BestArtifactSlot,
                secondManifest.BestArtifactSlot);
            Assert.False(File.Exists(
                WikiLanguageModelCommand.GetBestModelArtifactPath(
                    checkpointPath,
                    serialized.BestArtifactSlot == 0 ? 1 : 0)));
            run.Complete();
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
        }
    }

    [Fact]
    public void ResumeRejectsAnExplicitDTypeDifferentFromCheckpoint()
    {
        string checkpointPath = CreateCheckpointPath("dtype-mismatch");
        try
        {
            WikiTrainingConfiguration sourceConfig = CreateConfiguration(
                checkpointPath,
                WikiTrainingConfiguration.Mix16_32PrecisionMode,
                resume: false);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                sourceConfig,
                sourceConfig.VocabularySize,
                TensorDType.Float16);
            WikiLanguageModelCommand.WikiModelCheckpoint checkpoint =
                CreateCheckpoint(
                    sourceConfig,
                    source.state_dict(),
                    formatVersion: 5,
                    modelDType: TensorDType.Float16);
            torch.save(checkpoint, checkpointPath);

            WikiTrainingConfiguration compatibleResume = sourceConfig with
            {
                ResumeFromCheckpoint = true,
            };
            WikiLanguageModelCommand.WikiPrecisionSelection selection =
                WikiLanguageModelCommand.ResolvePrecisionForTraining(
                    compatibleResume);
            Assert.Equal(TensorPrecisionMode.Mix16_32, selection.Mode);
            Assert.Equal(TensorDType.Float16, selection.StorageDType);
            LanguageModel legacyStorageModel =
                WikiLanguageModelCommand.CreateModel(
                    compatibleResume,
                    compatibleResume.VocabularySize,
                    selection.Mode,
                    selection.StorageDType);
            Module legacyStorageModule =
                Assert.IsAssignableFrom<Module>(legacyStorageModel);
            Assert.Equal(TensorDType.Float16, legacyStorageModule.DType);
            Assert.Equal(
                TensorPrecisionMode.Mix16_32,
                legacyStorageModule.PrecisionMode);

            WikiTrainingConfiguration resumeConfig = sourceConfig with
            {
                ResumeFromCheckpoint = true,
                PrecisionMode = WikiTrainingConfiguration.Float32PrecisionMode,
            };
            InvalidDataException exception =
                Assert.Throws<InvalidDataException>(
                    () => WikiLanguageModelCommand
                        .ResolveModelDTypeForTraining(resumeConfig));

            Assert.Contains("does not match checkpoint", exception.Message);
            Assert.Contains("float32", exception.Message);
            Assert.Contains("mix16_32", exception.Message);
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
        }
    }

    [Fact]
    public void CheckpointRejectsOuterAndParameterDTypeDisagreement()
    {
        string checkpointPath = CreateCheckpointPath("state-dtype");
        try
        {
            WikiTrainingConfiguration config = CreateConfiguration(
                checkpointPath,
                WikiTrainingConfiguration.Mix16_32PrecisionMode,
                resume: true);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize,
                TensorDType.Float16);
            ModuleState valid = source.state_dict();
            ModuleParameterState[] parameters = valid.Parameters.ToArray();
            parameters[0] = parameters[0] with
            {
                DType = TensorDType.Float32,
            };
            ModuleState inconsistent = valid with
            {
                Parameters = parameters,
            };
            WikiLanguageModelCommand.WikiModelCheckpoint checkpoint =
                CreateCheckpoint(
                    config,
                    inconsistent,
                    formatVersion: 5,
                    modelDType: TensorDType.Float16);
            torch.save(checkpoint, checkpointPath);

            Assert.Equal(
                TensorDType.Float16,
                WikiLanguageModelCommand.ResolveModelDTypeForTraining(config));
            LanguageModel restored = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize,
                TensorDType.Float16);
            IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
                restored,
                config);
            WarmupCosineProgressLRScheduler scheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    optimizer,
                    config.WarmupPercent);
            ModuleState? bestState = null;
            float bestLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            long globalStep = 0;
            using var output = new StringWriter();

            InvalidDataException exception = Assert.Throws<InvalidDataException>(
                () => WikiLanguageModelCommand.RestoreTrainingCheckpoint(
                    config,
                    restored,
                    optimizer,
                    scheduler,
                    ref bestState,
                    ref bestLoss,
                    ref bestEpoch,
                    ref globalStep,
                    output));

            Assert.Contains("does not match model dtype", exception.Message);
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
        }
    }

    [Fact]
    public void CheckpointRejectsMismatchedCurrentAndBestSafeTensorSidecars()
    {
        string checkpointPath = CreateCheckpointPath("sidecar-mismatch");
        try
        {
            WikiTrainingConfiguration config = CreateConfiguration(
                checkpointPath,
                WikiTrainingConfiguration.Mix16_32PrecisionMode,
                resume: true);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize,
                TensorDType.Float16);
            ModuleState expected = source.state_dict();
            WikiLanguageModelCommand.WikiModelCheckpoint checkpoint =
                CreateCheckpoint(
                    config,
                    expected,
                    formatVersion: 5,
                    modelDType: TensorDType.Float16);
            torch.save(checkpoint, checkpointPath);
            ModuleState mismatched = PerturbFirstValue(expected);

            string currentSafePath =
                WikiLanguageModelCommand.GetSafeTensorsPath(checkpointPath);
            safetensors.torch.save_file(mismatched, currentSafePath);
            // The exact JSON state is authoritative for training resume. A
            // terminated save can leave the independently committed current
            // SafeTensors sidecar one checkpoint ahead, so dtype resolution
            // must not allocate or reject that redundant artifact.
            Assert.Equal(
                TensorDType.Float16,
                WikiLanguageModelCommand.ResolveModelDTypeForTraining(config));

            File.Delete(currentSafePath);
            safetensors.torch.save_file(
                mismatched,
                WikiLanguageModelCommand.GetBestSafeTensorsPath(
                    checkpointPath));
            InvalidDataException bestException =
                Assert.Throws<InvalidDataException>(
                    () => WikiLanguageModelCommand.LoadGenerationModelState(
                        checkpoint,
                        checkpointPath));
            Assert.Contains(
                "best-model SafeTensors sidecar",
                bestException.Message);
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
        }
    }

    [Theory]
    [InlineData(4, null, "mix16_32", "float32")]
    [InlineData(5, TensorDType.Float16, "float32", "mix16_32")]
    public void GenerateUsesCheckpointPrecisionForLegacyCheckpoints(
        int formatVersion,
        TensorDType? checkpointDType,
        string configuredDType,
        string expectedDType)
    {
        string checkpointPath = CreateCheckpointPath("generate");
        string tokenizerPath = Path.ChangeExtension(
            checkpointPath,
            ".tokenizer.json");
        string configurationPath = Path.ChangeExtension(
            checkpointPath,
            ".config.json");
        try
        {
            TensorDType physicalDType = checkpointDType
                ?? TensorDType.Float32;
            BpeTokenizer tokenizer = BpeTokenizer.Train(
                ["日本語の生成テスト"],
                BpeTokenizer.BaseVocabularySize);
            tokenizer.save(tokenizerPath);
            WikiTrainingConfiguration sourceConfig = CreateConfiguration(
                checkpointPath,
                physicalDType == TensorDType.Float16
                    ? WikiTrainingConfiguration.Mix16_32PrecisionMode
                    : WikiTrainingConfiguration.Float32PrecisionMode,
                resume: false);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                sourceConfig,
                tokenizer.VocabularySize,
                physicalDType);
            WikiLanguageModelCommand.WikiModelCheckpoint checkpoint =
                CreateCheckpoint(
                    sourceConfig,
                    source.state_dict(),
                    formatVersion,
                    checkpointDType);
            torch.save(checkpoint, checkpointPath);
            File.WriteAllText(
                configurationPath,
                $$"""
                {
                  "task": "gpt_rin_wiki_jp",
                  "dataPath": "unused",
                  "tokenizerPath": "{{tokenizerPath.Replace("\\", "\\\\")}}",
                  "checkpointPath": "{{checkpointPath.Replace("\\", "\\\\")}}",
                  "vocabularySize": {{tokenizer.VocabularySize}},
                  "contextLength": {{sourceConfig.ContextLength}},
                  "modelWidth": {{sourceConfig.ModelWidth}},
                  "heads": {{sourceConfig.Heads}},
                  "hiddenSize": {{sourceConfig.HiddenSize}},
                  "layers": {{sourceConfig.Layers}},
                  "modelArchitecture": "forgetmemoryv2",
                  "precisionMode": "{{configuredDType}}",
                  "forgetMemoryKeyWidth": {{sourceConfig.ForgetMemoryKeyWidth}},
                  "forgetMemoryValueWidth": {{sourceConfig.ForgetMemoryValueWidth}},
                  "forgetMemoryRetentionMinimum": {{sourceConfig.ForgetMemoryRetentionMinimum}},
                  "forgetMemoryRetentionMaximum": {{sourceConfig.ForgetMemoryRetentionMaximum}},
                  "dropout": 0,
                  "maxNewTokens": 1,
                  "temperature": 0,
                  "topK": 1
                }
                """);
            using var output = new StringWriter();
            using var error = new StringWriter();

            int exitCode = WikiLanguageModelCommand.Run(
                configurationPath,
                "日本",
                output,
                error);

            Assert.Equal(0, exitCode);
            Assert.Equal(string.Empty, error.ToString());
            Assert.Contains(
                $"precision {expectedDType}",
                output.ToString());
            Assert.Contains(
                "--generate uses the architecture and precision mode stored in the " +
                "checkpoint",
                output.ToString());
        }
        finally
        {
            DeleteCheckpointArtifacts(checkpointPath);
            if (File.Exists(tokenizerPath))
                File.Delete(tokenizerPath);
            if (File.Exists(configurationPath))
                File.Delete(configurationPath);
        }
    }

    private static WikiTrainingConfiguration CreateConfiguration(
        string checkpointPath,
        string? precisionMode,
        bool resume)
        => new()
        {
            CheckpointPath = checkpointPath,
            ResumeFromCheckpoint = resume,
            Epochs = 3,
            VocabularySize = BpeTokenizer.BaseVocabularySize,
            ContextLength = 2,
            ModelWidth = 4,
            Heads = 1,
            HiddenSize = 6,
            Layers = 1,
            ModelArchitecture =
                WikiTrainingConfiguration.ForgetMemoryV2Architecture,
            PrecisionMode = precisionMode,
            ForgetMemoryKeyWidth = 2,
            ForgetMemoryValueWidth = 2,
            ForgetMemoryRetentionMinimum = 0.5f,
            ForgetMemoryRetentionMaximum = 0.9f,
            Dropout = 0f,
            InitializationScale = 0.03f,
            Optimizer = WikiTrainingConfiguration.AdamWOptimizer,
            LearningRate = 0.003f,
            WeightDecay = 0.01f,
            WarmupPercent = 20f,
            Seed = 19,
        };

    private static WikiLanguageModelCommand.WikiModelCheckpoint
        CreateCheckpoint(
            WikiTrainingConfiguration config,
            ModuleState state,
            int formatVersion,
            TensorDType? modelDType)
        => new(
            FormatVersion: formatVersion,
            Epoch: 1,
            ValidationLoss: 1.5f,
            VocabularySize: config.VocabularySize,
            ContextLength: config.ContextLength,
            ModelWidth: config.ModelWidth,
            Heads: config.Heads,
            HiddenSize: config.HiddenSize,
            Layers: config.Layers,
            Dropout: config.Dropout,
            InitializationScale: config.InitializationScale,
            Model: state,
            ModelArchitecture: config.ModelArchitecture,
            HyenaFilterWidth: config.HyenaFilterWidth,
            ForgetMemoryKeyWidth: config.ForgetMemoryKeyWidth,
            ForgetMemoryValueWidth: config.ForgetMemoryValueWidth,
            ForgetMemoryRetentionMinimum:
                config.ForgetMemoryRetentionMinimum,
            ForgetMemoryRetentionMaximum:
                config.ForgetMemoryRetentionMaximum,
            CompletedEpoch: 1,
            CurrentModel: state,
            ModelDType: modelDType);

    private static void TrainOneStep(
        LanguageModel model,
        IOptimizer optimizer,
        WarmupCosineProgressLRScheduler scheduler,
        double progress)
    {
        optimizer.zero_grad();
        Tensor loss = nn.functional.cross_entropy(
            model.forward([1, 2], batch_size: 1, sequence_length: 2),
            [2, 1]);
        loss.backward();
        scheduler.step(progress);
        optimizer.step();
    }

    private static void AssertStatesBitwiseEqual(
        ModuleState expected,
        ModuleState actual)
    {
        Assert.Equal(expected.FormatVersion, actual.FormatVersion);
        Assert.Equal(expected.Parameters.Length, actual.Parameters.Length);
        for (int parameterIndex = 0;
            parameterIndex < expected.Parameters.Length;
            parameterIndex++)
        {
            ModuleParameterState first = expected.Parameters[parameterIndex];
            ModuleParameterState second = actual.Parameters[parameterIndex];
            Assert.Equal(first.Index, second.Index);
            Assert.Equal(first.Name, second.Name);
            Assert.Equal(first.Shape, second.Shape);
            Assert.Equal(first.DType, second.DType);
            Assert.Equal(first.Values.Length, second.Values.Length);
            for (int valueIndex = 0;
                valueIndex < first.Values.Length;
                valueIndex++)
            {
                Assert.Equal(
                    BitConverter.SingleToInt32Bits(first.Values[valueIndex]),
                    BitConverter.SingleToInt32Bits(second.Values[valueIndex]));
            }
        }
    }

    private static void AssertSafeStateEqualsQuantizedMaster(
        ModuleState safeState,
        ModuleState masterState)
    {
        Assert.Equal(masterState.FormatVersion, safeState.FormatVersion);
        Assert.Equal(
            masterState.Parameters.Length,
            safeState.Parameters.Length);
        for (int parameterIndex = 0;
            parameterIndex < masterState.Parameters.Length;
            parameterIndex++)
        {
            ModuleParameterState master =
                masterState.Parameters[parameterIndex];
            ModuleParameterState safe = safeState.Parameters[parameterIndex];
            Assert.Equal(TensorDType.BFloat16, master.DType);
            Assert.Equal(TensorDType.BFloat16, safe.DType);
            Assert.Equal(master.Index, safe.Index);
            Assert.Equal(master.Name, safe.Name);
            Assert.Equal(master.Shape, safe.Shape);
            Assert.Equal(master.Values.Length, safe.Values.Length);
            for (int valueIndex = 0;
                valueIndex < master.Values.Length;
                valueIndex++)
            {
                float expected = QuantizeBFloat16(master.Values[valueIndex]);
                Assert.Equal(
                    BitConverter.SingleToInt32Bits(expected),
                    BitConverter.SingleToInt32Bits(safe.Values[valueIndex]));
            }
        }
    }

    private static void AssertSafeStateEqualsExactMaster(
        ModuleState safeState,
        ModuleState masterState)
    {
        Assert.Equal(masterState.FormatVersion, safeState.FormatVersion);
        Assert.Equal(
            masterState.Parameters.Length,
            safeState.Parameters.Length);
        for (int parameterIndex = 0;
            parameterIndex < masterState.Parameters.Length;
            parameterIndex++)
        {
            ModuleParameterState master = masterState.Parameters[parameterIndex];
            ModuleParameterState safe = safeState.Parameters[parameterIndex];
            Assert.Equal(master.Index, safe.Index);
            Assert.Equal(master.Name, safe.Name);
            Assert.Equal(master.Shape, safe.Shape);
            Assert.Equal(TensorDType.Float32, safe.DType);
            Assert.Equal(master.Values, safe.Values);
        }
    }

    private static float QuantizeBFloat16(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle((rounded >> 16) << 16);
    }

    private static void AssertOptimizerStatesEqual(
        OptimizerStateDictionary expected,
        OptimizerStateDictionary actual)
    {
        Assert.Equal(expected.OptimizerType, actual.OptimizerType);
        Assert.Equal(expected.StateJsonText, actual.StateJsonText);
        Assert.Equal(expected.Children.Length, actual.Children.Length);
        for (int index = 0; index < expected.Children.Length; index++)
        {
            AssertOptimizerStatesEqual(
                expected.Children[index],
                actual.Children[index]);
        }
    }

    private static ModuleState PerturbFirstValue(ModuleState state)
    {
        ModuleParameterState[] parameters = state.Parameters
            .Select(parameter => parameter with
            {
                Shape = parameter.Shape.ToArray(),
                Values = parameter.Values.ToArray(),
            })
            .ToArray();
        parameters[0].Values[0] += 1f;
        return state with { Parameters = parameters };
    }

    private static string CreateCheckpointPath(string testName)
        => Path.Combine(
            Path.GetTempPath(),
            $"NNtrain-wiki-{testName}-{Guid.NewGuid():N}.json");

    private static void DeleteCheckpointArtifacts(string checkpointPath)
    {
        string[] paths =
        [
            checkpointPath,
            WikiLanguageModelCommand.GetSafeTensorsPath(checkpointPath),
            WikiLanguageModelCommand.GetBestSafeTensorsPath(checkpointPath),
            TrainingRunGuard.GetMarkerPath(checkpointPath),
        ];
        foreach (string path in paths)
        {
            if (File.Exists(path))
                File.Delete(path);
        }
        for (int slot = 0; slot < 2; slot++)
        {
            string currentArtifact =
                WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                    checkpointPath,
                    slot);
            string bestArtifact =
                WikiLanguageModelCommand.GetBestModelArtifactPath(
                    checkpointPath,
                    slot);
            if (File.Exists(currentArtifact))
                File.Delete(currentArtifact);
            if (File.Exists(bestArtifact))
                File.Delete(bestArtifact);
            for (int optimizerIndex = 0; optimizerIndex < 4;
                optimizerIndex++)
            {
                string optimizerArtifact =
                    WikiLanguageModelCommand.GetOptimizerArtifactPath(
                        checkpointPath,
                        slot,
                        optimizerIndex);
                if (File.Exists(optimizerArtifact))
                    File.Delete(optimizerArtifact);
                string optimizerBinaryArtifact =
                    WikiLanguageModelCommand.GetOptimizerBinaryArtifactPath(
                        checkpointPath,
                        slot,
                        optimizerIndex);
                if (File.Exists(optimizerBinaryArtifact))
                    File.Delete(optimizerBinaryArtifact);
            }
        }
    }
}
