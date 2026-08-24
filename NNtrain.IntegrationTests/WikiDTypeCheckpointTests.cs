using NNtrain;
using Xunit;

public sealed class WikiDTypeCheckpointTests
{
    [Fact]
    public void V2FactorySupportsFloat16DefaultAndExplicitFloat32()
    {
        WikiTrainingConfiguration defaultConfig = CreateConfiguration(
            checkpointPath: "unused.json",
            modelDType: null,
            resume: false);

        LanguageModel defaultModel =
            WikiLanguageModelCommand.CreateModel(
                defaultConfig,
                defaultConfig.VocabularySize);
        Module defaultModule = Assert.IsAssignableFrom<Module>(defaultModel);
        Assert.Equal(TensorDType.Float16, defaultModule.DType);
        Assert.All(
            defaultModel.parameters(),
            parameter => Assert.Equal(
                TensorDType.Float16,
                parameter.T.DType));

        WikiTrainingConfiguration float32Config = defaultConfig with
        {
            ModelDType = WikiTrainingConfiguration.Float32ModelDType,
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

    [Fact]
    public void LegacyV4V2CheckpointResumesAndGeneratesAsFloat32()
    {
        string checkpointPath = CreateCheckpointPath("legacy-v4");
        try
        {
            WikiTrainingConfiguration config = CreateConfiguration(
                checkpointPath,
                modelDType: null,
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
    public void AutoResumeV5Float16PreservesExactMastersAndOptimizerContinuity()
    {
        string checkpointPath = CreateCheckpointPath("f16-exact");
        try
        {
            WikiTrainingConfiguration sourceConfig = CreateConfiguration(
                checkpointPath,
                WikiTrainingConfiguration.Float16ModelDType,
                resume: false);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                sourceConfig,
                sourceConfig.VocabularySize,
                TensorDType.Float16);
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
                            (float)(Half)value)));

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
            Assert.Equal(5, serialized.FormatVersion);
            Assert.Equal(TensorDType.Float16, serialized.ModelDType);
            Assert.NotNull(serialized.CurrentModel);
            AssertStatesBitwiseEqual(
                exactSourceState,
                serialized.CurrentModel!);

            ModuleState safeState = safetensors.torch.load_file(
                WikiLanguageModelCommand.GetSafeTensorsPath(checkpointPath));
            AssertSafeStateEqualsQuantizedMaster(
                safeState,
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

            TensorDType resumeDType =
                WikiLanguageModelCommand.ResolveModelDTypeForTraining(
                    resumeConfig);
            Assert.Equal(TensorDType.Float16, resumeDType);
            LanguageModel restored =
                WikiLanguageModelCommand.CreateModel(
                    resumeConfig,
                    resumeConfig.VocabularySize,
                    resumeDType);
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
                WikiTrainingConfiguration.Float16ModelDType,
                resume: false);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                sourceConfig,
                sourceConfig.VocabularySize);
            WikiLanguageModelCommand.WikiModelCheckpoint checkpoint =
                CreateCheckpoint(
                    sourceConfig,
                    source.state_dict(),
                    formatVersion: 5,
                    modelDType: TensorDType.Float16);
            torch.save(checkpoint, checkpointPath);

            WikiTrainingConfiguration resumeConfig = sourceConfig with
            {
                ResumeFromCheckpoint = true,
                ModelDType = WikiTrainingConfiguration.Float32ModelDType,
            };
            InvalidDataException exception =
                Assert.Throws<InvalidDataException>(
                    () => WikiLanguageModelCommand
                        .ResolveModelDTypeForTraining(resumeConfig));

            Assert.Contains("does not match checkpoint", exception.Message);
            Assert.Contains("float32", exception.Message);
            Assert.Contains("float16", exception.Message);
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
                WikiTrainingConfiguration.Float16ModelDType,
                resume: true);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize);
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

            InvalidDataException exception =
                Assert.Throws<InvalidDataException>(
                    () => WikiLanguageModelCommand
                        .ResolveModelDTypeForTraining(config));

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
                WikiTrainingConfiguration.Float16ModelDType,
                resume: true);
            LanguageModel source = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize);
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
            InvalidDataException currentException =
                Assert.Throws<InvalidDataException>(
                    () => WikiLanguageModelCommand
                        .ResolveModelDTypeForTraining(config));
            Assert.Contains("SafeTensors sidecar", currentException.Message);

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
    [InlineData(4, null, "float16", "float32")]
    [InlineData(5, TensorDType.Float16, "float32", "float16")]
    public void GenerateUsesCheckpointDTypeForLegacyAndNewCheckpoints(
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
                    ? WikiTrainingConfiguration.Float16ModelDType
                    : WikiTrainingConfiguration.Float32ModelDType,
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
                  "modelDType": "{{configuredDType}}",
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
                $"dtype {expectedDType}",
                output.ToString());
            Assert.Contains(
                "--generate uses the architecture and dtype stored in the " +
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
        string? modelDType,
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
            ModelDType = modelDType,
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
            Assert.Equal(TensorDType.Float16, master.DType);
            Assert.Equal(TensorDType.Float16, safe.DType);
            Assert.Equal(master.Index, safe.Index);
            Assert.Equal(master.Name, safe.Name);
            Assert.Equal(master.Shape, safe.Shape);
            Assert.Equal(master.Values.Length, safe.Values.Length);
            for (int valueIndex = 0;
                valueIndex < master.Values.Length;
                valueIndex++)
            {
                float expected = (float)(Half)master.Values[valueIndex];
                Assert.Equal(
                    BitConverter.SingleToInt32Bits(expected),
                    BitConverter.SingleToInt32Bits(safe.Values[valueIndex]));
            }
        }
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
    }
}
