using NNtrain;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using Xunit;

public sealed class WikiLanguageModelCommandTests
{
    [Fact]
    public void ProductionFactoryDoesNotCreateCudaLanesForCpuAuthority()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor.CudaDeviceIndices = [3, 5];
            int factoryCalls = 0;
            using var session = ProductionTrainingSessionFactory.Create(
                TensorPrecisionMode.Float32,
                lastCommittedStep: -1,
                deviceIndex =>
                {
                    factoryCalls++;
                    return new TestStreamLane(deviceIndex);
                });

            Assert.Equal(0, factoryCalls);
            Assert.Empty(session.ExecutionSession.Lanes);
            Assert.Equal(
                ExecutionDeviceKind.Cpu,
                session.ExecutionSession.Options.Device);
            Assert.Equal(
                [3, 5],
                session.ExecutionSession.Options.CudaDevices);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ProductionFactoryAttachesEveryCudaLaneAndCleansPartialFailure()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.CudaDeviceIndices = [3, 5];
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            var created = new List<TestStreamLane>();

            using (var session = ProductionTrainingSessionFactory.Create(
                TensorPrecisionMode.Mix16_32,
                lastCommittedStep: -1,
                deviceIndex =>
                {
                    var lane = new TestStreamLane(deviceIndex);
                    created.Add(lane);
                    return lane;
                }))
            {
                Assert.Equal([3, 5], created.Select(lane => lane.DeviceIndex));
                Assert.Equal(2, session.ExecutionSession.Lanes.Count);
            }
            Assert.All(created, lane => Assert.True(lane.Disposed));

            created.Clear();
            Assert.Throws<InvalidOperationException>(() =>
                ProductionTrainingSessionFactory.Create(
                    TensorPrecisionMode.Mix16_32,
                    lastCommittedStep: -1,
                    deviceIndex =>
                    {
                        if (deviceIndex == 5)
                        {
                            throw new InvalidOperationException(
                                "scripted second-lane failure");
                        }
                        var lane = new TestStreamLane(deviceIndex);
                        created.Add(lane);
                        return lane;
                    }));
            Assert.Single(created);
            Assert.True(created[0].Disposed);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ProductionFactoryUsesExplicitCanonicalDeviceAuthority()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            // Deliberately contradict the requested session. Production code
            // must use its canonical specification, not these legacy globals.
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            Tensor.CudaDeviceIndices = [7];
            var created = new List<TestStreamLane>();

            using var session = ProductionTrainingSessionFactory.Create(
                TensorPrecisionMode.Mix8_32,
                lastCommittedStep: 12,
                TensorDevice.Cuda,
                [3, 5],
                deviceIndex =>
                {
                    var lane = new TestStreamLane(deviceIndex);
                    created.Add(lane);
                    return lane;
                });

            Assert.Equal(ExecutionDeviceKind.Cuda,
                session.ExecutionSession.Options.Device);
            Assert.Equal([3, 5],
                session.ExecutionSession.Options.CudaDevices);
            Assert.Equal([3, 5], created.Select(lane => lane.DeviceIndex));
            Assert.Equal(PrecisionMode.Mix8_32,
                session.ExecutionSession.Options.Precision.Mode);
            Assert.Equal(12, session.LastCommittedStep);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousIndices;
            Tensor.ExecutionDevice = previousDevice;
        }
    }

    [Fact]
    public void ExecutionAuthorityCanPrecedeCheckpointRestoreAndBeWrappedLater()
    {
        var lanes = new List<TestStreamLane>();
        ExecutionSession execution =
            ProductionTrainingSessionFactory.CreateExecutionSession(
                TensorPrecisionMode.Mix8_32,
                TensorDevice.Cuda,
                [3, 5],
                deviceIndex =>
                {
                    var lane = new TestStreamLane(deviceIndex);
                    lanes.Add(lane);
                    return lane;
                });

        using (execution)
        {
            using IDisposable scope = execution.Enter();
            Assert.Same(execution, ExecutionSession.Current);
            Assert.Equal([3, 5], execution.Options.CudaDevices);

            using (var training = new NNtrain.Training.Execution.TrainingSession(
                execution,
                ownsExecutionSession: false,
                lastCommittedStep: 27))
            {
                Assert.Equal(27, training.LastCommittedStep);
                Assert.Same(execution, training.ExecutionSession);
            }

            Assert.False(execution.IsDisposed);
            Assert.All(lanes, lane => Assert.False(lane.Disposed));
        }

        Assert.All(lanes, lane => Assert.True(lane.Disposed));
    }

    [Fact]
    public void WikiCudaRouteCreatesSessionOwnedDataParallelEngine()
    {
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousIndices = Tensor.CudaDeviceIndices.ToArray();
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = [3, 5];
            var config = new WikiTrainingConfiguration
            {
                Device = WikiTrainingConfiguration.CudaDevice,
                DeviceIndices = [3, 5],
                BatchSize = 4,
            };
            using var session =
                WikiLanguageModelCommand.CreateCudaDataParallelSession(
                    config,
                    TensorPrecisionMode.Mix16_32,
                    static deviceIndex => new TestStreamLane(deviceIndex));

            Assert.NotNull(session);
            Assert.Equal(
                [3, 5],
                session.ExecutionSession.Options.CudaDevices);
            var model = new GptRinWikiJp(
                vocabularySize: 16,
                contextLength: 2,
                dModel: 4,
                numHeads: 1,
                dHidden: 8,
                numLayers: 1,
                rng: new Random(107));
            CudaDataParallelEngine engine = session.OwnCudaDataParallel(
                model,
                session.ExecutionSession.Options.CudaDevices);

            session.Dispose();

            Assert.True(engine.IsDisposed);
            Assert.True(session.ExecutionSession.IsDisposed);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousIndices;
        }
    }

    private sealed class TestStreamLane(int deviceIndex)
        : IStreamExecutionLane
    {
        public bool Disposed { get; private set; }
        public ExecutionDeviceKind DeviceKind => ExecutionDeviceKind.Cuda;
        public int DeviceIndex { get; } = deviceIndex;
        public IDeviceMemoryManager MemoryManager { get; } =
            new TestMemoryManager(deviceIndex);
        public IKernelCapabilitySet Capabilities { get; } =
            new CudaKernelCapabilities(
                8,
                6,
                CudaKernelFeature.TensorCores);
        public IExecutionProfiler Profiler => NullExecutionProfiler.Instance;
        public nint ComputeStreamHandle => (nint)(DeviceIndex * 2 + 1);
        public nint CommunicationStreamHandle => (nint)(DeviceIndex * 2 + 2);
        public void ActivateComputeStream() { }
        public void SynchronizeComputeStream() { }
        public void SynchronizeCommunicationStream() { }
        public T OwnResource<T>(T resource) where T : class, IDisposable
            => resource;
        public void Dispose()
        {
            Disposed = true;
            MemoryManager.Dispose();
        }
    }

    private sealed class TestMemoryManager(int deviceIndex)
        : IDeviceMemoryManager
    {
        public int DeviceIndex { get; } = deviceIndex;
        public long AllocationCount => 0;
        public long AllocatedBytes => 0;
        public void Dispose() { }
    }

    [Fact]
    public void FiniteBatchPadsInputAndIgnoresPaddedTargets()
    {
        int[] tokens =
        [
            BpeTokenizer.BosTokenId,
            17,
            BpeTokenizer.EosTokenId,
        ];

        WikiLanguageModelCommand.LanguageBatch batch =
            WikiLanguageModelCommand.CreateBatch(
                tokens,
                [0],
                orderStart: 0,
                count: 1,
                contextLength: 4);

        Assert.Equal<int>([1, 17, 0, 0], batch.Input);
        Assert.Equal<int>([17, 2, -1, -1], batch.Target);
        Assert.Equal(2, batch.ValidTargetCount);
    }

    [Fact]
    public void StreamingBatchPadsInputAndKeepsOverlapToken()
    {
        var buffer = new List<int>
        {
            BpeTokenizer.BosTokenId,
            17,
            BpeTokenizer.EosTokenId,
        };

        WikiLanguageModelCommand.LanguageBatch batch =
            WikiLanguageModelCommand.CreateStreamingBatch(
                buffer,
                batchSize: 1,
                sequenceLength: 4);

        Assert.Equal<int>([1, 17, 0, 0], batch.Input);
        Assert.Equal<int>([17, 2, -1, -1], batch.Target);
        Assert.Equal(2, batch.ValidTargetCount);
        Assert.Equal<int>([BpeTokenizer.EosTokenId], buffer);
    }

    [Fact]
    public void DefaultConfigurationCreatesForgetMemoryV3Gpt()
    {
        var config = new WikiTrainingConfiguration
        {
            ContextLength = 8,
            ModelWidth = 12,
            Heads = 3,
            HiddenSize = 20,
            Layers = 2,
            ForgetMemoryKeyWidth = 5,
            ForgetMemoryValueWidth = 7,
            ForgetMemoryRetentionMinimum = 0.3f,
            ForgetMemoryRetentionMaximum = 0.9f,
        };

        LanguageModel created = WikiLanguageModelCommand.CreateModel(
            config,
            BpeTokenizer.BaseVocabularySize);

        ForgetMemoryV3Gpt model = Assert.IsType<ForgetMemoryV3Gpt>(created);
        Assert.Equal(TensorDType.BFloat16, model.DType);
        Assert.Equal(TensorPrecisionMode.Mix16_32, model.PrecisionMode);
        Assert.All(
            model.parameters(),
            parameter => Assert.Equal(
                TensorDType.BFloat16,
                parameter.T.DType));
        Assert.Equal(5, model.KeyWidth);
        Assert.Equal(7, model.ValueWidth);
        Assert.Equal(0.3f, model.Layers[0].RetentionFloor, precision: 6);
        Assert.Equal(0.9f, model.Layers[1].RetentionFloor, precision: 6);
    }

    [Fact]
    public void DefaultWikiJsonSelectsCustomForgetMemoryV3Model()
    {
        var config = new WikiTrainingConfiguration();

        LanguageModel model = WikiLanguageModelCommand.CreateModel(
            config,
            config.VocabularySize);

        Assert.True(config.IsForgetMemoryV3Architecture());
        ForgetMemoryV3Gpt typed = Assert.IsType<ForgetMemoryV3Gpt>(model);
        Assert.Equal(TensorDType.BFloat16, typed.DType);
        Assert.Equal(TensorPrecisionMode.Mix16_32, typed.PrecisionMode);
    }

    [Fact]
    public void ExplicitDrnArchitectureCreatesForgetMemoryDrnModel()
    {
        var config = new WikiTrainingConfiguration
        {
            ModelArchitecture = "forgetmemorydrn",
            ContextLength = 8,
            ModelWidth = 12,
            Heads = 3,
            HiddenSize = 20,
            Layers = 2,
            ForgetMemoryKeyWidth = 5,
            ForgetMemoryValueWidth = 7,
        };

        LanguageModel created = WikiLanguageModelCommand.CreateModel(
            config,
            BpeTokenizer.BaseVocabularySize);

        ForgetMemoryDRNGpt model = Assert.IsType<ForgetMemoryDRNGpt>(created);
        Assert.True(model.UseDrn);
        Assert.Equal(TensorDType.BFloat16, model.DType);
        Assert.Equal(TensorPrecisionMode.Mix16_32, model.PrecisionMode);
        Assert.Equal(5, model.KeyWidth);
        Assert.Equal(7, model.ValueWidth);
    }

    [Fact]
    public void RunPrintsTheSelectedConfigurationAndEffectiveModelSettings()
    {
        string configurationPath = Path.Combine(
            Path.GetTempPath(),
            $"NNtrain-wiki-settings-{Guid.NewGuid():N}.json");
        File.WriteAllText(
            configurationPath,
            """
            {
              "task": "gpt_rin_wiki_jp",
              "dataPath": "missing-wiki-data",
              "batchSize": 7,
              "contextLength": 12,
              "modelWidth": 12,
              "heads": 3,
              "hiddenSize": 20,
              "layers": 3
            }
            """);
        try
        {
            using var output = new StringWriter();
            using var error = new StringWriter();

            int exitCode = WikiLanguageModelCommand.Run(
                configurationPath,
                generatePrompt: null,
                output,
                error);

            Assert.Equal(2, exitCode);
            Assert.Contains(
                $"configuration = {Path.GetFullPath(configurationPath)}",
                output.ToString());
            Assert.Contains("simd = enabled", output.ToString());
            Assert.Contains(
                "thread parallelism = Parallel.For",
                output.ToString());
            Assert.Contains(
                "effective training = epochs 5, batch 7, context 12",
                output.ToString());
            Assert.Contains(
                "effective model = forgetmemoryv3, vocabulary 2048, " +
                "width 12, heads 3, " +
                "hidden 20, layers 3",
                output.ToString());
            Assert.Contains(
                "matrix delta memory key 16, value 16, retention 0.5-0.99",
                output.ToString());
            Assert.Contains(
                "Wikipedia data directory was not found",
                error.ToString());
        }
        finally
        {
            File.Delete(configurationPath);
        }
    }

    [Fact]
    public void CreatesNekoMuonWithAuxiliaryAdamWForGpt()
    {
        var model = new GptRinWikiJp(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 4,
            dModel: 8,
            numHeads: 2,
            dHidden: 16,
            numLayers: 1,
            rng: new Random(2));
        var config = new WikiTrainingConfiguration
        {
            Optimizer = "nekomuon",
            LearningRate = 4e-4f,
            AuxiliaryLearningRate = 2e-4f,
            NekoMuonNewtonSchulzInterval = 7,
        };

        IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
            model,
            config);

        CompositeOptimizer composite = Assert.IsType<CompositeOptimizer>(
            optimizer);
        Assert.Collection(
            composite.Optimizers,
            child => Assert.IsType<NekoMuon>(child),
            child => Assert.IsType<AdamW>(child));
        Assert.Equal(
            config.LearningRate,
            ((NekoMuon)composite.Optimizers[0]).LearningRate);
        Assert.Equal(
            7,
            ((NekoMuon)composite.Optimizers[0])
                .CaptureState()
                .Options
                .NewtonSchulzInterval);

        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
        IReadOnlyList<float> rates = scheduler.step(0.1d);
        float factor = rates[0] / config.LearningRate;

        Assert.Equal(0.5f, factor, precision: 6);
        Assert.Equal(
            config.LearningRate * 0.5f,
            ((NekoMuon)composite.Optimizers[0]).LearningRate,
            precision: 8);
        Assert.Equal(
            config.AuxiliaryLearningRate * 0.5f,
            ((AdamW)composite.Optimizers[1]).LearningRate,
            precision: 8);
    }

    [Theory]
    [InlineData(0.1d, 0.5f)]
    [InlineData(0.2d, 1f)]
    [InlineData(0.6d, 0.5f)]
    [InlineData(1d, 1e-6f)]
    public void WikiScheduleUsesTwentyPercentWarmupThenCosine(
        double progress,
        float expected)
    {
        float actual = WarmupCosineProgressLRScheduler.CalculateFactor(
            progress,
            warmupPercent: 20f);

        Assert.Equal(expected, actual, precision: 5);
    }

    [Theory]
    [InlineData(TensorPrecisionMode.BFloat16, true)]
    [InlineData(TensorPrecisionMode.Mix16_32, false)]
    [InlineData(TensorPrecisionMode.Float32, false)]
    public void PrecisionModeControlsAdamWMomentStorage(
        TensorPrecisionMode precisionMode,
        bool expectedBFloat16Moments)
    {
        TensorDType storageDType = precisionMode.ToStorageDType();
        var model = new GptRinWikiJp(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 2,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(29),
            dtype: storageDType);
        model.SetPrecisionMode(precisionMode);
        var config = new WikiTrainingConfiguration
        {
            Optimizer = "adamw",
            LearningRate = 0.01f,
            WarmupPercent = 20f,
            PrecisionMode = TensorPrecisionModeNames.Format(precisionMode),
        };
        IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
            model,
            config);

        AdamWState state = Assert.IsType<AdamW>(optimizer).CaptureState();
        Assert.Equal(
            expectedBFloat16Moments,
            state.Options.UseBFloat16FirstMoment);
        Assert.Equal(
            expectedBFloat16Moments,
            state.Options.UseBFloat16SecondMoment);

        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
        scheduler.step(0.6d);

        Assert.Equal(
            0.005f,
            Assert.IsType<AdamW>(optimizer).LearningRate,
            precision: 7);
    }

    [Fact]
    public void WikiCheckpointRestoresCurrentModelOptimizerSchedulerAndStep()
    {
        string checkpointPath = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-wiki-resume-{Guid.NewGuid():N}.json");
        try
        {
            var config = new WikiTrainingConfiguration
            {
                CheckpointPath = checkpointPath,
                ResumeFromCheckpoint = true,
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
            LanguageModel source =
                WikiLanguageModelCommand.CreateModel(
                    config,
                    config.VocabularySize);
            IOptimizer sourceOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(source, config);
            WarmupCosineProgressLRScheduler sourceScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    sourceOptimizer,
                    config.WarmupPercent);
            sourceOptimizer.zero_grad();
            Tensor loss = nn.functional.cross_entropy(
                source.forward([1, 2], 1, 2),
                [2, 1]);
            loss.backward();
            sourceScheduler.step(1d / 3d);
            sourceOptimizer.step();
            ModuleState expectedCurrent = source.state_dict();
            ModuleState expectedBest = expectedCurrent with
            {
                Parameters = expectedCurrent.Parameters
                    .Select((parameter, parameterIndex) => parameter with
                    {
                        Values = parameter.Values
                            .Select((value, valueIndex) =>
                                parameterIndex == 0 && valueIndex == 0
                                    ? value + 0.25f
                                    : value)
                            .ToArray(),
                    })
                    .ToArray(),
            };
            var expectedShardState = new CudaAdaptiveShardState(
                CudaAdaptiveShardState.CurrentFormatVersion,
                Devices: [0, 1],
                LastAllocation: [5, 3],
                ThroughputEma: [1.25d, 0.75d],
                HasObservation: true);
            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                config,
                config.VocabularySize,
                completedEpoch: 1,
                expectedBest,
                bestLoss: 1.25f,
                bestEpoch: 1,
                source,
                sourceOptimizer,
                sourceScheduler,
                globalStep: 7,
                adaptiveCudaShardState: expectedShardState);
            WikiLanguageModelCommand.WikiModelCheckpoint manifest =
                torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                    checkpointPath);
            string bestArtifactPath =
                WikiLanguageModelCommand.GetBestModelArtifactPath(
                    checkpointPath,
                    manifest.BestArtifactSlot);
            byte[] bestArtifactPayload = File.ReadAllBytes(bestArtifactPath);
            File.WriteAllBytes(bestArtifactPath, [0]);

            LanguageModel restored =
                WikiLanguageModelCommand.CreateModel(
                    config,
                    config.VocabularySize);
            IOptimizer restoredOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(restored, config);
            WarmupCosineProgressLRScheduler restoredScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    restoredOptimizer,
                    config.WarmupPercent);
            ModuleState? bestState = null;
            float bestLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            long globalStep = 0;
            using var output = new StringWriter();

            WikiLanguageModelCommand.WikiResumePosition position =
                WikiLanguageModelCommand.RestoreTrainingCheckpoint(
                    config,
                    restored,
                    restoredOptimizer,
                    restoredScheduler,
                    ref bestState,
                    ref bestLoss,
                    ref bestEpoch,
                    ref globalStep,
                    output);

            // Restore only consumes the current artifact. The deliberately
            // invalid best artifact is not parsed until best weights are
            // actually requested.
            File.WriteAllBytes(bestArtifactPath, bestArtifactPayload);
            Assert.Equal(2, position.Epoch);
            Assert.Equal(1, bestEpoch);
            Assert.Equal(1.25f, bestLoss);
            Assert.Equal(7, globalStep);
            Assert.Null(bestState);
            Assert.NotNull(position.AdaptiveCudaShardState);
            Assert.Equal(
                expectedShardState.Devices,
                position.AdaptiveCudaShardState.Devices);
            Assert.Equal(
                expectedShardState.LastAllocation,
                position.AdaptiveCudaShardState.LastAllocation);
            Assert.Equal(
                expectedShardState.ThroughputEma,
                position.AdaptiveCudaShardState.ThroughputEma);
            Assert.True(position.AdaptiveCudaShardState.HasObservation);
            Assert.Equal(
                expectedCurrent.Parameters[0].Values,
                restored.state_dict().Parameters[0].Values);
            ModuleState lazyBest =
                WikiLanguageModelCommand.LoadBestTrainingModelState(
                    checkpointPath);
            Assert.Equal(
                expectedBest.Parameters[0].Values,
                lazyBest.Parameters[0].Values);
            Assert.Equal(
                sourceOptimizer.state_dict().StateJsonText,
                restoredOptimizer.state_dict().StateJsonText);
            Assert.Equal(
                sourceScheduler.state_dict(),
                restoredScheduler.state_dict());
            Assert.True(File.Exists(
                WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                    checkpointPath,
                    manifest.ArtifactSlot)));
        }
        finally
        {
            if (File.Exists(checkpointPath))
                File.Delete(checkpointPath);
            string safeTensorsPath =
                WikiLanguageModelCommand.GetSafeTensorsPath(
                    checkpointPath);
            if (File.Exists(safeTensorsPath))
                File.Delete(safeTensorsPath);
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
                        WikiLanguageModelCommand
                            .GetOptimizerBinaryArtifactPath(
                                checkpointPath,
                                slot,
                                optimizerIndex);
                    if (File.Exists(optimizerBinaryArtifact))
                        File.Delete(optimizerBinaryArtifact);
                }
            }
        }
    }

    [Fact]
    public void WikiMidEpochResumeRestoresDropoutRandomStateAndSeed()
    {
        string directory = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-wiki-rng-resume-{Guid.NewGuid():N}");
        Directory.CreateDirectory(directory);
        string checkpointPath = Path.Combine(directory, "checkpoint.json");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cpu;
            var config = new WikiTrainingConfiguration
            {
                CheckpointPath = checkpointPath,
                ResumeFromCheckpoint = true,
                Epochs = 2,
                ContextLength = 2,
                ModelWidth = 4,
                Heads = 1,
                HiddenSize = 8,
                Layers = 1,
                VocabularySize = BpeTokenizer.BaseVocabularySize,
                ModelArchitecture =
                    WikiTrainingConfiguration.TransformerArchitecture,
                Optimizer = WikiTrainingConfiguration.AdamWOptimizer,
                LearningRate = 0.001f,
                Dropout = 0.25f,
                Seed = 73,
            };

            static void TrainStep(
                LanguageModel model,
                IOptimizer optimizer,
                WarmupCosineProgressLRScheduler scheduler,
                int[] input,
                int[] target,
                double progress)
            {
                model.train();
                optimizer.zero_grad();
                Tensor loss = model.forward_loss(
                    input,
                    target,
                    batch_size: 1,
                    sequence_length: 2);
                loss.backward();
                scheduler.step(progress);
                optimizer.step();
            }

            LanguageModel uninterrupted =
                WikiLanguageModelCommand.CreateModel(
                    config,
                    config.VocabularySize);
            IOptimizer uninterruptedOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(
                    uninterrupted,
                    config);
            WarmupCosineProgressLRScheduler uninterruptedScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    uninterruptedOptimizer,
                    config.WarmupPercent);

            TrainStep(
                uninterrupted,
                uninterruptedOptimizer,
                uninterruptedScheduler,
                [1, 2],
                [2, 3],
                progress: 0.25d);
            WikiLanguageModelCommand.SaveTrainingCheckpoint(
                config,
                config.VocabularySize,
                completedEpoch: 0,
                uninterrupted.state_dict(),
                bestLoss: 10f,
                bestEpoch: 0,
                uninterrupted,
                uninterruptedOptimizer,
                uninterruptedScheduler,
                globalStep: 1,
                currentEpoch: 1,
                completedBatchesInEpoch: 1,
                currentLossSum: 2d,
                currentTargetCount: 2);

            InvalidDataException seedMismatch =
                Assert.Throws<InvalidDataException>(() =>
                    WikiLanguageModelCommand.ResolvePrecisionForTraining(
                        config with { Seed = config.Seed + 1 }));
            Assert.Contains("checkpoint seed", seedMismatch.Message);

            TrainStep(
                uninterrupted,
                uninterruptedOptimizer,
                uninterruptedScheduler,
                [3, 4],
                [4, 5],
                progress: 0.5d);
            ModuleState expectedModel = uninterrupted.state_dict();
            OptimizerStateDictionary expectedOptimizer =
                uninterruptedOptimizer.state_dict();
            LRSchedulerStateDictionary expectedScheduler =
                uninterruptedScheduler.state_dict();

            LanguageModel resumed = WikiLanguageModelCommand.CreateModel(
                config,
                config.VocabularySize);
            IOptimizer resumedOptimizer =
                WikiLanguageModelCommand.CreateOptimizer(resumed, config);
            WarmupCosineProgressLRScheduler resumedScheduler =
                lr_scheduler.WarmupCosineProgressLR(
                    resumedOptimizer,
                    config.WarmupPercent);
            ModuleState? bestState = null;
            float bestLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            long globalStep = 0;
            using var output = new StringWriter();

            WikiLanguageModelCommand.WikiResumePosition position =
                WikiLanguageModelCommand.RestoreTrainingCheckpoint(
                    config,
                    resumed,
                    resumedOptimizer,
                    resumedScheduler,
                    ref bestState,
                    ref bestLoss,
                    ref bestEpoch,
                    ref globalStep,
                    output);
            Assert.Equal(1, position.Epoch);
            Assert.Equal(1, position.CompletedBatches);
            Assert.Equal(1, globalStep);

            TrainStep(
                resumed,
                resumedOptimizer,
                resumedScheduler,
                [3, 4],
                [4, 5],
                progress: 0.5d);

            ModuleState actualModel = resumed.state_dict();
            Assert.Equal(
                expectedModel.Parameters.Length,
                actualModel.Parameters.Length);
            for (int index = 0; index < expectedModel.Parameters.Length;
                index++)
            {
                Assert.Equal(
                    expectedModel.Parameters[index].Values,
                    actualModel.Parameters[index].Values);
            }
            Assert.Equal(
                expectedOptimizer.StateJsonText,
                resumedOptimizer.state_dict().StateJsonText);
            Assert.Equal(
                expectedScheduler,
                resumedScheduler.state_dict());
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void DatasetContinuationSplitsDocumentAndGeneratesFromFirstHalf()
    {
        const string document = "日本の歴史を説明します。ここからが文章の後半です。";
        BpeTokenizer tokenizer = BpeTokenizer.Train(
            Enumerable.Repeat(document, 4),
            vocabularySize: 280);
        var model = new GptRinWikiJp(
            tokenizer.VocabularySize,
            contextLength: 8,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(3));
        var config = new WikiTrainingConfiguration
        {
            ContextLength = 8,
            ModelWidth = 4,
            Heads = 1,
            HiddenSize = 8,
            Layers = 1,
            MaxNewTokens = 2,
            Temperature = 0f,
            TopK = 1,
        };

        WikiLanguageModelCommand.DatasetContinuation result =
            WikiLanguageModelCommand.CreateDatasetContinuation(
                model,
                tokenizer,
                [document],
                config,
                new Random(5));

        Assert.Equal(document.Length, result.DocumentLength);
        Assert.Equal(document.Length / 2, result.SplitIndex);
        Assert.Equal(document[..result.SplitIndex], result.PromptTail);
        Assert.StartsWith(
            document[result.SplitIndex..],
            result.ExpectedContinuation);
        Assert.NotNull(result.GeneratedContinuation);
        Assert.True(model.IsTraining);
    }
}
