using NNtrain;
using Xunit;

public sealed class WikiLanguageModelCommandTests
{
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
    public void DefaultConfigurationCreatesFrogetMemoryV2Gpt()
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

        IWikiLanguageModel created = WikiLanguageModelCommand.CreateModel(
            config,
            BpeTokenizer.BaseVocabularySize);

        FrogetMemoryV2Gpt model = Assert.IsType<FrogetMemoryV2Gpt>(created);
        Assert.Equal(5, model.KeyWidth);
        Assert.Equal(7, model.ValueWidth);
        Assert.Equal(0.3f, model.Layers[0].RetentionFloor, precision: 6);
        Assert.Equal(0.9f, model.Layers[1].RetentionFloor, precision: 6);
    }

    [Fact]
    public void DefaultWikiJsonSelectsCustomForgetMemoryV2Model()
    {
        var config = new WikiTrainingConfiguration();

        IWikiLanguageModel model = WikiLanguageModelCommand.CreateModel(
            config,
            config.VocabularySize);

        Assert.True(config.IsForgetMemoryV2Architecture());
        Assert.IsType<FrogetMemoryV2Gpt>(model);
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
                "effective model = forgetmemoryv2, vocabulary 2048, " +
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

        float factor = WikiLanguageModelCommand.SetScheduledLearningRates(
            optimizer,
            config,
            overallProgress: 0.1d);

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
        float actual = WikiLanguageModelCommand.CalculateLearningRateFactor(
            progress,
            warmupPercent: 20f);

        Assert.Equal(expected, actual, precision: 5);
    }

    [Fact]
    public void WikiScheduleUpdatesSingleAdamW()
    {
        var model = new GptRinWikiJp(
            BpeTokenizer.BaseVocabularySize,
            contextLength: 2,
            dModel: 4,
            numHeads: 1,
            dHidden: 8,
            numLayers: 1,
            rng: new Random(29));
        var config = new WikiTrainingConfiguration
        {
            Optimizer = "adamw",
            LearningRate = 0.01f,
            WarmupPercent = 20f,
            AdamWUseBFloat16FirstMoment = true,
            AdamWUseBFloat16SecondMoment = true,
        };
        IOptimizer optimizer = WikiLanguageModelCommand.CreateOptimizer(
            model,
            config);

        AdamWState state = Assert.IsType<AdamW>(optimizer).CaptureState();
        Assert.True(state.Options.UseBFloat16FirstMoment);
        Assert.True(state.Options.UseBFloat16SecondMoment);

        WikiLanguageModelCommand.SetScheduledLearningRates(
            optimizer,
            config,
            overallProgress: 0.6d);

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
            IWikiLanguageModel source =
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
                    .Select(parameter => parameter with
                    {
                        Values = parameter.Values.ToArray(),
                    })
                    .ToArray(),
            };
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
                globalStep: 7);

            IWikiLanguageModel restored =
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

            Assert.Equal(2, position.Epoch);
            Assert.Equal(1, bestEpoch);
            Assert.Equal(1.25f, bestLoss);
            Assert.Equal(7, globalStep);
            Assert.NotNull(bestState);
            Assert.Equal(
                expectedCurrent.Parameters[0].Values,
                restored.state_dict().Parameters[0].Values);
            Assert.Equal(
                sourceOptimizer.state_dict().StateJson,
                restoredOptimizer.state_dict().StateJson);
            Assert.Equal(
                sourceScheduler.state_dict(),
                restoredScheduler.state_dict());
            Assert.True(File.Exists(
                WikiLanguageModelCommand.GetSafeTensorsPath(
                    checkpointPath)));
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
