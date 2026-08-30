using System.Diagnostics;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;
using NNtrain.Training.Optimization;

namespace NNtrain;

internal static partial class WikiLanguageModelCommand
{
    /// <summary>
    /// Formats training wall time for the periodic progress line. Runs shorter
    /// than an hour stay in seconds so short profiling runs remain readable.
    /// </summary>
    private static string FormatElapsed(TimeSpan elapsed)
        => elapsed.TotalHours >= 1d
            ? $"{(int)elapsed.TotalHours}:{elapsed.Minutes:00}:{elapsed.Seconds:00}"
            : elapsed.TotalMinutes >= 1d
                ? $"{elapsed.Minutes}:{elapsed.Seconds:00}"
                : $"{elapsed.TotalSeconds:F1} sec";

    private static float GetLossGraphResumePosition(
        int resumeEpoch,
        long completedUnits,
        long totalUnits,
        int totalEpochs)
    {
        double epochProgress = totalUnits <= 0
            ? 0d
            : Math.Clamp((double)completedUnits / totalUnits, 0d, 1d);
        return (float)Math.Clamp(
            resumeEpoch - 1d + epochProgress,
            0d,
            totalEpochs);
    }

    internal static int Run(
        string configurationPath,
        string? generatePrompt,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false,
        bool resumeFromCheckpoint = false)
        => RunWithErrorHandling(
            () =>
            {
                CanonicalTrainingSpec loaded =
                    ConfigLoader.Load(configurationPath);
                CanonicalWikiTrainingSpec canonical = loaded
                    as CanonicalWikiTrainingSpec
                    ?? throw new InvalidDataException(
                        "Configuration is not a wiki-language-model task.");
                return RunCore(
                    canonical,
                    generatePrompt,
                    output,
                    error,
                    openLossGraph,
                    resumeFromCheckpoint);
            },
            error);

    internal static int Run(
        CanonicalWikiTrainingSpec configuration,
        string? generatePrompt,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false,
        bool resumeFromCheckpoint = false)
        => RunWithErrorHandling(
            () => RunCore(
                configuration,
                generatePrompt,
                output,
                error,
                openLossGraph,
                resumeFromCheckpoint),
            error);

    private static int RunCore(
        CanonicalWikiTrainingSpec canonical,
        string? generatePrompt,
        TextWriter output,
        TextWriter error,
        bool openLossGraph,
        bool resumeFromCheckpoint)
    {
        ArgumentNullException.ThrowIfNull(canonical);
        WikiTrainingConfiguration config = canonical.Configuration;
        if (resumeFromCheckpoint)
            config = config with { ResumeFromCheckpoint = true };
        torch.manual_seed(config.Seed);
        output.WriteLine($"configuration = {canonical.ConfigurationPath}");
        Tensor.SimdEnabled = config.UseSimd;
        Tensor.MaxDegreeOfParallelism = config.MaxDegreeOfParallelism;
        Tensor.ExecutionDevice = config.GetExecutionDevice();
        Tensor.CudaDeviceIndices = config.DeviceIndices ?? [config.DeviceIndex];
        output.WriteLine(
            $"simd = {(config.UseSimd ? "enabled" : "disabled")}, " +
            $"Vector256 hardware = " +
            $"{(Tensor.IsSimdHardwareAccelerated ? "available" : "unavailable")}");
        output.WriteLine(
            $"thread parallelism = Parallel.For, workers = " +
            $"{Tensor.EffectiveMaxDegreeOfParallelism}" +
            (config.MaxDegreeOfParallelism == 0 ? " (automatic)" : ""));
        output.WriteLine(
            $"device = {config.Device.ToLowerInvariant()}" +
            (Tensor.ExecutionDevice == TensorDevice.Cuda
                ? $" [{string.Join(",", Tensor.CudaDeviceIndices)}] " +
                    $"({Tensor.ExecutionDeviceName}; " +
                    "ForgetMemory training kernels CUDA, BF16 storage)"
                : string.Empty));
        if (Tensor.ExecutionDevice == TensorDevice.Cuda
            && Tensor.CudaDeviceIndices.Count > 1)
        {
            output.WriteLine(
                "CUDA data parallel = " +
                (config.AdaptiveCudaSharding
                    ? $"adaptive EMA (alpha " +
                        $"{config.CudaShardEmaAlpha:G}, minimum " +
                        $"{config.CudaMinimumRelativeShardSize:P0} of " +
                        $"even share, max shift " +
                        $"{config.CudaMaximumBatchAdjustmentPerStep}/step)"
                    : "fixed even shards"));
        }
        if (generatePrompt is not null)
            return GenerateOnly(config, generatePrompt, output);

        return Train(
            config,
            Path.ChangeExtension(canonical.ConfigurationPath, ".loss.html"),
            output,
            error,
            openLossGraph);
    }

    private static int RunWithErrorHandling(
        Func<int> run,
        TextWriter error)
    {
        try
        {
            return run();
        }
        catch (Exception exception) when (
            exception is not OutOfMemoryException
            and not StackOverflowException
            and not OperationCanceledException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }

    private static CudaAdaptiveShardingOptions
        CreateCudaAdaptiveShardingOptions(WikiTrainingConfiguration config)
        => new()
        {
            Enabled = config.AdaptiveCudaSharding,
            EmaAlpha = config.CudaShardEmaAlpha,
            MinimumRelativeShardSize =
                config.CudaMinimumRelativeShardSize,
            MaximumBatchAdjustmentPerStep =
                config.CudaMaximumBatchAdjustmentPerStep,
            GraphCacheBudgetBytes = checked(
                (long)config.CudaGraphCacheBudgetMiB * 1024L * 1024L),
        };

    internal static TrainingSession? CreateCudaDataParallelSession(
        WikiTrainingConfiguration config,
        TensorPrecisionMode precisionMode,
        Func<int, NNtrain.Runtime.Execution.IExecutionLane>?
            cudaLaneFactory = null)
    {
        if (Tensor.ExecutionDevice != TensorDevice.Cuda
            || Tensor.CudaDeviceIndices.Count <= 1
            || config.BatchSize <= 1)
        {
            return null;
        }

        return ProductionTrainingSessionFactory.Create(
            precisionMode,
            lastCommittedStep: -1,
            config.GetExecutionDevice(),
            config.DeviceIndices ?? [config.DeviceIndex],
            cudaLaneFactory);
    }

    private static int Train(
        WikiTrainingConfiguration config,
        string lossGraphPath,
        TextWriter output,
        TextWriter error,
        bool openLossGraph)
    {
        // Resolve resume metadata before any corpus scan or tokenizer work so
        // an incompatible checkpoint fails before expensive data processing.
        WikiPrecisionSelection precision =
            ResolvePrecisionForTraining(config);
        TensorPrecisionMode precisionMode = precision.Mode;
        PreflightCudaOptimizer(config, precisionMode);
        WriteEffectiveTrainingConfiguration(
            config,
            precisionMode,
            precision.Bfp8BlockSize,
            output);
        if (!Directory.Exists(config.DataPath))
        {
            throw new DirectoryNotFoundException(
                $"Wikipedia data directory was not found at " +
                $"'{config.DataPath}'.");
        }

        BpeTokenizer? tokenizer = null;
        if (File.Exists(config.TokenizerPath))
        {
            BpeTokenizer existing = tokenizers.load_bpe(config.TokenizerPath);
            if (existing.VocabularySize == config.VocabularySize)
            {
                tokenizer = existing;
                output.WriteLine($"tokenizer = loaded {config.TokenizerPath}");
            }
            else
            {
                output.WriteLine(
                    $"tokenizer vocabulary {existing.VocabularySize} does " +
                    $"not match configured {config.VocabularySize}; " +
                    "retraining tokenizer");
            }
        }
        if (tokenizer is null)
        {
            output.WriteLine(
                $"training BPE tokenizer: target vocabulary " +
                $"{config.VocabularySize}, up to " +
                $"{config.TokenizerTrainingDocuments} documents / " +
                $"{config.TokenizerTrainingBytes} bytes");
            var timer = Stopwatch.StartNew();
            tokenizer = tokenizers.train_bpe(
                ReadDocuments(
                    config.DataPath,
                    config.TextColumn,
                    config.TokenizerTrainingDocuments),
                config.VocabularySize,
                config.TokenizerTrainingBytes);
            tokenizer.save(config.TokenizerPath);
            timer.Stop();
            output.WriteLine(
                $"tokenizer = saved {config.TokenizerPath}, " +
                $"vocabulary {tokenizer.VocabularySize}, " +
                $"{timer.Elapsed.TotalSeconds:F2} sec");
        }

        if (config.MaxTrainingTokens == 0)
        {
            return TrainAllData(
                config,
                tokenizer,
                lossGraphPath,
                output,
                error,
                openLossGraph,
                precision);
        }

        output.WriteLine("loading and tokenizing Wikipedia documents...");
        TrainingCorpus corpus = LoadTrainingCorpus(config, tokenizer, output);
        int[] tokens = corpus.Tokens;
        int sequenceCount = TrainingRunner.DivideRoundUp(
            tokens.Length - 1,
            config.ContextLength);
        if (sequenceCount < 2)
        {
            throw new InvalidDataException(
                "The selected Wikipedia data does not contain two complete " +
                "training sequences.");
        }

        int validationSequences = config.ValidationFraction == 0f
            ? 0
            : Math.Max(
                1,
                (int)MathF.Floor(sequenceCount * config.ValidationFraction));
        validationSequences = Math.Min(validationSequences, sequenceCount - 1);
        int trainingSequences = sequenceCount - validationSequences;
        output.WriteLine(
            $"tokens = {tokens.Length:N0}, sequences = " +
            $"{trainingSequences:N0} train + " +
            $"{validationSequences:N0} validation, context " +
            $"{config.ContextLength}");
        output.WriteLine(
            $"dataset continuation = every " +
            $"{config.DatasetSampleEverySteps:N0} steps, sample pool " +
            $"{corpus.SampleDocuments.Length}");

        using ExecutionSession executionSession =
            ProductionTrainingSessionFactory.CreateExecutionSession(
                precisionMode,
                config.GetExecutionDevice(),
                config.DeviceIndices ?? [config.DeviceIndex]);
        using IDisposable executionScope = executionSession.Enter();
        LanguageModel model = CreateModel(
            config,
            tokenizer.VocabularySize,
            precisionMode,
            precision.StorageDType,
            precision.Bfp8BlockSize);
        if (Tensor.ExecutionDevice == TensorDevice.Cuda
            && model is Module modelModule)
        {
            modelModule.to(TensorDevice.Cuda);
        }
        OptimizerBundle optimizer = CreateOptimizerBundle(model, config);
        Parameter[] trainingParameters = model.parameters().ToArray();
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
        output.WriteLine(
            $"model = {model.GetType().Name} " +
            $"(custom {config.ModelArchitecture}), parameters " +
            $"{model.parameters().Sum(parameter => (long)parameter.T.Numel):N0}, " +
            $"width {config.ModelWidth}, heads {config.Heads}, " +
            $"hidden {config.HiddenSize}, layers {config.Layers}, " +
            $"context {config.ContextLength}, batch {config.BatchSize}, " +
            $"precision {FormatPrecisionMode(precisionMode)}");
        WriteOptimizerSummary(model, config, output);

        int[] order = Enumerable.Range(0, trainingSequences).ToArray();
        var sampleRandom = new Random(config.Seed ^ 0x6A09E667);
        long globalStep = 0;
        int batchesPerEpoch = TrainingRunner.DivideRoundUp(
            trainingSequences,
            config.BatchSize);
        int fullBatchesPerEpoch = trainingSequences / config.BatchSize;
        int updatesPerEpoch = TrainingRunner.DivideRoundUp(
            fullBatchesPerEpoch,
            config.GradientAccumulationSteps)
            + (trainingSequences % config.BatchSize == 0 ? 0 : 1);
        long totalTrainingSteps = checked(
            (long)config.Epochs * updatesPerEpoch);
        ModuleState? bestState = null;
        float bestLoss = float.PositiveInfinity;
        int bestEpoch = 0;
        WikiResumePosition resume = RestoreTrainingCheckpoint(
            config,
            model,
            optimizer,
            scheduler,
            ref bestState,
            ref bestLoss,
            ref bestEpoch,
            ref globalStep,
            output);
        double checkpointEpoch = GetLossGraphResumePosition(
            resume.Epoch,
            resume.CompletedBatches,
            batchesPerEpoch,
            config.Epochs);
        TrainingMetricReporter metricReporter = TrainingMetricReporter.Open(
            lossGraphPath,
            config.Epochs,
            config.ResumeFromCheckpoint,
            config.ResumeFromCheckpoint ? globalStep : -1,
            checkpointEpoch,
            config.ShowLossGraph);
        output.WriteLine($"metrics = {metricReporter.SidecarPath}");
        if (config.ShowLossGraph)
        {
            output.WriteLine(
                $"loss graph = {metricReporter.HtmlPath}, every " +
                $"{config.GraphUpdateSteps} step(s) or epoch end");
            if (openLossGraph)
                metricReporter.TryOpenHtml(error);
        }
        using TrainingSession trainingSession = new(
            executionSession,
            ownsExecutionSession: false,
            lastCommittedStep: globalStep);
        trainingSession.OwnOptimizer(optimizer);
        CudaDataParallelEngine? dataParallelEngine =
            Tensor.ExecutionDevice == TensorDevice.Cuda
                && Tensor.CudaDeviceIndices.Count > 0
                && config.BatchSize > 1
                ? trainingSession.OwnCudaDataParallel(
                    model,
                    Tensor.CudaDeviceIndices,
                    CreateCudaAdaptiveShardingOptions(config))
                : null;
        if (dataParallelEngine is not null
            && resume.AdaptiveCudaShardState is { } fixedShardState)
        {
            dataParallelEngine.RestoreAdaptiveShardingState(fixedShardState);
            output.WriteLine(
                "resume CUDA shard EMA = restored from checkpoint");
        }
        var stepExecutor = new TrainingStepExecutor(trainingSession);
        var microBatchCursor = new FixedWikiTrainingDataCursor(
            tokens,
            order,
            config.BatchSize,
            config.ContextLength);
        var batchCursor = new FixedWikiTrainingUpdateCursor(
            microBatchCursor,
            config.GradientAccumulationSteps);
        var stepOperations = new FixedWikiTrainingStepOperations(
            model,
            optimizer,
            trainingParameters,
            dataParallelEngine,
            scheduler,
            metricReporter,
            config.GraphUpdateSteps,
            totalTrainingSteps,
            config.BatchSize,
            config.ContextLength,
            globalStep);
        var cursorStepOperations =
            new CursorTrainingStepOperations<WikiTrainingUpdate>(
                batchCursor,
                stepOperations);
        var runProgress = new TrainingProgress();

        using NoGcTrainingWindow noGcWindow =
            TrainingRunner.BeginNoGcTrainingWindow();
        foreach (TrainingEpoch epochRun in TrainingRunner.Epochs(
            resume.Epoch,
            config.Epochs,
            resume.CompletedBatches))
        {
            int epoch = epochRun.Number;
            TrainingRunner.Shuffle(
                order.AsSpan(),
                new Random(TrainingRunner.CombineSeed(config.Seed, epoch)));
            model.train();
            double epochStartingLoss = epoch == resume.Epoch
                ? resume.LossSum
                : 0d;
            long epochStartingTargets = epoch == resume.Epoch
                ? resume.TargetCount
                : 0;
            int batchTotal = batchesPerEpoch;
            int firstBatch = epochRun.ResumeUnit;
            batchCursor.StartEpoch(firstBatch);
            stepOperations.StartEpoch(
                epoch,
                batchTotal,
                firstBatch,
                epochStartingLoss,
                epochStartingTargets);
            var timer = Stopwatch.StartNew();

            while (stepOperations.CompletedBatches < batchTotal)
            {
                TrainingStepState committedStep = stepExecutor.Execute(
                    checked(globalStep + 1),
                    cursorStepOperations);
                globalStep = committedStep.GlobalStep;
                noGcWindow.Pulse();
                float lossValue = stepOperations.LossValue;
                float gradientNorm = stepOperations.GradientNorm;
                IReadOnlyList<float> learningRates =
                    stepOperations.LearningRates;
                int completedBatches = stepOperations.CompletedBatches;
                bool epochEnd = completedBatches == batchTotal;
                if (TrainingRunner.ShouldSaveCheckpoint(
                    completedBatches,
                    batchTotal))
                {
                    ProductionTrainingSessionFactory
                        .EnsureCanPublishCheckpoint(
                            trainingSession,
                            globalStep);
                    SaveTrainingCheckpoint(
                        config,
                        tokenizer.VocabularySize,
                        epoch - 1,
                        bestState ?? EmptyModuleState(),
                        bestLoss,
                        bestEpoch,
                        model,
                        optimizer,
                        scheduler,
                        globalStep,
                        currentEpoch: epoch,
                        completedBatchesInEpoch: completedBatches,
                        currentLossSum: stepOperations.TotalLoss,
                        currentTargetCount:
                            stepOperations.CompletedTargets,
                        adaptiveCudaShardState: dataParallelEngine?
                            .CaptureAdaptiveShardingState());
                    bestState = null;
                    output.WriteLine(
                        $"training checkpoint = {config.CheckpointPath} " +
                        $"at epoch " +
                        $"{epoch - 1d + (double)completedBatches / batchTotal:F1}");
                    string snapshotPath = CheckpointSnapshot.Save(
                        config.CheckpointPath,
                        model.GetType().Name,
                        epoch - 1d
                            + (double)completedBatches / batchTotal,
                        model);
                    output.WriteLine(
                        $"model snapshot = {snapshotPath}");
                }
                if (globalStep % config.LogEveryBatches == 0
                    || epochEnd)
                {
                    output.WriteLine(
                        $"epoch {epoch}, step {globalStep:N0}, " +
                        $"batches {completedBatches}/{batchTotal}, " +
                        $"accumulation {stepOperations.MicroBatchCount}/" +
                        $"{config.GradientAccumulationSteps}, " +
                        $"loss = {lossValue:F6}, " +
                        $"lr = {string.Join('/', learningRates.Select(rate => $"{rate:G6}"))}, " +
                        $"grad norm = {gradientNorm:G6}, " +
                        $"clip = {MathF.Min(1f, 1f / MathF.Max(gradientNorm, 1e-12f)):G6}, " +
                        runProgress.Describe(
                            totalTrainingSteps == 0
                                ? 0d
                                : (double)globalStep / totalTrainingSteps) +
                        FormatOptimizerDiagnostics(optimizer));
                }
                if (!epochEnd)
                {
                    ProductionTrainingSessionFactory
                        .EnsureCanPublishCheckpoint(
                            trainingSession,
                            globalStep);
                    RunDatasetContinuationAfterCommittedStep(
                        globalStep,
                        model,
                        tokenizer,
                        corpus.SampleDocuments,
                        config,
                        sampleRandom,
                        output,
                        error);
                }
            }

            float trainingLoss = (float)(
                stepOperations.TotalLoss
                    / stepOperations.CompletedTargets);
            float validationLoss = validationSequences == 0
                ? trainingLoss
                : Evaluate(
                    model,
                    tokens,
                    trainingSequences,
                    validationSequences,
                    config);
            timer.Stop();
            output.WriteLine(
                $"epoch {epoch}, train loss = {trainingLoss:F6}, " +
                $"validation loss = {validationLoss:F6}, " +
                $"time = {timer.Elapsed.TotalSeconds:F2} sec");
            float graphTrainingLoss = stepOperations.GraphWindowTargets == 0
                ? trainingLoss
                : stepOperations.GraphWindowLoss
                    / stepOperations.GraphWindowTargets;
            metricReporter.AppendCommittedEpochLosses(
                globalStep,
                epoch,
                graphTrainingLoss,
                validationLoss);

            if (bestEpoch == 0 || validationLoss < bestLoss)
            {
                bestLoss = validationLoss;
                bestEpoch = epoch;
                bestState = EmptyModuleState();
            }
            ProductionTrainingSessionFactory
                .EnsureCanPublishCheckpoint(
                    trainingSession,
                    globalStep);
            SaveTrainingCheckpoint(
                config,
                tokenizer.VocabularySize,
                epoch,
                bestState ?? EmptyModuleState(),
                bestLoss,
                bestEpoch,
                model,
                 optimizer,
                 scheduler,
                 globalStep,
                 adaptiveCudaShardState: dataParallelEngine?
                     .CaptureAdaptiveShardingState());
            bestState = null;
            string epochSnapshotPath = CheckpointSnapshot.Save(
                config.CheckpointPath,
                model.GetType().Name,
                epoch,
                model);
            output.WriteLine(
                $"model snapshot = {epochSnapshotPath}");
            ProductionTrainingSessionFactory
                .EnsureCanPublishCheckpoint(
                    trainingSession,
                    globalStep);
            RunDatasetContinuationAfterCommittedStep(
                globalStep,
                model,
                tokenizer,
                corpus.SampleDocuments,
                config,
                sampleRandom,
                output,
                error);
        }

        if (bestEpoch == 0)
            throw new InvalidOperationException("Training did not produce a model state.");
        LoadBestTrainingModelInto(config.CheckpointPath, model);
        output.WriteLine(
            $"best model = epoch {bestEpoch}, validation loss " +
            $"{bestLoss:F6}");
        output.WriteLine($"checkpoint = {config.CheckpointPath}");
        WriteFinalDatasetContinuation(
            model,
            tokenizer,
            corpus.SampleDocuments,
            config,
            sampleRandom,
            globalStep,
            output);
        return 0;
    }

    private static int GenerateOnly(
        WikiTrainingConfiguration config,
        string prompt,
        TextWriter output)
    {
        if (!File.Exists(config.TokenizerPath))
        {
            throw new FileNotFoundException(
                "BPE tokenizer file was not found.",
                config.TokenizerPath);
        }
        if (!File.Exists(config.CheckpointPath))
        {
            throw new FileNotFoundException(
                "Wiki model checkpoint was not found.",
                config.CheckpointPath);
        }

        BpeTokenizer tokenizer = tokenizers.load_bpe(config.TokenizerPath);
        WikiModelCheckpoint checkpoint = LoadCheckpoint(config.CheckpointPath);
        if (checkpoint.VocabularySize != tokenizer.VocabularySize)
        {
            throw new InvalidDataException(
                "Checkpoint and tokenizer vocabulary sizes do not match.");
        }

        TensorPrecisionMode checkpointMode =
            GetCheckpointPrecisionMode(checkpoint);
        TensorDType checkpointDType = checkpointMode.ToStorageDType();
        bool configuredPrecisionDiffers =
            config.GetPrecisionMode() != checkpointMode;

        using ExecutionSession executionSession =
            ProductionTrainingSessionFactory.CreateExecutionSession(
                checkpointMode,
                config.GetExecutionDevice(),
                config.DeviceIndices ?? [config.DeviceIndex]);
        using IDisposable executionScope = executionSession.Enter();
        LanguageModel model = CreateModel(
            checkpoint,
            config.Seed,
            config.Bfp8BlockSize);
        LoadGenerationModelInto(
            checkpoint,
            config.CheckpointPath,
            model);
        if (Tensor.ExecutionDevice == TensorDevice.Cuda
            && model is Module modelModule)
        {
            modelModule.to(TensorDevice.Cuda);
        }
        output.WriteLine(
            $"checkpoint = epoch {checkpoint.Epoch}, validation loss " +
            $"{checkpoint.ValidationLoss:F6}");
        output.WriteLine(
            $"checkpoint model = {GetCheckpointArchitecture(checkpoint)}, " +
            $"vocabulary {checkpoint.VocabularySize}, " +
            $"width {checkpoint.ModelWidth}, heads {checkpoint.Heads}, " +
            $"hidden {checkpoint.HiddenSize}, layers {checkpoint.Layers}, " +
            $"context {checkpoint.ContextLength}, " +
            $"precision {FormatPrecisionMode(checkpointMode)}");
        if (!CheckpointArchitectureMatchesConfiguration(checkpoint, config)
            || configuredPrecisionDiffers)
        {
            output.WriteLine(
                "note: --generate uses the architecture and precision mode stored " +
                "in the checkpoint. JSON model settings take effect when " +
                "a new training run starts and its checkpoint is saved.");
        }
        WriteGeneration(model, tokenizer, prompt, config, output);
        return 0;
    }

    private static int TrainAllData(
        WikiTrainingConfiguration config,
        BpeTokenizer tokenizer,
        string lossGraphPath,
        TextWriter output,
        TextWriter error,
        bool openLossGraph,
        WikiPrecisionSelection precision)
    {
        TensorPrecisionMode precisionMode = precision.Mode;
        long availableDocuments = WikiParquetCorpus.CountRowsAsync(
            config.DataPath).GetAwaiter().GetResult();
        long documentsPerEpoch = config.MaxTrainingDocuments == 0
            ? availableDocuments
            : Math.Min(availableDocuments, config.MaxTrainingDocuments);
        output.WriteLine(
            $"streaming corpus = {documentsPerEpoch:N0} documents/epoch, " +
            $"{FormatDocumentTokenLimit(config.MaxDocumentTokens)}");

        using ExecutionSession executionSession =
            ProductionTrainingSessionFactory.CreateExecutionSession(
                precisionMode,
                config.GetExecutionDevice(),
                config.DeviceIndices ?? [config.DeviceIndex]);
        using IDisposable executionScope = executionSession.Enter();
        LanguageModel model = CreateModel(
            config,
            tokenizer.VocabularySize,
            precisionMode,
            precision.StorageDType,
            precision.Bfp8BlockSize);
        if (Tensor.ExecutionDevice == TensorDevice.Cuda
            && model is Module modelModule)
        {
            modelModule.to(TensorDevice.Cuda);
        }
        OptimizerBundle optimizer = CreateOptimizerBundle(model, config);
        Parameter[] trainingParameters = model.parameters().ToArray();
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                config.WarmupPercent);
        output.WriteLine(
            $"model = {model.GetType().Name} " +
            $"(custom {config.ModelArchitecture}), parameters " +
            $"{model.parameters().Sum(parameter => (long)parameter.T.Numel):N0}, " +
            $"width {config.ModelWidth}, heads {config.Heads}, " +
            $"hidden {config.HiddenSize}, layers {config.Layers}, " +
            $"context {config.ContextLength}, batch {config.BatchSize}, " +
            $"precision {FormatPrecisionMode(precisionMode)}");
        WriteOptimizerSummary(model, config, output);

        var sampleDocuments = new List<string>(config.DatasetSamplePoolSize);
        var reservoirRandom = new Random(
            config.Seed ^ unchecked((int)0xBB67AE85));
        var generationRandom = new Random(config.Seed ^ 0x6A09E667);
        int eligibleSampleDocuments = 0;
        long globalStep = 0;
        ModuleState? bestState = null;
        float bestLoss = float.PositiveInfinity;
        int bestEpoch = 0;
        WikiResumePosition resume = RestoreTrainingCheckpoint(
            config,
            model,
            optimizer,
            scheduler,
            ref bestState,
            ref bestLoss,
            ref bestEpoch,
            ref globalStep,
            output);
        double checkpointEpoch = GetLossGraphResumePosition(
            resume.Epoch,
            resume.CompletedDocuments,
            documentsPerEpoch,
            config.Epochs);
        TrainingMetricReporter metricReporter = TrainingMetricReporter.Open(
            lossGraphPath,
            config.Epochs,
            config.ResumeFromCheckpoint,
            config.ResumeFromCheckpoint ? globalStep : -1,
            checkpointEpoch,
            config.ShowLossGraph);
        output.WriteLine($"metrics = {metricReporter.SidecarPath}");
        if (config.ShowLossGraph)
        {
            output.WriteLine(
                $"loss graph = {metricReporter.HtmlPath}, every " +
                $"{config.GraphUpdateSteps} step(s) or epoch end");
            if (openLossGraph)
                metricReporter.TryOpenHtml(error);
        }
        using TrainingSession trainingSession = new(
            executionSession,
            ownsExecutionSession: false,
            lastCommittedStep: globalStep);
        trainingSession.OwnOptimizer(optimizer);
        CudaDataParallelEngine? dataParallelEngine =
            Tensor.ExecutionDevice == TensorDevice.Cuda
                && Tensor.CudaDeviceIndices.Count > 0
                && config.BatchSize > 1
                ? trainingSession.OwnCudaDataParallel(
                    model,
                    Tensor.CudaDeviceIndices,
                    CreateCudaAdaptiveShardingOptions(config))
                : null;
        if (dataParallelEngine is not null
            && resume.AdaptiveCudaShardState is { } streamingShardState)
        {
            dataParallelEngine.RestoreAdaptiveShardingState(
                streamingShardState);
            output.WriteLine(
                "resume CUDA shard EMA = restored from checkpoint");
        }
        var stepExecutor = new TrainingStepExecutor(trainingSession);
        var runProgress = new TrainingProgress();
        // Keep the fixed-step graph window continuous across epoch
        // boundaries. Flushing it at every boundary compared a short tail
        // sample with a short head sample and exaggerated corpus-order shifts.
        double graphWindowLoss = 0d;
        long graphWindowTargets = 0;

        using NoGcTrainingWindow noGcWindow =
            TrainingRunner.BeginNoGcTrainingWindow();
        foreach (TrainingEpoch epochRun in TrainingRunner.Epochs(
            resume.Epoch,
            config.Epochs,
            resume.CompletedBatches))
        {
            int epoch = epochRun.Number;
            model.train();
            var buffer = new List<int>(
                config.BatchSize * config.ContextLength
                + (config.MaxDocumentTokens > 0
                    ? config.MaxDocumentTokens
                    : config.ContextLength)
                + 2);
            if (epoch == resume.Epoch)
                buffer.AddRange(resume.TokenBuffer);
            double totalLoss = epoch == resume.Epoch
                ? resume.LossSum
                : 0d;
            long completedTargets = epoch == resume.Epoch
                ? resume.TargetCount
                : 0;
            long documentsProcessed = epoch == resume.Epoch
                ? resume.CompletedDocuments
                : 0;
            int completedBatchesInEpoch = epochRun.ResumeUnit;
            var timer = Stopwatch.StartNew();
            var microBatchCursor = new StreamingWikiTrainingDataCursor(buffer);
            microBatchCursor.StartEpoch(completedBatchesInEpoch);
            var updateCursor = new BufferedWikiTrainingUpdateCursor();
            var pendingMicroBatches = new List<WikiTrainingBatch>(
                config.GradientAccumulationSteps);
            var stepOperations = new StreamingWikiTrainingStepOperations(
                model,
                optimizer,
                trainingParameters,
                dataParallelEngine,
                scheduler,
                metricReporter,
                documentsPerEpoch,
                config.Epochs,
                config.GraphUpdateSteps,
                config.BatchSize,
                config.ContextLength,
                globalStep,
                graphWindowLoss,
                graphWindowTargets);
            var cursorStepOperations =
                new CursorTrainingStepOperations<WikiTrainingUpdate>(
                    updateCursor,
                    stepOperations);
            stepOperations.StartEpoch(
                epoch,
                totalLoss,
                completedTargets,
                completedBatchesInEpoch);

            void CommitPendingUpdate()
            {
                if (pendingMicroBatches.Count == 0)
                    return;
                updateCursor.ConfigureNext(pendingMicroBatches.ToArray());
                pendingMicroBatches.Clear();
                TrainingStepState committedStep = stepExecutor.Execute(
                    checked(globalStep + 1),
                    cursorStepOperations);
                globalStep = committedStep.GlobalStep;
                noGcWindow.Pulse();
                totalLoss = stepOperations.TotalLoss;
                completedTargets = stepOperations.CompletedTargets;
                completedBatchesInEpoch =
                    stepOperations.CompletedBatches;
                graphWindowLoss = stepOperations.GraphWindowLoss;
                graphWindowTargets = stepOperations.GraphWindowTargets;
                float lossValue = stepOperations.LossValue;
                float gradientNorm = stepOperations.GradientNorm;
                IReadOnlyList<float> learningRates =
                    stepOperations.LearningRates;
                double overallProgress = stepOperations.OverallProgress;
                if (globalStep % config.LogEveryBatches == 0)
                {
                    IReadOnlyList<int> shardBatches =
                        stepOperations.LastShardBatchSizes;
                    output.WriteLine(
                        $"epoch {epoch}, step {globalStep:N0}, " +
                        $"documents {documentsProcessed:N0}/" +
                        $"{documentsPerEpoch:N0}, loss = {lossValue:F6}, " +
                        $"accumulation {stepOperations.MicroBatchCount}/" +
                        $"{config.GradientAccumulationSteps}, " +
                        $"lr = {string.Join('/', learningRates.Select(rate => $"{rate:G6}"))}, " +
                        $"grad norm = {gradientNorm:G6}, " +
                        $"clip = {MathF.Min(1f, 1f / MathF.Max(gradientNorm, 1e-12f)):G6}, " +
                        runProgress.Describe(overallProgress) +
                        (shardBatches.Count > 1
                            ? $", gpu batches = " +
                                string.Join('/', shardBatches)
                            : string.Empty) +
                        FormatOptimizerDiagnostics(optimizer));
                }
                ProductionTrainingSessionFactory
                    .EnsureCanPublishCheckpoint(
                        trainingSession,
                        globalStep);
                RunDatasetContinuationAfterCommittedStep(
                    globalStep,
                    model,
                    tokenizer,
                    sampleDocuments,
                    config,
                    generationRandom,
                    output,
                    error);
            }

            void TrainBatch(
                int batchSize,
                int sequenceLength)
            {
                microBatchCursor.ConfigureNext(
                    batchSize,
                    sequenceLength,
                    documentsProcessed);
                WikiTrainingBatch next = microBatchCursor.AcquireNext();
                if (pendingMicroBatches.Count > 0
                    && pendingMicroBatches[0].BatchSize != next.BatchSize)
                {
                    CommitPendingUpdate();
                }
                pendingMicroBatches.Add(next);
                if (pendingMicroBatches.Count
                    == config.GradientAccumulationSteps)
                {
                    CommitPendingUpdate();
                }
            }

            int? maximumDocuments = config.MaxTrainingDocuments == 0
                ? null
                : config.MaxTrainingDocuments;
            long documentsToSkip = documentsProcessed;
            // Seeded per epoch so a resumed run replays the same order and the
            // skip count still lands on the document it left off at.
            int documentShuffleSeed = TrainingRunner.CombineSeed(
                config.Seed,
                epoch,
                ShuffleSeedSalt);
            int corpusShuffleSeed = TrainingRunner.CombineSeed(
                config.Seed,
                epoch,
                CorpusShuffleSeedSalt);
            foreach (string document in ShuffleDocuments(
                ReadDocuments(
                    config.DataPath,
                    config.TextColumn,
                    maximumDocuments,
                    corpusShuffleSeed),
                config.ShuffleBufferSize,
                new Random(documentShuffleSeed)))
            {
                if (documentsToSkip > 0)
                {
                    documentsToSkip--;
                    continue;
                }
                documentsProcessed++;
                if (epoch == resume.Epoch
                    && TryGetDocumentSplit(document, out _))
                {
                    AddReservoirSample(
                        document,
                        sampleDocuments,
                        ref eligibleSampleDocuments,
                        config.DatasetSamplePoolSize,
                        reservoirRandom);
                }

                AppendDocument(
                    buffer,
                    tokenizer,
                    document,
                    config.MaxDocumentTokens);

                int previousTenth = (int)((documentsProcessed - 1) * 10
                    / documentsPerEpoch);
                int currentTenth = (int)(documentsProcessed * 10
                    / documentsPerEpoch);
                bool shouldSaveDocumentCheckpoint =
                    currentTenth > previousTenth && currentTenth < 10;
                bool documentCheckpointSaved = false;

                void SaveDocumentCheckpoint()
                {
                    CommitPendingUpdate();
                    ProductionTrainingSessionFactory
                        .EnsureCanPublishCheckpoint(
                            trainingSession,
                            globalStep);
                    SaveTrainingCheckpoint(
                        config,
                        tokenizer.VocabularySize,
                        epoch - 1,
                        bestState ?? EmptyModuleState(),
                        bestLoss,
                        bestEpoch,
                        model,
                        optimizer,
                        scheduler,
                        globalStep,
                        currentEpoch: epoch,
                        completedBatchesInEpoch,
                        currentLossSum: totalLoss,
                        currentTargetCount: completedTargets,
                        completedDocumentsInEpoch: documentsProcessed,
                        currentTokenBuffer: buffer.ToArray(),
                        adaptiveCudaShardState: dataParallelEngine?
                            .CaptureAdaptiveShardingState());
                    bestState = null;
                    output.WriteLine(
                        $"training checkpoint = {config.CheckpointPath} " +
                        $"at epoch " +
                        $"{epoch - 1d + (double)documentsProcessed / documentsPerEpoch:F1}");
                    string snapshotPath = CheckpointSnapshot.Save(
                        config.CheckpointPath,
                        model.GetType().Name,
                        epoch - 1d
                            + (double)documentsProcessed / documentsPerEpoch,
                        model);
                    output.WriteLine(
                        $"model snapshot = {snapshotPath}");
                    documentCheckpointSaved = true;
                }

                while ((buffer.Count - 1) / config.ContextLength
                    >= config.BatchSize)
                {
                    int completeSequences =
                        (buffer.Count - 1) / config.ContextLength;
                    bool isLastDocumentBatch =
                        completeSequences < config.BatchSize * 2;
                    TrainBatch(
                        config.BatchSize,
                        config.ContextLength);
                    if (shouldSaveDocumentCheckpoint
                        && isLastDocumentBatch)
                    {
                        SaveDocumentCheckpoint();
                    }
                }
                if (shouldSaveDocumentCheckpoint
                    && !documentCheckpointSaved)
                {
                    SaveDocumentCheckpoint();
                }
            }

            while (buffer.Count > 1)
            {
                int remainingTargets = buffer.Count - 1;
                int remainingSequences = TrainingRunner.DivideRoundUp(
                    remainingTargets,
                    config.ContextLength);
                int batchSize = Math.Min(
                    config.BatchSize,
                    remainingSequences);
                TrainBatch(batchSize, config.ContextLength);
            }
            CommitPendingUpdate();

            if (completedTargets == 0)
            {
                throw new InvalidDataException(
                    "Wikipedia corpus did not produce trainable token pairs.");
            }
            float trainingLoss = (float)(totalLoss / completedTargets);
            timer.Stop();
            output.WriteLine(
                $"epoch {epoch}, train loss = {trainingLoss:F6}, " +
                $"documents = {documentsProcessed:N0}, targets = " +
                $"{completedTargets:N0}, time = " +
                $"{timer.Elapsed.TotalSeconds:F2} sec");
            if (epoch == config.Epochs)
            {
                if (graphWindowTargets > 0)
                {
                    metricReporter.AppendCommittedLoss(
                        globalStep,
                        epoch,
                        MetricKinds.TrainLoss,
                        graphWindowLoss / graphWindowTargets);
                    graphWindowLoss = 0d;
                    graphWindowTargets = 0;
                }
            }

            if (bestEpoch == 0 || trainingLoss < bestLoss)
            {
                bestLoss = trainingLoss;
                bestEpoch = epoch;
                bestState = EmptyModuleState();
            }
            ProductionTrainingSessionFactory
                .EnsureCanPublishCheckpoint(
                    trainingSession,
                    globalStep);
            SaveTrainingCheckpoint(
                config,
                tokenizer.VocabularySize,
                epoch,
                bestState ?? EmptyModuleState(),
                bestLoss,
                bestEpoch,
                model,
                 optimizer,
                 scheduler,
                 globalStep,
                 adaptiveCudaShardState: dataParallelEngine?
                     .CaptureAdaptiveShardingState());
            bestState = null;
            string epochSnapshotPath = CheckpointSnapshot.Save(
                config.CheckpointPath,
                model.GetType().Name,
                epoch,
                model);
            output.WriteLine(
                $"model snapshot = {epochSnapshotPath}");
        }

        if (bestEpoch == 0)
            throw new InvalidOperationException("Training did not produce a model state.");
        LoadBestTrainingModelInto(config.CheckpointPath, model);
        output.WriteLine(
            $"best model = epoch {bestEpoch}, train loss {bestLoss:F6}");
        output.WriteLine($"checkpoint = {config.CheckpointPath}");
        WriteFinalDatasetContinuation(
            model,
            tokenizer,
            sampleDocuments,
            config,
            generationRandom,
            globalStep,
            output);
        return 0;
    }

    private static TrainingCorpus LoadTrainingCorpus(
        WikiTrainingConfiguration config,
        BpeTokenizer tokenizer,
        TextWriter output)
    {
        var tokens = new List<int>(config.MaxTrainingTokens);
        var sampleDocuments = new List<string>(config.DatasetSamplePoolSize);
        var sampleRandom = new Random(
            config.Seed ^ unchecked((int)0xBB67AE85));
        int eligibleSampleDocuments = 0;
        int documentCount = 0;
        int documentShuffleSeed = TrainingRunner.CombineSeed(
            config.Seed,
            1,
            ShuffleSeedSalt);
        int corpusShuffleSeed = TrainingRunner.CombineSeed(
            config.Seed,
            1,
            CorpusShuffleSeedSalt);
        foreach (string document in ShuffleDocuments(
            ReadDocuments(
                config.DataPath,
                config.TextColumn,
                config.MaxTrainingDocuments == 0
                    ? null
                    : config.MaxTrainingDocuments,
                corpusShuffleSeed),
            config.ShuffleBufferSize,
            new Random(documentShuffleSeed)))
        {
            if (tokens.Count >= config.MaxTrainingTokens)
                break;
            if (TryGetDocumentSplit(document, out _))
            {
                eligibleSampleDocuments++;
                if (sampleDocuments.Count < config.DatasetSamplePoolSize)
                {
                    sampleDocuments.Add(document);
                }
                else
                {
                    int replacement = sampleRandom.Next(
                        eligibleSampleDocuments);
                    if (replacement < sampleDocuments.Count)
                        sampleDocuments[replacement] = document;
                }
            }

            tokens.Add(BpeTokenizer.BosTokenId);
            int[] documentTokens = tokenizer.Encode(document);
            foreach (int token in documentTokens)
            {
                if (tokens.Count >= config.MaxTrainingTokens - 1)
                    break;
                tokens.Add(token);
            }
            if (tokens.Count < config.MaxTrainingTokens)
                tokens.Add(BpeTokenizer.EosTokenId);
            documentCount++;
            if (documentCount % 100 == 0)
            {
                output.Write(
                    $"\rdocuments {documentCount:N0}, tokens " +
                    $"{tokens.Count:N0}");
            }
        }
        if (documentCount >= 100)
            output.WriteLine();
        return new TrainingCorpus(
            tokens.ToArray(),
            sampleDocuments.ToArray());
    }

    private static float Evaluate(
        LanguageModel model,
        int[] tokens,
        int validationStartSequence,
        int validationSequences,
        WikiTrainingConfiguration config)
    {
        model.eval();
        float totalLoss = 0f;
        int completedTargets = 0;
        using (torch.no_grad())
        {
            for (int start = 0; start < validationSequences; start += config.BatchSize)
            {
                int count = Math.Min(config.BatchSize, validationSequences - start);
                int[] order = Enumerable.Range(
                    validationStartSequence + start,
                    count).ToArray();
                LanguageBatch values = CreateBatch(
                    tokens,
                    order,
                    0,
                    count,
                    config.ContextLength);
                Tensor logits = model.forward(
                    values.Input,
                    count,
                    config.ContextLength);
                Tensor loss = nn.functional.cross_entropy(
                    logits,
                    values.Target);
                totalLoss += loss.item() * values.ValidTargetCount;
                completedTargets += values.ValidTargetCount;
            }
        }
        return totalLoss / completedTargets;
    }

    private static void WriteEffectiveTrainingConfiguration(
        WikiTrainingConfiguration config,
        TensorPrecisionMode precisionMode,
        int bfp8BlockSize,
        TextWriter output)
    {
        string architectureDetails = config.IsForgetMemoryArchitecture()
            ? $", matrix delta memory key {config.ForgetMemoryKeyWidth}, " +
                $"value {config.ForgetMemoryValueWidth}, retention " +
                $"{config.ForgetMemoryRetentionMinimum:G}-" +
                $"{config.ForgetMemoryRetentionMaximum:G}"
            : config.IsArchitecture(WikiTrainingConfiguration.HyenaArchitecture)
                ? $", Hyena filter width {config.HyenaFilterWidth}, " +
                    $"convolution {config.HyenaConvolutionAlgorithm}"
                : config.IsArchitecture(
                    WikiTrainingConfiguration.ForgetScanArchitecture)
                    ? ", associative forget scan"
                    : string.Empty;
        output.WriteLine(
            $"effective training = epochs {config.Epochs}, " +
            $"microbatch {config.BatchSize}, accumulation " +
            $"{config.GradientAccumulationSteps}, effective batch " +
            $"{checked(config.BatchSize * config.GradientAccumulationSteps)}, " +
            $"context {config.ContextLength}, " +
            $"{FormatDocumentTokenLimit(config.MaxDocumentTokens)}");
        output.WriteLine(
            $"checkpoint = {config.CheckpointPath}, " +
            $"resume {(config.ResumeFromCheckpoint ? "enabled" : "disabled")}, " +
            $"auto-resume {(config.AutoResume ? "enabled" : "disabled")}");
        output.WriteLine(
            $"effective model = {config.ModelArchitecture}, " +
            $"vocabulary {config.VocabularySize}, " +
            $"width {config.ModelWidth}, heads {config.Heads}, " +
            $"hidden {config.HiddenSize}, layers {config.Layers}, " +
            $"dropout {config.Dropout:G}, precision " +
            $"{FormatPrecisionMode(precisionMode)}" +
            (precisionMode == TensorPrecisionMode.Bfp8
                ? " (tensor scale)"
                : precisionMode == TensorPrecisionMode.Mix8_32
                    ? $" (block {bfp8BlockSize})"
                    : string.Empty) +
            (config.ResumeFromCheckpoint
                ? " (checkpoint)"
                : config.Precision is null
                    && config.PrecisionMode is null
                    && config.ModelDType is null
                    ? " (default)"
                    : string.Empty) +
            architectureDetails);
        output.WriteLine(
            $"special tokens = {BpeTokenizer.PadToken}:" +
            $"{BpeTokenizer.PadTokenId}, {BpeTokenizer.BosToken}:" +
            $"{BpeTokenizer.BosTokenId}, {BpeTokenizer.EosToken}:" +
            $"{BpeTokenizer.EosTokenId}; padded targets use ignoreIndex " +
            $"{Tensor.DefaultCrossEntropyIgnoreIndex}");
    }

    private static string TakeHead(string text, int count)
    {
        if (text.Length <= count)
            return text;
        int length = count;
        if (length > 0 && char.IsHighSurrogate(text[length - 1]))
            length--;
        return text[..length] + "…";
    }

    private static string TakeTail(string text, int count)
    {
        if (text.Length <= count)
            return text;
        int start = text.Length - count;
        if (start < text.Length && char.IsLowSurrogate(text[start]))
            start++;
        return "…" + text[start..];
    }

    private static void WriteGeneration(
        LanguageModel model,
        BpeTokenizer tokenizer,
        string prompt,
        WikiTrainingConfiguration config,
        TextWriter output)
    {
        string generated = model.generate(
            prompt,
            tokenizer,
            config.MaxNewTokens,
            config.Temperature,
            config.TopK,
            new Random(config.Seed ^ 0x27D4EB2D));
        output.WriteLine("generated text:");
        output.WriteLine(generated);
    }

    private readonly record struct TrainingCorpus(
        int[] Tokens,
        string[] SampleDocuments);

}
