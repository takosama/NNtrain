using System.Diagnostics;
using System.Text.Json;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Optimization;

namespace NNtrain;

internal static partial class Program
{
    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool openLossGraph = false)
        => Run(
            args,
            output,
            error,
            openLossGraph,
            ConfigLoader.Load);

    internal static int Run(
        string[] args,
        TextWriter output,
        TextWriter error,
        bool openLossGraph,
        Func<string, CanonicalTrainingSpec> loadConfiguration)
    {
        ArgumentNullException.ThrowIfNull(args);
        ArgumentNullException.ThrowIfNull(output);
        ArgumentNullException.ThrowIfNull(error);
        ArgumentNullException.ThrowIfNull(loadConfiguration);

        string configurationPath;
        string? generatePrompt = null;
        string? generationConfigurationPath = null;
        bool resumeFromCheckpoint = false;
        bool autoResume = false;
        if (args.Length == 0)
        {
            configurationPath = FindDefaultConfiguration();
        }
        else if (args.Length == 2
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
        }
        else if (args.Length == 1
            && string.Equals(
                args[0],
                "--resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = FindDefaultConfiguration();
            resumeFromCheckpoint = true;
        }
        else if (args.Length == 1
            && string.Equals(
                args[0],
                "--auto-resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = FindDefaultConfiguration();
            autoResume = true;
        }
        else if (args.Length == 3
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                args[2],
                "--resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
            resumeFromCheckpoint = true;
        }
        else if (args.Length == 3
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                args[2],
                "--auto-resume",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
            autoResume = true;
        }
        else if (args.Length == 2
            && string.Equals(
                args[0],
                "--generate-config",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = string.Empty;
            generationConfigurationPath = args[1];
        }
        else if (args.Length == 4
            && string.Equals(
                args[0],
                "--config",
                StringComparison.OrdinalIgnoreCase)
            && string.Equals(
                args[2],
                "--generate",
                StringComparison.OrdinalIgnoreCase))
        {
            configurationPath = args[1];
            generatePrompt = args[3];
        }
        else
        {
            error.WriteLine(
                "Usage: NNtrain.Cli [--config <training-config.json>] " +
                "[--resume | --auto-resume | --generate <prompt>] " +
                "[--generate-config <generate.json>]");
            return 1;
        }

        try
        {
            if (generationConfigurationPath is not null)
                return GenerateCommand.Run(generationConfigurationPath, output, error);
            CanonicalTrainingSpec canonical =
                loadConfiguration(configurationPath);
            if (canonical is CanonicalWikiTrainingSpec wiki)
            {
                if (generatePrompt is not null)
                {
                    return WikiLanguageModelCommand.Run(
                        wiki,
                        generatePrompt,
                        output,
                        error,
                        openLossGraph);
                }
                WikiTrainingConfiguration wikiConfig = wiki.Configuration;
                using TrainingRunGuard wikiRun = TrainingRunGuard.Begin(
                    wiki.ConfigurationPath,
                    wikiConfig.CheckpointPath);
                bool resumeWiki = ResolveAutomaticResume(
                    resumeFromCheckpoint,
                    autoResume || wikiConfig.AutoResume,
                    wikiRun,
                    wikiConfig.CheckpointPath,
                    output);
                int wikiExitCode = WikiLanguageModelCommand.Run(
                    wiki,
                    generatePrompt: null,
                    output,
                    error,
                    openLossGraph,
                    resumeWiki);
                if (wikiExitCode == 0)
                    wikiRun.Complete();
                return wikiExitCode;
            }
            if (generatePrompt is not null)
            {
                throw new ArgumentException(
                    "--generate can only be used with a gpt_rin_wiki_jp " +
                    "configuration.");
            }

            CanonicalClassificationTrainingSpec classification = canonical
                as CanonicalClassificationTrainingSpec
                ?? throw new InvalidDataException(
                    "Configuration is not an image-classification task.");
            TrainingConfiguration config = classification.Configuration;
            using TrainingRunGuard classificationRun = TrainingRunGuard.Begin(
                classification.ConfigurationPath,
                config.CheckpointPath);
            bool resumeClassification = ResolveAutomaticResume(
                resumeFromCheckpoint,
                autoResume || config.AutoResume,
                classificationRun,
                config.CheckpointPath,
                output);
            if (resumeClassification)
                config = config with { ResumeFromCheckpoint = true };
            torch.manual_seed(config.Seed);
            Tensor.SimdEnabled = config.UseSimd;
            IImageClassificationDataset trainData = CreateDataset(
                config.TrainingData,
                "Training");
            IImageClassificationDataset evalData = CreateDataset(
                config.EvaluationData,
                "Evaluation");

            ValidateDatasetCompatibility(trainData, evalData);
            config.Model.ValidateForModelWidth(trainData.Columns);
            using ExecutionSession executionSession =
                ProductionTrainingSessionFactory.CreateExecutionSession(
                    classification.PrecisionMode,
                    Tensor.ExecutionDevice,
                    Tensor.CudaDeviceIndices);
            using IDisposable executionScope = executionSession.Enter();
            TransformerClassifier model = nn.transformer_classifier(
                seq_len: trainData.Rows,
                d_model: trainData.Columns,
                num_heads: config.Model.Heads,
                dim_feedforward: config.Model.HiddenSize,
                num_layers: config.Model.Layers,
                num_classes: trainData.ClassCount,
                generator: new Random(config.Model.Seed),
                init_scale: config.Model.InitializationScale,
                dropout: config.Model.Dropout);
            model.to(
                classification.PrecisionMode,
                classification.Bfp8BlockSize);
            if (Tensor.ExecutionDevice == TensorDevice.Cuda)
                model.to(TensorDevice.Cuda);
            output.WriteLine(
                $"model = {model.GetType().Name} (image classifier)");
            output.WriteLine(
                $"precision = " +
                $"{TensorPrecisionModeNames.Format(classification.PrecisionMode)}" +
                (classification.PrecisionMode is TensorPrecisionMode.Bfp8
                    ? " (tensor scale)"
                    : classification.PrecisionMode == TensorPrecisionMode.Mix8_32
                        ? $" (block {classification.Bfp8BlockSize})"
                        : string.Empty));
            OptimizerBundle optimizer = CreateOptimizerBundle(
                model,
                config);
            ILRScheduler scheduler =
                lr_scheduler.LinearWarmupCosineAnnealingLR(
                    optimizer,
                    total_epochs: config.Epochs,
                    warmup_epochs: config.WarmupEpochs,
                    min_lr_ratio: config.MinimumLearningRateRatio);
            DataLoader evalLoader = torch.utils.data.DataLoader(
                evalData,
                batch_size: config.ResolvedMicroBatchSize);
            string graphPath = Path.ChangeExtension(
                Path.GetFullPath(configurationPath),
                ".loss.html");
            output.WriteLine(
                $"workers = {Environment.ProcessorCount}");
            output.WriteLine(
                $"simd = {GetSimdStatus()}");
            output.WriteLine(
                $"micro batch = {config.ResolvedMicroBatchSize} samples x " +
                $"{config.MicroBatchCount} accumulation(s), effective " +
                $"batch {config.EffectiveBatchSize}");
            output.WriteLine(
                $"optimizer = {config.Optimizer.ToLowerInvariant()}");
            output.WriteLine(
                $"learning rate = {config.LearningRate:F6}");
            if (config.IsOptimizer(
                TrainingConfiguration.NekoMuonOptimizer))
            {
                output.WriteLine(
                    "auxiliary optimizer = adamw");
                output.WriteLine(
                    "auxiliary learning rate = " +
                    $"{config.AuxiliaryLearningRate:F6}");
            }
            output.WriteLine(
                $"label smoothing = {config.LabelSmoothing:F3}");
            output.WriteLine(
                $"weight decay = {config.WeightDecay:F6}");
            if (config.IsOptimizer(
                TrainingConfiguration.GainShareAdamWOptimizer))
            {
                output.WriteLine(
                    $"gainshare = block depth " +
                    $"{config.GainShareBlockDepth}, " +
                    $"betas ({config.GainShareBeta1:F3}, " +
                    $"{config.GainShareBeta2:F3}), " +
                    $"eps {config.GainShareEpsilon:E1}, " +
                    $"rho {config.GainShareRho:F3}, " +
                    $"gamma {config.GainShareGamma:F3}, " +
                    $"scale [{config.GainShareMinScale:F3}, " +
                    $"{config.GainShareMaxScale:F3}]");
            }
            output.WriteLine(
                $"dropout = {config.Model.Dropout:F3}");
            output.WriteLine(
                $"learning-rate schedule = warmup {config.WarmupEpochs} " +
                $"epoch(s), cosine to " +
                $"{config.MinimumLearningRateRatio:P1}");
            output.WriteLine(
                $"checkpoint = {config.CheckpointPath}, " +
                $"resume {(config.ResumeFromCheckpoint ? "enabled" : "disabled")}, " +
                $"auto-resume {(config.AutoResume ? "enabled" : "disabled")}");
            output.WriteLine(
                config.EarlyStoppingPatience > 0
                    ? $"early stopping = patience " +
                        $"{config.EarlyStoppingPatience} epoch(s)"
                    : "early stopping = disabled");
            if (trainData is Cifar100)
            {
                var cifar100 = (Cifar100)trainData;
                Cifar100Options augmentation = cifar100.Options;
                output.WriteLine(
                    $"tokenization = {augmentation.PatchSize}x" +
                    $"{augmentation.PatchSize} patches, " +
                    $"{trainData.Rows} tokens x " +
                    $"{trainData.Columns} features");
                output.WriteLine(
                    $"normalization = " +
                    (augmentation.Normalize ? "cifar100" : "disabled"));
                output.WriteLine(
                    GetCifar100AugmentationDescription(augmentation));
            }

            bool hasBestModel = false;
            float bestEvaluationLoss = float.PositiveInfinity;
            float earlyStoppingReferenceLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            int epochsWithoutImprovement = 0;
            int firstEpoch = 1;
            int firstUpdate = 0;
            double resumedTrainingLossSum = 0d;
            int resumedTrainingCorrect = 0;
            int resumedTrainingSamples = 0;
            int classificationMicroBatchesPerEpoch =
                TrainingRunner.DivideRoundUp(
                    trainData.Count,
                    config.ResolvedMicroBatchSize);
            int classificationUpdatesPerEpoch =
                TrainingRunner.DivideRoundUp(
                    classificationMicroBatchesPerEpoch,
                    config.MicroBatchCount);
            long globalStep = 0;

            if (config.ResumeFromCheckpoint)
            {
                if (!File.Exists(config.CheckpointPath))
                {
                    throw new FileNotFoundException(
                        "Training checkpoint was not found.",
                        config.CheckpointPath);
                }
                ClassificationTrainingCheckpoint checkpoint =
                    ClassificationCheckpoint.LoadIntoModel(
                        config.CheckpointPath,
                        model,
                        optimizer,
                        output);
                bool hasPartialEpoch = checkpoint.CurrentEpoch
                    > checkpoint.CompletedEpoch;
                if (!hasPartialEpoch
                    && checkpoint.CompletedEpoch >= config.Epochs)
                {
                    throw new InvalidDataException(
                        $"Checkpoint already completed epoch " +
                        $"{checkpoint.CompletedEpoch}, but configuration " +
                        $"requests only {config.Epochs} epoch(s).");
                }
                scheduler.load_state_dict(checkpoint.Scheduler);
                hasBestModel = checkpoint.BestEpoch > 0;
                bestEpoch = checkpoint.BestEpoch;
                bestEvaluationLoss = checkpoint.BestEvaluationLoss;
                earlyStoppingReferenceLoss =
                    checkpoint.EarlyStoppingReferenceLoss;
                epochsWithoutImprovement =
                    checkpoint.EpochsWithoutImprovement;
                firstEpoch = hasPartialEpoch
                    ? checkpoint.CurrentEpoch
                    : checkpoint.CompletedEpoch + 1;
                firstUpdate = hasPartialEpoch
                    ? checkpoint.CompletedUpdatesInEpoch
                    : 0;
                globalStep = checked(
                    (long)checkpoint.CompletedEpoch
                        * classificationUpdatesPerEpoch
                    + checkpoint.CompletedUpdatesInEpoch);
                if (hasPartialEpoch)
                {
                    resumedTrainingLossSum =
                        checkpoint.CurrentTrainingLossSum;
                    resumedTrainingCorrect =
                        checkpoint.CurrentTrainingCorrect;
                    resumedTrainingSamples =
                        checkpoint.CurrentTrainingSamples;
                }
                output.WriteLine(
                    $"resumed checkpoint = {config.CheckpointPath}, " +
                    $"next epoch {firstEpoch}" +
                    (firstUpdate == 0
                        ? string.Empty
                        : $", update {firstUpdate + 1}"));
            }

            double checkpointEpoch = firstEpoch - 1d
                + (double)firstUpdate / classificationUpdatesPerEpoch;
            TrainingMetricReporter metricReporter =
                TrainingMetricReporter.Open(
                    graphPath,
                    config.Epochs,
                    config.ResumeFromCheckpoint,
                    config.ResumeFromCheckpoint ? globalStep : -1,
                    checkpointEpoch,
                    config.ShowLossGraph);
            output.WriteLine($"metrics = {metricReporter.SidecarPath}");
            if (config.ShowLossGraph)
            {
                output.WriteLine($"loss graph = {metricReporter.HtmlPath}");
                if (openLossGraph)
                    metricReporter.TryOpenHtml(error);
            }
            using TrainingSession trainingSession = new(
                executionSession,
                ownsExecutionSession: false,
                lastCommittedStep: globalStep);
            trainingSession.OwnOptimizer(optimizer);
            var stepExecutor = new TrainingStepExecutor(trainingSession);
            var batchCursor = new ClassificationTrainingDataCursor();
            var stepOperations =
                new ClassificationTrainingStepOperations(
                    model,
                    optimizer,
                    scheduler,
                    output,
                    config.LabelSmoothing,
                    trainData.ClassCount,
                    globalStep);
            var cursorStepOperations =
                new CursorTrainingStepOperations<ClassificationUpdateBatch>(
                    batchCursor,
                    stepOperations);

            foreach (TrainingEpoch epochRun in TrainingRunner.Epochs(
                firstEpoch,
                config.Epochs,
                firstUpdate))
            {
                int epoch = epochRun.Number;
                DataLoader trainLoader = torch.utils.data.DataLoader(
                    trainData,
                    batch_size: config.ResolvedMicroBatchSize,
                    shuffle: true,
                    training: true,
                    generator: new Random(
                        TrainingRunner.CombineSeed(config.Seed, epoch)),
                    augmentation_generator: new Random(
                        TrainingRunner.CombineSeed(
                            config.Seed ^ 0x51F15EED,
                            epoch)));
                int resumeUpdate = epochRun.ResumeUnit;
                IReadOnlyList<float> scheduledRates =
                    scheduler.get_last_lr();
                var learningRates = new LearningRates(
                    scheduledRates[0],
                    scheduledRates.Count > 1 ? scheduledRates[1] : null);
                model.train();
                double trainLoss = epoch == firstEpoch
                    ? resumedTrainingLossSum
                    : 0d;
                int trainCorrect = epoch == firstEpoch
                    ? resumedTrainingCorrect
                    : 0;
                int completedTrainingSamples = epoch == firstEpoch
                    ? resumedTrainingSamples
                    : 0;
                var trainTimer = Stopwatch.StartNew();
                int microBatchSize = trainLoader.batch_size;
                int microBatchTotal = trainLoader.Count;
                int updateTotal = TrainingRunner.DivideRoundUp(
                    microBatchTotal,
                    config.MicroBatchCount);
                if (updateTotal != classificationUpdatesPerEpoch)
                {
                    throw new InvalidOperationException(
                        "Classification update count changed after metric " +
                        "resume position was established.");
                }
                using IEnumerator<DataBatch> trainingBatches =
                    trainLoader.GetEnumerator();
                int microBatchesToSkip = Math.Min(
                    microBatchTotal,
                    resumeUpdate * config.MicroBatchCount);
                batchCursor.StartEpoch(
                    trainingBatches,
                    microBatchesToSkip);
                stepOperations.StartEpoch(
                    epoch,
                    trainLoss,
                    trainCorrect,
                    completedTrainingSamples,
                    advanceSchedule: resumeUpdate == 0);

                for (int update = resumeUpdate;
                    update < updateTotal;
                    update++)
                {
                    int firstMicroBatch = update * config.MicroBatchCount;
                    int microBatchesInUpdate = Math.Min(
                        config.MicroBatchCount,
                        microBatchTotal - firstMicroBatch);
                    int updateStart = firstMicroBatch * microBatchSize;
                    int samplesInUpdate = Math.Min(
                        config.EffectiveBatchSize,
                        trainData.Count - updateStart);
                    batchCursor.ConfigureNext(
                        update,
                        updateTotal,
                        firstMicroBatch,
                        microBatchesInUpdate,
                        microBatchTotal,
                        samplesInUpdate);
                    TrainingStepState committedStep = stepExecutor.Execute(
                        checked(globalStep + 1),
                        cursorStepOperations);
                    globalStep = committedStep.GlobalStep;
                    trainLoss = stepOperations.TrainingLossSum;
                    trainCorrect = stepOperations.TrainingCorrect;
                    completedTrainingSamples =
                        stepOperations.TrainingSamples;
                    IReadOnlyList<float> currentRates =
                        stepOperations.LearningRates;
                    learningRates = new LearningRates(
                        currentRates[0],
                        currentRates.Count > 1
                            ? currentRates[1]
                            : null);
                    int completedUpdates = update + 1;
                    if (TrainingRunner.ShouldSaveCheckpoint(
                        completedUpdates,
                        updateTotal))
                    {
                        ProductionTrainingSessionFactory
                            .EnsureCanPublishCheckpoint(
                                trainingSession,
                                globalStep);
                        ClassificationCheckpoint.Save(
                            config.CheckpointPath,
                            model,
                            optimizer,
                            CreateClassificationCheckpoint(
                                completedEpoch: epoch - 1,
                                currentEpoch: epoch,
                                completedUpdates,
                                scheduler,
                                bestEpoch,
                                bestEvaluationLoss,
                                earlyStoppingReferenceLoss,
                                epochsWithoutImprovement,
                                trainLoss,
                                trainCorrect,
                                completedTrainingSamples),
                            publishCurrentAsBest: false);
                        output.WriteLine(
                            $"training checkpoint = " +
                            $"{config.CheckpointPath} at epoch " +
                            $"{epoch - 1d + (double)completedUpdates / updateTotal:F1}");
                        string snapshotPath = CheckpointSnapshot.Save(
                            config.CheckpointPath,
                            model.GetType().Name,
                            epoch - 1d
                                + (double)completedUpdates / updateTotal,
                            model);
                        output.WriteLine(
                            $"model snapshot = {snapshotPath}");
                    }
                }

                trainTimer.Stop();

                float evalLoss = 0f;
                int evalCorrectCount = 0;
                int completedEvaluationSamples = 0;
                int lastEvalPercent = -1;
                var evalTimer = Stopwatch.StartNew();
                model.eval();
                using (torch.no_grad())
                {
                    foreach (DataBatch samples in evalLoader)
                    {
                        int samplesInBatch = samples.target.Length;
                        Tensor logits = model.forward(samples.input);
                        Tensor loss = nn.functional.cross_entropy(
                            logits,
                            samples.target);
                        using CudaClassificationCorrectCountReadback?
                            correctCountReadback = BeginCountCorrect(
                                logits,
                                samples.target,
                                evalData.ClassCount);
                        evalLoss += loss.item() * samplesInBatch;
                        evalCorrectCount += CompleteCountCorrect(
                            logits,
                            samples.target,
                            evalData.ClassCount,
                            correctCountReadback);

                        completedEvaluationSamples += samplesInBatch;
                        int evalPercent =
                            completedEvaluationSamples
                            * 100
                            / evalData.Count;
                        if (evalPercent > lastEvalPercent)
                        {
                            output.Write(
                                $"\repoch {epoch}, " +
                                $"eval {evalPercent,3}%");
                            lastEvalPercent = evalPercent;
                        }
                    }
                }

                evalTimer.Stop();
                float averageTrainingLoss =
                    (float)(trainLoss / completedTrainingSamples);
                float averageEvaluationLoss = evalLoss / evalData.Count;
                bool publishCurrentAsBest = !hasBestModel
                    || averageEvaluationLoss < bestEvaluationLoss;
                if (publishCurrentAsBest)
                {
                    bestEvaluationLoss = averageEvaluationLoss;
                    bestEpoch = epoch;
                    hasBestModel = true;
                }

                bool meaningfulImprovement =
                    averageEvaluationLoss
                        < earlyStoppingReferenceLoss
                            - config.EarlyStoppingMinimumDelta;
                if (meaningfulImprovement)
                {
                    earlyStoppingReferenceLoss = averageEvaluationLoss;
                    epochsWithoutImprovement = 0;
                }
                else
                {
                    epochsWithoutImprovement++;
                }

                metricReporter.AppendCommittedEpochLosses(
                    globalStep,
                    epoch,
                    averageTrainingLoss,
                    averageEvaluationLoss);
                output.WriteLine();
                output.WriteLine(
                    $"epoch {epoch}, " +
                    $"train loss = {averageTrainingLoss:F6}, " +
                    $"train acc = {100f * trainCorrect / trainData.Count:F2}%, " +
                    $"eval loss = {averageEvaluationLoss:F6}, " +
                    $"eval acc = {100f * evalCorrectCount / evalData.Count:F2}%, " +
                    $"lr = {learningRates.Primary:F8}, " +
                    $"train time = {trainTimer.Elapsed.TotalSeconds:F2} sec, " +
                    $"eval time = {evalTimer.Elapsed.TotalSeconds:F2} sec");

                ProductionTrainingSessionFactory
                    .EnsureCanPublishCheckpoint(
                        trainingSession,
                        globalStep);
                ClassificationCheckpoint.Save(
                    config.CheckpointPath,
                    model,
                    optimizer,
                    CreateClassificationCheckpoint(
                        completedEpoch: epoch,
                        currentEpoch: 0,
                        completedUpdatesInEpoch: 0,
                        scheduler,
                        bestEpoch,
                        bestEvaluationLoss,
                        earlyStoppingReferenceLoss,
                        epochsWithoutImprovement,
                        currentTrainingLossSum: 0d,
                        currentTrainingCorrect: 0,
                        currentTrainingSamples: 0),
                    publishCurrentAsBest);
                output.WriteLine(
                    $"training checkpoint = {config.CheckpointPath}");

                if (config.EarlyStoppingPatience > 0
                    && epochsWithoutImprovement
                        >= config.EarlyStoppingPatience)
                {
                    output.WriteLine(
                        $"early stopping at epoch {epoch}: eval loss did " +
                        $"not improve for " +
                        $"{config.EarlyStoppingPatience} epoch(s).");
                    break;
                }
            }

            if (hasBestModel)
            {
                ClassificationCheckpoint.LoadBestModel(
                    config.CheckpointPath,
                    model);
                string checkpointPath = config.Checkpoint is null
                    ? Path.ChangeExtension(
                        Path.GetFullPath(configurationPath),
                        ".best-model.json")
                    : GetBestModelCheckpointPath(config.CheckpointPath);
                SaveModelCheckpoint(
                    checkpointPath,
                    bestEpoch,
                    bestEvaluationLoss,
                    model);
                output.WriteLine(
                    $"best model = epoch {bestEpoch}, " +
                    $"eval loss {bestEvaluationLoss:F6}");
                output.WriteLine($"checkpoint = {checkpointPath}");
            }

            classificationRun.Complete();
            return 0;
        }
        catch (Exception exception) when (
            exception is IOException
            or System.Text.Json.JsonException
            or ArgumentException)
        {
            error.WriteLine($"Error: {exception.Message}");
            return 2;
        }
    }

    internal static bool ResolveAutomaticResume(
        bool explicitResume,
        bool autoResume,
        TrainingRunGuard run,
        string checkpointPath,
        TextWriter output)
    {
        ArgumentNullException.ThrowIfNull(run);
        ArgumentNullException.ThrowIfNull(output);
        if (explicitResume)
            return true;
        if (!autoResume)
            return false;
        if (!run.WasInterrupted)
        {
            output.WriteLine(
                "auto-resume = no interrupted training run detected; " +
                "starting from the configured initial state");
            return false;
        }
        if (!File.Exists(checkpointPath))
        {
            output.WriteLine(
                $"auto-resume = interrupted run marker found at " +
                $"{run.MarkerPath}, but checkpoint is missing; starting " +
                "from the configured initial state");
            return false;
        }

        output.WriteLine(
            $"auto-resume = interrupted training detected; restoring " +
            $"latest checkpoint {Path.GetFullPath(checkpointPath)}");
        return true;
    }

    private static ClassificationTrainingCheckpoint
        CreateClassificationCheckpoint(
            int completedEpoch,
            int currentEpoch,
            int completedUpdatesInEpoch,
            ILRScheduler scheduler,
            int bestEpoch,
            float bestEvaluationLoss,
            float earlyStoppingReferenceLoss,
            int epochsWithoutImprovement,
            double currentTrainingLossSum,
            int currentTrainingCorrect,
            int currentTrainingSamples)
        => new(
            ClassificationTrainingCheckpoint.CurrentFormatVersion,
            completedEpoch,
            new ModuleState(ModuleState.CurrentFormatVersion, []),
            new OptimizerStateDictionary("ArtifactStream", null, []),
            scheduler.state_dict(),
            null,
            bestEpoch,
            bestEvaluationLoss,
            earlyStoppingReferenceLoss,
            epochsWithoutImprovement,
            currentEpoch,
            completedUpdatesInEpoch,
            currentTrainingLossSum,
            currentTrainingCorrect,
            currentTrainingSamples);

    private static string GetSimdStatus()
    {
        if (!Tensor.SimdEnabled)
            return "disabled (scalar)";

        return Tensor.IsSimdHardwareAccelerated
            ? "enabled (Vector256)"
            : "enabled, hardware unavailable (scalar)";
    }

    internal static IOptimizer CreateOptimizer(
        TransformerClassifier model,
        TrainingConfiguration configuration)
        => CreateOptimizerBundle(model, configuration).RootOptimizer;

    internal static OptimizerBundle CreateOptimizerBundle(
        TransformerClassifier model,
        TrainingConfiguration configuration)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(configuration);

        if (configuration.IsOptimizer(
            TrainingConfiguration.GainShareAdamWOptimizer))
        {
            IOptimizer optimizer = optim.GainShareAdamW(
                model.make_gainshare_parameter_groups(
                    configuration.GainShareBlockDepth),
                lr: configuration.LearningRate,
                beta1: configuration.GainShareBeta1,
                beta2: configuration.GainShareBeta2,
                eps: configuration.GainShareEpsilon,
                rho: configuration.GainShareRho,
                gamma: configuration.GainShareGamma,
                min_scale: configuration.GainShareMinScale,
                max_scale: configuration.GainShareMaxScale,
                weight_decay: configuration.WeightDecay);
            return OptimizerBundle.Wrap(optimizer, ["all"]);
        }

        if (configuration.IsOptimizer(
            TrainingConfiguration.NekoMuonOptimizer))
        {
            IOptimizer nekoMuon = optim.NekoMuon(
                model.HiddenWeightParameters,
                lr: configuration.LearningRate,
                weight_decay: configuration.WeightDecay);
            IOptimizer auxiliaryAdamW = optim.AdamW(
                model.AuxiliaryParameters,
                lr: configuration.AuxiliaryLearningRate,
                beta1: 0.9f,
                beta2: 0.95f,
                eps: 1e-8f,
                weight_decay: configuration.WeightDecay);
            return new OptimizerBundle(
            [
                new OptimizerGroup("hidden", nekoMuon),
                new OptimizerGroup("auxiliary", auxiliaryAdamW),
            ]);
        }

        if (configuration.IsOptimizer(TrainingConfiguration.LionOptimizer))
        {
            IOptimizer optimizer = optim.Lion(
                model.parameters(),
                lr: configuration.LearningRate,
                weight_decay: configuration.WeightDecay);
            return OptimizerBundle.Wrap(optimizer, ["all"]);
        }

        IOptimizer adamW = optim.AdamW(
            model.parameters(),
            lr: configuration.LearningRate,
            weight_decay: configuration.WeightDecay);
        return OptimizerBundle.Wrap(adamW, ["all"]);
    }

    private static void SaveModelCheckpoint(
        string path,
        int epoch,
        float evaluationLoss,
        Module model)
    {
        string fullPath = Path.GetFullPath(path);
        var repository = new NNtrain.Training.Persistence.CheckpointRepository(
            fullPath);
        using NNtrain.Training.Persistence.CheckpointWriteTransaction
            transaction = repository.BeginWrite();
        transaction.PublishArtifact(
            Path.ChangeExtension(fullPath, ".safetensors"),
            stagingPath => SafeTensorFile.SaveModel(
                model,
                stagingPath,
                artifactDTypeOverride:
                    model.PrecisionMode is TensorPrecisionMode.Mix16_32
                            or TensorPrecisionMode.Mix8_32
                            or TensorPrecisionMode.Bfp8
                        ? TensorDType.Float32
                        : null));
        transaction.CommitManifest(
            stagingPath => SaveModelCheckpointManifest(
                stagingPath,
                epoch,
                evaluationLoss,
                model));
    }

    private static void SaveModelCheckpointManifest(
        string path,
        int epoch,
        float evaluationLoss,
        Module model)
    {
        using var stream = new FileStream(
            path,
            FileMode.Create,
            FileAccess.Write,
            FileShare.None,
            bufferSize: 1024 * 1024,
            FileOptions.SequentialScan);
        using var writer = new Utf8JsonWriter(stream);
        using var staging = new CheckpointFloatStagingBuffer();
        writer.WriteStartObject();
        StreamingModuleStateJsonWriter.WriteProperty(
            writer,
            nameof(ModelCheckpoint.FormatVersion),
            ModelCheckpoint.CurrentFormatVersion);
        StreamingModuleStateJsonWriter.WriteProperty(
            writer,
            nameof(ModelCheckpoint.Epoch),
            epoch);
        StreamingModuleStateJsonWriter.WriteProperty(
            writer,
            nameof(ModelCheckpoint.EvaluationLoss),
            evaluationLoss);
        writer.WritePropertyName(nameof(ModelCheckpoint.Model));
        StreamingModuleStateJsonWriter.Write(writer, model, staging);
        writer.WriteEndObject();
        writer.Flush();
        stream.Flush(flushToDisk: true);
    }

    internal static string GetBestModelCheckpointPath(
        string trainingCheckpointPath)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(trainingCheckpointPath);
        string fullPath = Path.GetFullPath(trainingCheckpointPath);
        string directory = Path.GetDirectoryName(fullPath)
            ?? Environment.CurrentDirectory;
        string fileName = Path.GetFileName(fullPath);
        const string defaultSuffix = ".checkpoint.json";
        string stem = fileName.EndsWith(
            defaultSuffix,
            StringComparison.OrdinalIgnoreCase)
            ? fileName[..^defaultSuffix.Length]
            : Path.GetFileNameWithoutExtension(fileName);
        return Path.Combine(directory, $"{stem}.best-model.json");
    }

    internal static string FindDefaultConfiguration()
    {
        string[] fileNames =
        [
            "training.wiki-jp.json",
            "training.example.json",
        ];

        foreach (string startPath in new[]
        {
            Environment.CurrentDirectory,
            AppContext.BaseDirectory,
        })
        {
            DirectoryInfo? directory = new DirectoryInfo(startPath);
            while (directory is not null)
            {
                foreach (string fileName in fileNames)
                {
                    string candidate = Path.Combine(
                        directory.FullName,
                        fileName);
                    if (File.Exists(candidate))
                        return candidate;
                }

                directory = directory.Parent;
            }
        }

        return Path.Combine(
            Environment.CurrentDirectory,
            fileNames[0]);
    }

    private static IImageClassificationDataset CreateDataset(
        DatasetConfiguration configuration,
        string role)
    {
        if (configuration.IsType(DatasetConfiguration.MnistType))
        {
            EnsureDataFile(configuration.ImagePath, $"{role} image");
            EnsureDataFile(configuration.LabelPath, $"{role} label");
            return datasets.mnist(
                images: configuration.ImagePath,
                labels: configuration.LabelPath);
        }

        if (configuration.IsType(DatasetConfiguration.Cifar100Type))
        {
            EnsureDataFile(configuration.DataPath, $"{role} CIFAR-100");
            return datasets.cifar100(
                data: configuration.DataPath,
                patch_size: configuration.PatchSize,
                normalize: configuration.Normalize,
                random_crop_padding:
                    configuration.Augmentation.RandomCropPadding,
                horizontal_flip:
                    configuration.Augmentation.HorizontalFlip,
                vertical_flip:
                    configuration.Augmentation.VerticalFlip);
        }

        throw new ArgumentException(
            $"Unsupported {role.ToLowerInvariant()} dataset type " +
            $"'{configuration.Type}'.",
            nameof(configuration));
    }

    private static void ValidateDatasetCompatibility(
        IImageClassificationDataset training,
        IImageClassificationDataset evaluation)
    {
        if (training.Rows != evaluation.Rows
            || training.Columns != evaluation.Columns)
        {
            throw new ArgumentException(
                $"Training input shape '{training.Rows}x" +
                $"{training.Columns}' does not match evaluation input " +
                $"shape '{evaluation.Rows}x{evaluation.Columns}'.");
        }

        if (training.ClassCount != evaluation.ClassCount)
        {
            throw new ArgumentException(
                $"Training class count '{training.ClassCount}' does not " +
                $"match evaluation class count " +
                $"'{evaluation.ClassCount}'.");
        }
    }

    private static string GetCifar100AugmentationDescription(
        Cifar100Options options)
    {
        var operations = new List<string>();
        if (options.RandomCropPadding > 0)
        {
            operations.Add(
                $"random crop (padding {options.RandomCropPadding})");
        }
        if (options.HorizontalFlip)
            operations.Add("horizontal flip");
        if (options.VerticalFlip)
            operations.Add("vertical flip");

        return operations.Count == 0
            ? "augmentation = disabled"
            : $"augmentation = {string.Join(", ", operations)}";
    }

    private static BatchSamples ReadBatch(
        IImageClassificationDataset dataset,
        int[]? order,
        int start,
        int count,
        Random? trainingRandom = null)
    {
        var inputValues = new float[count * dataset.ImageSize];
        var answers = new int[count];

        for (int offset = 0; offset < count; offset++)
        {
            int index = order is null ? start + offset : order[start + offset];
            Span<float> destination = inputValues.AsSpan(
                offset * dataset.ImageSize,
                dataset.ImageSize);
            int answer = trainingRandom is null
                ? dataset.ReadSample(index, destination)
                : dataset.ReadTrainingSample(
                    index,
                    destination,
                    trainingRandom);
            answers[offset] = answer;
        }

        return new BatchSamples(
            Tensor.FromOwnedData(
                inputValues,
                [count, dataset.Rows, dataset.Columns],
                "classifierInput"),
            answers);
    }

    private static int CountCorrect(
        IReadOnlyList<float> logits,
        IReadOnlyList<int> answers,
        int classCount)
    {
        int correct = 0;
        for (int sample = 0; sample < answers.Count; sample++)
        {
            if (ArgMax(logits, sample * classCount, classCount)
                == answers[sample])
            {
                correct++;
            }
        }

        return correct;
    }

    private static CudaClassificationCorrectCountReadback?
        BeginCountCorrect(
            Tensor logits,
            int[] answers,
            int classCount)
        => Tensor.ExecutionDevice == TensorDevice.Cuda
            ? logits.BeginCudaClassificationCorrectCount(
                answers,
                classCount)
            : null;

    private static int CompleteCountCorrect(
        Tensor logits,
        IReadOnlyList<int> answers,
        int classCount,
        CudaClassificationCorrectCountReadback? readback)
        => readback is not null
            ? readback.CompleteAndReturn()
            : CountCorrect(logits.Data, answers, classCount);

    internal static int CountCorrectForMetrics(
        Tensor logits,
        int[] answers,
        int classCount)
    {
        ArgumentNullException.ThrowIfNull(logits);
        ArgumentNullException.ThrowIfNull(answers);
        using CudaClassificationCorrectCountReadback? readback =
            BeginCountCorrect(logits, answers, classCount);
        return CompleteCountCorrect(
            logits,
            answers,
            classCount,
            readback);
    }

    private static int ArgMax(
        IReadOnlyList<float> values,
        int offset,
        int count)
    {
        int result = 0;
        float best = values[offset];
        for (int index = 1; index < count; index++)
        {
            if (values[offset + index] > best)
            {
                best = values[offset + index];
                result = index;
            }
        }

        return result;
    }

    private static void EnsureDataFile(string path, string role)
    {
        if (!File.Exists(path))
        {
            throw new FileNotFoundException(
                $"{role} data file was not found at '{path}'. " +
                "Check the corresponding path in the training " +
                "configuration.",
                path);
        }
    }

    private readonly record struct BatchSamples(
        Tensor Input,
        int[] Answers);

    internal readonly record struct LearningRates(
        float Primary,
        float? Auxiliary);

    private sealed record ModelCheckpoint(
        int FormatVersion,
        int Epoch,
        float EvaluationLoss,
        ModuleState Model)
    {
        internal const int CurrentFormatVersion = 1;
    }
}
