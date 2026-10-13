using System.Diagnostics;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Optimization;

namespace NNtrain;

internal static partial class ClassificationTrainingCommand
{
    internal static int Run(
        CanonicalClassificationTrainingSpec classification,
        string configurationPath,
        bool resumeFromCheckpoint,
        bool autoResume,
        TextWriter output,
        TextWriter error,
        bool openLossGraph)
    {
        TrainingConfiguration config = classification.Configuration;
        using TrainingRunGuard classificationRun = TrainingRunGuard.Begin(
            classification.ConfigurationPath,
            config.CheckpointPath);
        bool resumeClassification = TrainingResumePolicy.Resolve(
            resumeFromCheckpoint,
            autoResume || config.AutoResume,
            classificationRun,
            config.CheckpointPath,
            output);
        if (resumeClassification)
            config = config with { ResumeFromCheckpoint = true };
        torch.manual_seed(config.Seed);
        Tensor.SimdEnabled = config.UseSimd;
        IImageClassificationDataset trainData = ClassificationDatasets.Create(
            config.TrainingData,
            "Training");
        IImageClassificationDataset evalData = ClassificationDatasets.Create(
            config.EvaluationData,
            "Evaluation");

        ClassificationDatasets.ValidateCompatibility(trainData, evalData);
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
        ClassificationTrainingConsole.WriteModel(
            model,
            classification,
            output);
        OptimizerBundle optimizer = ClassificationOptimizerFactory.CreateBundle(
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
        ClassificationTrainingConsole.WriteSettings(
            config,
            trainData,
            output);

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
            ClassificationOptimizerFactory.ApplyMuonPolicyAfterResume(
                config,
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

        var checkpointSchedule = new CheckpointSchedule(config.CheckpointIntervalMinutes);
        output.WriteLine($"checkpoint interval = {config.CheckpointIntervalMinutes:G} minutes (and epoch end)");
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
                if (completedUpdates < updateTotal && checkpointSchedule.IsDue)
                {
                    ProductionTrainingSessionFactory
                        .EnsureCanPublishCheckpoint(
                            trainingSession,
                            globalStep);
                    ClassificationCheckpoint.Save(
                        config.CheckpointPath,
                        model,
                        optimizer,
                        ClassificationCheckpointStateFactory.Create(
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
                    checkpointSchedule.RecordSaved();
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
                        correctCountReadback = ClassificationMetrics.BeginCountCorrect(
                            logits,
                            samples.target,
                            evalData.ClassCount);
                    evalLoss += loss.item() * samplesInBatch;
                    evalCorrectCount += ClassificationMetrics.CompleteCountCorrect(
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
                ClassificationCheckpointStateFactory.Create(
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

            string epochSnapshotPath = CheckpointSnapshot.Save(
                config.CheckpointPath, model.GetType().Name, epoch, model);
            output.WriteLine($"model snapshot = {epochSnapshotPath}");
            checkpointSchedule.RecordSaved();

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
                : ClassificationModelCheckpoint.GetBestModelCheckpointPath(config.CheckpointPath);
            ClassificationModelCheckpoint.Save(
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

    private readonly record struct LearningRates(
        float Primary,
        float? Auxiliary);
}
