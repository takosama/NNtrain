using System.Diagnostics;
using System.Security.Cryptography;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;
using NNtrain.Training.Optimization;
using NNtrain.Training.Persistence;

namespace NNtrain.Benchmarks;

internal static class PerformanceBaselineRunner
{
    internal static BaselineScenarioResult Run(BaselineWorkerJob job)
    {
        ArgumentNullException.ThrowIfNull(job);
        Validate(job);
        BaselineScenario scenario = job.Scenario;
        BaselineModelConfiguration configuration = job.Model;
        TensorPrecisionMode precision = TensorPrecisionModeNames.Parse(
            configuration.Precision);
        TensorDType storageDType = precision.ToStorageDType();

        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDeviceIndices = Tensor.CudaDeviceIndices.ToArray();
        bool previousSimd = Tensor.SimdEnabled;
        try
        {
            Tensor.ExecutionDevice = scenario.Device == BaselineDeviceKind.Cuda
                ? TensorDevice.Cuda
                : TensorDevice.Cpu;
            Tensor.CudaDeviceIndices = scenario.DeviceIndices;
            Tensor.SimdEnabled = true;

            using ExecutionSession execution = CreateExecutionSession(
                scenario,
                precision);
            using IDisposable executionScope = execution.Enter();

            IReadOnlyList<BaselineGpu> selectedGpus = ResolveSelectedGpus(
                scenario, job.Gpus);
            if (scenario.Soak is not null)
            {
                return RunSoak(
                    job,
                    precision,
                    storageDType,
                    selectedGpus,
                    execution);
            }
            var runs = new List<BaselineRunResult>(scenario.Repetitions);
            BaselinePhaseProbe? probe = null;
            for (int repetition = 1; repetition <= scenario.Repetitions;
                repetition++)
            {
                using BaselineFixture fixture = CreateFixture(
                    configuration,
                    precision,
                    storageDType,
                    scenario);
                Console.WriteLine(
                    $"{scenario.Name}: repetition {repetition}/" +
                    $"{scenario.Repetitions}, warming up " +
                    $"{scenario.WarmupSteps} steps");
                for (int warmup = 0; warmup < scenario.WarmupSteps; warmup++)
                {
                    _ = MeasureStep(
                        fixture,
                        configuration,
                        scenario,
                        diagnosticLabel: $"warmup-{warmup + 1}");
                }
                CudaTrainingGraphTelemetry? graphBeforeMeasurement =
                    fixture.DataParallelEngine?.TrainingGraphTelemetry;

                DateTimeOffset started = DateTimeOffset.UtcNow;
                var measurements = new List<BaselineStepMeasurement>(
                    scenario.MeasuredSteps);
                int progressInterval = Math.Max(
                    1, scenario.MeasuredSteps / 20);
                for (int step = 1; step <= scenario.MeasuredSteps; step++)
                {
                    RawStep raw = MeasureStep(
                        fixture,
                        configuration,
                        scenario,
                        diagnosticLabel: $"steady-{step}");
                    measurements.Add(raw.ToMeasurement(step));
                    if (step == 1
                        || step == scenario.MeasuredSteps
                        || step % progressInterval == 0)
                    {
                        Console.WriteLine(
                            $"{scenario.Name} run {repetition}/" +
                            $"{scenario.Repetitions} step {step,3}/" +
                            $"{scenario.MeasuredSteps}: " +
                            $"{raw.TotalMilliseconds,9:F2} ms, " +
                            $"loss={raw.Loss:F6}, " +
                            $"fwd+bwd=" +
                            $"{raw.ForwardBackwardMilliseconds:F2}, " +
                            $"clip={raw.ClipMilliseconds:F2}, " +
                            $"optimizer={raw.OptimizerMilliseconds:F2}");
                    }
                }
                CudaTrainingGraphTelemetry? graphAfterMeasurement =
                    fixture.DataParallelEngine?.TrainingGraphTelemetry;
                runs.Add(CreateRun(
                    repetition,
                    started,
                    DateTimeOffset.UtcNow,
                    fixture.DataParallelEngine?.LastShardBatchSizes ?? [],
                    CreateGraphTelemetry(
                        graphBeforeMeasurement,
                        graphAfterMeasurement,
                        scenario.MeasuredSteps),
                    measurements));
                if (scenario.CollectPhaseProbe
                    && repetition == scenario.Repetitions)
                {
                    // Reuse the already-warmed final fixture. Constructing a
                    // fourth full model/optimizer/graph after the official
                    // three runs can retain enough session-generation memory
                    // to make the diagnostic eager backward fail even though
                    // every measured run itself completed. The probe remains
                    // outside all measured intervals and runs only after the
                    // final BaselineRunResult has been committed.
                    probe = CreatePhaseProbe(
                        fixture,
                        configuration,
                        scenario,
                        runs);
                }
            }
            BaselineConditions conditions = CreateConditions(
                job, precision, storageDType, selectedGpus);
            BaselineDistribution aggregateStep = BaselineDistribution.From(
                runs.SelectMany(run => run.Measurements)
                    .Select(measurement => measurement.TotalMilliseconds));
            BaselineValidationResult? validation =
                scenario.PerformanceGate is { } performanceGate
                    ? PerformanceBaselineGatePolicy.Evaluate(
                        performanceGate,
                        scenario,
                        configuration,
                        runs)
                    : null;
            return new BaselineScenarioResult(
                conditions,
                runs,
                aggregateStep,
                probe,
                CreateNotes(scenario),
                validation);
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDeviceIndices;
            Tensor.SimdEnabled = previousSimd;
        }
    }

    private static BaselineScenarioResult RunSoak(
        BaselineWorkerJob job,
        TensorPrecisionMode precision,
        TensorDType storageDType,
        IReadOnlyList<BaselineGpu> selectedGpus,
        ExecutionSession execution)
    {
        BaselineScenario scenario = job.Scenario;
        BaselineModelConfiguration configuration = job.Model;
        BaselineSoakConfiguration soak = scenario.Soak
            ?? throw new InvalidOperationException(
                "A soak scenario requires soak configuration.");
        string artifactDirectory = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-checkpoint-soak-{Guid.NewGuid():N}");
        string checkpointPath = Path.Combine(
            artifactDirectory, "wiki-soak-checkpoint.json");
        string sidecarPath = Path.Combine(
            artifactDirectory, "metrics.jsonl");
        string htmlPath = Path.Combine(
            artifactDirectory, "loss.html");
        Directory.CreateDirectory(artifactDirectory);

        var measurements = new List<BaselineStepMeasurement>(
            soak.TotalCommittedSteps);
        var runtimeErrors = new List<string>();
        BaselineFixture? fixture = null;
        TrainingSession? trainingSession = null;
        MetricJournalJsonlRepository? metrics = null;
        bool zeroShardObserved = false;
        bool generationObserved = false;
        int generatedTokens = 0;
        double? generationMilliseconds = null;
        bool restartObserved = false;
        bool resumeArtifactValidated = false;
        bool sidecarContinuityValidated = false;
        bool htmlContinuityValidated = false;
        string htmlContinuityStatus = "not checked";
        long resumeArtifactBytes = 0;
        string? resumeArtifactSha256 = null;
        BaselineCheckpointResult checkpointResult =
            EmptyCheckpointResult(artifactDirectory);
        int sidecarEntriesBeforeRestart = 0;
        int sidecarEntriesAfterResume = 0;
        double committedLossSum = 0d;
        long committedTargetCount = 0;
        long completedDocuments = 0;
        DateTimeOffset started = DateTimeOffset.UtcNow;
        IReadOnlyList<int> finalShards = [];

        try
        {
            fixture = CreateFixture(
                configuration,
                precision,
                storageDType,
                scenario);
            fixture.PrewarmCompiledTrainingPlan(
                configuration,
                globalStep: 0,
                CreateAdaptiveShardingOptions(configuration),
                adaptiveState: null);
            trainingSession = new TrainingSession(
                execution,
                ownsExecutionSession: false,
                lastCommittedStep: -1);
            metrics = new MetricJournalJsonlRepository(sidecarPath);
            TrainingMetricReporter htmlMetrics =
                TrainingMetricReporter.Open(
                    htmlPath,
                    totalEpochs: 1,
                    resume: false,
                    checkpointGlobalStep: -1,
                    checkpointEpoch: 0d,
                    renderHtml: true);
            int progressInterval = Math.Max(
                1, soak.TotalCommittedSteps / 20);

            for (int committedIndex = 0;
                committedIndex < soak.TotalCommittedSteps;
                committedIndex++)
            {
                // Production Wiki training numbers the first committed step
                // as one. Keep the soak on that same public/checkpoint scale
                // so the configured step-2000 event is the 2000th commit, not
                // the 2001st commit hidden behind a zero-based loop index.
                long globalStep = checked((long)committedIndex + 1L);
                RawStep raw;
                using (TrainingStep step = trainingSession.BeginStep(globalStep))
                {
                    try
                    {
                        step.Advance(TrainingStepPhase.BatchAcquired);
                        raw = MeasureStep(
                            fixture,
                            configuration,
                            scenario,
                            globalStep,
                            scheduleProgress: globalStep
                                / soak.TotalCommittedSteps);
                        step.Advance(TrainingStepPhase.GradientsCleared);
                        step.Advance(TrainingStepPhase.ForwardCompleted);
                        step.Advance(TrainingStepPhase.BackwardCompleted);
                        step.Advance(TrainingStepPhase.GradientsReduced);
                        step.Advance(TrainingStepPhase.GradientsClipped);
                        step.Advance(TrainingStepPhase.ScheduleApplied);
                        step.Advance(TrainingStepPhase.OptimizerCommitted);
                        metrics.AppendAndFlush(new MetricJournalEntry(
                            globalStep,
                            globalStep / (double)soak.TotalCommittedSteps,
                            globalStep / (double)soak.TotalCommittedSteps,
                            MetricKinds.TrainLoss,
                            raw.Loss,
                            DateTimeOffset.UtcNow));
                        if (ShouldWriteSoakHtmlMetric(
                                committedIndex,
                                globalStep,
                                soak))
                        {
                            htmlMetrics.AppendCommittedLoss(
                                globalStep,
                                globalStep
                                    / (double)soak.TotalCommittedSteps,
                                MetricKinds.TrainLoss,
                                raw.Loss);
                        }
                        step.Advance(TrainingStepPhase.MetricsCommitted);
                    }
                    catch (Exception exception)
                    {
                        step.Fault(exception);
                        throw;
                    }
                }

                if (!float.IsFinite(raw.Loss))
                {
                    throw new InvalidOperationException(
                        $"Non-finite loss at committed global step " +
                        $"{globalStep}: {raw.Loss}.");
                }
                committedLossSum += raw.Loss;
                committedTargetCount = checked(
                    committedTargetCount
                    + (long)configuration.Batch * configuration.Sequence);
                completedDocuments = checked(
                    completedDocuments + configuration.Batch);
                bool isWarmup = committedIndex
                    < soak.PerformanceWarmupSteps;
                measurements.Add(raw.ToMeasurement(
                    checked(committedIndex + 1), isWarmup));
                IReadOnlyList<int> shards =
                    fixture.DataParallelEngine?.LastShardBatchSizes ?? [];
                if (shards.Count != scenario.DeviceIndices.Length
                    || shards.Any(value => value <= 0))
                {
                    zeroShardObserved = true;
                }
                finalShards = shards.ToArray();
                if (globalStep == soak.GenerationStep)
                {
                    int promptLength = Math.Max(
                        1, Math.Min(configuration.Sequence, 8));
                    int[] prompt = Enumerable.Repeat(1, promptLength).ToArray();
                    long generationStarted = Stopwatch.GetTimestamp();
                    int[] generated = fixture.Model.GenerateTokenIds(
                        prompt,
                        soak.GenerationTokens,
                        temperature: 0f,
                        topK: 1,
                        stopTokenId: null,
                        random: new Random(configuration.Seed + committedIndex));
                    Synchronize(scenario);
                    generationMilliseconds = Elapsed(generationStarted);
                    generatedTokens = generated.Length - prompt.Length;
                    generationObserved = generatedTokens
                        == soak.GenerationTokens;
                }

                if (globalStep == soak.RestartStep)
                {
                    Synchronize(scenario);
                    CudaAdaptiveShardState expectedAdaptiveState =
                        fixture.DataParallelEngine!
                            .CaptureAdaptiveShardingState();
                    TrainingRandomState? expectedRandomState =
                        fixture.Model.CaptureTrainingRandomState();
                    LRSchedulerStateDictionary expectedSchedulerState =
                        fixture.Scheduler.state_dict();
                    int expectedNekoMuonStep =
                        fixture.NekoMuon.GetDiagnostics().Step;
                    int expectedAdamWStep = fixture.AdamW.StreamingStep;
                    int expectedParameterCount = fixture.Parameters.Length;
                    int expectedCompletedBatches = checked(committedIndex + 1);
                    int[] expectedTokenBuffer = fixture.Input.ToArray();
                    MetricJournalLoadResult beforeRestart = metrics.Load();
                    sidecarEntriesBeforeRestart =
                        beforeRestart.Journal.Count;
                    if (beforeRestart.IgnoredCorruptTail
                        || sidecarEntriesBeforeRestart
                            != expectedCompletedBatches
                        || beforeRestart.Journal.Entries[^1].GlobalStep
                            != globalStep)
                    {
                        throw new InvalidDataException(
                            "Metrics sidecar was not committed through the " +
                            "checkpoint step.");
                    }

                    WikiTrainingConfiguration checkpointConfiguration =
                        CreateSoakCheckpointConfiguration(
                            configuration,
                            precision,
                            checkpointPath,
                            scenario.DeviceIndices,
                            resume: false);
                    long saveStarted = Stopwatch.GetTimestamp();
                    WikiLanguageModelCommand.SaveTrainingCheckpoint(
                        checkpointConfiguration,
                        configuration.Vocabulary,
                        completedEpoch: 0,
                        new ModuleState(ModuleState.CurrentFormatVersion, []),
                        bestLoss: raw.Loss,
                        bestEpoch: 0,
                        fixture.Model,
                        fixture.Optimizer,
                        fixture.Scheduler,
                        globalStep,
                        currentEpoch: 1,
                        completedBatchesInEpoch: expectedCompletedBatches,
                        currentLossSum: committedLossSum,
                        currentTargetCount: committedTargetCount,
                        completedDocumentsInEpoch: completedDocuments,
                        currentTokenBuffer: expectedTokenBuffer,
                        adaptiveCudaShardState: expectedAdaptiveState,
                        checkpointFaultInjector:
                            soak.InjectCheckpointFailureAfterFirstArtifact
                                ? new SoakCheckpointFaultInjector()
                                : null);
                    Synchronize(scenario);
                    double saveMilliseconds = Elapsed(saveStarted);
                    TrainingRandomState? probeRandomState =
                        fixture.Model.CaptureTrainingRandomState();
                    float lossBeforeReload = fixture.DataParallelEngine!
                        .ForwardBackward(
                            fixture.Input,
                            fixture.Target,
                            configuration.Batch,
                            configuration.Sequence,
                            Tensor.DefaultCrossEntropyIgnoreIndex,
                            checked(globalStep + 1));
                    fixture.Optimizer.zero_grad();
                    fixture.Model.RestoreTrainingRandomState(probeRandomState);
                    CheckpointArtifactSnapshot checkpointSnapshot =
                        CaptureCheckpointArtifacts(checkpointPath);
                    BaselineCheckpointArtifact manifestArtifact =
                        checkpointSnapshot.Artifacts.Single(artifact =>
                            string.Equals(
                                artifact.Name,
                                Path.GetFileName(checkpointPath),
                                StringComparison.OrdinalIgnoreCase));
                    resumeArtifactBytes = manifestArtifact.Bytes;
                    resumeArtifactSha256 = manifestArtifact.Sha256;
                    resumeArtifactValidated =
                        checkpointSnapshot.Manifest.FormatVersion == 8
                        && checkpointSnapshot
                            .ArtifactFirstManifestLastValidated;
                    checkpointResult = checkpointResult with
                    {
                        FormatVersion =
                            checkpointSnapshot.Manifest.FormatVersion,
                        SaveMilliseconds = saveMilliseconds,
                        TotalBytes = checkpointSnapshot.TotalBytes,
                        Artifacts = checkpointSnapshot.Artifacts,
                        ArtifactFirstManifestLastValidated =
                            checkpointSnapshot
                                .ArtifactFirstManifestLastValidated,
                    };

                    trainingSession.Dispose();
                    trainingSession = null;
                    fixture.Dispose();
                    fixture = null;
                    Synchronize(scenario);
                    bool oldFixtureDisposedBeforeReload = true;

                    long loadStarted = Stopwatch.GetTimestamp();
                    fixture = CreateFixture(
                        configuration,
                        precision,
                        storageDType,
                        scenario);
                    WikiTrainingConfiguration resumeConfiguration =
                        checkpointConfiguration with
                        {
                            ResumeFromCheckpoint = true,
                            AutoResume = false,
                        };
                    ModuleState? restoredBestState = null;
                    float restoredBestLoss = float.PositiveInfinity;
                    int restoredBestEpoch = -1;
                    long restoredGlobalStep = -1;
                    using var resumeOutput = new StringWriter();
                    WikiLanguageModelCommand.WikiResumePosition restored =
                        WikiLanguageModelCommand.RestoreTrainingCheckpoint(
                            resumeConfiguration,
                            fixture.Model,
                            fixture.Optimizer,
                            fixture.Scheduler,
                            ref restoredBestState,
                            ref restoredBestLoss,
                            ref restoredBestEpoch,
                            ref restoredGlobalStep,
                            resumeOutput);
                    bool cursorValidated =
                        restoredGlobalStep == globalStep
                        && restored.Epoch == 1
                        && restored.CompletedBatches
                            == expectedCompletedBatches
                        && BitConverter.DoubleToInt64Bits(restored.LossSum)
                            == BitConverter.DoubleToInt64Bits(
                                committedLossSum)
                        && restored.TargetCount == committedTargetCount
                        && restored.CompletedDocuments
                            == completedDocuments
                        && restored.TokenBuffer.AsSpan().SequenceEqual(
                            expectedTokenBuffer)
                        && restoredBestState is null
                        && restoredBestEpoch == 0
                        && BitConverter.SingleToInt32Bits(restoredBestLoss)
                            == BitConverter.SingleToInt32Bits(raw.Loss);
                    bool randomValidated = TrainingRandomStatesEqual(
                        expectedRandomState,
                        fixture.Model.CaptureTrainingRandomState());
                    bool schedulerValidated =
                        fixture.Scheduler.state_dict()
                            == expectedSchedulerState;
                    bool optimizerStateValidated =
                        fixture.NekoMuon.GetDiagnostics().Step
                            == expectedNekoMuonStep
                        && fixture.AdamW.StreamingStep
                            == expectedAdamWStep;
                    (bool modelPayloadValidated,
                        bool optimizerPayloadValidated) =
                        ValidateReloadedCheckpointPayloads(
                            checkpointPath,
                            checkpointSnapshot.Manifest,
                            fixture.Model,
                            fixture.Optimizer,
                            artifactDirectory);
                    bool adaptiveStateValidated =
                        AdaptiveShardStatesEqual(
                            expectedAdaptiveState,
                            restored.AdaptiveCudaShardState);
                    if (restored.AdaptiveCudaShardState is not
                        { } restoredAdaptiveState)
                    {
                        throw new InvalidDataException(
                            "Checkpoint did not restore adaptive CUDA shard " +
                            "state.");
                    }
                    fixture.DataParallelEngine!.RestoreAdaptiveShardingState(
                        restoredAdaptiveState);
                    // prepare() is the optimizer residency contract: it does
                    // not return until every persistent moment, descriptor,
                    // scratch buffer, and parameter master is present on all
                    // configured devices.
                    fixture.PrepareForTraining(configuration.Batch);
                    bool optimizerValidated = optimizerStateValidated
                        && optimizerPayloadValidated;
                    bool precisionValidated =
                        fixture.Model.PrecisionMode == precision
                        && fixture.Model.DType == storageDType
                        && checkpointSnapshot.Manifest.PrecisionMode
                            == precision
                        && checkpointSnapshot.Manifest.ModelDType
                            == storageDType;
                    bool bfp8BlockSizeValidated =
                        ValidateBfp8BlockSize(
                            fixture.Parameters,
                            precision,
                            configuration.Bfp8BlockSize)
                        && (precision != TensorPrecisionMode.Mix8_32
                            || checkpointSnapshot.Manifest.Bfp8BlockSize
                                == configuration.Bfp8BlockSize);
                    bool modelValidated =
                        fixture.Model is GptRinWikiJp
                        && modelPayloadValidated
                        && fixture.Parameters.Length
                            == expectedParameterCount
                        && checkpointSnapshot.Manifest.VocabularySize
                            == fixture.Model.VocabularySize
                        && checkpointSnapshot.Manifest.ContextLength
                            == fixture.Model.ContextLength
                        && checkpointSnapshot.Manifest.ModelWidth
                            == fixture.Model.ModelWidth;
                    bool deviceResidencyValidated =
                        ValidateCudaResidency(
                            fixture.Parameters,
                            scenario.DeviceIndices);
                    Synchronize(scenario);
                    double loadMilliseconds = Elapsed(loadStarted);

                    if (!cursorValidated
                        || !randomValidated
                        || !schedulerValidated
                        || !optimizerValidated
                        || !adaptiveStateValidated
                        || !precisionValidated
                        || !bfp8BlockSizeValidated
                        || !modelValidated
                        || !deviceResidencyValidated)
                    {
                        throw new InvalidDataException(
                            "The full Wiki checkpoint restart failed one or " +
                            "more state/residency validations.");
                    }
                    checkpointResult = checkpointResult with
                    {
                        Validated = resumeArtifactValidated,
                        LoadMilliseconds = loadMilliseconds,
                        CursorValidated = cursorValidated,
                        TrainingRandomValidated = randomValidated,
                        SchedulerValidated = schedulerValidated,
                        AdaptiveShardStateValidated =
                            adaptiveStateValidated,
                        ModelValidated = modelValidated,
                        OptimizerValidated = optimizerValidated,
                        PrecisionValidated = precisionValidated,
                        Bfp8BlockSizeValidated =
                            bfp8BlockSizeValidated,
                        DeviceResidencyValidated =
                            deviceResidencyValidated,
                        OldFixtureDisposedBeforeReload =
                            oldFixtureDisposedBeforeReload,
                        ArtifactDirectory = null,
                    };

                    TrainingRandomState? restoredRandomState =
                        fixture.Model.CaptureTrainingRandomState();
                    float lossAfterReload = fixture.PrewarmCompiledTrainingPlan(
                        configuration,
                        checked(globalStep + 1),
                        CreateAdaptiveShardingOptions(configuration),
                        restoredAdaptiveState);
                    fixture.Model.RestoreTrainingRandomState(
                        restoredRandomState);
                    float lossDelta = MathF.Abs(
                        lossBeforeReload - lossAfterReload);
                    Console.WriteLine(
                        $"checkpoint forward continuity: before=" +
                        $"{lossBeforeReload:G9}, after={lossAfterReload:G9}, " +
                        $"delta={lossDelta:G9}");
                    // A reconstructed CUDA Graph receives new per-operation
                    // dropout seeds, so training-mode loss is diagnostic and
                    // is not expected to be bitwise equal. Exact streamed
                    // model and optimizer artifact hashes above are the
                    // authoritative resume-continuity check.
                    metrics = new MetricJournalJsonlRepository(sidecarPath);
                    MetricJournalLoadResult resumedMetrics = metrics.Load();
                    sidecarContinuityValidated =
                        !resumedMetrics.IgnoredCorruptTail
                        && resumedMetrics.Journal.Count
                            == sidecarEntriesBeforeRestart
                        && resumedMetrics.Journal.Entries[^1].GlobalStep
                            == restoredGlobalStep;
                    string htmlSidecarPath =
                        TrainingMetricReporter.GetSidecarPath(htmlPath);
                    MetricJournalLoadResult htmlBeforeResume =
                        new MetricJournalJsonlRepository(htmlSidecarPath)
                            .Load();
                    int htmlEntriesBeforeResume =
                        htmlBeforeResume.Journal.Count;
                    htmlMetrics = TrainingMetricReporter.Open(
                        htmlPath,
                        totalEpochs: 1,
                        resume: true,
                        checkpointGlobalStep: globalStep,
                        checkpointEpoch: globalStep
                            / (double)soak.TotalCommittedSteps,
                        renderHtml: true);
                    MetricJournalLoadResult htmlAfterResume =
                        new MetricJournalJsonlRepository(htmlSidecarPath)
                            .Load();
                    htmlContinuityValidated =
                        htmlEntriesBeforeResume > 0
                        && !htmlBeforeResume.IgnoredCorruptTail
                        && !htmlAfterResume.IgnoredCorruptTail
                        && htmlAfterResume.Journal.Count
                            == htmlEntriesBeforeResume
                        && File.Exists(htmlPath)
                        && new FileInfo(htmlPath).Length > 0;
                    trainingSession = new TrainingSession(
                        execution,
                        ownsExecutionSession: false,
                        lastCommittedStep: restoredGlobalStep);
                    restartObserved = true;
                }

                if (committedIndex == 0
                    || committedIndex + 1 == soak.TotalCommittedSteps
                    || (committedIndex + 1) % progressInterval == 0)
                {
                    Console.WriteLine(
                        $"{scenario.Name} committed {committedIndex + 1,4}/" +
                        $"{soak.TotalCommittedSteps}: " +
                        $"{raw.TotalMilliseconds,9:F2} ms, " +
                        $"loss={raw.Loss:F6}, shards=[" +
                        $"{string.Join(',', finalShards)}]");
                }
            }

            if (metrics is not null)
            {
                MetricJournalLoadResult finalMetrics = metrics.Load();
                sidecarEntriesAfterResume = finalMetrics.Journal.Count;
                sidecarContinuityValidated &=
                    !finalMetrics.IgnoredCorruptTail
                    && finalMetrics.Journal.Count == measurements.Count
                    && finalMetrics.Journal.Entries[^1].GlobalStep
                        == measurements.Count;

                string htmlSidecarPath =
                    TrainingMetricReporter.GetSidecarPath(htmlPath);
                MetricJournalLoadResult htmlFinal =
                    new MetricJournalJsonlRepository(htmlSidecarPath).Load();
                IReadOnlyList<LossGraph.LossPoint> renderedPoints =
                    new LossGraph(htmlPath, totalEpochs: 1)
                        .ImportExisting(resumeEpoch: 1f);
                htmlContinuityValidated &=
                    !htmlFinal.IgnoredCorruptTail
                    && htmlFinal.Journal.Count > 1
                    && renderedPoints.Count == htmlFinal.Journal.Count
                    && htmlFinal.Journal.Entries[0].GlobalStep == 1
                    && htmlFinal.Journal.Entries[^1].GlobalStep
                        == measurements.Count;
                htmlContinuityStatus =
                    $"sidecar={htmlFinal.Journal.Count}, rendered=" +
                    $"{renderedPoints.Count}, bytes=" +
                    $"{new FileInfo(htmlPath).Length}";
            }
        }
        catch (Exception exception)
        {
            AddRuntimeErrors(exception, runtimeErrors);
        }
        finally
        {
            try
            {
                trainingSession?.Dispose();
            }
            catch (Exception exception)
            {
                AddRuntimeErrors(exception, runtimeErrors);
            }
            try
            {
                fixture?.Dispose();
            }
            catch (Exception exception)
            {
                AddRuntimeErrors(exception, runtimeErrors);
            }
        }

        if (runtimeErrors.Count > 0)
        {
            try
            {
                IReadOnlyList<BaselineCheckpointArtifact> partialArtifacts =
                    checkpointResult.Artifacts.Count > 0
                        ? checkpointResult.Artifacts
                        : CapturePartialCheckpointArtifacts(checkpointPath);
                checkpointResult = checkpointResult with
                {
                    Artifacts = partialArtifacts,
                    TotalBytes = partialArtifacts.Sum(artifact =>
                        artifact.Bytes),
                    ArtifactsRetainedAfterFailure = true,
                    ArtifactDirectory = artifactDirectory,
                };
            }
            catch (Exception exception)
            {
                AddRuntimeErrors(exception, runtimeErrors);
                checkpointResult = checkpointResult with
                {
                    ArtifactsRetainedAfterFailure = true,
                    ArtifactDirectory = artifactDirectory,
                };
            }
        }

        BaselineStepMeasurement[] performanceMeasurements = measurements
            .Where(value => !value.IsWarmup)
            .ToArray();
        int window = Math.Min(
            soak.TrendWindowSteps,
            performanceMeasurements.Length / 2);
        BaselineDistribution firstWindow = BaselineDistribution.From(
            performanceMeasurements.Take(window)
                .Select(value => value.TotalMilliseconds));
        BaselineDistribution lastWindow = BaselineDistribution.From(
            performanceMeasurements.TakeLast(window)
                .Select(value => value.TotalMilliseconds));
        double ratio = window == 0 || firstWindow.P50 <= 0d
            ? 0d
            : lastWindow.P50 / firstWindow.P50;
        IReadOnlyList<BaselineDeviceMemorySummary> postWarmupMemory =
            SummarizePostWarmupDeviceMemory(
                measurements,
                soak.PerformanceWarmupSteps);
        var soakResult = new BaselineSoakResult(
            soak.TotalCommittedSteps,
            measurements.Count,
            soak.PerformanceWarmupSteps,
            soak.TrendWindowSteps,
            firstWindow,
            lastWindow,
            ratio,
            soak.GenerationStep,
            generationObserved,
            generatedTokens,
            generationMilliseconds,
            soak.RestartStep,
            restartObserved,
            resumeArtifactValidated,
            resumeArtifactBytes,
            resumeArtifactSha256,
            checkpointResult,
            sidecarEntriesBeforeRestart,
            sidecarEntriesAfterResume,
            sidecarContinuityValidated,
            true,
            htmlContinuityValidated,
            htmlContinuityStatus,
            postWarmupMemory,
            zeroShardObserved,
            runtimeErrors);
        IReadOnlyList<BaselineGateResult> gates = CreateSoakGates(
            soak,
            soakResult,
            selectedGpus.Count);
        var validation = new BaselineValidationResult(
            "full Wiki v8 streaming-checkpoint two-GPU soak; the old model, " +
            "optimizer, and data-parallel engine are destroyed before a " +
            "fresh fixture is restored",
            gates.All(gate => gate.Passed == true),
            gates,
            soakResult);

        BaselineRunResult run = CreateRun(
            repetition: 1,
            started,
            DateTimeOffset.UtcNow,
            finalShards,
            trainingGraph: null,
            measurements);
        BaselineDistribution aggregate = BaselineDistribution.From(
            performanceMeasurements.Select(value =>
                value.TotalMilliseconds));
        BaselineScenarioResult result = new(
            CreateConditions(job, precision, storageDType, selectedGpus),
            [run],
            aggregate,
            PhaseProbe: null,
            CreateNotes(scenario),
            validation);

        if (runtimeErrors.Count == 0)
        {
            try
            {
                Directory.Delete(artifactDirectory, recursive: true);
            }
            catch (Exception exception)
            {
                Console.Error.WriteLine(
                    $"Could not remove soak artifacts at " +
                    $"{artifactDirectory}: {exception.Message}");
            }
        }
        else
        {
            Console.Error.WriteLine(
                $"Soak failed; checkpoint diagnostics were retained at " +
                $"{artifactDirectory}.");
        }
        return result;
    }

    private static string ComputeFileSha256(string path)
    {
        using var stream = new FileStream(
            path,
            FileMode.Open,
            FileAccess.Read,
            FileShare.Read,
            bufferSize: 4 * 1024 * 1024,
            FileOptions.SequentialScan);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private static (bool Model, bool Optimizer)
        ValidateReloadedCheckpointPayloads(
            string checkpointPath,
            WikiLanguageModelCommand.WikiModelCheckpoint manifest,
            Module model,
            IOptimizer optimizer,
            string artifactDirectory)
    {
        string currentArtifact =
            WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                checkpointPath, manifest.ArtifactSlot);
        string verificationModel = Path.Combine(
            artifactDirectory, "reload-model-verification.safetensors");
        var verificationOptimizerPaths = new List<string>();
        try
        {
            SafeTensorFile.SaveModel(
                model,
                verificationModel,
                artifactDTypeOverride: TensorDType.Float32);
            bool modelValid = new FileInfo(currentArtifact).Length
                    == new FileInfo(verificationModel).Length
                && string.Equals(
                    ComputeFileSha256(currentArtifact),
                    ComputeFileSha256(verificationModel),
                    StringComparison.Ordinal);

            IReadOnlyList<IOptimizer> leaves =
                OptimizerBundle.GetCheckpointLeafOptimizers(optimizer);
            bool optimizerValid =
                leaves.Count == manifest.OptimizerStateTypes?.Length;
            for (int index = 0; index < leaves.Count; index++)
            {
                string source = WikiLanguageModelCommand
                    .GetOptimizerBinaryArtifactPath(
                        checkpointPath, manifest.ArtifactSlot, index);
                string verification = Path.Combine(
                    artifactDirectory,
                    $"reload-optimizer-{index}-verification.bin");
                verificationOptimizerPaths.Add(verification);
                using (var stream = new FileStream(
                    verification,
                    FileMode.Create,
                    FileAccess.Write,
                    FileShare.None,
                    bufferSize: 4 * 1024 * 1024,
                    FileOptions.SequentialScan))
                {
                    OptimizerStateStream.SaveStateBinary(
                        leaves[index], stream);
                    stream.Flush(flushToDisk: true);
                }
                optimizerValid &= new FileInfo(source).Length
                        == new FileInfo(verification).Length
                    && string.Equals(
                        ComputeFileSha256(source),
                        ComputeFileSha256(verification),
                        StringComparison.Ordinal);
            }
            return (modelValid, optimizerValid);
        }
        finally
        {
            if (File.Exists(verificationModel))
                File.Delete(verificationModel);
            foreach (string path in verificationOptimizerPaths)
            {
                if (File.Exists(path))
                    File.Delete(path);
            }
        }
    }

    private static WikiTrainingConfiguration
        CreateSoakCheckpointConfiguration(
            BaselineModelConfiguration configuration,
            TensorPrecisionMode precision,
            string checkpointPath,
            IReadOnlyList<int> deviceIndices,
            bool resume)
        => new()
        {
            CheckpointPath = checkpointPath,
            ResumeFromCheckpoint = resume,
            AutoResume = false,
            VocabularySize = configuration.Vocabulary,
            Epochs = 1,
            BatchSize = configuration.Batch,
            ContextLength = configuration.Sequence,
            ModelWidth = configuration.Width,
            Heads = configuration.Heads,
            HiddenSize = configuration.Hidden,
            Layers = configuration.Layers,
            ModelArchitecture =
                WikiTrainingConfiguration.TransformerArchitecture,
            Precision = TensorPrecisionModeNames.Format(precision),
            Bfp8BlockSize = configuration.Bfp8BlockSize,
            TieWordEmbeddings = configuration.TieWordEmbeddings,
            Device = WikiTrainingConfiguration.CudaDevice,
            DeviceIndex = deviceIndices[0],
            DeviceIndices = deviceIndices.ToArray(),
            AdaptiveCudaSharding = configuration.AdaptiveCudaSharding,
            CudaShardEmaAlpha = configuration.CudaShardEmaAlpha,
            CudaMinimumRelativeShardSize =
                configuration.CudaMinimumRelativeShardSize,
            CudaMaximumBatchAdjustmentPerStep =
                configuration.CudaMaximumBatchAdjustmentPerStep,
            CudaGraphCacheBudgetMiB =
                configuration.CudaGraphCacheBudgetMiB,
            Dropout = configuration.Dropout,
            InitializationScale = configuration.InitializationScale,
            Optimizer = WikiTrainingConfiguration.NekoMuonOptimizer,
            LearningRate = configuration.LearningRate,
            AuxiliaryLearningRate = configuration.AuxiliaryLearningRate,
            NekoMuonNewtonSchulzInterval =
                configuration.NewtonSchulzInterval,
            NekoMuonNewtonSchulzDepthMode = "fixed",
            NekoMuonNewtonSchulzDepth = 5f,
            WarmupPercent = 0f,
            WeightDecay = configuration.WeightDecay,
            Seed = configuration.Seed,
            ValidationFraction = 0f,
        };

    private static CheckpointArtifactSnapshot CaptureCheckpointArtifacts(
        string checkpointPath)
    {
        WikiLanguageModelCommand.WikiModelCheckpoint manifest =
            torch.load<WikiLanguageModelCommand.WikiModelCheckpoint>(
                checkpointPath);
        if (manifest.FormatVersion != 8
            || manifest.ArtifactSlot is < 0 or > 1
            || manifest.BestArtifactSlot is < 0 or > 1
            || manifest.OptimizerStateTypes is not { Length: > 0 })
        {
            throw new InvalidDataException(
                "Soak restart requires a committed Wiki checkpoint v8 " +
                "manifest.");
        }

        var paths = new List<string>
        {
            Path.GetFullPath(checkpointPath),
            WikiLanguageModelCommand.GetCurrentModelArtifactPath(
                checkpointPath,
                manifest.ArtifactSlot),
            WikiLanguageModelCommand.GetBestModelArtifactPath(
                checkpointPath,
                manifest.BestArtifactSlot),
        };
        for (int index = 0;
            index < manifest.OptimizerStateTypes.Length;
            index++)
        {
            paths.Add(
                WikiLanguageModelCommand.GetOptimizerBinaryArtifactPath(
                    checkpointPath,
                    manifest.ArtifactSlot,
                    index));
        }
        foreach (string path in paths)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException(
                    "A checkpoint artifact referenced by the v8 manifest " +
                    "was not published.",
                    path);
            }
        }

        BaselineCheckpointArtifact[] artifacts = paths
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Select(path => new BaselineCheckpointArtifact(
                Path.GetFileName(path),
                new FileInfo(path).Length,
                ComputeFileSha256(path)))
            .ToArray();
        string directory = Path.GetDirectoryName(
            Path.GetFullPath(checkpointPath))!;
        string stem = Path.GetFileNameWithoutExtension(checkpointPath);
        bool transactionDebris = Directory.EnumerateFiles(
                directory,
                $"{stem}*")
            .Any(path => path.Contains(
                    ".transaction.",
                    StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(".tmp", StringComparison.OrdinalIgnoreCase)
                || path.EndsWith(
                    ".backup",
                    StringComparison.OrdinalIgnoreCase));
        DateTime manifestWrite = File.GetLastWriteTimeUtc(checkpointPath);
        bool manifestPublishedLast = paths.Skip(1).All(path =>
            File.GetLastWriteTimeUtc(path) <= manifestWrite);
        return new CheckpointArtifactSnapshot(
            manifest,
            artifacts,
            artifacts.Sum(artifact => artifact.Bytes),
            !transactionDebris && manifestPublishedLast);
    }

    private static IReadOnlyList<BaselineCheckpointArtifact>
        CapturePartialCheckpointArtifacts(string checkpointPath)
    {
        string fullPath = Path.GetFullPath(checkpointPath);
        string directory = Path.GetDirectoryName(fullPath)!;
        if (!Directory.Exists(directory))
            return [];
        string stem = Path.GetFileNameWithoutExtension(fullPath);
        return Directory.EnumerateFiles(directory, $"{stem}*")
            .Order(StringComparer.OrdinalIgnoreCase)
            .Select(path => new BaselineCheckpointArtifact(
                Path.GetFileName(path),
                new FileInfo(path).Length,
                ComputeFileSha256(path)))
            .ToArray();
    }

    private static bool TrainingRandomStatesEqual(
        TrainingRandomState? expected,
        TrainingRandomState? actual)
        => expected is null || actual is null
            ? expected is null && actual is null
            : expected.FormatVersion == actual.FormatVersion
                && expected.RootSeed == actual.RootSeed
                && expected.HostState == actual.HostState
                && expected.DeviceStates.Length
                    == actual.DeviceStates.Length
                && expected.DeviceStates.Zip(
                    actual.DeviceStates,
                    (left, right) => left.DeviceIndex == right.DeviceIndex
                        && left.State == right.State)
                    .All(equal => equal);

    private static bool AdaptiveShardStatesEqual(
        CudaAdaptiveShardState expected,
        CudaAdaptiveShardState? actual)
        => actual is not null
            && expected.FormatVersion == actual.FormatVersion
            && expected.Devices.AsSpan().SequenceEqual(actual.Devices)
            && expected.LastAllocation.AsSpan()
                .SequenceEqual(actual.LastAllocation)
            && expected.ThroughputEma.AsSpan()
                .SequenceEqual(actual.ThroughputEma)
            && expected.HasObservation == actual.HasObservation
            && expected.ObservationCount == actual.ObservationCount
            && expected.LastAdjustmentObservation
                == actual.LastAdjustmentObservation
            && expected.PendingAllocation.AsSpan()
                .SequenceEqual(actual.PendingAllocation)
            && expected.PendingConfirmationCount
                == actual.PendingConfirmationCount
            && expected.LastCandidateObservation
                == actual.LastCandidateObservation
            && expected.OversizedGraphAllocation.AsSpan()
                .SequenceEqual(actual.OversizedGraphAllocation)
            && expected.OversizedGraphPinnedBytes
                == actual.OversizedGraphPinnedBytes;

    private static bool ValidateBfp8BlockSize(
        IReadOnlyList<Parameter> parameters,
        TensorPrecisionMode precision,
        int expectedBlockSize)
        => precision != TensorPrecisionMode.Mix8_32
            || parameters.All(parameter =>
                parameter.T.Bfp8Quantization is
                {
                    Granularity: Bfp8ScaleGranularity.Block,
                    BlockSize: var blockSize,
                }
                && blockSize == expectedBlockSize);

    private static bool ValidateCudaResidency(
        IReadOnlyList<Parameter> parameters,
        IReadOnlyList<int> deviceIndices)
        => parameters.Count > 0
            && parameters.All(parameter =>
            {
                int[] residents = parameter.T.GetResidentCudaDeviceIndices();
                return deviceIndices.All(device =>
                    residents.Contains(device));
            });

    private static BaselineCheckpointResult EmptyCheckpointResult(
        string artifactDirectory)
        => new(
            FormatVersion: 0,
            Validated: false,
            SaveMilliseconds: null,
            LoadMilliseconds: null,
            TotalBytes: 0,
            Artifacts: [],
            ArtifactFirstManifestLastValidated: false,
            CursorValidated: false,
            TrainingRandomValidated: false,
            SchedulerValidated: false,
            AdaptiveShardStateValidated: false,
            ModelValidated: false,
            OptimizerValidated: false,
            PrecisionValidated: false,
            Bfp8BlockSizeValidated: false,
            DeviceResidencyValidated: false,
            OldFixtureDisposedBeforeReload: false,
            ArtifactsRetainedAfterFailure: false,
            ArtifactDirectory: artifactDirectory);

    private static void AddRuntimeErrors(
        Exception exception,
        ICollection<string> errors)
    {
        if (exception is AggregateException aggregate)
        {
            foreach (Exception inner in aggregate.Flatten().InnerExceptions)
                AddRuntimeErrors(inner, errors);
            return;
        }
        string status = exception is NativeCudaException cuda
            ? $" CUDA-status={cuda.Status}"
            : string.Empty;
        errors.Add($"{exception.GetType().Name}:{status} {exception.Message}");
        if (exception.InnerException is not null)
            AddRuntimeErrors(exception.InnerException, errors);
    }

    private static IReadOnlyList<BaselineDeviceMemorySummary>
        SummarizePostWarmupDeviceMemory(
            IReadOnlyList<BaselineStepMeasurement> measurements,
            int warmupSteps)
    {
        if (measurements.Count == 0)
            return [];
        int baselineIndex = Math.Clamp(
            warmupSteps - 1,
            0,
            measurements.Count - 1);
        var result = new List<BaselineDeviceMemorySummary>();
        foreach (BaselineDeviceMemoryObservation baseline in
            measurements[baselineIndex].DeviceMemory)
        {
            if (!baseline.TotalBytes.HasValue
                || !baseline.UsedBytes.HasValue)
            {
                continue;
            }
            BaselineDeviceMemoryObservation[] values = measurements
                .Skip(baselineIndex)
                .SelectMany(value => value.DeviceMemory)
                .Where(value => value.Device == baseline.Device
                    && value.UsedBytes.HasValue)
                .ToArray();
            if (values.Length == 0)
                continue;
            long start = baseline.UsedBytes.Value;
            long peak = values.Max(value => value.UsedBytes!.Value);
            result.Add(new BaselineDeviceMemorySummary(
                baseline.Device,
                baseline.TotalBytes.Value,
                start,
                peak,
                values[^1].UsedBytes!.Value,
                Math.Max(0, peak - start),
                values.Length));
        }
        return result.OrderBy(value => value.Device).ToArray();
    }

    private static IReadOnlyList<BaselineGateResult> CreateSoakGates(
        BaselineSoakConfiguration configuration,
        BaselineSoakResult result,
        int selectedGpuCount)
    {
        bool enoughTrendSamples = result.FirstWindow.Count
                == configuration.TrendWindowSteps
            && result.LastWindow.Count == configuration.TrendWindowSteps;
        bool vramAvailable = result.PostWarmupDeviceMemory.Count
            == selectedGpuCount;
        bool vramPassed = vramAvailable
            && result.PostWarmupDeviceMemory.All(value =>
                value.PeakGrowthBytes
                    <= configuration.MaximumPostWarmupVramGrowthBytes);
        return
        [
            new BaselineGateResult(
                "committed-step-count",
                result.CompletedCommittedSteps
                    == configuration.TotalCommittedSteps,
                result.CompletedCommittedSteps.ToString(),
                configuration.TotalCommittedSteps.ToString()),
            new BaselineGateResult(
                "step-time-trend",
                enoughTrendSamples
                    && result.LastToFirstP50Ratio
                        <= configuration.MaximumLastToFirstP50Ratio,
                enoughTrendSamples
                    ? $"ratio={result.LastToFirstP50Ratio:F4}, first=" +
                        $"{result.FirstWindow.P50:F3}ms, last=" +
                        $"{result.LastWindow.P50:F3}ms"
                    : "insufficient samples",
                $"last p50 / first p50 <= " +
                    $"{configuration.MaximumLastToFirstP50Ratio:F4}"),
            new BaselineGateResult(
                "post-warmup-vram-growth",
                vramPassed,
                vramAvailable
                    ? string.Join(
                        ", ",
                        result.PostWarmupDeviceMemory.Select(value =>
                            $"cuda:{value.Device} +" +
                            $"{value.PeakGrowthBytes / 1048576d:F1} MiB"))
                    : "cudaMemGetInfo unavailable",
                $"each GPU <= " +
                    $"{configuration.MaximumPostWarmupVramGrowthBytes
                        / 1048576d:F1} MiB"),
            new BaselineGateResult(
                "generation-event",
                result.GenerationObserved,
                result.GenerationObserved
                    ? $"{result.GeneratedTokens} token(s) at global step " +
                        $"{result.GenerationStep} in " +
                        $"{result.GenerationMilliseconds:F3} ms"
                    : "not observed",
                $"post-commit generation at global step " +
                    $"{configuration.GenerationStep}"),
            new BaselineGateResult(
                "wiki-v8-streaming-checkpoint",
                result.RestartObserved
                    && result.ResumeArtifactValidated
                    && result.Checkpoint.Validated,
                $"restart={result.RestartObserved}, " +
                    $"validated={result.Checkpoint.Validated}, " +
                    $"format=v{result.Checkpoint.FormatVersion}, " +
                    $"artifacts={result.Checkpoint.Artifacts.Count}, " +
                    $"bytes={result.Checkpoint.TotalBytes}, " +
                    $"save={result.Checkpoint.SaveMilliseconds:F3}ms, " +
                    $"load={result.Checkpoint.LoadMilliseconds:F3}ms",
                $"validated full Wiki v8 restart after global step " +
                    $"{configuration.RestartStep}",
                "Artifact-first/manifest-last, cursor, training RNG, " +
                "zero-warmup cosine scheduler, adaptive shards, model, " +
                "optimizer, precision/BFP8 metadata, and two-GPU residency " +
                "must all round-trip after the old fixture is disposed. " +
                string.Join(
                    ", ",
                    result.Checkpoint.Artifacts.Select(artifact =>
                        $"{artifact.Name}:{artifact.Bytes}:" +
                        artifact.Sha256))),
            new BaselineGateResult(
                "metrics-sidecar-continuity",
                result.SidecarContinuityValidated,
                $"before={result.SidecarEntriesBeforeRestart}, " +
                    $"final={result.SidecarEntriesAfterResume}",
                "strictly continuous JSONL entries through restart"),
            new BaselineGateResult(
                "html-continuity",
                result.HtmlContinuityChecked
                    ? result.HtmlContinuityValidated
                    : null,
                result.HtmlContinuityStatus,
                "production HTML projection retains pre/post-resume metrics"),
            new BaselineGateResult(
                "nonzero-gpu-shards",
                !result.ZeroShardObserved,
                result.ZeroShardObserved
                    ? "a zero/missing shard was observed"
                    : "all selected GPUs received positive shards",
                "no zero shard"),
            new BaselineGateResult(
                "runtime-errors",
                result.RuntimeErrors.Count == 0,
                result.RuntimeErrors.Count == 0
                    ? "none"
                    : string.Join(" | ", result.RuntimeErrors),
                "no OOM, CUDA 600/700, illegal access, or other runtime error"),
        ];
    }

    private sealed record CheckpointArtifactSnapshot(
        WikiLanguageModelCommand.WikiModelCheckpoint Manifest,
        IReadOnlyList<BaselineCheckpointArtifact> Artifacts,
        long TotalBytes,
        bool ArtifactFirstManifestLastValidated);

    private sealed class SoakCheckpointFaultInjector
        : ICheckpointFaultInjector
    {
        public void OnCheckpointFaultPoint(CheckpointFaultContext context)
        {
            if (context.Point == CheckpointFaultPoint.AfterArtifactPublish
                && context.ArtifactIndex == 0)
            {
                throw new IOException(
                    "Intentional soak checkpoint failure after publishing " +
                    "the first artifact.");
            }
        }
    }

    private static bool ShouldWriteSoakHtmlMetric(
        int committedIndex,
        long globalStep,
        BaselineSoakConfiguration soak)
        => committedIndex == 0
            || globalStep == soak.RestartStep
            || globalStep == soak.RestartStep + 1L
            || committedIndex + 1 == soak.TotalCommittedSteps
            || (committedIndex + 1) % 100 == 0;

    private static RawStep MeasureStep(
        BaselineFixture fixture,
        BaselineModelConfiguration configuration,
        BaselineScenario scenario,
        long? globalStep = null,
        string? diagnosticLabel = null,
        double? scheduleProgress = null)
    {
        Synchronize(scenario);
        long managedBefore = GC.GetTotalAllocatedBytes(precise: false);
        NativeCudaAllocationTelemetry nativeBefore =
            NativeCudaRuntime.AllocationTelemetry;
        NativeCudaTransferTelemetry transferBefore =
            NativeCudaRuntime.TransferTelemetry;
        NativeCudaTransferTelemetry gradientCollectiveBefore =
            NativeCudaRuntime.GradientCollectiveTransferTelemetry;
        bool logTransferPhases = scenario.Soak is null
            && scenario.Device == BaselineDeviceKind.Cuda
            && scenario.Name.Contains(
                "smoke", StringComparison.OrdinalIgnoreCase);
        long totalStarted = Stopwatch.GetTimestamp();

        long phaseStarted = Stopwatch.GetTimestamp();
        RunStepPhase("zero_grad", fixture.Optimizer.zero_grad);
        double zeroGrad = Elapsed(phaseStarted);
        NativeCudaTransferTelemetry transferAfterZero =
            NativeCudaRuntime.TransferTelemetry;

        double? forward = null;
        double? lossPhase = null;
        double? backward = null;
        float lossValue;
        phaseStarted = Stopwatch.GetTimestamp();
        if (scenario.Device == BaselineDeviceKind.Cuda)
        {
            lossValue = RunStepPhase(
                "forward_backward_reduce",
                () => globalStep.HasValue
                    ? fixture.DataParallelEngine!.ForwardBackward(
                        fixture.Input,
                        fixture.Target,
                        configuration.Batch,
                        configuration.Sequence,
                        Tensor.DefaultCrossEntropyIgnoreIndex,
                        globalStep.Value)
                    : fixture.DataParallelEngine!.ForwardBackward(
                        fixture.Input,
                        fixture.Target,
                        configuration.Batch,
                        configuration.Sequence));
        }
        else
        {
            long forwardStarted = Stopwatch.GetTimestamp();
            Tensor logits = fixture.Model.Forward(
                fixture.Input,
                configuration.Batch,
                configuration.Sequence);
            forward = Elapsed(forwardStarted);
            long lossStarted = Stopwatch.GetTimestamp();
            Tensor loss = logits.CrossEntropyWithLogits(fixture.Target);
            lossValue = loss.item();
            lossPhase = Elapsed(lossStarted);
            long backwardStarted = Stopwatch.GetTimestamp();
            loss.Backward();
            backward = Elapsed(backwardStarted);
        }
        double forwardBackward = Elapsed(phaseStarted);
        NativeCudaTransferTelemetry transferAfterForwardBackward =
            NativeCudaRuntime.TransferTelemetry;

        phaseStarted = Stopwatch.GetTimestamp();
        _ = RunStepPhase(
            "gradient_clip",
            () => nn.utils.clip_grad_norm_(
                fixture.Parameters, max_norm: 1f));
        double clip = Elapsed(phaseStarted);
        NativeCudaTransferTelemetry transferAfterClip =
            NativeCudaRuntime.TransferTelemetry;

        if (scheduleProgress is double progress)
        {
            _ = RunStepPhase(
                "schedule",
                () => fixture.Scheduler.step(progress));
        }

        phaseStarted = Stopwatch.GetTimestamp();
        RunStepPhase("optimizer", fixture.Optimizer.step);
        double optimizerMilliseconds = Elapsed(phaseStarted);
        NativeCudaTransferTelemetry transferAfterOptimizer =
            NativeCudaRuntime.TransferTelemetry;

        // Composite CUDA optimizers enqueue both children before one shared
        // completion barrier. Splitting the wall-clock interval at the child
        // calls would either omit that barrier or reintroduce it twice, so the
        // normal-path baseline records the fused interval as AdamW and leaves
        // the NekoMuon sub-field at zero. The separate synchronized phase
        // probe reports only the authoritative total optimizer interval.
        double neko = 0d;
        double adam = optimizerMilliseconds;

        Synchronize(scenario);
        double total = Elapsed(totalStarted);
        long managedBytes = Math.Max(
            0,
            GC.GetTotalAllocatedBytes(precise: false) - managedBefore);
        NativeCudaAllocationTelemetry native =
            NativeCudaRuntime.AllocationTelemetry - nativeBefore;
        NativeCudaTransferTelemetry transfers =
            NativeCudaRuntime.TransferTelemetry - transferBefore;
        NativeCudaTransferTelemetry gradientCollectiveTransfers =
            NativeCudaRuntime.GradientCollectiveTransferTelemetry
                - gradientCollectiveBefore;
        if (logTransferPhases)
        {
            Console.WriteLine(
                $"{scenario.Name} transfer phases " +
                $"[{diagnosticLabel ?? "step"}]: " +
                $"zero={FormatTransferDelta(
                    transferAfterZero - transferBefore)}, " +
                $"forward_backward={FormatTransferDelta(
                    transferAfterForwardBackward - transferAfterZero)}, " +
                $"clip={FormatTransferDelta(
                    transferAfterClip - transferAfterForwardBackward)}, " +
                $"optimizer={FormatTransferDelta(
                    transferAfterOptimizer - transferAfterClip)}, " +
                $"total={FormatTransferDelta(transfers)}, " +
                $"gradient_collective={FormatTransferDelta(
                    gradientCollectiveTransfers)}");
        }
        return new RawStep(
            lossValue,
            total,
            zeroGrad,
            forwardBackward,
            forward,
            lossPhase,
            backward,
            clip,
            optimizerMilliseconds,
            neko,
            adam,
            managedBytes,
            native,
            transfers,
            gradientCollectiveTransfers,
            CaptureDeviceMemory(scenario));
    }

    private static string FormatTransferDelta(
        NativeCudaTransferTelemetry value)
        => $"H2D {value.HostToDeviceCopyCount}/{value.HostToDeviceBytes}B " +
            $"D2H {value.DeviceToHostCopyCount}/{value.DeviceToHostBytes}B";

    private static void RunStepPhase(string name, Action action)
    {
        try
        {
            action();
        }
        catch (Exception exception)
        {
            throw new InvalidOperationException(
                $"Training phase '{name}' failed.", exception);
        }
    }

    private static T RunStepPhase<T>(string name, Func<T> action)
    {
        try
        {
            return action();
        }
        catch (Exception exception)
        {
            throw new InvalidOperationException(
                $"Training phase '{name}' failed.", exception);
        }
    }

    private static BaselinePhaseProbe CreatePhaseProbe(
        BaselineFixture fixture,
        BaselineModelConfiguration configuration,
        BaselineScenario scenario,
        IReadOnlyList<BaselineRunResult> runs)
    {
        BaselineStepMeasurement[] measured = runs
            .SelectMany(run => run.Measurements).ToArray();
        double clipP50 = BaselineDistribution.From(
            measured.Select(step => step.ClipMilliseconds)).P50;
        double optimizerP50 = BaselineDistribution.From(
            measured.Select(step => step.OptimizerMilliseconds)).P50;
        if (scenario.Device == BaselineDeviceKind.Cpu)
        {
            return new BaselinePhaseProbe(
                "measured CPU steps (p50; no diagnostic synchronization)",
                OptionalDistribution(measured.Select(step =>
                    step.ForwardMilliseconds))?.P50,
                OptionalDistribution(measured.Select(step =>
                    step.LossPhaseMilliseconds))?.P50,
                OptionalDistribution(measured.Select(step =>
                    step.BackwardMilliseconds))?.P50,
                null,
                null,
                clipP50,
                optimizerP50,
                null,
                null,
                "not applicable on CPU",
                new BaselineTransferTelemetry(0, 0, 0, 0),
                []);
        }

        fixture.Optimizer.zero_grad();
        Synchronize(scenario);
        CudaDataParallelProfile profile =
            fixture.DataParallelEngine!.ForwardBackwardProfiled(
                fixture.Input,
                fixture.Target,
                configuration.Batch,
                configuration.Sequence);
        long clipStarted = Stopwatch.GetTimestamp();
        _ = nn.utils.clip_grad_norm_(fixture.Parameters, max_norm: 1f);
        Synchronize(scenario);
        double diagnosticClip = Elapsed(clipStarted);
        double forward = profile.Shards.Max(shard =>
            shard.ForwardMilliseconds);
        double loss = profile.Shards.Max(shard => shard.LossMilliseconds);
        double backward = profile.Shards.Max(shard =>
            shard.BackwardMilliseconds);
        double hostPreparation = profile.Shards.Max(shard =>
            shard.DataPreparationMilliseconds);
        BaselineShardProbe[] shards = profile.Shards
            .Select(shard => new BaselineShardProbe(
                shard.Device,
                shard.BatchSize,
                shard.DataPreparationMilliseconds,
                shard.ForwardMilliseconds,
                shard.LossMilliseconds,
                shard.BackwardMilliseconds))
            .ToArray();
        (double transferMilliseconds, NativeCudaTransferTelemetry transfers) =
            MeasureEquivalentInputTransfer(
                fixture,
                configuration,
                scenario,
                profile.Shards);
        return new BaselinePhaseProbe(
            "one diagnostic CUDA step with phase synchronization",
            forward,
            loss,
            backward,
            profile.AllReduceMilliseconds,
            profile.GradientPreparationMilliseconds,
            diagnosticClip,
            optimizerP50,
            transferMilliseconds,
            hostPreparation,
            "isolated synchronized H2D probe using the measured shard payloads; " +
            "normal-step transfer counts/bytes are recorded independently",
            ToTransferTelemetry(transfers),
            shards);
    }

    private static (
        double Milliseconds,
        NativeCudaTransferTelemetry Transfers) MeasureEquivalentInputTransfer(
            BaselineFixture fixture,
            BaselineModelConfiguration configuration,
            BaselineScenario scenario,
            IReadOnlyList<CudaShardProfile> shards)
    {
        var inputs = new List<NativeCudaBuffer<int>>(shards.Count);
        var targets = new List<NativeCudaBuffer<int>>(shards.Count);
        try
        {
            foreach (CudaShardProfile shard in shards)
            {
                int elements = checked(shard.BatchSize * configuration.Sequence);
                NativeCudaDevice device = NativeCudaDevice.GetOrCreate(
                    shard.Device);
                inputs.Add(device.Allocate<int>(elements));
                targets.Add(device.Allocate<int>(elements));
            }

            Synchronize(scenario);
            NativeCudaTransferTelemetry before =
                NativeCudaRuntime.TransferTelemetry;
            long started = Stopwatch.GetTimestamp();
            int offset = 0;
            for (int index = 0; index < shards.Count; index++)
            {
                int elements = checked(
                    shards[index].BatchSize * configuration.Sequence);
                inputs[index].CopyFromCPU(
                    fixture.Input.AsSpan(offset, elements));
                targets[index].CopyFromCPU(
                    fixture.Target.AsSpan(offset, elements));
                offset = checked(offset + elements);
            }
            Synchronize(scenario);
            return (
                Elapsed(started),
                NativeCudaRuntime.TransferTelemetry - before);
        }
        finally
        {
            List<Exception>? failures = null;
            for (int index = inputs.Count - 1; index >= 0; index--)
            {
                try
                {
                    inputs[index].Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            for (int index = targets.Count - 1; index >= 0; index--)
            {
                try
                {
                    targets[index].Dispose();
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "Diagnostic H2D probe cleanup failed.", failures);
            }
        }
    }

    private static BaselineTransferTelemetry ToTransferTelemetry(
        NativeCudaTransferTelemetry telemetry)
        => new(
            telemetry.HostToDeviceCopyCount,
            telemetry.HostToDeviceBytes,
            telemetry.DeviceToHostCopyCount,
            telemetry.DeviceToHostBytes);

    private static IReadOnlyList<BaselineDeviceMemoryObservation>
        CaptureDeviceMemory(BaselineScenario scenario)
    {
        if (scenario.Device != BaselineDeviceKind.Cuda)
            return [];
        var observations = new BaselineDeviceMemoryObservation[
            scenario.DeviceIndices.Length];
        for (int index = 0; index < scenario.DeviceIndices.Length; index++)
        {
            int deviceIndex = scenario.DeviceIndices[index];
            try
            {
                NativeCudaDevice device = NativeCudaDevice.GetOrCreate(
                    deviceIndex);
                long total = device.MemorySize;
                long free = device.GetFreeMemory();
                observations[index] = new BaselineDeviceMemoryObservation(
                    deviceIndex,
                    total,
                    free,
                    Math.Max(0, total - free),
                    null);
            }
            catch (Exception exception)
            {
                observations[index] = new BaselineDeviceMemoryObservation(
                    deviceIndex,
                    null,
                    null,
                    null,
                    exception.Message);
            }
        }
        return observations;
    }

    private static IReadOnlyList<BaselineDeviceMemorySummary>
        SummarizeDeviceMemory(
            IEnumerable<BaselineDeviceMemoryObservation> observations)
        => observations
            .Where(value => value.TotalBytes.HasValue
                && value.UsedBytes.HasValue)
            .GroupBy(value => value.Device)
            .OrderBy(group => group.Key)
            .Select(group =>
            {
                BaselineDeviceMemoryObservation[] values = group.ToArray();
                long start = values[0].UsedBytes!.Value;
                long peak = values.Max(value => value.UsedBytes!.Value);
                long end = values[^1].UsedBytes!.Value;
                return new BaselineDeviceMemorySummary(
                    group.Key,
                    values[0].TotalBytes!.Value,
                    start,
                    peak,
                    end,
                    Math.Max(0, peak - start),
                    values.Length);
            })
            .ToArray();

    private static BaselineRunResult CreateRun(
        int repetition,
        DateTimeOffset started,
        DateTimeOffset finished,
        IReadOnlyList<int> finalShardBatchSizes,
        BaselineTrainingGraphTelemetry? trainingGraph,
        IReadOnlyList<BaselineStepMeasurement> measurements)
    {
        BaselineStepMeasurement[] measured = measurements
            .Where(value => !value.IsWarmup)
            .ToArray();
        if (measured.Length == 0)
            measured = measurements.ToArray();
        return new BaselineRunResult(
            repetition,
            started,
            finished,
            Distribution(measured, value => value.TotalMilliseconds),
            Distribution(measured, value => value.ZeroGradMilliseconds),
            Distribution(
                measured, value => value.ForwardBackwardMilliseconds),
            OptionalDistribution(measured.Select(value =>
                value.ForwardMilliseconds)),
            OptionalDistribution(measured.Select(value =>
                value.LossPhaseMilliseconds)),
            OptionalDistribution(measured.Select(value =>
                value.BackwardMilliseconds)),
            OptionalDistribution(measured.Select(value =>
                value.ReduceWaitMilliseconds)),
            OptionalDistribution(measured.Select(value =>
                value.TransferMilliseconds)),
            Distribution(measured, value => value.ClipMilliseconds),
            Distribution(measured, value => value.OptimizerMilliseconds),
            Distribution(measured, value => value.NekoMuonMilliseconds),
            Distribution(measured, value => value.AdamWMilliseconds),
            Distribution(measured, value => value.ManagedAllocationBytes),
            Distribution(measured, value => value.NativeAllocationCount),
            Distribution(measured, value => value.NativeAllocationBytes),
            Distribution(measured, value => value.NativeFreeCount),
            Distribution(measured, value => value.NativeFreeBytes),
            Distribution(measured, value => value.HostToDeviceCopyCount),
            Distribution(measured, value => value.HostToDeviceBytes),
            Distribution(measured, value => value.DeviceToHostCopyCount),
            Distribution(measured, value => value.DeviceToHostBytes),
            SummarizeDeviceMemory(measured.SelectMany(value =>
                value.DeviceMemory)),
            finalShardBatchSizes.ToArray(),
            trainingGraph,
            measurements);
    }

    private static BaselineTrainingGraphTelemetry? CreateGraphTelemetry(
        CudaTrainingGraphTelemetry? before,
        CudaTrainingGraphTelemetry? after,
        int measuredSteps)
    {
        if (before is not { } start || after is not { } finish)
            return null;
        long measuredCaptures = finish.CaptureCount - start.CaptureCount;
        long measuredReplays = finish.ReplayCount - start.ReplayCount;
        long measuredFallbacks = finish.FallbackCount - start.FallbackCount;
        long measuredReadyEvents = finish.CapturedReadyEventRecordCount
            - start.CapturedReadyEventRecordCount;
        double measuredReadyEventMilliseconds = Math.Max(
            0d,
            finish.CapturedReadyEventRecordMilliseconds
                - start.CapturedReadyEventRecordMilliseconds);
        bool fullyCompiledReplay = start.CachedCompiledPlanCount > 0
            && measuredCaptures == 0
            && measuredFallbacks == 0
            && measuredReplays == measuredSteps;
        return new BaselineTrainingGraphTelemetry(
            finish.CaptureCount,
            finish.ReplayCount,
            finish.FallbackCount,
            finish.CachedCompiledPlanCount,
            finish.GraphPinnedBytes,
            finish.CapturedReadyEventRecordCount,
            finish.CapturedReadyEventRecordMilliseconds,
            measuredCaptures,
            measuredReplays,
            measuredFallbacks,
            measuredReadyEvents,
            measuredReadyEventMilliseconds,
            fullyCompiledReplay);
    }

    private static BaselineConditions CreateConditions(
        BaselineWorkerJob job,
        TensorPrecisionMode precision,
        TensorDType storageDType,
        IReadOnlyList<BaselineGpu> selectedGpus)
    {
        BaselineModelConfiguration model = job.Model;
        BaselineScenario scenario = job.Scenario;
        long tokens = checked((long)model.Batch * model.Sequence);
        return new BaselineConditions(
            scenario.Name,
            job.Commit,
            job.ConfigurationPath,
            job.ConfigurationSha256,
            scenario.Device == BaselineDeviceKind.Cuda ? "cuda" : "cpu",
            scenario.DeviceIndices,
            selectedGpus,
            TensorPrecisionModeNames.Format(precision),
            storageDType.ToString(),
            precision == TensorPrecisionMode.Mix8_32
                ? model.Bfp8BlockSize
                : null,
            scenario.Device != BaselineDeviceKind.Cuda
                ? "managed eager CPU"
                : "CUDA Graph preferred for every supported precision; " +
                    "run telemetry verifies capture/replay/fallback",
            model.Batch,
            model.Sequence,
            model.Vocabulary,
            model.Width,
            model.Heads,
            model.Hidden,
            model.Layers,
            "NekoMuon+AdamW (FP32 moments)",
            model.NewtonSchulzDepth,
            model.NewtonSchulzDepthMode,
            model.NewtonSchulzInterval,
            scenario.WarmupSteps,
            scenario.MeasuredSteps,
            scenario.Repetitions,
            scenario.PerformanceGate is null
                ? null
                : "median of per-repetition step p50 values",
            scenario.PerformanceGate?.FrozenStepP50Milliseconds,
            scenario.PerformanceGate?.MaximumBaselineRatio,
            scenario.PerformanceGate?
                .MaximumAllowedStepP50Milliseconds,
            model.Seed,
            model.AdaptiveCudaSharding,
            model.CudaShardEmaAlpha,
            model.CudaMinimumRelativeShardSize,
            model.CudaMaximumBatchAdjustmentPerStep,
            model.CudaGraphCacheBudgetMiB,
            scenario.Device == BaselineDeviceKind.Cuda
                ? checked(tokens * sizeof(int) * 2)
                : 0,
            scenario.Device == BaselineDeviceKind.Cuda
                ? checked((long)scenario.DeviceIndices.Length * sizeof(float))
                : 0,
            "fixed synthetic token/target arrays; dataset I/O excluded",
            job.EffectiveOverrides ?? []);
    }

    private static IReadOnlyList<string> CreateNotes(BaselineScenario scenario)
    {
        var notes = new List<string>
        {
            "Step p50/p95 use normal training entry points; model construction, " +
            "worker startup, warmup, and the diagnostic phase probe are excluded.",
            "p50 uses the upper middle sample and p95 uses nearest-rank " +
            "ceil(N*0.95)-1, matching TransformerCudaProfiler output.",
            "Managed allocation is the process-wide GC allocation counter delta. " +
            "Native allocation is NNtrain CUDA allocator telemetry and does not " +
            "include allocations performed internally by CUDA libraries.",
            "Every measured step records actual NNtrain H2D/D2H copy counts and " +
            "bytes plus cudaMemGetInfo observations for each selected device.",
            "Every repetition starts from the same seed with a new model, " +
            "optimizer, token batch, and explicitly owned data-parallel engine.",
            "The diagnostic phase probe uses another fresh fixture and is " +
            "excluded from all measured repetitions.",
            "Optimizer p50/p95 is the authoritative fused CompositeOptimizer " +
            "interval. Legacy child fields encode zero for NekoMuon and the " +
            "fused interval for AdamW because the shared CUDA completion " +
            "barrier cannot be assigned to either child without changing the " +
            "measured execution path.",
        };
        if (scenario.Device == BaselineDeviceKind.Cuda)
        {
            notes.Add(
                "Forward/backward/reduce-wait are reported by one separate " +
                "synchronizing diagnostic probe; they must not be summed with " +
                "the normal-path step p50/p95.");
            notes.Add(
                "The diagnostic transfer duration is an isolated synchronized " +
                "copy of the same per-shard token/target payload. Normal-path " +
                "step timing remains unmodified and reports exact transfer " +
                "counts/bytes rather than attributing asynchronous copy time.");
            notes.Add(
                "CUDA Graph telemetry records both lifetime totals and measured-" +
                "interval deltas. A run is marked fully compiled only when every " +
                "measured training call replayed a warmup-compiled plan with no " +
                "capture or fallback in the measured interval.");
        }
        if (scenario.Soak is not null)
        {
            notes.Add(
                "The soak commits exactly TotalCommittedSteps; its leading " +
                "PerformanceWarmupSteps are present in raw measurements with " +
                "isWarmup=true and excluded from aggregate/trend distributions.");
            notes.Add(
                "After the committed midpoint step, the soak synchronizes both " +
                "GPUs, publishes a Wiki v8 streaming checkpoint, disposes the " +
                "entire fixture (model, optimizer, and data-parallel engine), " +
                "constructs a fresh fixture, restores the checkpoint, and " +
                "continues. It validates every artifact hash/size, cursor, " +
                "training RNG, zero-warmup cosine scheduler, adaptive shards, " +
                "precision/BFP8 metadata, two-GPU residency, JSONL continuity, " +
                "and rendered HTML continuity. Failed runs retain their " +
                "temporary checkpoint directory for diagnosis.");
        }
        return notes;
    }

    private static BaselineFixture CreateFixture(
        BaselineModelConfiguration configuration,
        TensorPrecisionMode precision,
        TensorDType storageDType,
        BaselineScenario scenario)
    {
        var trainingRandom = new CheckpointableRandom(configuration.Seed);
        var model = new GptRinWikiJp(
            configuration.Vocabulary,
            configuration.Sequence,
            configuration.Width,
            configuration.Heads,
            configuration.Hidden,
            configuration.Layers,
            trainingRandom,
            configuration.InitializationScale,
            configuration.Dropout,
            storageDType,
            configuration.TieWordEmbeddings);
        trainingRandom.BeginRuntime();
        model.AttachTrainingRandom(trainingRandom);
        // The benchmark must exercise the same physical storage contract as
        // production. In particular, a constructor-created BFP8 model uses
        // tensor-wide scales, while mix8_32 requires block scales. Merely
        // changing the mode metadata either rejects the model or benchmarks
        // the wrong representation.
        model.to(precision, configuration.Bfp8BlockSize);
        var nekoMuon = new NekoMuon(
            model.HiddenWeightParameters,
            new NekoMuonOptions
            {
                LearningRate = configuration.LearningRate,
                WeightDecay = configuration.WeightDecay,
                MaxNewtonSchulzSteps = configuration.NewtonSchulzDepth,
                NewtonSchulzInterval = configuration.NewtonSchulzInterval,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = configuration.NewtonSchulzDepth,
            });
        var adamW = new AdamW(
            model.AuxiliaryParameters,
            new AdamWOptions
            {
                LearningRate = configuration.AuxiliaryLearningRate,
                WeightDecay = configuration.WeightDecay,
                UseBFloat16FirstMoment = false,
                UseBFloat16SecondMoment = false,
        });
        var optimizer = new CompositeOptimizer(nekoMuon, adamW);
        WarmupCosineProgressLRScheduler scheduler =
            lr_scheduler.WarmupCosineProgressLR(
                optimizer,
                warmup_percent: 0f);
        Parameter[] parameters = model.parameters().ToArray();
        (int[] input, int[] target) = CreateBatch(configuration);
        CudaDataParallelEngine? dataParallelEngine =
            scenario.Device == BaselineDeviceKind.Cuda
                ? new CudaDataParallelEngine(
                    model,
                    scenario.DeviceIndices,
                    CreateAdaptiveShardingOptions(configuration))
                : null;
        return new BaselineFixture(
            scenario,
            model,
            nekoMuon,
            adamW,
            optimizer,
            scheduler,
            parameters,
            input,
            target,
            dataParallelEngine);
    }

    private static CudaAdaptiveShardingOptions CreateAdaptiveShardingOptions(
        BaselineModelConfiguration configuration)
        => new()
        {
            Enabled = configuration.AdaptiveCudaSharding,
            EmaAlpha = configuration.CudaShardEmaAlpha,
            MinimumRelativeShardSize =
                configuration.CudaMinimumRelativeShardSize,
            MaximumBatchAdjustmentPerStep =
                configuration.CudaMaximumBatchAdjustmentPerStep,
            GraphCacheBudgetBytes = checked(
                (long)configuration.CudaGraphCacheBudgetMiB
                * 1024L
                * 1024L),
        };

    private static ExecutionSession CreateExecutionSession(
        BaselineScenario scenario,
        TensorPrecisionMode precision)
    {
        ExecutionDeviceKind device = scenario.Device ==
                BaselineDeviceKind.Cuda
            ? ExecutionDeviceKind.Cuda
            : ExecutionDeviceKind.Cpu;
        var session = new ExecutionSession(new ExecutionOptions
        {
            Device = device,
            CudaDevices = new DeviceSet(scenario.DeviceIndices),
            Precision = PrecisionPolicy.Parse(
                TensorPrecisionModeNames.Format(precision)),
        });
        if (device != ExecutionDeviceKind.Cuda)
            return session;

        try
        {
            foreach (int deviceIndex in scenario.DeviceIndices)
            {
                session.AttachLane(
                    CudaExecutionLaneFactory.Create(deviceIndex));
            }
            return session;
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    private static IReadOnlyList<BaselineGpu> ResolveSelectedGpus(
        BaselineScenario scenario,
        IReadOnlyList<BaselineGpu> discovered)
    {
        if (scenario.Device == BaselineDeviceKind.Cpu)
            return [];
        int count = Tensor.CudaDeviceCount;
        foreach (int device in scenario.DeviceIndices)
        {
            if (device >= count)
            {
                throw new InvalidOperationException(
                    $"Scenario '{scenario.Name}' requires cuda:{device}, but " +
                    $"only {count} CUDA device(s) are available.");
            }
        }
        return scenario.DeviceIndices.Select(device =>
        {
            BaselineGpu? known = discovered.FirstOrDefault(gpu =>
                gpu.Index == device);
            return known ?? new BaselineGpu(
                device,
                ForgetMemoryV2Cuda.GetAccelerator(device).Name,
                null,
                null);
        }).ToArray();
    }

    private static void Validate(BaselineWorkerJob job)
    {
        BaselineScenario scenario = job.Scenario;
        BaselineModelConfiguration model = job.Model;
        if (scenario.WarmupSteps < 0)
            throw new ArgumentOutOfRangeException(nameof(scenario.WarmupSteps));
        if (scenario.MeasuredSteps <= 0)
            throw new ArgumentOutOfRangeException(nameof(scenario.MeasuredSteps));
        if (scenario.Repetitions <= 0)
            throw new ArgumentOutOfRangeException(nameof(scenario.Repetitions));
        if (scenario.DeviceIndices.Length == 0
            || scenario.DeviceIndices.Any(device => device < 0)
            || scenario.DeviceIndices.Distinct().Count()
                != scenario.DeviceIndices.Length)
        {
            throw new ArgumentException("Device indices must be unique and non-negative.");
        }
        if (model.Batch <= 0 || model.Sequence <= 0 || model.Vocabulary <= 0)
            throw new ArgumentException("Batch, sequence, and vocabulary must be positive.");
        if (model.Bfp8BlockSize <= 0)
        {
            throw new ArgumentException(
                "BFP8 block size must be positive.");
        }
        if (model.CudaGraphCacheBudgetMiB <= 0)
        {
            throw new ArgumentException(
                "CUDA Graph cache budget must be positive.");
        }
        if (model.Batch < scenario.DeviceIndices.Length
            && scenario.Device == BaselineDeviceKind.Cuda)
        {
            throw new ArgumentException(
                "Batch must provide at least one shard per CUDA device.");
        }
        if (scenario.PerformanceGate is { } performanceGate)
        {
            if (scenario.Soak is not null
                || performanceGate.RequiredCudaDeviceCount <= 0
                || performanceGate.RequiredWarmupSteps < 0
                || performanceGate.RequiredMeasuredSteps <= 0
                || performanceGate.RequiredRepetitions <= 0
                || !double.IsFinite(
                    performanceGate.FrozenStepP50Milliseconds)
                || performanceGate.FrozenStepP50Milliseconds <= 0d
                || !double.IsFinite(
                    performanceGate.MaximumBaselineRatio)
                || performanceGate.MaximumBaselineRatio <= 0d)
            {
                throw new ArgumentException(
                    "Performance-gate configuration is internally " +
                    "inconsistent.");
            }
        }
        if (scenario.Soak is { } soak)
        {
            if (scenario.Device != BaselineDeviceKind.Cuda
                || scenario.DeviceIndices.Length != 2)
            {
                throw new ArgumentException(
                    "The soak harness requires exactly two CUDA devices.");
            }
            if (soak.TotalCommittedSteps <= 0
                || scenario.MeasuredSteps != soak.TotalCommittedSteps
                || soak.PerformanceWarmupSteps < 0
                || soak.PerformanceWarmupSteps >= soak.TotalCommittedSteps
                || soak.TrendWindowSteps <= 0
                || soak.TrendWindowSteps * 2
                    > soak.TotalCommittedSteps
                        - soak.PerformanceWarmupSteps
                || soak.GenerationStep <= 0
                || soak.GenerationStep > soak.TotalCommittedSteps
                || soak.GenerationTokens <= 0
                || soak.RestartStep <= 0
                || soak.RestartStep > soak.TotalCommittedSteps
                || soak.MaximumPostWarmupVramGrowthBytes < 0
                || !double.IsFinite(soak.MaximumLastToFirstP50Ratio)
                || soak.MaximumLastToFirstP50Ratio <= 0d)
            {
                throw new ArgumentException(
                    "Soak configuration is internally inconsistent.");
            }
        }
    }

    private static (int[] Input, int[] Target) CreateBatch(
        BaselineModelConfiguration configuration)
    {
        var random = new Random(configuration.Seed ^ 0x5A17);
        int count = checked(configuration.Batch * configuration.Sequence);
        int[] input = Enumerable.Range(0, count)
            .Select(_ => random.Next(configuration.Vocabulary)).ToArray();
        int[] target = Enumerable.Range(0, count)
            .Select(_ => random.Next(configuration.Vocabulary)).ToArray();
        return (input, target);
    }

    private static void Synchronize(BaselineScenario scenario)
    {
        if (scenario.Device != BaselineDeviceKind.Cuda)
            return;
        foreach (int device in scenario.DeviceIndices)
            ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
    }

    private static double Elapsed(long started)
        => Stopwatch.GetElapsedTime(started).TotalMilliseconds;

    private static BaselineDistribution Distribution<T>(
        IEnumerable<T> values,
        Func<T, double> selector)
        => BaselineDistribution.From(values.Select(selector));

    private static BaselineDistribution? OptionalDistribution(
        IEnumerable<double?> values)
    {
        double[] present = values.Where(value => value.HasValue)
            .Select(value => value!.Value).ToArray();
        return present.Length == 0 ? null : BaselineDistribution.From(present);
    }

    private sealed class BaselineFixture : IDisposable
    {
        private readonly BaselineScenario _scenario;
        private int _disposed;

        internal BaselineFixture(
            BaselineScenario scenario,
            GptRinWikiJp model,
            NekoMuon nekoMuon,
            AdamW adamW,
            CompositeOptimizer optimizer,
            WarmupCosineProgressLRScheduler scheduler,
            Parameter[] parameters,
            int[] input,
            int[] target,
            CudaDataParallelEngine? dataParallelEngine)
        {
            _scenario = scenario;
            Model = model;
            NekoMuon = nekoMuon;
            AdamW = adamW;
            Optimizer = optimizer;
            Scheduler = scheduler;
            Parameters = parameters;
            Input = input;
            Target = target;
            DataParallelEngine = dataParallelEngine;
        }

        internal GptRinWikiJp Model { get; private set; }

        internal NekoMuon NekoMuon { get; private set; }

        internal AdamW AdamW { get; private set; }

        internal CompositeOptimizer Optimizer { get; private set; }

        internal WarmupCosineProgressLRScheduler Scheduler
        {
            get;
            private set;
        }

        internal Parameter[] Parameters { get; private set; }

        internal int[] Input { get; private set; }

        internal int[] Target { get; private set; }

        internal CudaDataParallelEngine? DataParallelEngine
        {
            get;
            private set;
        }

        internal void PrepareForTraining(int batchSize)
        {
            DataParallelEngine?.PrepareForTraining(batchSize);
            Optimizer.prepare();
        }

        internal float PrewarmCompiledTrainingPlan(
            BaselineModelConfiguration configuration,
            long globalStep,
            CudaAdaptiveShardingOptions options,
            CudaAdaptiveShardState? adaptiveState)
        {
            ArgumentNullException.ThrowIfNull(configuration);
            ArgumentNullException.ThrowIfNull(options);
            CudaDataParallelEngine engine = DataParallelEngine
                ?? throw new InvalidOperationException(
                    "CUDA graph prewarm requires a data-parallel engine.");
            PrepareForTraining(configuration.Batch);
            float loss = engine.ForwardBackward(
                Input,
                Target,
                configuration.Batch,
                configuration.Sequence,
                Tensor.DefaultCrossEntropyIgnoreIndex,
                globalStep);
            Optimizer.zero_grad();
            if (adaptiveState is null)
                engine.ConfigureAdaptiveSharding(options);
            else
                engine.RestoreAdaptiveShardingState(adaptiveState);
            return loss;
        }

        internal void RestartDataParallelEngine(
            CudaAdaptiveShardState adaptiveState,
            CudaAdaptiveShardingOptions options)
        {
            ArgumentNullException.ThrowIfNull(adaptiveState);
            ArgumentNullException.ThrowIfNull(options);
            if (_scenario.Device != BaselineDeviceKind.Cuda)
            {
                throw new InvalidOperationException(
                    "Only a CUDA fixture owns a data-parallel engine.");
            }
            DataParallelEngine?.Dispose();
            DataParallelEngine = new CudaDataParallelEngine(
                Model,
                _scenario.DeviceIndices,
                options);
            DataParallelEngine.RestoreAdaptiveShardingState(adaptiveState);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) != 0)
                return;

            List<Exception>? failures = null;
            if (_scenario.Device == BaselineDeviceKind.Cuda)
            {
                try
                {
                    Synchronize(_scenario);
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
            }
            try
            {
                DataParallelEngine?.Dispose();
            }
            catch (Exception exception)
            {
                (failures ??= []).Add(exception);
            }
            finally
            {
                DataParallelEngine = null;
            }

            if (_scenario.Device == BaselineDeviceKind.Cuda)
            {
                // Optimizers currently expose no IDisposable contract. Their
                // owned-state restore paths deterministically retire resident
                // moments, batched descriptors, and scratch allocations. This
                // cleanup is outside all measured intervals.
                try
                {
                    NekoMuon.RestoreStateOwned(
                        NekoMuon.CaptureStateForStreaming());
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
                try
                {
                    AdamW.RestoreStateOwned(
                        AdamW.CaptureStateForStreaming());
                }
                catch (Exception exception)
                {
                    (failures ??= []).Add(exception);
                }
                foreach (Parameter parameter in Parameters)
                {
                    try
                    {
                        parameter.T.InvalidateCudaBuffers();
                    }
                    catch (Exception exception)
                    {
                        (failures ??= []).Add(exception);
                    }
                }
            }

            Model = null!;
            NekoMuon = null!;
            AdamW = null!;
            Optimizer = null!;
            Scheduler = null!;
            Parameters = [];
            Input = [];
            Target = [];

            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);
            GC.WaitForPendingFinalizers();
            GC.Collect(
                GC.MaxGeneration,
                GCCollectionMode.Aggressive,
                blocking: true,
                compacting: true);

            if (_scenario.Device == BaselineDeviceKind.Cuda)
            {
                foreach (int device in _scenario.DeviceIndices)
                {
                    try
                    {
                        Tensor.ClearCudaFloatBufferPool(device);
                    }
                    catch (Exception exception)
                    {
                        (failures ??= []).Add(exception);
                    }
                }
            }
            if (failures is not null)
            {
                throw new AggregateException(
                    "Benchmark fixture cleanup failed.",
                    failures);
            }
        }
    }

    private sealed record RawStep(
        float Loss,
        double TotalMilliseconds,
        double ZeroGradMilliseconds,
        double ForwardBackwardMilliseconds,
        double? ForwardMilliseconds,
        double? LossPhaseMilliseconds,
        double? BackwardMilliseconds,
        double ClipMilliseconds,
        double OptimizerMilliseconds,
        double NekoMuonMilliseconds,
        double AdamWMilliseconds,
        long ManagedAllocationBytes,
        NativeCudaAllocationTelemetry NativeAllocations,
        NativeCudaTransferTelemetry Transfers,
        NativeCudaTransferTelemetry GradientCollectiveTransfers,
        IReadOnlyList<BaselineDeviceMemoryObservation> DeviceMemory)
    {
        internal BaselineStepMeasurement ToMeasurement(
            int step,
            bool isWarmup = false)
            => new(
                step,
                isWarmup,
                Loss,
                TotalMilliseconds,
                ZeroGradMilliseconds,
                ForwardBackwardMilliseconds,
                ForwardMilliseconds,
                LossPhaseMilliseconds,
                BackwardMilliseconds,
                null,
                null,
                ClipMilliseconds,
                OptimizerMilliseconds,
                NekoMuonMilliseconds,
                AdamWMilliseconds,
                ManagedAllocationBytes,
                NativeAllocations.AllocationCount,
                NativeAllocations.AllocationBytes,
                NativeAllocations.FreeCount,
                NativeAllocations.FreeBytes,
                Transfers.HostToDeviceCopyCount,
                Transfers.HostToDeviceBytes,
                Transfers.DeviceToHostCopyCount,
                Transfers.DeviceToHostBytes,
                DeviceMemory)
            {
                GradientCollectiveHostToDeviceCopyCount =
                    GradientCollectiveTransfers.HostToDeviceCopyCount,
                GradientCollectiveHostToDeviceBytes =
                    GradientCollectiveTransfers.HostToDeviceBytes,
                GradientCollectiveDeviceToHostCopyCount =
                    GradientCollectiveTransfers.DeviceToHostCopyCount,
                GradientCollectiveDeviceToHostBytes =
                    GradientCollectiveTransfers.DeviceToHostBytes,
            };
    }
}
