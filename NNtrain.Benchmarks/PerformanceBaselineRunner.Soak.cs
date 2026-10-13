using System.Diagnostics;
using System.Security.Cryptography;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Execution;
using NNtrain.Training.Metrics;
using NNtrain.Training.Optimization;
using NNtrain.Training.Persistence;

namespace NNtrain.Benchmarks;

internal static partial class PerformanceBaselineRunner
{
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
}
