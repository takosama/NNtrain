using System.Text.Json.Serialization;

namespace NNtrain.Benchmarks;

internal static class PerformanceBaselineSchema
{
    internal const string Version = "nntrain.performance-baseline/v2";
}

internal sealed record PerformanceBaselineDocument(
    string SchemaVersion,
    DateTimeOffset CreatedUtc,
    string Preset,
    string Commit,
    bool WorkingTreeDirty,
    BaselineHost Host,
    IReadOnlyList<BaselineScenarioResult> Scenarios);

internal sealed record BaselineHost(
    string OperatingSystem,
    string Runtime,
    string ProcessArchitecture,
    string Processor,
    IReadOnlyList<BaselineGpu> Gpus);

internal sealed record BaselineGpu(
    int Index,
    string Name,
    string? ComputeCapability,
    string? SmArchitecture);

internal sealed record BaselineScenarioResult(
    BaselineConditions Conditions,
    IReadOnlyList<BaselineRunResult> Runs,
    BaselineDistribution AggregateStep,
    BaselinePhaseProbe? PhaseProbe,
    IReadOnlyList<string> Notes,
    BaselineValidationResult? Validation = null);

internal sealed record BaselineConditions(
    string Scenario,
    string Commit,
    string? ConfigurationPath,
    string? ConfigurationSha256,
    string Device,
    IReadOnlyList<int> DeviceIndices,
    IReadOnlyList<BaselineGpu> Gpus,
    string Precision,
    string StorageDType,
    int? Bfp8BlockSize,
    string TrainingExecutionPlan,
    int Batch,
    int Sequence,
    int Vocabulary,
    int Width,
    int Heads,
    int Hidden,
    int Layers,
    string Optimizer,
    int NewtonSchulzSteps,
    string NewtonSchulzDepthMode,
    int NewtonSchulzInterval,
    int WarmupSteps,
    int MeasuredSteps,
    int Repetitions,
    string? PerformanceGateStatistic,
    double? FrozenBaselineStepP50Milliseconds,
    double? MaximumBaselineRatio,
    double? MaximumAllowedStepP50Milliseconds,
    int Seed,
    bool AdaptiveCudaSharding,
    double CudaShardEmaAlpha,
    double CudaMinimumRelativeShardSize,
    int CudaMaximumBatchAdjustmentPerStep,
    int CudaGraphCacheBudgetMiB,
    long ExpectedHostToDeviceBytesPerStep,
    long ExpectedLossReadbackBytesPerStep,
    string InputSource,
    IReadOnlyList<BaselineEffectiveOverride> EffectiveOverrides);

internal sealed record BaselineEffectiveOverride(
    string Setting,
    string ConfiguredValue,
    string EffectiveValue,
    bool Changed,
    string Reason);

internal sealed record BaselineRunResult(
    int Repetition,
    DateTimeOffset StartedUtc,
    DateTimeOffset FinishedUtc,
    BaselineDistribution Step,
    BaselineDistribution ZeroGrad,
    BaselineDistribution ForwardBackward,
    BaselineDistribution? Forward,
    BaselineDistribution? LossPhase,
    BaselineDistribution? Backward,
    BaselineDistribution? ReduceWait,
    BaselineDistribution? Transfer,
    BaselineDistribution Clip,
    BaselineDistribution Optimizer,
    BaselineDistribution NekoMuon,
    BaselineDistribution AdamW,
    BaselineDistribution ManagedAllocationBytes,
    BaselineDistribution NativeAllocationCount,
    BaselineDistribution NativeAllocationBytes,
    BaselineDistribution NativeFreeCount,
    BaselineDistribution NativeFreeBytes,
    BaselineDistribution HostToDeviceCopyCount,
    BaselineDistribution HostToDeviceBytes,
    BaselineDistribution DeviceToHostCopyCount,
    BaselineDistribution DeviceToHostBytes,
    IReadOnlyList<BaselineDeviceMemorySummary> DeviceMemory,
    IReadOnlyList<int> FinalShardBatchSizes,
    BaselineTrainingGraphTelemetry? TrainingGraph,
    IReadOnlyList<BaselineStepMeasurement> Measurements);

internal sealed record BaselineTrainingGraphTelemetry(
    long CaptureCount,
    long ReplayCount,
    long FallbackCount,
    int CachedCompiledPlanCount,
    long GraphPinnedBytes,
    long CapturedReadyEventRecordCount,
    double CapturedReadyEventRecordMilliseconds,
    long MeasuredCaptureCount,
    long MeasuredReplayCount,
    long MeasuredFallbackCount,
    long MeasuredReadyEventRecordCount,
    double MeasuredReadyEventRecordMilliseconds,
    bool MeasuredIntervalFullyCompiledReplay);

internal sealed record BaselineStepMeasurement(
    int Step,
    bool IsWarmup,
    double Loss,
    double TotalMilliseconds,
    double ZeroGradMilliseconds,
    double ForwardBackwardMilliseconds,
    double? ForwardMilliseconds,
    double? LossPhaseMilliseconds,
    double? BackwardMilliseconds,
    double? ReduceWaitMilliseconds,
    double? TransferMilliseconds,
    double ClipMilliseconds,
    double OptimizerMilliseconds,
    double NekoMuonMilliseconds,
    double AdamWMilliseconds,
    long ManagedAllocationBytes,
    long NativeAllocationCount,
    long NativeAllocationBytes,
    long NativeFreeCount,
    long NativeFreeBytes,
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes,
    IReadOnlyList<BaselineDeviceMemoryObservation> DeviceMemory)
{
    public long GradientCollectiveHostToDeviceCopyCount { get; init; }

    public long GradientCollectiveHostToDeviceBytes { get; init; }

    public long GradientCollectiveDeviceToHostCopyCount { get; init; }

    public long GradientCollectiveDeviceToHostBytes { get; init; }
}

internal sealed record BaselineDeviceMemoryObservation(
    int Device,
    long? TotalBytes,
    long? FreeBytes,
    long? UsedBytes,
    string? Error);

internal sealed record BaselineDeviceMemorySummary(
    int Device,
    long TotalBytes,
    long StartUsedBytes,
    long PeakUsedBytes,
    long EndUsedBytes,
    long PeakGrowthBytes,
    int ObservationCount);

internal sealed record BaselineTransferTelemetry(
    long HostToDeviceCopyCount,
    long HostToDeviceBytes,
    long DeviceToHostCopyCount,
    long DeviceToHostBytes)
{
    public long GradientCollectiveHostToDeviceCopyCount { get; init; }

    public long GradientCollectiveHostToDeviceBytes { get; init; }

    public long GradientCollectiveDeviceToHostCopyCount { get; init; }

    public long GradientCollectiveDeviceToHostBytes { get; init; }
}

internal sealed record BaselinePhaseProbe(
    string MeasurementKind,
    double? ForwardMilliseconds,
    double? LossMilliseconds,
    double? BackwardMilliseconds,
    double? ReduceWaitMilliseconds,
    double? GradientPreparationMilliseconds,
    double? ClipMilliseconds,
    double? OptimizerMilliseconds,
    double? TransferMilliseconds,
    double? HostDataPreparationMilliseconds,
    string TransferStatus,
    BaselineTransferTelemetry Transfers,
    IReadOnlyList<BaselineShardProbe> Shards);

internal sealed record BaselineShardProbe(
    int Device,
    int Batch,
    double HostDataPreparationMilliseconds,
    double ForwardMilliseconds,
    double LossMilliseconds,
    double BackwardMilliseconds);

internal sealed record BaselineDistribution(
    int Count,
    double Mean,
    double P50,
    double P95,
    double Minimum,
    double Maximum)
{
    internal static BaselineDistribution From(IEnumerable<double> values)
    {
        double[] ordered = values.OrderBy(value => value).ToArray();
        if (ordered.Length == 0)
            return new BaselineDistribution(0, 0d, 0d, 0d, 0d, 0d);
        return new BaselineDistribution(
            ordered.Length,
            ordered.Average(),
            ordered[ordered.Length / 2],
            Percentile(ordered, 0.95d),
            ordered[0],
            ordered[^1]);
    }

    private static double Percentile(double[] ordered, double percentile)
    {
        int index = Math.Clamp(
            (int)Math.Ceiling(ordered.Length * percentile) - 1,
            0,
            ordered.Length - 1);
        return ordered[index];
    }
}

internal sealed record BaselineWorkerJob(
    string Preset,
    string Commit,
    BaselineModelConfiguration Model,
    BaselineScenario Scenario,
    IReadOnlyList<BaselineGpu> Gpus,
    string? ConfigurationPath,
    string? ConfigurationSha256,
    IReadOnlyList<BaselineEffectiveOverride>? EffectiveOverrides = null);

internal sealed record BaselineScenario(
    string Name,
    BaselineDeviceKind Device,
    int[] DeviceIndices,
    int WarmupSteps,
    int MeasuredSteps,
    int Repetitions,
    bool CollectPhaseProbe,
    BaselineSoakConfiguration? Soak = null,
    BaselinePerformanceGateConfiguration? PerformanceGate = null);

internal sealed record BaselinePerformanceGateConfiguration(
    int RequiredCudaDeviceCount,
    int RequiredWarmupSteps,
    int RequiredMeasuredSteps,
    int RequiredRepetitions,
    int RequiredBatch,
    int RequiredSequence,
    string RequiredPrecision,
    string RequiredNewtonSchulzDepthMode,
    int RequiredNewtonSchulzSteps,
    double FrozenStepP50Milliseconds,
    double MaximumBaselineRatio)
{
    internal double MaximumAllowedStepP50Milliseconds
        => FrozenStepP50Milliseconds * MaximumBaselineRatio;
}

internal sealed record BaselineSoakConfiguration(
    int TotalCommittedSteps,
    int PerformanceWarmupSteps,
    int TrendWindowSteps,
    int GenerationStep,
    int GenerationTokens,
    int RestartStep,
    long MaximumPostWarmupVramGrowthBytes,
    double MaximumLastToFirstP50Ratio,
    bool InjectCheckpointFailureAfterFirstArtifact = false);

internal sealed record BaselineValidationResult(
    string Scope,
    bool Passed,
    IReadOnlyList<BaselineGateResult> Gates,
    BaselineSoakResult? Soak);

internal sealed record BaselineGateResult(
    string Name,
    bool? Passed,
    string Actual,
    string Required,
    string? Detail = null);

internal sealed record BaselineSoakResult(
    int RequestedCommittedSteps,
    int CompletedCommittedSteps,
    int PerformanceWarmupSteps,
    int TrendWindowSteps,
    BaselineDistribution FirstWindow,
    BaselineDistribution LastWindow,
    double LastToFirstP50Ratio,
    int GenerationStep,
    bool GenerationObserved,
    int GeneratedTokens,
    double? GenerationMilliseconds,
    int RestartStep,
    bool RestartObserved,
    bool ResumeArtifactValidated,
    long ResumeArtifactBytes,
    string? ResumeArtifactSha256,
    BaselineCheckpointResult Checkpoint,
    int SidecarEntriesBeforeRestart,
    int SidecarEntriesAfterResume,
    bool SidecarContinuityValidated,
    bool HtmlContinuityChecked,
    bool HtmlContinuityValidated,
    string HtmlContinuityStatus,
    IReadOnlyList<BaselineDeviceMemorySummary> PostWarmupDeviceMemory,
    bool ZeroShardObserved,
    IReadOnlyList<string> RuntimeErrors);

internal sealed record BaselineCheckpointResult(
    int FormatVersion,
    bool Validated,
    double? SaveMilliseconds,
    double? LoadMilliseconds,
    long TotalBytes,
    IReadOnlyList<BaselineCheckpointArtifact> Artifacts,
    bool ArtifactFirstManifestLastValidated,
    bool CursorValidated,
    bool TrainingRandomValidated,
    bool SchedulerValidated,
    bool AdaptiveShardStateValidated,
    bool ModelValidated,
    bool OptimizerValidated,
    bool PrecisionValidated,
    bool Bfp8BlockSizeValidated,
    bool DeviceResidencyValidated,
    bool OldFixtureDisposedBeforeReload,
    bool ArtifactsRetainedAfterFailure,
    string? ArtifactDirectory);

internal sealed record BaselineCheckpointArtifact(
    string Name,
    long Bytes,
    string Sha256);

[JsonConverter(typeof(JsonStringEnumConverter<BaselineDeviceKind>))]
internal enum BaselineDeviceKind
{
    Cpu,
    Cuda,
}

internal sealed record BaselineModelConfiguration(
    int Vocabulary,
    int Batch,
    int Sequence,
    int Width,
    int Heads,
    int Hidden,
    int Layers,
    int Seed,
    float Dropout,
    float InitializationScale,
    bool TieWordEmbeddings,
    string Precision,
    float LearningRate,
    float AuxiliaryLearningRate,
    float NekoMuonBetaFast,
    float WeightDecay,
    string NewtonSchulzDepthMode,
    int NewtonSchulzDepth,
    int NewtonSchulzInterval,
    bool AdaptiveCudaSharding,
    double CudaShardEmaAlpha,
    double CudaMinimumRelativeShardSize,
    int CudaMaximumBatchAdjustmentPerStep,
    int CudaGraphCacheBudgetMiB,
    int Bfp8BlockSize);
