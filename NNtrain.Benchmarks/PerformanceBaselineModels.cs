using System.Text.Json.Serialization;

namespace NNtrain.Benchmarks;

internal static class PerformanceBaselineSchema
{
    internal const string Version = "nntrain.performance-baseline/v1";
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
    IReadOnlyList<string> Notes);

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
    int Seed,
    bool AdaptiveCudaSharding,
    double CudaShardEmaAlpha,
    double CudaMinimumRelativeShardSize,
    int CudaMaximumBatchAdjustmentPerStep,
    long ExpectedHostToDeviceBytesPerStep,
    long ExpectedLossReadbackBytesPerStep,
    string InputSource);

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
    BaselineDistribution NativeAllocationBytes,
    IReadOnlyList<int> FinalShardBatchSizes,
    IReadOnlyList<BaselineStepMeasurement> Measurements);

internal sealed record BaselineStepMeasurement(
    int Step,
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
    long NativeFreeBytes);

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
    string? ConfigurationSha256);

internal sealed record BaselineScenario(
    string Name,
    BaselineDeviceKind Device,
    int[] DeviceIndices,
    int WarmupSteps,
    int MeasuredSteps,
    int Repetitions,
    bool CollectPhaseProbe);

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
    float WeightDecay,
    int NewtonSchulzInterval,
    bool AdaptiveCudaSharding,
    double CudaShardEmaAlpha,
    double CudaMinimumRelativeShardSize,
    int CudaMaximumBatchAdjustmentPerStep);
