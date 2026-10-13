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
}
