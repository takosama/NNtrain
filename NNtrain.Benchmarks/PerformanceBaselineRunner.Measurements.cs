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
