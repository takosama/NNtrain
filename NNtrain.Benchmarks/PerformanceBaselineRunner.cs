using System.Diagnostics;

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

            IReadOnlyList<BaselineGpu> selectedGpus = ResolveSelectedGpus(
                scenario, job.Gpus);
            var runs = new List<BaselineRunResult>(scenario.Repetitions);
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
                        scenario);
                }

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
                        scenario);
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
                runs.Add(CreateRun(
                    repetition,
                    started,
                    DateTimeOffset.UtcNow,
                    fixture.DataParallelEngine?.LastShardBatchSizes ?? [],
                    measurements));
            }

            BaselinePhaseProbe? probe = null;
            if (scenario.CollectPhaseProbe)
            {
                using BaselineFixture probeFixture = CreateFixture(
                    configuration,
                    precision,
                    storageDType,
                    scenario);
                if (scenario.Device == BaselineDeviceKind.Cuda)
                {
                    for (int warmup = 0;
                        warmup < scenario.WarmupSteps;
                        warmup++)
                    {
                        _ = MeasureStep(
                            probeFixture,
                            configuration,
                            scenario);
                    }
                }
                probe = CreatePhaseProbe(
                    probeFixture,
                    configuration,
                    scenario,
                    runs);
            }
            BaselineConditions conditions = CreateConditions(
                job, precision, storageDType, selectedGpus);
            BaselineDistribution aggregateStep = BaselineDistribution.From(
                runs.SelectMany(run => run.Measurements)
                    .Select(measurement => measurement.TotalMilliseconds));
            return new BaselineScenarioResult(
                conditions,
                runs,
                aggregateStep,
                probe,
                CreateNotes(scenario));
        }
        finally
        {
            Tensor.ExecutionDevice = previousDevice;
            Tensor.CudaDeviceIndices = previousDeviceIndices;
            Tensor.SimdEnabled = previousSimd;
        }
    }

    private static RawStep MeasureStep(
        BaselineFixture fixture,
        BaselineModelConfiguration configuration,
        BaselineScenario scenario)
    {
        Synchronize(scenario);
        long managedBefore = GC.GetTotalAllocatedBytes(precise: false);
        NativeCudaAllocationTelemetry nativeBefore =
            NativeCudaRuntime.AllocationTelemetry;
        long totalStarted = Stopwatch.GetTimestamp();

        long phaseStarted = Stopwatch.GetTimestamp();
        fixture.Optimizer.zero_grad();
        Synchronize(scenario);
        double zeroGrad = Elapsed(phaseStarted);

        double? forward = null;
        double? lossPhase = null;
        double? backward = null;
        float lossValue;
        phaseStarted = Stopwatch.GetTimestamp();
        if (scenario.Device == BaselineDeviceKind.Cuda)
        {
            lossValue = fixture.DataParallelEngine!.ForwardBackward(
                fixture.Input,
                fixture.Target,
                configuration.Batch,
                configuration.Sequence);
            Synchronize(scenario);
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

        phaseStarted = Stopwatch.GetTimestamp();
        _ = nn.utils.clip_grad_norm_(fixture.Parameters, max_norm: 1f);
        Synchronize(scenario);
        double clip = Elapsed(phaseStarted);

        phaseStarted = Stopwatch.GetTimestamp();
        fixture.NekoMuon.step();
        Synchronize(scenario);
        double neko = Elapsed(phaseStarted);
        phaseStarted = Stopwatch.GetTimestamp();
        fixture.AdamW.step();
        Synchronize(scenario);
        double adam = Elapsed(phaseStarted);
        double optimizerMilliseconds = neko + adam;

        double total = Elapsed(totalStarted);
        long managedBytes = Math.Max(
            0,
            GC.GetTotalAllocatedBytes(precise: false) - managedBefore);
        NativeCudaAllocationTelemetry native =
            NativeCudaRuntime.AllocationTelemetry - nativeBefore;
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
            native);
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
        return new BaselinePhaseProbe(
            "one diagnostic CUDA step with phase synchronization",
            forward,
            loss,
            backward,
            profile.AllReduceMilliseconds,
            profile.GradientPreparationMilliseconds,
            diagnosticClip,
            optimizerP50,
            null,
            hostPreparation,
            "H2D copies are queued inside kernels and are not isolated by the " +
            "current public profiler; expected bytes are recorded in conditions",
            shards);
    }

    private static BaselineRunResult CreateRun(
        int repetition,
        DateTimeOffset started,
        DateTimeOffset finished,
        IReadOnlyList<int> finalShardBatchSizes,
        IReadOnlyList<BaselineStepMeasurement> measurements)
        => new(
            repetition,
            started,
            finished,
            Distribution(measurements, value => value.TotalMilliseconds),
            Distribution(measurements, value => value.ZeroGradMilliseconds),
            Distribution(
                measurements, value => value.ForwardBackwardMilliseconds),
            OptionalDistribution(measurements.Select(value =>
                value.ForwardMilliseconds)),
            OptionalDistribution(measurements.Select(value =>
                value.LossPhaseMilliseconds)),
            OptionalDistribution(measurements.Select(value =>
                value.BackwardMilliseconds)),
            OptionalDistribution(measurements.Select(value =>
                value.ReduceWaitMilliseconds)),
            OptionalDistribution(measurements.Select(value =>
                value.TransferMilliseconds)),
            Distribution(measurements, value => value.ClipMilliseconds),
            Distribution(measurements, value => value.OptimizerMilliseconds),
            Distribution(measurements, value => value.NekoMuonMilliseconds),
            Distribution(measurements, value => value.AdamWMilliseconds),
            Distribution(measurements, value => value.ManagedAllocationBytes),
            Distribution(measurements, value => value.NativeAllocationBytes),
            finalShardBatchSizes.ToArray(),
            measurements);

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
            model.Batch,
            model.Sequence,
            model.Vocabulary,
            model.Width,
            model.Heads,
            model.Hidden,
            model.Layers,
            "NekoMuon+AdamW (FP32 moments)",
            5,
            "fixed",
            model.NewtonSchulzInterval,
            scenario.WarmupSteps,
            scenario.MeasuredSteps,
            scenario.Repetitions,
            model.Seed,
            model.AdaptiveCudaSharding,
            model.CudaShardEmaAlpha,
            model.CudaMinimumRelativeShardSize,
            model.CudaMaximumBatchAdjustmentPerStep,
            scenario.Device == BaselineDeviceKind.Cuda
                ? checked(tokens * sizeof(int) * 2)
                : 0,
            scenario.Device == BaselineDeviceKind.Cuda
                ? checked((long)scenario.DeviceIndices.Length * sizeof(float))
                : 0,
            "fixed synthetic token/target arrays; dataset I/O excluded");
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
            "Every repetition starts from the same seed with a new model, " +
            "optimizer, token batch, and explicitly owned data-parallel engine.",
            "The diagnostic phase probe uses another fresh fixture and is " +
            "excluded from all measured repetitions.",
        };
        if (scenario.Device == BaselineDeviceKind.Cuda)
        {
            notes.Add(
                "Forward/backward/reduce-wait are reported by one separate " +
                "synchronizing diagnostic probe; they must not be summed with " +
                "the normal-path step p50/p95.");
            notes.Add(
                "Transfer duration is null because the current profiler cannot " +
                "isolate asynchronous token/target H2D copies without changing " +
                "the measured execution path.");
        }
        return notes;
    }

    private static BaselineFixture CreateFixture(
        BaselineModelConfiguration configuration,
        TensorPrecisionMode precision,
        TensorDType storageDType,
        BaselineScenario scenario)
    {
        var model = new GptRinWikiJp(
            configuration.Vocabulary,
            configuration.Sequence,
            configuration.Width,
            configuration.Heads,
            configuration.Hidden,
            configuration.Layers,
            new Random(configuration.Seed),
            configuration.InitializationScale,
            configuration.Dropout,
            storageDType,
            configuration.TieWordEmbeddings);
        model.SetPrecisionMode(precision);
        var nekoMuon = new NekoMuon(
            model.HiddenWeightParameters,
            new NekoMuonOptions
            {
                LearningRate = configuration.LearningRate,
                WeightDecay = configuration.WeightDecay,
                MaxNewtonSchulzSteps = 5,
                NewtonSchulzInterval = configuration.NewtonSchulzInterval,
                NewtonSchulzDepthMode =
                    NekoMuonNewtonSchulzDepthMode.Fixed,
                NewtonSchulzDepth = 5f,
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
        };

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
        if (model.Batch < scenario.DeviceIndices.Length
            && scenario.Device == BaselineDeviceKind.Cuda)
        {
            throw new ArgumentException(
                "Batch must provide at least one shard per CUDA device.");
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
            Parameters = parameters;
            Input = input;
            Target = target;
            DataParallelEngine = dataParallelEngine;
        }

        internal GptRinWikiJp Model { get; private set; }

        internal NekoMuon NekoMuon { get; private set; }

        internal AdamW AdamW { get; private set; }

        internal CompositeOptimizer Optimizer { get; private set; }

        internal Parameter[] Parameters { get; private set; }

        internal int[] Input { get; private set; }

        internal int[] Target { get; private set; }

        internal CudaDataParallelEngine? DataParallelEngine
        {
            get;
            private set;
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
        NativeCudaAllocationTelemetry NativeAllocations)
    {
        internal BaselineStepMeasurement ToMeasurement(int step)
            => new(
                step,
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
                NativeAllocations.FreeBytes);
    }
}
