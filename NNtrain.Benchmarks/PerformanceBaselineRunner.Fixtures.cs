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
                BetaFast = configuration.NekoMuonBetaFast,
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
}
