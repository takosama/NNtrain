using System.Diagnostics;
using System.Text.Json;
using NNtrain.Cuda.Execution;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Optimization;

namespace NNtrain.Benchmarks;

/// <summary>Bounded, fixed-seed training benchmark using the actual CLI factories.</summary>
internal static class ForgetMemoryDrnCudaProfiler
{
    internal static void Run(string configurationPath, int warmupSteps,
        int measuredSteps, bool detail = false, string? resultPath = null,
        bool realData = false, float learningRateMultiplier = 1f)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(warmupSteps);
        ArgumentOutOfRangeException.ThrowIfLessThan(measuredSteps, 1);
        if (!float.IsFinite(learningRateMultiplier) || learningRateMultiplier <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRateMultiplier));
        if (realData && (warmupSteps != 0 || measuredSteps > 500 || detail || resultPath is null))
            throw new ArgumentException("Real probes require 0 warmup, 1–500 updates, no detail, and a new result path.");
        string path = Path.GetFullPath(configurationPath);
        if (resultPath is not null && (File.Exists(resultPath)
            || string.Equals(Path.GetFullPath(resultPath), path, StringComparison.OrdinalIgnoreCase)))
            throw new IOException("Benchmark output must be a new artifact.");
        WikiTrainingConfiguration config = WikiTrainingConfiguration.Load(path);
        using IDisposable dispatchScope = CudaDispatchPolicy.Push(WikiLanguageModelCommand.CreateCudaDispatchPolicy(config));
        if (!config.IsForgetMemoryDrnArchitecture()
            || !(config.IsOptimizer(WikiTrainingConfiguration.NekoMuonOptimizer)
                || config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer)))
            throw new ArgumentException("DRN profiling requires forgetmemorydrn with Muon or NekoMuon.");
        int[] devices = config.DeviceIndices ?? [config.DeviceIndex];
        if (config.GetExecutionDevice() != TensorDevice.Cuda
            || devices.Any(device => device < 0 || device >= Tensor.CudaDeviceCount))
            throw new InvalidOperationException("The configured CUDA devices are unavailable.");
        TensorDevice previousDevice = Tensor.ExecutionDevice;
        int[] previousDevices = Tensor.CudaDeviceIndices.ToArray();
        bool previousSimd = Tensor.SimdEnabled;
        int previousWorkers = Tensor.MaxDegreeOfParallelism;
        try
        {
            Tensor.ExecutionDevice = TensorDevice.Cuda;
            Tensor.CudaDeviceIndices = devices;
            Tensor.SimdEnabled = config.UseSimd;
            Tensor.MaxDegreeOfParallelism = config.MaxDegreeOfParallelism;
            RunCore(path, config, devices, warmupSteps, measuredSteps, detail, resultPath,
                realData, learningRateMultiplier);
        }
        finally
        {
            Tensor.CudaDeviceIndices = previousDevices;
            Tensor.ExecutionDevice = previousDevice;
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousWorkers;
        }
    }

    private static void RunCore(string path, WikiTrainingConfiguration config,
        int[] devices, int warmupSteps, int measuredSteps, bool detail, string? resultPath,
        bool realData, float learningRateMultiplier)
    {
        DrnRealDataProbe? probe = realData ? DrnRealDataProbe.Load(config, measuredSteps) : null;
        string? checkpointHash = realData ? HashFile(config.CheckpointPath) : null;
        string nativeHash = HashFile(Path.Combine(AppContext.BaseDirectory, "NNtrain.CudaKernels.dll"));
        TensorPrecisionMode precision = config.GetPrecisionMode();
        WikiLanguageModelCommand.PreflightCudaOptimizer(config, precision);
        using var session = new ExecutionSession(new ExecutionOptions
        {
            Device = ExecutionDeviceKind.Cuda,
            CudaDevices = new DeviceSet(devices),
            Precision = PrecisionPolicy.Parse(TensorPrecisionModeNames.Format(precision)),
        }, devices.Select(device => CudaExecutionLaneFactory.Create(device)));
        using IDisposable scope = session.Enter();
        // This factory attaches CheckpointableRandom and performs physical BFP8 conversion.
        LanguageModel model = WikiLanguageModelCommand.CreateModel(config, config.VocabularySize);
        model.to(TensorDevice.Cuda);
        OptimizerBundle bundle = WikiLanguageModelCommand.CreateOptimizerBundle(model, config);
        IOptimizer optimizer = bundle.RootOptimizer;
        NekoMuon nekoMuon = bundle.LeavesOfType<NekoMuon>().Single();
        AdamW adamW = bundle.LeavesOfType<AdamW>().Single();
        long globalStep = 0;
        if (realData)
        {
            var scheduler = lr_scheduler.WarmupCosineProgressLR(bundle, config.WarmupPercent);
            ModuleState? bestState = null;
            float bestLoss = float.PositiveInfinity;
            int bestEpoch = 0;
            WikiLanguageModelCommand.RestoreTrainingCheckpoint(config with { ResumeFromCheckpoint = true },
                model, bundle, scheduler, ref bestState, ref bestLoss, ref bestEpoch, ref globalStep, Console.Out);
            // Fixed configured rates isolate LR effects; do not silently replay a different scheduler.
            nekoMuon.SetLearningRate(config.LearningRate * learningRateMultiplier);
            adamW.SetLearningRate(config.AuxiliaryLearningRate * learningRateMultiplier);
        }
        long initialGlobalStep = globalStep;
        using var engine = new CudaDataParallelEngine(model, devices, new CudaAdaptiveShardingOptions
        {
            Enabled = config.AdaptiveCudaSharding,
            EmaAlpha = config.CudaShardEmaAlpha,
            MinimumRelativeShardSize = config.CudaMinimumRelativeShardSize,
            MaximumBatchAdjustmentPerStep = config.CudaMaximumBatchAdjustmentPerStep,
            GraphCacheBudgetBytes = checked((long)config.CudaGraphCacheBudgetMiB * 1024 * 1024),
        });
        try
        {
            if (probe is not null) Console.WriteLine("probe: preparing replicas");
            engine.PrepareForTraining(config.BatchSize);
            if (probe is not null) Console.WriteLine("probe: preparing optimizer");
            optimizer.prepare();
            if (probe is not null) Console.WriteLine("probe: prepared, evaluating initial model");
            Parameter[] parameters = model.Parameters().ToArray();
            var random = new Random(config.Seed ^ 0x5A17);
            var batches = Enumerable.Range(0, config.GradientAccumulationSteps).Select(_ =>
            {
                int length = checked(config.BatchSize * config.ContextLength);
                int[] input = Enumerable.Range(0, length).Select(_ => random.Next(config.VocabularySize)).ToArray();
                int[] target = Enumerable.Range(0, length).Select(_ => random.Next(config.VocabularySize)).ToArray();
                return new CudaLanguageModelMicroBatch(input, target, config.BatchSize, config.ContextLength);
            }).ToArray();
            int probeOffset = 0;
            var evaluations = new List<object>();
            if (probe is not null)
            {
                float initialLoss = DrnRealDataProbe.Evaluate(model, probe.Evaluation);
                evaluations.Add(new { Updates = 0, TrainingSeconds = 0d, Loss = initialLoss });
                Console.WriteLine($"real probe initial eval={initialLoss:F6}, token SHA256={probe.TokenHash}");
            }
            void Synchronize()
            {
                foreach (int device in devices)
                    ForgetMemoryV2Cuda.GetAccelerator(device).Synchronize();
            }
            float ForwardBackward()
            {
                var current = probe is null ? batches
                    : probe.Training.AsSpan(probeOffset, config.GradientAccumulationSteps).ToArray();
                probeOffset += config.GradientAccumulationSteps;
                return current.Length == 1
                    ? engine.ForwardBackward(current[0].Input, current[0].Target,
                        config.BatchSize, config.ContextLength, Tensor.DefaultCrossEntropyIgnoreIndex, globalStep++)
                    : engine.ForwardBackwardAccumulated(current,
                        Tensor.DefaultCrossEntropyIgnoreIndex, globalStep++);
            }
            StepSample Step(bool synchronizedPhases)
            {
                Synchronize();
                var nativeBefore = NativeCudaRuntime.AllocationTelemetry;
                var transferBefore = NativeCudaRuntime.TransferTelemetry;
                using IDisposable guard = DeviceTransferGuard.EnterTrainingStep(devices.Length);
                var timer = Stopwatch.StartNew();
                optimizer.zero_grad();
                if (synchronizedPhases) Synchronize();
                double zero = timer.Elapsed.TotalMilliseconds;
                float loss = ForwardBackward();
                if (synchronizedPhases) Synchronize();
                double backward = timer.Elapsed.TotalMilliseconds;
                float norm = nn.utils.clip_grad_norm_(parameters, max_norm: 1f);
                if (synchronizedPhases) Synchronize();
                double clip = timer.Elapsed.TotalMilliseconds;
                double matrix = clip;
                if (synchronizedPhases)
                {
                    nekoMuon.step();
                    Synchronize();
                    matrix = timer.Elapsed.TotalMilliseconds;
                    adamW.step();
                }
                else
                    optimizer.step(); // Preserve production composite batching/overlap.
                Synchronize();
                double total = timer.Elapsed.TotalMilliseconds;
                if (!float.IsFinite(loss) || !float.IsFinite(norm))
                    throw new InvalidOperationException($"Non-finite loss/norm at step {globalStep}: {loss}/{norm}.");
                return new StepSample(globalStep, total, loss, norm, zero, backward - zero,
                    clip - backward, matrix - clip, total - matrix,
                    NativeCudaRuntime.AllocationTelemetry - nativeBefore,
                    NativeCudaRuntime.TransferTelemetry - transferBefore);
            }

            Console.WriteLine($"DRN configuration = {path}");
            Console.WriteLine($"shape batch={config.BatchSize}, accumulation={config.GradientAccumulationSteps}, " +
                $"sequence={config.ContextLength}, width={config.ModelWidth}, hidden={config.HiddenSize}, " +
                $"layers={config.Layers}, key/value={config.ForgetMemoryKeyWidth}/{config.ForgetMemoryValueWidth}, " +
                $"vocabulary={config.VocabularySize}, precision={TensorPrecisionModeNames.Format(precision)} " +
                $"(block {config.Bfp8BlockSize}), GPUs=[{string.Join(',', devices)}], seed={config.Seed}");
            bool ordinaryMuon = config.IsOptimizer(WikiTrainingConfiguration.MuonOptimizer);
            Console.WriteLine($"{(ordinaryMuon ? "Muon" : "NekoMuon")} lr={config.LearningRate:G}, " +
                $"beta={(ordinaryMuon ? .95f : config.NekoMuonBetaFast):G}, " +
                $"NS interval={config.NekoMuonNewtonSchulzInterval}, mode={config.GetNekoMuonNewtonSchulzDepthMode()}, " +
                $"depth={config.GetNekoMuonNewtonSchulzDepth():G}; AdamW lr={config.AuxiliaryLearningRate:G}, beta2=0.95; clip=1");
            Console.WriteLine(probe is null
                ? "Synthetic fixed-seed full-length tokens; fixed configured learning rates; no I/O or LR schedule."
                : $"Real FineWeb probe from checkpoint step {initialGlobalStep}, fixed LR multiplier {learningRateMultiplier}; "
                    + "disjoint probe train/eval documents (not guaranteed unseen during prior training); loading/evaluation excluded from training timer.");
            for (int step = 0; step < warmupSteps; step++)
                Console.WriteLine($"warmup {step + 1}/{warmupSteps} = {Step(false).Total:F2} ms");
            CudaTrainingGraphTelemetry graphBefore = engine.TrainingGraphTelemetry;
            var samples = new StepSample[measuredSteps];
            var peakUsed = new long[devices.Length];
            for (int step = 0; step < samples.Length; step++)
            {
                samples[step] = Step(false);
                for (int d = 0; d < devices.Length; d++)
                {
                    var accelerator = ForgetMemoryV2Cuda.GetAccelerator(devices[d]);
                    peakUsed[d] = Math.Max(peakUsed[d], accelerator.MemorySize - accelerator.GetFreeMemory());
                }
                Console.WriteLine($"step {step + 1}/{measuredSteps} = {samples[step].Total:F2} ms, " +
                    $"loss={samples[step].Loss:F6}, norm={samples[step].GradientNorm:F5}, " +
                    $"native alloc/free={samples[step].Native.AllocationCount}/{samples[step].Native.FreeCount}");
            }
            CudaTrainingGraphTelemetry graphAfter = engine.TrainingGraphTelemetry;
            var vram = devices.Select(device =>
            {
                var accelerator = ForgetMemoryV2Cuda.GetAccelerator(device);
                var lane = (CudaExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Cuda, device);
                return new { Device = device, UsedBytes = accelerator.MemorySize - accelerator.GetFreeMemory(),
                    Allocator = lane.Memory.Telemetry };
            }).ToArray();
            Console.WriteLine($"wall mean={samples.Average(x => x.Total):F2} ms, p50={Median(samples.Select(x => x.Total)):F2} ms; " +
                $"graph capture/replay/fallback={graphAfter.CaptureCount - graphBefore.CaptureCount}/" +
                $"{graphAfter.ReplayCount - graphBefore.ReplayCount}/{graphAfter.FallbackCount - graphBefore.FallbackCount}");
            Console.WriteLine("VRAM used MiB = " + string.Join(", ", vram.Select(x => $"GPU{x.Device}:{x.UsedBytes / 1048576d:F1}")));
            if (probe is not null)
            {
                // End-only evaluation: release captured activations before running
                // a different inference shape, without perturbing timed training.
                engine.ReleaseCheckpointTransientMemory();
                float eval = DrnRealDataProbe.Evaluate(model, probe.Evaluation);
                double seconds = samples.Sum(x => x.Total) / 1000;
                evaluations.Add(new { Updates = measuredSteps, TrainingSeconds = seconds, Loss = eval });
                Console.WriteLine($"eval after {measuredSteps} updates / {seconds:F2} training seconds = {eval:F6}");
            }

            // One complete NS interval prevents attribution from accidentally excluding expensive NS updates.
            int phaseSteps = probe is null ? Math.Max(5, config.NekoMuonNewtonSchulzInterval) : 0;
            var phases = new StepSample[phaseSteps];
            for (int step = 0; step < phases.Length; step++) phases[step] = Step(true);
            Console.WriteLine($"synchronized diagnostic ({phaseSteps} steps, extra synchronization/no composite overlap):");
            if (phases.Length > 0) Console.WriteLine($"zero={phases.Average(x => x.Zero):F2}, fwd+bwd+reduce={phases.Average(x => x.ForwardBackward):F2}, " +
                $"clip={phases.Average(x => x.Clip):F2}, NekoMuon={phases.Average(x => x.NekoMuon):F2}, " +
                $"AdamW={phases.Average(x => x.AdamW):F2}, total={phases.Average(x => x.Total):F2} ms");
            CudaDataParallelProfile? eagerProfile = null;
            CudaBfp8CodecTelemetrySnapshot? eagerCodecs = null;
            IReadOnlyList<CudaOperationProfileSample> operations = [];
            if (detail)
            {
                optimizer.zero_grad();
                var codecBefore = CudaBfp8CodecTelemetry.Snapshot;
                using (CudaOperationProfiler.Begin())
                {
                    eagerProfile = engine.ForwardBackwardProfiled(batches[0].Input, batches[0].Target,
                        config.BatchSize, config.ContextLength);
                    operations = CudaOperationProfiler.Snapshot();
                }
                eagerCodecs = CudaBfp8CodecTelemetry.Snapshot - codecBefore;
                Console.WriteLine("eager operation diagnostic (one microbatch; not comparable to graph wall time):");
                Console.WriteLine(JsonSerializer.Serialize(eagerProfile));
                foreach (var group in operations.GroupBy(x => x.Operation)
                    .OrderByDescending(group => group.Max(x => x.TotalMilliseconds)).Take(20))
                    Console.WriteLine($"{group.Key}: max-device {group.Max(x => x.TotalMilliseconds):F2} ms, calls={group.Sum(x => x.Count)}");
            }
            if (resultPath is not null)
            {
                string output = Path.GetFullPath(resultPath);
                if (string.Equals(output, path, StringComparison.OrdinalIgnoreCase) || File.Exists(output))
                    throw new IOException("Benchmark output must be a new file, not an existing artifact.");
                Directory.CreateDirectory(Path.GetDirectoryName(output)!);
                File.WriteAllText(output, JsonSerializer.Serialize(new
                {
                    Configuration = path, TimestampUtc = DateTimeOffset.UtcNow,
                    Conditions = config, WarmupSteps = warmupSteps, MeasuredSteps = measuredSteps,
                    Workload = probe is null ? "synthetic full-length" : "real corpus checkpoint probe",
                    InitialGlobalStep = initialGlobalStep, LearningRateMultiplier = learningRateMultiplier,
                    ProbeTokenHash = probe?.TokenHash, CorpusLoadSeconds = probe?.LoadSeconds,
                    Evaluations = evaluations, SampledPeakUsedBytes = peakUsed,
                    CheckpointManifestSha256 = checkpointHash, NativeDllSha256 = nativeHash,
                    DispatchPolicy = new { CudaDispatchPolicy.Current.DrnRetainedHistoryBudgetBytes,
                        CudaDispatchPolicy.Current.DisableDrnChunkBackward,
                        CudaDispatchPolicy.Current.DisableDirectBfp8Elementwise,
                        CudaDispatchPolicy.Current.DisableDirectDrnMix8LossHead,
                        CudaDispatchPolicy.Current.DisableBfp8LayerNormParameterCache },
                    MeanMilliseconds = samples.Average(x => x.Total), MedianMilliseconds = Median(samples.Select(x => x.Total)),
                    Samples = samples, GraphBefore = graphBefore, GraphAfter = graphAfter, Vram = vram,
                    SynchronizedPhases = phases, EagerProfile = eagerProfile, EagerCodecs = eagerCodecs, Operations = operations,
                }, new JsonSerializerOptions { WriteIndented = true }));
                Console.WriteLine($"result = {output}");
            }
        }
        finally
        {
            try { nekoMuon.DisposeCudaResources(); }
            finally { adamW.DisposeCudaResources(); }
        }
    }

    private static double Median(IEnumerable<double> values)
    {
        double[] sorted = values.Order().ToArray();
        return (sorted[(sorted.Length - 1) / 2] + sorted[sorted.Length / 2]) / 2;
    }

    private static string HashFile(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(stream));
    }

    private sealed record StepSample(long Step, double Total, float Loss, float GradientNorm,
        double Zero, double ForwardBackward, double Clip, double NekoMuon, double AdamW,
        NativeCudaAllocationTelemetry Native, NativeCudaTransferTelemetry Transfers);
}
