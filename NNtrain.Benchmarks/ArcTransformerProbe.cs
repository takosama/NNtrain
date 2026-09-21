using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using NNtrain.Training.Optimization;

namespace NNtrain.Benchmarks;

/// <summary>
/// Bounded training probe. The CLI model and optimizer factories preserve the
/// production mathematical configuration; only explicitly requested dimensions
/// are reduced. No corpus, tokenizer, checkpoint or metrics files are accessed.
/// </summary>
internal static class ArcTransformerProbe
{
    private const string NextFeatureNames = "inline,dq,dkv,pv,norm,scatter,direct,storage,sg,tiles,lru,retired";

    internal static void Run(string[] args)
    {
        if (args.Length < 2)
            throw new ArgumentException("Expected config and NEW result.json, followed by optional "
                + "--batch N --sequence N --layers N --accumulation N --warmup N --steps N --device arc|cpu "
                + "--arc-mode pre-xmx|optimized (or historical reference modes) --xmx-mode auto|legacy|narrow|wide "
                + "--attention-mode legacy|unrolled|panel|optimized --next-features none|" + NextFeatureNames + " "
                + "--attention-workspace MiB --event-limit N --pool-mib MiB --deferred-mib MiB --profile --compare-cpu. "
                + "--next-features replaces the enabled feature set; use none to disable all listed features. "
                + "The direct feature takes precedence over --xmx-mode for supported shapes.");
        string configPath = Path.GetFullPath(args[0]);
        string resultPath = Path.GetFullPath(args[1]);
        if (File.Exists(resultPath) || string.Equals(configPath, resultPath, StringComparison.OrdinalIgnoreCase))
            throw new IOException("Probe output must be a new artifact; refusing to replace an existing file.");
        var flags = new Dictionary<string, string>(StringComparer.Ordinal);
        bool compareCpu = false, detailed = false;
        for (int i = 2; i < args.Length; i++)
        {
            if (args[i] == "--compare-cpu") { compareCpu = true; continue; }
            if (args[i] == "--profile") { detailed = true; continue; }
            if (args[i] is not ("--batch" or "--sequence" or "--layers" or "--accumulation" or "--warmup" or "--steps" or "--device" or "--arc-mode" or "--xmx-mode" or "--attention-mode" or "--next-features" or "--attention-workspace" or "--event-limit" or "--pool-mib" or "--deferred-mib"))
                throw new ArgumentException($"Unknown probe argument: {args[i]}");
            if (++i >= args.Length) throw new ArgumentException("Missing probe argument value.");
            if (!flags.TryAdd(args[i - 1], args[i])) throw new ArgumentException($"Repeated argument: {args[i - 1]}");
        }
        int Number(string key, int fallback, int minimum = 1)
        {
            int value = flags.TryGetValue(key, out string? text) ? int.Parse(text, CultureInfo.InvariantCulture) : fallback;
            if (value < minimum) throw new ArgumentOutOfRangeException(key);
            return value;
        }
        WikiTrainingConfiguration original = WikiTrainingConfiguration.Load(configPath);
        if (!string.Equals(original.ModelArchitecture, "transformer", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException("This probe requires the Transformer configuration.");
        int warmup = Number("--warmup", 1, 0);
        int steps = Number("--steps", 3);
        if (warmup > 100 || steps > 210) throw new ArgumentException("Probe is bounded to 100 warmup and 210 measured updates.");
        string deviceText = flags.GetValueOrDefault("--device", "arc");
        string arcMode = flags.GetValueOrDefault("--arc-mode", "optimized");
        ArcExecutionOptions arcOptions = arcMode switch {
            "reference" => ArcExecutionOptions.Reference,
            "muon" => new() { ResidentTensors = false, TiledMatrices = false, StreamingAttention = false, ChunkedLossHead = false, PackedUploads = false, FusedNormalization = false, XmxMatrices = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "tiled" => new() { ResidentTensors = false, ChunkedLossHead = false, PackedUploads = false, FusedNormalization = false, XmxMatrices = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "chunked" => new() { ResidentTensors = false, PackedUploads = false, FusedNormalization = false, XmxMatrices = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "packed" => new() { ResidentTensors = false, FusedNormalization = false, XmxMatrices = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "staged" => new() { ResidentTensors = false, XmxMatrices = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "resident" => new() { CoalescedAttention = false, BatchedAttention = false, XmxMatrices = false, ParallelReductions = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "coalesced" => new() { BatchedAttention = false, XmxMatrices = false, ParallelReductions = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "batched" => new() { XmxMatrices = false, ParallelReductions = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "reduced" => new() { XmxMatrices = false, WideMatrices = false, BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "synchronous" => new() { BatchDispatch = false, LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "small-loss" => new() { LossChunkRows = 128, DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "previous" => new() { DeepMatrices = false, CausalAttentionBounds = false, ParallelWeightGradients = false },
            "causal-only" => new() { DeepMatrices = false, ParallelWeightGradients = false },
            "deep-only" => new() { CausalAttentionBounds = false, ParallelWeightGradients = false },
            "no-split" => new() { ParallelWeightGradients = false },
            "attention128" => new() { AttentionWorkspaceMiB = 128 },
            "attention256" => new() { AttentionWorkspaceMiB = 256 },
            "deferred" => new() { DeferredReleaseBytes = 128L * 1024 * 1024 },
            "pre-profile" => new() { MixedBackwardMatrixOperands = false, DeferredReleaseBytes = 0 },
            "mixed" => new() { MixedBackwardMatrixOperands = true, DeferredReleaseBytes = 128L * 1024 * 1024 },
            "compact" => new() { CompactAttentionTiles = true },
            "pool512" => new() { BufferPoolBytes = 512L * 1024 * 1024 },
            "pooled" => new() { BufferPoolBytes = 512L * 1024 * 1024, CompactAttentionTiles = true },
            "pre-xmx" => new() { XmxGemmMode = ArcXmxGemmMode.Legacy, UnrolledAttentionTiles = false, PanelAttention = false },
            "optimized" => new(),
            _ => throw new ArgumentException("Unknown --arc-mode. Use pre-xmx, pre-profile, deferred, mixed, compact, pool512, pooled, attention128, attention256, optimized, or a historical reference mode."),
        };
        // Freeze the historical modes independently of changing production defaults.
        if (arcMode != "optimized")
            arcOptions = arcOptions with { InlineMatrixGradient = false, FusedAttentionDkv = false,
                FusedAttentionDq = false, FusedAttentionPv = false, BlockResidualNorm = false, FusedNormGradient = false,
                DirectXmxMatrices = false, PackedMatrixStorage = false, SubgroupAttentionReduction = false,
                TunedFp32Attention = false, LruBufferPool = false, ReuseRetiredBuffers = false };
        if (arcMode is not ("optimized" or "pre-xmx"))
            arcOptions = arcOptions with {
                CompactAttentionTiles = arcMode is "compact" or "pooled",
                XmxGemmMode = ArcXmxGemmMode.Legacy,
                UnrolledAttentionTiles = false,
                PanelAttention = false,
                BufferPoolBytes = arcMode is "pool512" or "pooled" ? 512L * 1024 * 1024
                    : arcMode == "reference" ? 0 : 256L * 1024 * 1024,
            };
        if (arcMode is not ("optimized" or "pre-xmx" or "mixed" or "compact" or "pool512" or "pooled"))
            arcOptions = arcOptions with { MixedBackwardMatrixOperands = false, DeferredReleaseBytes = arcMode == "deferred" ? 128L * 1024 * 1024 : 0 };
        ArcXmxGemmMode xmxMode = flags.TryGetValue("--xmx-mode", out string? xmxText) ? xmxText switch {
            "auto" => ArcXmxGemmMode.Auto, "legacy" => ArcXmxGemmMode.Legacy,
            "narrow" => ArcXmxGemmMode.Narrow, "wide" => ArcXmxGemmMode.Wide,
            _ => throw new ArgumentException("--xmx-mode must be auto, legacy, narrow or wide.")
        } : arcOptions.XmxGemmMode;
        arcOptions = arcOptions with { DetailedProfiling = detailed, XmxGemmMode = xmxMode };
        if (flags.TryGetValue("--attention-mode", out string? attentionText))
            arcOptions = attentionText switch {
                "legacy" => arcOptions with { UnrolledAttentionTiles = false, PanelAttention = false },
                "unrolled" => arcOptions with { UnrolledAttentionTiles = true, PanelAttention = false },
                "panel" => arcOptions with { UnrolledAttentionTiles = false, PanelAttention = true },
                "optimized" => arcOptions with { UnrolledAttentionTiles = true, PanelAttention = true },
                _ => throw new ArgumentException("--attention-mode must be legacy, unrolled, panel or optimized.")
            };
        TensorDevice device = deviceText switch { "arc" => TensorDevice.Arc, "cpu" => TensorDevice.Cpu,
            _ => throw new ArgumentException("Probe device must be arc or cpu.") };
        if (flags.TryGetValue("--next-features", out string? featuresText))
        {
            var features = featuresText.Split(',').ToHashSet(StringComparer.Ordinal);
            if (features.Any(f => f is not ("none" or "inline" or "dq" or "dkv" or "pv" or "norm" or "scatter" or "direct" or "storage" or "sg" or "tiles" or "lru" or "retired"))
                || features.Contains("none") && features.Count != 1)
                throw new ArgumentException("--next-features accepts none or a comma-separated subset of " + NextFeatureNames + ".");
            arcOptions = arcOptions with {
                InlineMatrixGradient = features.Contains("inline"), FusedAttentionDq = features.Contains("dq"),
                FusedAttentionDkv = features.Contains("dkv"), FusedAttentionPv = features.Contains("pv"),
                BlockResidualNorm = features.Contains("norm"), FusedNormGradient = features.Contains("scatter"), DirectXmxMatrices = features.Contains("direct"),
                PackedMatrixStorage = features.Contains("storage"), SubgroupAttentionReduction = features.Contains("sg"),
                TunedFp32Attention = features.Contains("tiles"), LruBufferPool = features.Contains("lru"),
                ReuseRetiredBuffers = features.Contains("retired") };
        }
        arcOptions = arcOptions with { AttentionWorkspaceMiB = Number("--attention-workspace", arcOptions.AttentionWorkspaceMiB, 8) };
        arcOptions = arcOptions with {
            QueuedKernelLimit = Number("--event-limit", arcOptions.QueuedKernelLimit, 16),
            BufferPoolBytes = Number("--pool-mib", checked((int)(arcOptions.BufferPoolBytes / 1048576)), 0) * 1048576L,
            DeferredReleaseBytes = Number("--deferred-mib", checked((int)(arcOptions.DeferredReleaseBytes / 1048576)), 0) * 1048576L };
        WikiTrainingConfiguration config = original with
        {
            BatchSize = Number("--batch", original.BatchSize),
            ContextLength = Number("--sequence", original.ContextLength),
            Layers = Number("--layers", original.Layers),
            GradientAccumulationSteps = Number("--accumulation", original.GradientAccumulationSteps),
            Device = deviceText,
            DeviceIndices = [original.DeviceIndex],
        };
        if (config.BatchSize > original.BatchSize || config.ContextLength > original.ContextLength
            || config.Layers > original.Layers || config.GradientAccumulationSteps > original.GradientAccumulationSteps)
            throw new ArgumentException("Shape overrides may only reduce the configured dimensions.");
        var random = new Random(config.Seed ^ 0x5A17);
        var batches = Enumerable.Range(0, config.GradientAccumulationSteps).Select(_ =>
        {
            int n = checked(config.BatchSize * config.ContextLength);
            var input = new int[n]; var target = new int[n];
            for (int i = 0; i < n; i++) { input[i] = random.Next(3, config.VocabularySize); target[i] = random.Next(3, config.VocabularySize); }
            return (Input: input, Target: target);
        }).ToArray();
        Console.WriteLine($"Arc Transformer probe: batch={config.BatchSize}, accumulation={config.GradientAccumulationSteps}, sequence={config.ContextLength}, layers={config.Layers}, width={config.ModelWidth}, hidden={config.HiddenSize}, heads={config.Heads}, vocabulary={config.VocabularySize}, precision={config.GetPrecisionMode()}, block={config.Bfp8BlockSize}, optimizer={config.Optimizer}, warmup={warmup}, measured={steps}");
        Console.WriteLine("Synthetic fixed-seed tokens, fresh model, fixed configured learning rates; no tokenizer/corpus/checkpoint/metrics I/O. Phase timing is synchronous wall time; finite-gradient scan is excluded.");
        if (device == TensorDevice.Arc)
        {
            ArcDeviceInfo info = ArcDevices.Enumerate()[original.DeviceIndex];
            Console.WriteLine($"Arc capability: minimum subgroup={info.MinimumSubgroupSize}, XMX matrix extension={info.SupportsXmx}");
        }
        bool previousSimd = Tensor.SimdEnabled;
        int previousWorkers = Tensor.MaxDegreeOfParallelism;
        var results = new List<RunResult>();
        try
        {
            Tensor.SimdEnabled = config.UseSimd;
            Tensor.MaxDegreeOfParallelism = config.MaxDegreeOfParallelism;
            results.Add(Measure(config, device, batches, warmup, steps, arcOptions));
            if (compareCpu && device != TensorDevice.Cpu)
                results.Add(Measure(config with { Device = "cpu" }, TensorDevice.Cpu, batches, warmup, steps, arcOptions));
        }
        finally
        {
            Tensor.SimdEnabled = previousSimd;
            Tensor.MaxDegreeOfParallelism = previousWorkers;
        }
        object report = new
        {
            SchemaVersion = 1,
            TimestampUtc = DateTimeOffset.UtcNow,
            ConfigurationPath = configPath,
            ConfigurationSha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(configPath))),
            BinarySha256 = new[] { typeof(Tensor).Assembly, typeof(ArcExecutionLane).Assembly, typeof(ArcTransformerProbe).Assembly }
                .ToDictionary(assembly => assembly.GetName().Name!, assembly => Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(assembly.Location)))),
            Seed = config.Seed,
            Shape = new { config.BatchSize, config.GradientAccumulationSteps, config.ContextLength, config.ModelWidth,
                config.Heads, config.HiddenSize, config.Layers, config.VocabularySize, config.Dropout, config.TieWordEmbeddings },
            Precision = new { Mode = TensorPrecisionModeNames.Format(config.GetPrecisionMode()), config.Bfp8BlockSize },
            Optimizer = new { config.Optimizer, config.LearningRate, config.AuxiliaryLearningRate, config.WeightDecay,
                config.NekoMuonNewtonSchulzInterval, DepthMode = config.GetNekoMuonNewtonSchulzDepthMode().ToString(), Depth = config.GetNekoMuonNewtonSchulzDepth(), config.NekoMuonBetaFast },
            WarmupSteps = warmup, MeasuredSteps = steps,
            ArcOptions = arcOptions,
            Notes = "Fixed synthetic full-length tokens. Fresh seeded model; no LR scheduling, evaluation, checkpoint or corpus I/O. Gradient clipping max_norm=1 matches WikiLanguageModelCommand.TrainingStep. Host process working set is not dedicated VRAM. Lane byte counters describe backend allocations/transfers, not driver-reported VRAM. Kernel timings are OpenCL event GPU durations. Allocation milliseconds include upload; transfer milliseconds include pending kernel waits, so these counters are not disjoint. Amdahl fractions use sums of measured synchronous phase wall durations.",
            Overrides = flags,
            Results = results,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(resultPath)!);
        using (var output = new FileStream(resultPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
            JsonSerializer.Serialize(output, report, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Arc probe saved: {resultPath}");
    }

    private static RunResult Measure(WikiTrainingConfiguration config, TensorDevice device,
        (int[] Input, int[] Target)[] batches, int warmup, int steps, ArcExecutionOptions arcOptions)
    {
        GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
        using ExecutionSession session = ProductionTrainingSessionFactory.CreateExecutionSession(
            config.GetPrecisionMode(), device, [config.DeviceIndex], arcOptions: arcOptions);
        using IDisposable scope = session.Enter();
        ArcExecutionLane? lane = device == TensorDevice.Arc
            ? (ArcExecutionLane)session.GetRequiredLane(ExecutionDeviceKind.Arc, config.DeviceIndex) : null;
        LanguageModel model = WikiLanguageModelCommand.CreateModel(config, config.VocabularySize);
        model.to(device);
        OptimizerBundle bundle = WikiLanguageModelCommand.CreateOptimizerBundle(model, config);
        IOptimizer optimizer = bundle.RootOptimizer;
        Parameter[] parameters = model.Parameters().ToArray();
        optimizer.prepare();
        var samples = new List<StepSample>();
        long parameterCount = parameters.Sum(p => (long)p.T.numel());
        Console.WriteLine($"device={lane?.Device.Name ?? "CPU"}, parameters={parameterCount:N0}");
        try
        {
            for (int step = 0; step < warmup + steps; step++)
            {
                bool warming = step < warmup;
                lane?.ResetDetailedProfile();
                var before = CaptureLane(lane);
                var kernelsBefore = lane?.KernelTimings.ToDictionary(p => p.Key, p => p.Value) ?? [];
                long allocatedBefore = GC.GetTotalAllocatedBytes(false);
                int[] gcBefore = [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];
                var total = Stopwatch.StartNew();
                double Time(string phase, Action action) { if(lane?.DetailedProfiler is {} profiler) profiler.Phase=phase; long start = Stopwatch.GetTimestamp(); action(); lane?.Synchronize(); return Stopwatch.GetElapsedTime(start).TotalMilliseconds; }
                double zero = Time("zero-grad", optimizer.zero_grad);
                double forward = 0, backward = 0;
                float meanLoss = 0;
                foreach (var batch in batches)
                {
                    Tensor? loss = null;
                    forward += Time("forward", () => loss = model.forward_loss(batch.Input, batch.Target, config.BatchSize, config.ContextLength));
                    if (config.Layers >= 16) Console.WriteLine($"  step {step + 1}: forward completed, cumulative {forward:F0} ms");
                    meanLoss += loss!.item() / batches.Length;
                    backward += Time("backward", () => loss.BackwardAndRelease([1f / batches.Length]));
                    if (config.Layers >= 16) Console.WriteLine($"  step {step + 1}: backward completed, cumulative {backward:F0} ms");
                }
                float norm = 0;
                double clip = Time("clip", () => norm = nn.utils.clip_grad_norm_(parameters, 1f));
                double update = Time("optimizer", optimizer.step);
                total.Stop();
                var after = CaptureLane(lane);
                var profile = lane?.DetailedProfiler?.Snapshot();
                bool finite = float.IsFinite(meanLoss) && float.IsFinite(norm);
                if (Tensor.ArcResident) finite &= ArcTrainingMath.GradientsFinite(parameters);
                else foreach (Parameter p in parameters)
                    foreach (float value in p.T.Grad) finite &= float.IsFinite(value);
                if (!finite) throw new InvalidOperationException($"Non-finite loss/gradient at probe step {step + 1}.");
                using var process = Process.GetCurrentProcess();
                var sample = new StepSample(step + 1, meanLoss, norm, finite,
                    total.Elapsed.TotalMilliseconds, zero, forward, backward, clip, update,
                    GC.GetTotalAllocatedBytes(false) - allocatedBefore, GC.GetTotalMemory(false), process.WorkingSet64,
                    [GC.CollectionCount(0) - gcBefore[0], GC.CollectionCount(1) - gcBefore[1], GC.CollectionCount(2) - gcBefore[2]],
                    SubtractLane(after, before), after,
                    lane?.KernelTimings.ToDictionary(p => p.Key, p => p.Value - kernelsBefore.GetValueOrDefault(p.Key)) ?? [], profile);
                if (!warming) samples.Add(sample);
                Console.WriteLine($"{device} {(warming ? "warmup" : "measure")} {step + 1}/{warmup + steps}: step={sample.TotalMs:F2} ms, forward={forward:F2}, backward={backward:F2}, clip={clip:F2}, optimizer={update:F2}, loss={meanLoss:F6}, norm={norm:G6}, managed={sample.ManagedBytes / 1048576d:F1} MiB");
            }
            double sum = samples.Sum(s => s.TotalMs);
            var phases = new Dictionary<string, double>
            {
                ["zeroGrad"] = samples.Sum(s => s.ZeroGradMs), ["forward"] = samples.Sum(s => s.ForwardMs),
                ["backward"] = samples.Sum(s => s.BackwardMs), ["clip"] = samples.Sum(s => s.ClipMs),
                ["optimizer"] = samples.Sum(s => s.OptimizerMs),
            };
            var amdahl = phases.OrderByDescending(p => p.Value).Select(p => new PhaseShare(p.Key, p.Value / samples.Count,
                p.Value / sum, 1d / (1d - p.Value / sum), 1d / (1d - p.Value / sum + p.Value / sum / 2d))).ToArray();
            double median = Median(samples.Select(s => s.TotalMs));
            Console.WriteLine($"{device} p50={median:F2} ms/update, {1000d * config.BatchSize * config.ContextLength * config.GradientAccumulationSteps / median:F1} tokens/s; largest={amdahl[0].Phase} ({amdahl[0].Fraction:P1}), 2x that phase => {amdahl[0].TotalSpeedupIfPhase2X:F3}x total");
            if (arcOptions.DetailedProfiling)
            {
                Console.WriteLine("Largest device kernel groups (mean ms/update; GPU events, not additive with host waits):");
                foreach (var group in samples.SelectMany(s => s.Profile ?? []).Where(p => p.Kind == "gpu-kernel")
                    .GroupBy(p => p.Detail).OrderByDescending(g => g.Sum(p => p.Milliseconds)).Take(10))
                    Console.WriteLine($"  {group.Sum(p => p.Milliseconds) / samples.Count:F2} ms: {group.Key}");
            }
            return new RunResult(device.ToString(), lane?.Device.Name ?? "CPU", lane?.Device.DriverVersion,
                parameterCount, median, samples.Average(s => s.TotalMs),
                1000d * config.BatchSize * config.ContextLength * config.GradientAccumulationSteps / median, amdahl, samples);
        }
        finally
        {
            foreach (IOptimizer leaf in bundle.Optimizers)
                if (leaf is IDisposable disposable) disposable.Dispose();
        }
    }

    // Optional fields permit old/new lanes to use the same probe without
    // manufacturing zero measurements for telemetry that is not implemented.
    private static Dictionary<string, double?> CaptureLane(ArcExecutionLane? lane)
    {
        string[] names = ["AllocatedBytes", "AllocationCount", "KernelLaunchCount", "H2DBytes", "D2HBytes",
            "PeakAllocatedBytes", "KernelMilliseconds", "TransferMilliseconds", "AllocationMilliseconds",
            "CachedBytes", "RetiredBytes", "PoolHits", "RetiredReuseCount", "BufferReuseCount"];
        return names.ToDictionary(name => name, name => lane is not null
            && lane.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance)?.GetValue(lane) is object value
            ? (double?)Convert.ToDouble(value, CultureInfo.InvariantCulture) : null);
    }

    private static Dictionary<string, double?> SubtractLane(Dictionary<string, double?> after, Dictionary<string, double?> before)
        => after.ToDictionary(p => p.Key, p => p.Value - before[p.Key]);

    private static double Median(IEnumerable<double> values)
    {
        double[] ordered = values.Order().ToArray();
        return ordered.Length % 2 == 0 ? (ordered[ordered.Length / 2 - 1] + ordered[ordered.Length / 2]) * .5 : ordered[ordered.Length / 2];
    }

    private sealed record StepSample(int Step, float Loss, float GradientNorm, bool FiniteGradient,
        double TotalMs, double ZeroGradMs, double ForwardMs, double BackwardMs, double ClipMs, double OptimizerMs,
        long ManagedAllocatedBytes, long ManagedBytes, long WorkingSetBytes, int[] GcCollections,
        Dictionary<string, double?> LaneDelta, Dictionary<string, double?> LaneSnapshot,
        Dictionary<string, double> KernelGpuMs, IReadOnlyList<ArcProfileEntry>? Profile);
    private sealed record PhaseShare(string Phase, double MeanMs, double Fraction, double MaximumTotalSpeedup,
        double TotalSpeedupIfPhase2X);
    private sealed record RunResult(string Device, string DeviceName, string? DriverVersion, long ParameterCount,
        double StepP50Ms, double StepMeanMs, double TokensPerSecond, PhaseShare[] Amdahl, List<StepSample> Samples);
}
