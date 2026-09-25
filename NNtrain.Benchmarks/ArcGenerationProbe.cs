using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;

namespace NNtrain.Benchmarks;

/// <summary>
/// Fixed-seed, bounded Transformer generation with separate prefill and decode
/// timing. Synthetic weights are explicit; no tokenizer or training data is read.
/// </summary>
internal static class ArcGenerationProbe
{
    internal static void Run(string[] args)
    {
        if (args.Length < 2)
            throw new ArgumentException("Expected training config and NEW result.json, followed by "
                + "--tokens N --prompt N --warmup N --runs N --mode single|tensorParallel|auto "
                + "--kv-cache on|off --gemv on|off --fused-gemv on|off --packed-embedding on|off --small-row-norm on|off "
                + "--top-k N --temperature F --profile --layers N --safetensors path.");
        string configPath = Path.GetFullPath(args[0]);
        string resultPath = Path.GetFullPath(args[1]);
        if (File.Exists(resultPath)
            || string.Equals(configPath, resultPath, StringComparison.OrdinalIgnoreCase))
            throw new IOException("Probe output must be a new artifact; refusing to replace an existing file.");
        var flags = new Dictionary<string, string>(StringComparer.Ordinal);
        bool detailed = false;
        for (int i = 2; i < args.Length; ++i)
        {
            string flag = args[i];
            if (flag == "--profile")
            {
                if (detailed) throw new ArgumentException("Repeated --profile.");
                detailed = true;
                continue;
            }
            if (flag is not ("--tokens" or "--prompt" or "--warmup" or "--runs"
                or "--mode" or "--kv-cache" or "--gemv" or "--fused-gemv" or "--packed-embedding" or "--small-row-norm"
                or "--top-k" or "--temperature" or "--layers" or "--safetensors"))
                throw new ArgumentException($"Unknown generation probe argument: {flag}");
            if (++i >= args.Length || !flags.TryAdd(flag, args[i]))
                throw new ArgumentException($"Missing value or repeated argument: {flag}");
        }
        int Number(string name, int fallback, int minimum = 1)
        {
            int value = flags.TryGetValue(name, out string? text)
                ? int.Parse(text, CultureInfo.InvariantCulture) : fallback;
            if (value < minimum) throw new ArgumentOutOfRangeException(name);
            return value;
        }
        bool Switch(string name, bool fallback) => flags.TryGetValue(name, out string? text)
            ? text switch { "on" => true, "off" => false,
                _ => throw new ArgumentException($"{name} expects on or off.") }
            : fallback;

        WikiTrainingConfiguration original = WikiTrainingConfiguration.Load(configPath);
        if (!string.Equals(original.ModelArchitecture, "transformer", StringComparison.OrdinalIgnoreCase))
            throw new NotSupportedException("The Arc generation probe requires a Transformer configuration.");
        int tokenCount = Number("--tokens", 200);
        int promptLength = Number("--prompt", 8);
        int warmup = Number("--warmup", 1, 0);
        int runs = Number("--runs", 3);
        int layers = Number("--layers", original.Layers);
        int topK = Number("--top-k", 1, 0);
        float temperature = flags.TryGetValue("--temperature", out string? temperatureText)
            ? float.Parse(temperatureText, CultureInfo.InvariantCulture) : 0f;
        if (!float.IsFinite(temperature) || temperature < 0)
            throw new ArgumentOutOfRangeException("--temperature", "Temperature must be finite and non-negative.");
        if (tokenCount > 4096 || promptLength > original.ContextLength || warmup > 10 || runs > 30
            || layers > original.Layers)
            throw new ArgumentException("Probe bounds: at most 4096 generated tokens, context-length prompt, "
                + "10 warmup runs, 30 measured runs and the configured number of layers.");
        string mode = flags.GetValueOrDefault("--mode", "single");
        ArcInferenceRouting.ValidateOptions(null, mode);
        WikiTrainingConfiguration config = original with { Device = "arc", Layers = layers,
            ArcInferenceMode = mode };
        ArcInferenceSettings settings = ArcInferenceRouting.Resolve(config);
        ArcExecutionOptions options = new() { DetailedProfiling = detailed };
        options = options with { InferenceKvCache = Switch("--kv-cache", options.InferenceKvCache),
            InferenceGemv = Switch("--gemv", options.InferenceGemv),
            InferenceFusedGemv = Switch("--fused-gemv", options.InferenceFusedGemv),
            InferencePackedEmbedding = Switch("--packed-embedding", options.InferencePackedEmbedding),
            InferenceSmallRowNorm = Switch("--small-row-norm", options.InferenceSmallRowNorm) };
        string? checkpointPath = flags.TryGetValue("--safetensors", out string? checkpoint)
            ? Path.GetFullPath(checkpoint) : null;
        if (checkpointPath is not null && !File.Exists(checkpointPath))
            throw new FileNotFoundException("The requested SafeTensors checkpoint was not found.", checkpointPath);
        if (checkpointPath is not null && layers != original.Layers)
            throw new ArgumentException("--safetensors cannot be combined with a different layer count.");

        int[] prompt = Enumerable.Range(0, promptLength)
            .Select(index => (int)((17L + 37L * index) % config.VocabularySize)).ToArray();
        bool previousSimd = Tensor.SimdEnabled;
        int previousWorkers = Tensor.MaxDegreeOfParallelism;
        object measurement;
        try
        {
            Tensor.SimdEnabled = config.UseSimd;
            Tensor.MaxDegreeOfParallelism = config.MaxDegreeOfParallelism;
            measurement = Measure(config, settings, options, prompt, tokenCount, warmup, runs, checkpointPath,
                temperature, topK);
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
            ConfigurationSha256 = HashFile(configPath),
            BinarySha256 = new[] { typeof(Tensor).Assembly, typeof(ArcExecutionLane).Assembly,
                typeof(ArcGenerationProbe).Assembly, typeof(WikiTrainingConfiguration).Assembly,
                typeof(ExecutionSession).Assembly }
                .Distinct().ToDictionary(assembly => assembly.GetName().Name!, assembly => HashFile(assembly.Location)),
            ModelSource = checkpointPath is null ? "fixed-seed synthetic weights" : "SafeTensors checkpoint",
            CheckpointPath = checkpointPath,
            CheckpointSha256 = checkpointPath is null ? null : HashFile(checkpointPath),
            Seed = config.Seed,
            Shape = new { BatchSize = 1, config.ContextLength, config.ModelWidth, config.Heads,
                config.HiddenSize, config.Layers, config.VocabularySize, config.TieWordEmbeddings },
            Precision = new { Mode = TensorPrecisionModeNames.Format(config.GetPrecisionMode()), config.Bfp8BlockSize },
            RequestedMode = mode,
            PromptTokenIds = prompt,
            MaxNewTokens = tokenCount,
            Sampling = new { Temperature = temperature, TopK = topK, StopTokenId = (int?)null },
            WarmupRuns = warmup,
            MeasuredRuns = runs,
            ArcOptions = options,
            Overrides = flags,
            Notes = "Fresh fixed-seed synthetic Transformer unless --safetensors is provided. "
                + "Prompt IDs are synthetic and fixed; dropout is disabled by Eval. Model creation, upload, route "
                + "selection and JSON export are outside measured runs. All runs use the specified sampling "
                + "with the same RNG seed and exactly "
                + "MaxNewTokens; no EOS stop. KV caches belong to one generation call and are not shared between runs. "
                + "Token timestamps are measured at the streaming callback; first-token and decode metrics are null "
                + "if the model only implements the base post-generation callback. Total includes generation cleanup "
                + "and final lane synchronization. Backend allocation peaks are session-cumulative native buffer "
                + "allocation peaks, not driver-reported dedicated VRAM. Kernel event durations overlap host wall "
                + "timings and must not be added to them. Synthetic weights establish throughput, not text quality.",
            Result = measurement,
        };
        Directory.CreateDirectory(Path.GetDirectoryName(resultPath)!);
        using (var output = new FileStream(resultPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
            JsonSerializer.Serialize(output, report, new JsonSerializerOptions { WriteIndented = true });
        Console.WriteLine($"Arc generation probe saved: {resultPath}");
    }

    private static object Measure(WikiTrainingConfiguration config, ArcInferenceSettings settings,
        ArcExecutionOptions options, int[] prompt, int tokenCount, int warmup, int runs, string? checkpointPath,
        float temperature, int topK)
    {
        using IDisposable execution = settings.UsesTwoDevices
            ? Tensor.BeginArcInferenceExecution(settings.DeviceIndices, config.GetPrecisionMode(), options)
            : Tensor.BeginArcExecution(settings.DeviceIndices[0], config.GetPrecisionMode(), options);
        ExecutionSession session = ExecutionSession.Current!;
        ArcExecutionLane[] lanes = session.Lanes.OfType<ArcExecutionLane>().OrderBy(lane => lane.DeviceIndex).ToArray();
        var model = (GptRinWikiJp)WikiLanguageModelCommand.CreateModel(config, config.VocabularySize);
        if (checkpointPath is not null)
            model.load_state_dict(safetensors.torch.load_file(checkpointPath));
        model.to(new TorchDevice(TensorDevice.Arc, settings.DeviceIndices[0]));
        model.Eval();
        long parameterCount = model.Parameters().Sum(parameter => (long)parameter.T.numel());
        object? selection = null;
        if (settings.UsesTwoDevices)
        {
            if (!model.CanUseArcTensorParallel(out string reason))
            {
                if (settings.Mode != "auto") throw new NotSupportedException(reason);
                selection = new { SelectedMode = "single", Reason = reason };
            }
            else if (settings.Mode == "auto")
            {
                using var routeOutput = new StringWriter(CultureInfo.InvariantCulture);
                ArcInferenceRouting.ConfigureModelTokens(model, prompt, tokenCount, settings, routeOutput);
                string selectionLog = routeOutput.ToString();
                Console.Write(selectionLog);
                selection = new { Method = "shared CLI ArcInferenceRouting.ConfigureModelTokens", Log = selectionLog };
            }
            else model.ArcTensorParallelEnabled = true;
        }
        string selectedMode = model.ArcTensorParallelEnabled ? "tensorParallel" : "single";
        bool streamingCallback = typeof(GptRinWikiJp).GetMethod("GenerateTokenIds",
            BindingFlags.Instance | BindingFlags.NonPublic, null,
            [typeof(IEnumerable<int>), typeof(int), typeof(float), typeof(int), typeof(int?),
                typeof(Random), typeof(Action<int>)], null)?.DeclaringType == typeof(GptRinWikiJp);
        Console.WriteLine($"Arc generation: {selectedMode}, KV cache={options.InferenceKvCache}, "
            + $"layers={config.Layers}, prompt={prompt.Length}, new tokens={tokenCount}, "
            + $"parameters={parameterCount:N0}, {(checkpointPath is null ? "synthetic weights" : "checkpoint weights")}");
        Console.WriteLine(string.Join(", ", lanes.Select(lane => $"Arc[{lane.DeviceIndex}]={lane.Device.Name}")));

        var warmupSamples = new List<GenerationSample>();
        var samples = new List<GenerationSample>();
        for (int run = 0; run < warmup + runs; ++run)
        {
            foreach (ArcExecutionLane lane in lanes)
            {
                lane.Synchronize();
                lane.ResetDetailedProfile();
                if (lane.DetailedProfiler is { } profile) profile.Phase = "generation";
            }
            var before = lanes.ToDictionary(lane => lane.DeviceIndex, CaptureLane);
            var kernelBefore = lanes.ToDictionary(lane => lane.DeviceIndex,
                lane => new Dictionary<string, double>(lane.KernelTimings));
            long managedBefore = GC.GetTotalAllocatedBytes(false);
            int[] gcBefore = Enumerable.Range(0, GC.MaxGeneration + 1).Select(GC.CollectionCount).ToArray();
            var callbackTimes = new List<double>(tokenCount);
            var callbackTokens = new List<int>(tokenCount);
            long started = Stopwatch.GetTimestamp();
            int[] generated = model.generate_token_ids(prompt, tokenCount, temperature, topK, null,
                new Random(config.Seed ^ 0x27D4EB2D), token => {
                    callbackTimes.Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
                    callbackTokens.Add(token);
                });
            foreach (ArcExecutionLane lane in lanes) lane.Synchronize();
            double total = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            if (generated.Length != prompt.Length + tokenCount
                || callbackTokens.Count != tokenCount
                || !callbackTokens.SequenceEqual(generated.Skip(prompt.Length)))
                throw new InvalidOperationException("Generation result or streaming callback token count is inconsistent.");
            double? firstToken = streamingCallback ? callbackTimes[0] : null;
            double? decode = streamingCallback && tokenCount > 1
                ? callbackTimes[^1] - callbackTimes[0] : null;
            var after = lanes.ToDictionary(lane => lane.DeviceIndex, CaptureLane);
            var sample = new GenerationSample(run < warmup ? run + 1 : run - warmup + 1,
                total, tokenCount * 1000d / total, firstToken, decode,
                decode is > 0 ? (tokenCount - 1) * 1000d / decode : null,
                callbackTimes.ToArray(), generated, GC.GetTotalAllocatedBytes(false) - managedBefore,
                GC.GetTotalMemory(false), Environment.WorkingSet,
                Enumerable.Range(0, gcBefore.Length).Select(i => GC.CollectionCount(i) - gcBefore[i]).ToArray(),
                lanes.ToDictionary(lane => lane.DeviceIndex,
                    lane => SubtractLane(after[lane.DeviceIndex], before[lane.DeviceIndex])), after,
                lanes.ToDictionary(lane => lane.DeviceIndex,
                    lane => lane.KernelTimings.ToDictionary(pair => pair.Key,
                        pair => pair.Value - kernelBefore[lane.DeviceIndex].GetValueOrDefault(pair.Key))),
                lanes.ToDictionary(lane => lane.DeviceIndex, lane => lane.DetailedProfiler?.Snapshot()));
            if (run < warmup) warmupSamples.Add(sample); else samples.Add(sample);
            Console.WriteLine($"{(run < warmup ? "warmup" : "run")} {sample.Run}: "
                + $"{total:F2} ms, {sample.TokensPerSecond:F2} tokens/s"
                + (firstToken.HasValue ? $", first={firstToken:F2} ms, decode={sample.DecodeTokensPerSecond:F2} tokens/s" : ""));
        }
        double median = Median(samples.Select(sample => sample.TotalMs));
        bool deterministic = samples.All(sample => sample.GeneratedTokenIds.SequenceEqual(samples[0].GeneratedTokenIds));
        Console.WriteLine($"Arc generation p50={median:F2} ms, {tokenCount * 1000d / median:F2} tokens/s, "
            + $"deterministic={deterministic}");
        if (options.DetailedProfiling)
        {
            Console.WriteLine("Largest kernel groups (mean GPU event ms/run per device):");
            foreach (var group in samples.SelectMany(sample => sample.KernelGpuMs
                .SelectMany(device => device.Value.Select(kernel => new { Device = device.Key, kernel.Key, kernel.Value })))
                .GroupBy(kernel => (kernel.Device, kernel.Key))
                .OrderByDescending(group => group.Sum(kernel => kernel.Value)).Take(12))
                Console.WriteLine($"  Arc[{group.Key.Device}] {group.Sum(kernel => kernel.Value) / samples.Count:F2} ms: {group.Key.Key}");
        }
        return new
        {
            SelectedMode = selectedMode,
            RouteSelection = selection,
            StreamingCallback = streamingCallback,
            Devices = lanes.Select(lane => new { lane.DeviceIndex, lane.Device.Name, lane.Device.DriverVersion }).ToArray(),
            ParameterCount = parameterCount,
            TotalP50Ms = median,
            TotalMeanMs = samples.Average(sample => sample.TotalMs),
            TokensPerSecond = tokenCount * 1000d / median,
            FirstTokenP50Ms = streamingCallback ? Median(samples.Select(sample => sample.FirstTokenMs!.Value)) : (double?)null,
            DecodeTokensPerSecondP50 = streamingCallback && tokenCount > 1
                ? Median(samples.Select(sample => sample.DecodeTokensPerSecond!.Value)) : (double?)null,
            DeterministicAcrossMeasuredRuns = deterministic,
            WarmupSamples = warmupSamples,
            Samples = samples,
        };
    }

    private static Dictionary<string, double?> CaptureLane(ArcExecutionLane lane)
    {
        string[] names = ["AllocatedBytes", "AllocationCount", "KernelLaunchCount", "H2DBytes", "D2HBytes",
            "PeakAllocatedBytes", "RetainedMatrixPanelBytes", "KernelMilliseconds", "TransferMilliseconds",
            "AllocationMilliseconds", "CachedBytes", "RetiredBytes", "PoolHits", "RetiredReuseCount",
            "BufferReuseCount", "RequestedBytes", "NativeAllocatedBytes", "NativeReleasedBytes", "NativeReleaseCount"];
        return names.ToDictionary(name => name,
            name => lane.GetType().GetProperty(name, BindingFlags.Public | BindingFlags.Instance)?.GetValue(lane) is object value
                ? (double?)Convert.ToDouble(value, CultureInfo.InvariantCulture) : null);
    }

    private static Dictionary<string, double?> SubtractLane(Dictionary<string, double?> after,
        Dictionary<string, double?> before) => after.ToDictionary(pair => pair.Key, pair => pair.Value - before[pair.Key]);

    private static double Median(IEnumerable<double> values)
    {
        double[] ordered = values.Order().ToArray();
        return ordered.Length % 2 == 0
            ? (ordered[ordered.Length / 2 - 1] + ordered[ordered.Length / 2]) * .5
            : ordered[ordered.Length / 2];
    }

    private static string HashFile(string path)
    {
        using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(input));
    }

    private sealed record GenerationSample(int Run, double TotalMs, double TokensPerSecond,
        double? FirstTokenMs, double? DecodeMs, double? DecodeTokensPerSecond, double[] CallbackElapsedMs,
        int[] GeneratedTokenIds, long ManagedAllocatedBytes, long ManagedBytes, long WorkingSetBytes, int[] GcCollections,
        Dictionary<int, Dictionary<string, double?>> LaneDelta,
        Dictionary<int, Dictionary<string, double?>> LaneSnapshot,
        Dictionary<int, Dictionary<string, double>> KernelGpuMs,
        Dictionary<int, IReadOnlyList<ArcProfileEntry>?> Profile);
}
