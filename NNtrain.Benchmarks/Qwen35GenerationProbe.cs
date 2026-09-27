using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using NNtrain.Arc;
using NNtrain;

namespace NNtrain.Benchmarks;

/// <summary>Measures a real Qwen3.5 GGUF with fixed prompts, greedy output and reset state.</summary>
internal static class Qwen35GenerationProbe
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true,
        PropertyNameCaseInsensitive = true,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        Converters = { new JsonStringEnumConverter() }
    };

    internal static void Run(string[] args)
    {
        var flags = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i++)
        {
            string flag = args[i];
            if (flag is not ("--model" or "--output" or "--prompt" or "--tokens" or "--runs"
                or "--devices" or "--kernel" or "--fused-delta" or "--queue"
                or "--adapter" or "--options" or "--profile" or "--teacher" or "--logits" or "--logits-only"))
                throw new ArgumentException($"Unknown Qwen3.5 generation probe argument: {flag}");
            if (++i >= args.Length || !flags.TryAdd(flag, args[i]))
                throw new ArgumentException($"Missing value or repeated argument: {flag}");
        }

        string Required(string flag) => flags.TryGetValue(flag, out string? value)
            && !string.IsNullOrWhiteSpace(value) ? value
            : throw new ArgumentException($"Qwen3.5 generation probe requires {flag}.");
        int Number(string flag, int fallback, int maximum, int minimum = 1)
        {
            int value = flags.TryGetValue(flag, out string? text)
                ? int.Parse(text, CultureInfo.InvariantCulture) : fallback;
            if (value < minimum || value > maximum)
                throw new ArgumentOutOfRangeException(flag, $"Expected {minimum}..{maximum}.");
            return value;
        }
        bool Boolean(string flag, bool fallback) => !flags.TryGetValue(flag, out string? value) ? fallback
            : value switch { "on" => true, "off" => false, _ => throw new ArgumentException($"{flag} expects on or off.") };

        string modelPath = Path.GetFullPath(Required("--model"));
        string outputPath = Path.GetFullPath(Required("--output"));
        if (!File.Exists(modelPath)) throw new FileNotFoundException("GGUF model was not found.", modelPath);
        if (File.Exists(outputPath) || string.Equals(modelPath, outputPath, StringComparison.OrdinalIgnoreCase))
            throw new IOException("Probe output must be a new artifact; refusing to replace an existing file.");
        string? adapterPath = flags.TryGetValue("--adapter", out string? adapter) ? Path.GetFullPath(adapter) : null;
        if (adapterPath is not null && !File.Exists(adapterPath)) throw new FileNotFoundException("Adapter was not found.", adapterPath);
        int tokens = Number("--tokens", 64, 4096);
        int runs = Number("--runs", 3, 30);
        int logitsCount = Number("--logits", 0, 256, minimum: 0);
        bool logitsOnly = Boolean("--logits-only", false);
        int[]? teacherIds = flags.TryGetValue("--teacher", out string? teacherPath)
            ? JsonSerializer.Deserialize<int[]>(File.ReadAllText(teacherPath)) ?? throw new ArgumentException("Teacher IDs are null.") : null;
        if (logitsOnly && (logitsCount == 0 || teacherIds is null))
            throw new ArgumentException("--logits-only on requires --teacher PATH and --logits N > 0.");
        var defaults = flags.TryGetValue("--options", out string? optionsText)
            ? JsonSerializer.Deserialize<Qwen35ExecutionOptions>(optionsText.TrimStart().StartsWith('{') ? optionsText : File.ReadAllText(optionsText), JsonOptions)
                ?? throw new ArgumentException("Options JSON is null.")
            : new Qwen35ExecutionOptions();
        if (defaults.LoraTraining) throw new ArgumentException("Inference probes must not enable LoraTraining.");
        int queue = Number("--queue", defaults.QueuedKernelLimit, 4096, minimum: 16);
        string prompt = flags.GetValueOrDefault("--prompt", "<|im_start|>user\n国会議事堂への行き方を教えて<|im_end|>\n<|im_start|>assistant\n");
        int[] devices = flags.GetValueOrDefault("--devices", "0,1").Split(',')
            .Select(value => int.Parse(value, CultureInfo.InvariantCulture)).ToArray();
        if (devices.Length == 0 || devices.Any(index => index < 0) || devices.Distinct().Count() != devices.Length)
            throw new ArgumentException("--devices requires distinct nonnegative indices, such as 0,1.");
        Qwen35QuantizedKernel kernel = flags.GetValueOrDefault("--kernel", defaults.QuantizedKernel.ToString().ToLowerInvariant()) switch
        {
            "reference" => Qwen35QuantizedKernel.Reference,
            "cooperative" => Qwen35QuantizedKernel.Cooperative,
            "subgroup" => Qwen35QuantizedKernel.Subgroup,
            "auto" => Qwen35QuantizedKernel.Auto,
            _ => throw new ArgumentException("--kernel expects reference, cooperative, subgroup or auto.")
        };
        bool fusedDelta = flags.TryGetValue("--fused-delta", out string? fused) ? fused switch
        {
            "on" => true,
            "off" => false,
            _ => throw new ArgumentException("--fused-delta expects on or off.")
        } : defaults.FusedDelta;
        var options = defaults with
        {
            QuantizedKernel = kernel,
            FusedDelta = fusedDelta,
            QueuedKernelLimit = queue,
            DetailedProfiling = Boolean("--profile", defaults.DetailedProfiling)
        };

        DateTimeOffset startedUtc = DateTimeOffset.UtcNow;
        var modelFile = new FileInfo(modelPath);
        long modelBytes = modelFile.Length;
        DateTime modelLastWriteUtc = modelFile.LastWriteTimeUtc;
        long inspectionStart = Stopwatch.GetTimestamp();
        Qwen35GgufDescriptor descriptor = Qwen35Gguf.Inspect(modelPath);
        double inspectionMilliseconds = Stopwatch.GetElapsedTime(inspectionStart).TotalMilliseconds;
        long tokenizerStart = Stopwatch.GetTimestamp();
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(modelPath);
        int[] promptIds = tokenizer.Encode(prompt);
        double tokenizerMilliseconds = Stopwatch.GetElapsedTime(tokenizerStart).TotalMilliseconds;
        if (promptIds.Length == 0 || (long)promptIds.Length + tokens > descriptor.ContextLength)
            throw new ArgumentException("The prompt and exact requested output count must fit within the context window.");
        if (logitsCount > 0 && ((teacherIds?.Length ?? tokens) < logitsCount - 1
            || (long)promptIds.Length + logitsCount - 1 > descriptor.ContextLength))
            throw new ArgumentException("Not enough teacher IDs or context capacity for logits snapshots.");
        if (teacherIds?.Any(id => id < 0 || id >= descriptor.VocabularySize) == true)
            throw new ArgumentException("Teacher IDs contain an out-of-vocabulary token.");
        string logitsPath = outputPath + ".logits.f32";
        if (logitsCount > 0 && File.Exists(logitsPath)) throw new IOException("Logits output already exists.");
        ArcDeviceInfo[] available = ArcDevices.Enumerate().ToArray();
        if (devices.Any(index => index >= available.Length))
            throw new ArgumentException("A requested Arc device is unavailable.");
        var selectedDevices = devices.Select(index => available[index]).Select(device => new
        {
            device.Index, device.Name, device.DriverVersion,
            device.GlobalMemoryBytes, device.MaximumAllocationBytes, device.MinimumSubgroupSize
        }).ToArray();

        Console.WriteLine($"Qwen3.5 generation: prompt={promptIds.Length}, new tokens={tokens}, runs={runs}, "
            + $"devices={string.Join(',', devices)}, kernel={kernel}, fused delta={fusedDelta}, queue={queue}");
        var samples = new List<GenerationSample>(runs);
        double loadMilliseconds;
        double adapterLoadMilliseconds = 0;
        MemorySnapshot afterLoad;
        MemorySnapshot afterAdapter;
        var logitSnapshots = new List<object>();
        long loadStart = Stopwatch.GetTimestamp();
        using (Qwen35QuantizedModel model = Qwen35QuantizedModel.Load(modelPath, devices, Console.WriteLine, options))
        {
            loadMilliseconds = Stopwatch.GetElapsedTime(loadStart).TotalMilliseconds;
            afterLoad = Snapshot(model);
            if (adapterPath is not null)
            {
                long adapterStart = Stopwatch.GetTimestamp();
                model.LoadLora(adapterPath);
                adapterLoadMilliseconds = Stopwatch.GetElapsedTime(adapterStart).TotalMilliseconds;
                Console.WriteLine($"Loaded adapter: step={model.LoraStep}, load={adapterLoadMilliseconds:F2} ms");
            }
            afterAdapter = Snapshot(model);
            int[]? firstRunIds = null;
            for (int run = 0; run < (logitsOnly ? 0 : runs); run++)
            {
                MemorySnapshot before = Snapshot(model);
                IReadOnlyDictionary<string, double> kernelsBefore = model.KernelMilliseconds;
                var tokenTimes = new double[tokens];
                var callbackIds = new int[tokens];
                int callbackCount = 0;
                DateTimeOffset runStartedUtc = DateTimeOffset.UtcNow;
                long runStart = Stopwatch.GetTimestamp();
                int[] allIds = model.GenerateTokenIds(promptIds, tokens, eosTokenId: null, onToken: tokenId =>
                {
                    if (callbackCount >= tokens)
                        throw new InvalidOperationException("Generation emitted more tokens than requested.");
                    tokenTimes[callbackCount] = Stopwatch.GetElapsedTime(runStart).TotalMilliseconds;
                    callbackIds[callbackCount++] = tokenId;
                });
                double totalMilliseconds = Stopwatch.GetElapsedTime(runStart).TotalMilliseconds;
                MemorySnapshot after = Snapshot(model);
                IReadOnlyDictionary<string, double> kernelsAfter = model.KernelMilliseconds;
                int[] generatedIds = allIds.Skip(promptIds.Length).ToArray();
                if (callbackCount != tokens || !allIds.Take(promptIds.Length).SequenceEqual(promptIds)
                    || !generatedIds.SequenceEqual(callbackIds))
                    throw new InvalidOperationException("Generation did not produce the exact requested tokens and matching callbacks.");
                double firstTokenMilliseconds = tokenTimes[0];
                double? decodeMilliseconds = tokens > 1 ? tokenTimes[^1] - tokenTimes[0] : null;
                double? decodeTokensPerSecond = decodeMilliseconds is > 0
                    ? (tokens - 1) * 1000d / decodeMilliseconds.Value : null;
                var kernelDifferences = kernelsAfter.Keys.Union(kernelsBefore.Keys)
                    .Select(name => new KeyValuePair<string, double>(name,
                        kernelsAfter.GetValueOrDefault(name) - kernelsBefore.GetValueOrDefault(name)))
                    .Where(pair => pair.Value != 0)
                    .OrderByDescending(pair => pair.Value).ThenBy(pair => pair.Key, StringComparer.Ordinal)
                    .ToDictionary(pair => pair.Key, pair => pair.Value);
                var timestamps = Enumerable.Range(0, tokens).Select(index => new TokenSample(
                    index, generatedIds[index], tokenTimes[index],
                    index == 0 ? null : tokenTimes[index] - tokenTimes[index - 1])).ToArray();
                bool matchesFirst = firstRunIds is null || generatedIds.SequenceEqual(firstRunIds);
                firstRunIds ??= generatedIds;
                var sample = new GenerationSample(
                    run, run == 0 ? "first run after process-local model load" : "reused model with sequence state reset",
                    runStartedUtc, totalMilliseconds, firstTokenMilliseconds,
                    firstTokenMilliseconds > 0 ? promptIds.Length * 1000d / firstTokenMilliseconds : null,
                    decodeMilliseconds, decodeTokensPerSecond, tokens * 1000d / totalMilliseconds,
                    allIds, generatedIds, tokenizer.Decode(generatedIds), matchesFirst, timestamps,
                    Difference(after.UploadedBytes, before.UploadedBytes),
                    Difference(after.DownloadedBytes, before.DownloadedBytes),
                    before, after, kernelDifferences);
                samples.Add(sample);
                Console.WriteLine(FormattableString.Invariant(
                    $"Run {run + 1}/{runs}: first token={firstTokenMilliseconds:F2} ms, decode={decodeTokensPerSecond:F3} token/s, total={totalMilliseconds:F2} ms, matches first IDs={matchesFirst}"));
            }
            if (logitsCount > 0)
            {
                int[] continuation = teacherIds ?? firstRunIds!;
                model.Reset();
                for (int i = 0; i < promptIds.Length - 1; i++) model.ForwardToken(promptIds[i], returnLogits: false);
                using var logitsFile = new FileStream(logitsPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
                using var writer = new BinaryWriter(logitsFile);
                for (int step = 0; step < logitsCount; step++)
                {
                    int inputId = step == 0 ? promptIds[^1] : continuation[step - 1];
                    float[] logits = model.ForwardToken(inputId);
                    if (logits.Length != descriptor.VocabularySize || logits.Any(value => !float.IsFinite(value)))
                        throw new ArithmeticException("Invalid logits shape or non-finite value.");
                    foreach (float value in logits) writer.Write(value);
                    int[] top = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).ThenBy(i => i).Take(10).ToArray();
                    logitSnapshots.Add(new { Step = step, InputTokenId = inputId, VocabularySize = logits.Length,
                        ByteOffset = (long)step * logits.Length * sizeof(float), GreedyTokenId = top[0],
                        Rms = Math.Sqrt(logits.Select(value => (double)value * value).Average()),
                        Top10 = top.Select(i => new { TokenId = i, Logit = logits[i] }).ToArray() });
                }
            }
        }

        // Hash after measurement so model-file hashing does not warm the file
        // cache before the reported load. It is outside every reported timing.
        string modelSha256 = HashFile(modelPath);
        modelFile.Refresh();
        if (modelFile.Length != modelBytes || modelFile.LastWriteTimeUtc != modelLastWriteUtc)
            throw new IOException("The GGUF model file changed during the benchmark.");
        object report = new
        {
            SchemaVersion = 2,
            StartedUtc = startedUtc,
            CompletedUtc = DateTimeOffset.UtcNow,
            CommandArguments = args,
            Model = new { Path = modelPath, Bytes = modelBytes, LastWriteUtc = modelLastWriteUtc, Sha256 = modelSha256 },
            Adapter = adapterPath is null ? null : new { Path = adapterPath, Bytes = new FileInfo(adapterPath).Length, Sha256 = HashFile(adapterPath) },
            BinarySha256 = new[] { typeof(Qwen35QuantizedModel).Assembly, typeof(ArcExecutionLane).Assembly,
                typeof(Qwen35GenerationProbe).Assembly }.Distinct()
                .ToDictionary(assembly => assembly.GetName().Name!, assembly => HashFile(assembly.Location)),
            Runtime = new { Framework = RuntimeInformation.FrameworkDescription,
                OperatingSystem = RuntimeInformation.OSDescription, Architecture = RuntimeInformation.ProcessArchitecture.ToString() },
            Devices = selectedDevices,
            Options = options,
            Shape = new { descriptor.LayerCount, descriptor.EmbeddingLength, descriptor.FeedForwardLength,
                descriptor.ContextLength, descriptor.VocabularySize, descriptor.HeadCount, descriptor.KvHeadCount,
                descriptor.HeadWidth, descriptor.LinearKeyHeads, descriptor.LinearValueHeads,
                descriptor.LinearHeadWidth, descriptor.ConvKernel, descriptor.FullAttentionInterval },
            Prompt = prompt,
            PromptTokenIds = promptIds,
            RequestedNewTokens = tokens,
            MeasuredRuns = samples.Count,
            Sampling = new { Method = "GPU greedy argmax", Temperature = 0, EosTokenId = (int?)null,
                Policy = "Emit exactly the requested count, including tokens after EOS; no early EOS stop." },
            InspectionMilliseconds = inspectionMilliseconds,
            TokenizerLoadAndEncodeMilliseconds = tokenizerMilliseconds,
            LoadMilliseconds = loadMilliseconds,
            AdapterLoadMilliseconds = adapterLoadMilliseconds,
            AfterLoad = afterLoad,
            AfterAdapter = afterAdapter,
            RunsHaveIdenticalGeneratedTokenIds = samples.All(sample => sample.MatchesFirstRunTokenIds),
            Samples = samples,
            Logits = logitsCount == 0 ? null : new { Path = logitsPath, Sha256 = HashFile(logitsPath),
                Format = "little-endian float32, row-major [snapshot, vocabulary]", TeacherSource = teacherPath ?? "first generated run",
                Snapshots = logitSnapshots },
            Notes = "The model is loaded once per process. The first run has no generation warmup; later runs reuse "
                + "weights and allocation capacity, and GenerateTokenIds resets sequence state. OS file caches and GPU "
                + "driver caches are not flushed. PrefillThroughFirstTokenMilliseconds and PrefillTokensPerSecond "
                + "include state reset, the full prompt, first head projection and argmax. Decode timing spans the first "
                + "to last output callback and excludes the first generated token. Total includes generation cleanup; "
                + "tokenization, loading, snapshots, console output, model hashing and JSON export are outside timed runs. "
                + "Callback bookkeeping is included. Kernel durations are sums of completed per-device events and "
                + "must not be added to wall-clock durations. Memory counters are backend allocation bytes; peaks "
                + "are cumulative across the model lifetime and include cached or retired buffers. They are not "
                + "driver-reported VRAM measurements. Upload/download differences include cross-context transport. "
                + "A model SHA256 is computed after all runs. Generated IDs are retained because different kernel "
                + "reduction orders can change greedy continuations. No equality across kernel modes is assumed. "
                + "Adapter loading and optional fixed-continuation complete-logit downloads are outside all timed generation runs."
        };
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        using (var output = new FileStream(outputPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
            JsonSerializer.Serialize(output, report, JsonOptions);
        Console.WriteLine($"Qwen3.5 generation probe saved: {outputPath}");
    }

    private static MemorySnapshot Snapshot(Qwen35QuantizedModel model) => new(
        model.ResidentWeightBytes.ToArray(), model.ResidentAuxiliaryWeightBytes.ToArray(),
        model.ResidentStateBytes.ToArray(), model.LiveDeviceBytes.ToArray(), model.PeakDeviceBytes.ToArray(),
        model.UploadedBytes.ToArray(), model.DownloadedBytes.ToArray(), SnapshotLanes(model));

    private static LaneSnapshot[] SnapshotLanes(Qwen35QuantizedModel model)
    {
        var field = typeof(Qwen35QuantizedModel).GetField("_lanes", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException("The benchmark expects Qwen35QuantizedModel._lanes for diagnostic snapshots.");
        return ((IEnumerable<ArcExecutionLane>)field.GetValue(model)!).Select(lane => new LaneSnapshot(
            lane.DeviceIndex, lane.AllocationCount, lane.RequestedBytes, lane.NativeAllocatedBytes,
            lane.NativeReleasedBytes, lane.NativeReleaseCount, lane.AllocatedBytes, lane.KernelLaunchCount,
            lane.CachedBytes, lane.RetiredBytes, lane.PoolHits, lane.RetiredReuseCount,
            lane.H2DBytes, lane.D2HBytes, lane.PeakAllocatedBytes, lane.KernelMilliseconds,
            lane.TransferMilliseconds, lane.AllocationMilliseconds,
            lane.DetailedProfiler?.Snapshot().ToArray() ?? [])).ToArray();
    }

    private static long[] Difference(long[] after, long[] before)
        => after.Select((value, index) => value - before[index]).ToArray();

    private static string HashFile(string path)
    {
        using FileStream stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
    }

    private sealed record MemorySnapshot(long[] ResidentWeightBytes, long[] ResidentAuxiliaryWeightBytes,
        long[] ResidentStateBytes, long[] LiveDeviceBytes, long[] PeakDeviceBytes, long[] UploadedBytes, long[] DownloadedBytes,
        LaneSnapshot[] Lanes);

    private sealed record LaneSnapshot(int DeviceIndex, long AllocationCount, long RequestedBytes,
        long NativeAllocatedBytes, long NativeReleasedBytes, long NativeReleaseCount, long AllocatedBytes,
        long KernelLaunchCount, long CachedBytes, long RetiredBytes, long PoolHits, long RetiredReuseCount,
        long H2DBytes, long D2HBytes, long PeakAllocatedBytes, double KernelMilliseconds,
        double TransferMilliseconds, double AllocationMilliseconds, ArcProfileEntry[] ProfileEntries);

    private sealed record TokenSample(int GeneratedIndex, int TokenId, double ElapsedMilliseconds,
        double? SincePreviousTokenMilliseconds);

    private sealed record GenerationSample(int RunIndex, string Kind, DateTimeOffset StartedUtc,
        double TotalMilliseconds, double PrefillThroughFirstTokenMilliseconds, double? PrefillTokensPerSecond,
        double? DecodeMilliseconds, double? DecodeTokensPerSecond, double OverallGeneratedTokensPerSecond,
        int[] AllTokenIds, int[] GeneratedTokenIds, string GeneratedText, bool MatchesFirstRunTokenIds,
        TokenSample[] Tokens, long[] UploadedBytes, long[] DownloadedBytes,
        MemorySnapshot Before, MemorySnapshot After, IReadOnlyDictionary<string, double> KernelMilliseconds);
}
