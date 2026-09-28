using System.Diagnostics;
using System.Globalization;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

/// <summary>Real full-target LoRA updates with exact input and runtime evidence.</summary>
internal static class Qwen35LoraTrainingProbe
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
            if (flag is not ("--model" or "--data" or "--examples" or "--output" or "--options" or "--prompt-mode"
                or "--checkpoint" or "--resume" or "--devices" or "--rank" or "--alpha" or "--seed"))
                throw new ArgumentException($"Unknown Qwen3.5 training probe argument: {flag}");
            if (++i >= args.Length || !flags.TryAdd(flag, args[i]) || string.IsNullOrWhiteSpace(args[i]))
                throw new ArgumentException($"Missing value or repeated argument: {flag}");
        }
        string Required(string flag) => flags.TryGetValue(flag, out string? value) ? value
            : throw new ArgumentException($"Qwen3.5 training probe requires {flag}.");
        string? OptionalPath(string flag) => flags.TryGetValue(flag, out string? value) ? Path.GetFullPath(value) : null;
        int Number(string flag, int fallback) => flags.TryGetValue(flag, out string? value)
            ? int.Parse(value, CultureInfo.InvariantCulture) : fallback;
        string modelPath = Path.GetFullPath(Required("--model"));
        string dataPath = Path.GetFullPath(Required("--data"));
        string outputPath = Path.GetFullPath(Required("--output"));
        string? checkpointPath = OptionalPath("--checkpoint"), resumePath = OptionalPath("--resume");
        string[] inputs = new[] { modelPath, dataPath, resumePath }.OfType<string>().ToArray();
        foreach (string input in inputs)
            if (!File.Exists(input)) throw new FileNotFoundException("Benchmark input was not found.", input);
        string[] outputs = new[] { outputPath, checkpointPath }.OfType<string>().ToArray();
        if (outputs.Distinct(StringComparer.OrdinalIgnoreCase).Count() != outputs.Length
            || outputs.Any(path => inputs.Contains(path, StringComparer.OrdinalIgnoreCase) || File.Exists(path) || Directory.Exists(path)))
            throw new IOException("JSON and checkpoint outputs must be distinct new files, separate from every input.");
        if (resumePath is not null && new[] { "--rank", "--alpha", "--seed" }.Any(flags.ContainsKey))
            throw new ArgumentException("--resume preserves checkpoint LoRA options; omit --rank, --alpha and --seed.");
        int[] selected = flags.GetValueOrDefault("--examples", "1,2,3,1").Split(',')
            .Select(value => int.Parse(value, CultureInfo.InvariantCulture)).ToArray();
        if (selected.Length == 0 || selected.Any(index => index < 1))
            throw new ArgumentException("--examples requires one-based nonempty JSONL record numbers.");
        int[] devices = flags.GetValueOrDefault("--devices", "0,1").Split(',')
            .Select(value => int.Parse(value, CultureInfo.InvariantCulture)).ToArray();
        if (devices.Length == 0 || devices.Any(index => index < 0) || devices.Distinct().Count() != devices.Length)
            throw new ArgumentException("--devices requires distinct nonnegative device indices.");
        string promptMode = flags.GetValueOrDefault("--prompt-mode", "wrapped");
        if (promptMode is not ("wrapped" or "direct"))
            throw new ArgumentException("--prompt-mode must be wrapped or direct.");
        Qwen35ExecutionOptions options = (flags.TryGetValue("--options", out string? optionsText)
            ? JsonSerializer.Deserialize<Qwen35ExecutionOptions>(optionsText.TrimStart().StartsWith('{')
                ? optionsText : File.ReadAllText(optionsText), JsonOptions)
                ?? throw new ArgumentException("Options JSON is null.")
            : new Qwen35ExecutionOptions { CollectKernelTimings = true }) with { LoraTraining = true };
        var loraOptions = new Qwen35LoraOptions
        {
            Rank = Number("--rank", 8), Seed = Number("--seed", 1),
            Alpha = flags.TryGetValue("--alpha", out string? alpha) ? float.Parse(alpha, CultureInfo.InvariantCulture) : 16,
            LearningRate = .0001f, GradientClip = 1
        };
        loraOptions.Validate();

        // Hold input handles so a checkpoint/report can never replace them while
        // this benchmark is running, and Windows writers cannot mutate evidence.
        using FileStream modelInput = File.OpenRead(modelPath);
        using FileStream dataInput = File.OpenRead(dataPath);
        using FileStream? resumeInput = resumePath is null ? null : File.OpenRead(resumePath);
        string dataSha = HashStream(dataInput);
        string? resumeSha = resumeInput is null ? null : HashStream(resumeInput);
        var binaries = new[] { typeof(Qwen35LoraTrainingProbe).Assembly, typeof(Qwen35QuantizedModel).Assembly,
            typeof(ArcExecutionLane).Assembly, typeof(NNtrain.Runtime.Execution.ExecutionSession).Assembly }
            .Distinct().ToDictionary(assembly => assembly.GetName().Name!, assembly => HashFile(assembly.Location));
        string sourcePath = SourcePath();
        var source = new { Path = sourcePath, Sha256 = File.Exists(sourcePath) ? HashFile(sourcePath) : null,
            Note = "Source file observed at runtime; assembly SHA-256 identifies the actual compiled implementation." };
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(modelPath);
        if (tokenizer.EosTokenId is not int eos) throw new InvalidDataException("Tokenizer requires an EOS token for training examples.");
        var examples = new List<Example>();
        int physicalLine = 0;
        foreach (string line in File.ReadLines(dataPath))
        {
            physicalLine++;
            if (string.IsNullOrWhiteSpace(line)) continue;
            using JsonDocument document = JsonDocument.Parse(line);
            string prompt = document.RootElement.GetProperty("prompt").GetString()
                ?? throw new InvalidDataException($"Null prompt on line {physicalLine}.");
            string response = document.RootElement.GetProperty("response").GetString()
                ?? throw new InvalidDataException($"Null response on line {physicalLine}.");
            int[] prefix = tokenizer.Encode(promptMode == "direct" ? prompt
                : "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n");
            int[] tokens = [.. prefix, .. tokenizer.Encode(response), eos];
            examples.Add(new(examples.Count + 1, physicalLine, HashText(line), tokens, prefix.Length));
        }
        if (selected.Any(index => index > examples.Count)) throw new ArgumentException("Selected example exceeds the dataset record count.");
        var descriptor = Qwen35Gguf.Inspect(modelPath);
        foreach (Example example in selected.Select(index => examples[index - 1]))
            if (example.TokenIds.Length < 2 || example.TokenIds.Length - 1 > descriptor.ContextLength
                || example.ResponseStart < 1 || example.ResponseStart >= example.TokenIds.Length)
                throw new ArgumentException($"Example {example.Record} does not fit the model context; no truncation is performed.");

        Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
        using var output = new FileStream(outputPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read);
        var samples = new List<Dictionary<string, object?>>();
        var report = new Dictionary<string, object?>
        {
            ["SchemaVersion"] = 1, ["Status"] = "loading", ["StartedUtc"] = DateTimeOffset.UtcNow,
            ["CommandArguments"] = args, ["PromptMode"] = promptMode,
            ["Model"] = new { Path = modelPath, Bytes = modelInput.Length },
            ["Data"] = new { Path = dataPath, Sha256 = dataSha, RecordCount = examples.Count },
            ["Resume"] = resumePath is null ? null : new { Path = resumePath, Sha256 = resumeSha },
            ["Source"] = source, ["BinarySha256"] = binaries, ["Options"] = options,
            ["Runtime"] = new { Framework = RuntimeInformation.FrameworkDescription, OperatingSystem = RuntimeInformation.OSDescription,
                Architecture = RuntimeInformation.ProcessArchitecture.ToString() },
            ["SelectedExamples"] = selected, ["Samples"] = samples,
            ["Notes"] = "Each example is one complete optimizer update; repeated rows continue training, not reset-state repetitions. "
                + "Only TrainLora and completion synchronization are timed per step. Tokenization, diagnostic snapshots, JSON, console output, "
                + "hashes and checkpoint saving are excluded. No data truncation or shortening is performed. "
                + "Kernel sums across devices are not wall-clock durations. Device memory counters include cached/retired allocations, "
                + "and are backend accounting rather than driver-reported VRAM. Peaks are cumulative. "
                + "Kernel timing collection defaults on for this diagnostic command; supplied options can disable it, leaving kernel deltas empty. "
                + "First checkpoint save can include lazy base-model fingerprinting. OS/driver caches are not flushed."
        };
        void WriteReport()
        {
            output.Position = 0; output.SetLength(0);
            JsonSerializer.Serialize(output, report, JsonOptions);
            output.Flush();
        }
        WriteReport();
        try
        {
            long start = Stopwatch.GetTimestamp();
            using var model = Qwen35QuantizedModel.Load(modelPath, devices, Console.WriteLine, options);
            report["LoadSeconds"] = Stopwatch.GetElapsedTime(start).TotalSeconds;
            start = Stopwatch.GetTimestamp();
            if (resumePath is null) model.AttachLora(loraOptions);
            else model.LoadLora(resumePath);
            report["AdapterSetupSeconds"] = Stopwatch.GetElapsedTime(start).TotalSeconds;
            Qwen35LoraOptions effectiveLora = Field<Qwen35LoraOptions>(model, "_loraOptions");
            ArcExecutionLane[] lanes = Field<IEnumerable<ArcExecutionLane>>(model, "_lanes").ToArray();
            report["LoraOptions"] = effectiveLora;
            report["InitialStep"] = model.LoraStep;
            report["LoraTargets"] = model.LoraTargets;
            report["Lanes"] = lanes.Select(lane => new { lane.Device, lane.Options }).ToArray();
            report["KernelTimingsEnabled"] = lanes.Any(lane => lane.Options.CollectKernelTimings || lane.Options.DetailedProfiling);
            report["ResidentWeightBytes"] = model.ResidentWeightBytes;
            report["ResidentAuxiliaryWeightBytes"] = model.ResidentAuxiliaryWeightBytes;
            report["Status"] = "training";
            WriteReport();
            foreach (int index in selected)
            {
                foreach (ArcExecutionLane lane in lanes) { lane.Synchronize(); lane.ResetDetailedProfile(); }
                var before = lanes.Select(Snapshot).ToArray();
                var kernelsBefore = model.KernelMilliseconds.ToDictionary(pair => pair.Key, pair => pair.Value);
                Example example = examples[index - 1];
                start = Stopwatch.GetTimestamp();
                Qwen35LoraStepResult result = model.TrainLora(example.TokenIds, example.ResponseStart);
                foreach (ArcExecutionLane lane in lanes) lane.Synchronize();
                double seconds = Stopwatch.GetElapsedTime(start).TotalSeconds;
                var after = lanes.Select(Snapshot).ToArray();
                var iq2Cache = model.LastIq2ProjectionCacheStats;
                var iq2GpuCache = model.LastIq2GpuProjectionCacheStats;
                var kernels = model.KernelMilliseconds.Select(pair => new
                {
                    Kernel = pair.Key, Milliseconds = pair.Value - kernelsBefore.GetValueOrDefault(pair.Key)
                }).Where(pair => pair.Milliseconds != 0).OrderByDescending(pair => pair.Milliseconds).ToArray();
                samples.Add(new()
                {
                    ["Case"] = samples.Count + 1, ["Example"] = example.Record, ["PhysicalLine"] = example.PhysicalLine,
                    ["ExampleSourceSha256"] = example.SourceSha256, ["TokenIds"] = example.TokenIds,
                    ["Tokens"] = example.TokenIds.Length, ["ResponseStartIndex"] = example.ResponseStart,
                    ["SupervisedTokens"] = result.SupervisedTokens, ["Step"] = result.Step,
                    ["Seconds"] = seconds, ["Loss"] = result.Loss, ["GradientNorm"] = result.GradientNorm,
                    ["TrainingTokensPerSecond"] = (example.TokenIds.Length - 1) / seconds,
                    ["Iq2ProjectionCache"] = new
                    {
                        iq2Cache.Captured, iq2Cache.Reused, iq2Cache.PeakBytes,
                        ConfiguredBudgetBytes = (long)options.TrainingIQ2ProjectionCacheMiB * 1024 * 1024,
                        Selection = options.TrainingIQ2ProjectionCachePrioritize
                            ? "highest-input-width" : "forward-order"
                    },
                    ["Iq2GpuProjectionCache"] = new
                    {
                        iq2GpuCache.Captured, iq2GpuCache.Reused, iq2GpuCache.PeakBytes,
                        PeakBytesByDevice = model.LastIq2GpuProjectionCachePeakBytesByDevice,
                        BudgetBytesByDevice = model.LastIq2GpuProjectionCacheBudgetBytesByDevice,
                        ConfiguredBudgetBytesPerDevice = (long)options.TrainingIQ2GpuProjectionCacheMiB * 1024 * 1024
                    },
                    ["Kernels"] = kernels, ["GpuBefore"] = before, ["GpuAfter"] = after,
                    ["GpuDelta"] = after.Select((snapshot, i) => Delta(snapshot, before[i])).ToArray(),
                    ["DetailedProfile"] = lanes.Select(lane => new { lane.DeviceIndex, Entries = lane.DetailedProfiler?.Snapshot() }).ToArray()
                });
                WriteReport();
                Console.WriteLine(FormattableString.Invariant(
                    $"Step {result.Step}, example {index}: {seconds:F3} s, tokens={example.TokenIds.Length}, loss={result.Loss:F8}, norm={result.GradientNorm:F8}"));
            }
            report["TotalTrainingSeconds"] = samples.Sum(sample => (double)sample["Seconds"]!);
            if (checkpointPath is not null)
            {
                Directory.CreateDirectory(Path.GetDirectoryName(checkpointPath)!);
                string staging = checkpointPath + "." + Guid.NewGuid().ToString("N") + ".probe";
                try
                {
                    start = Stopwatch.GetTimestamp();
                    model.SaveLora(staging);
                    // SaveLora supports replacing checkpoints. This benchmark only
                    // publishes a new destination and never overwrites a prior run.
                    File.Move(staging, checkpointPath, overwrite: false);
                    report["CheckpointSaveSeconds"] = Stopwatch.GetElapsedTime(start).TotalSeconds;
                    report["Checkpoint"] = new { Path = checkpointPath, Bytes = new FileInfo(checkpointPath).Length,
                        Sha256 = HashFile(checkpointPath), Step = model.LoraStep };
                }
                finally { if (File.Exists(staging)) File.Delete(staging); }
            }
            string modelSha = HashStream(modelInput);
            report["Model"] = new { Path = modelPath, Bytes = modelInput.Length, Sha256 = modelSha };
            string evidenceId = HashText(JsonSerializer.Serialize(new
            {
                ModelSha256 = modelSha, DataSha256 = dataSha, ResumeSha256 = resumeSha,
                Source = source, Binaries = binaries, Options = options, Lora = effectiveLora,
                PromptMode = promptMode, Devices = devices, Examples = selected
            }, JsonOptions));
            report["EvidenceId"] = evidenceId;
            foreach (var sample in samples) sample["TestId"] = evidenceId + "/step-" + sample["Case"];
            report["FinalStep"] = model.LoraStep;
            report["Status"] = "completed";
            report["CompletedUtc"] = DateTimeOffset.UtcNow;
            WriteReport();
            Console.WriteLine($"Qwen3.5 LoRA training probe saved: {outputPath}");
        }
        catch (Exception error)
        {
            report["Status"] = "failed"; report["Error"] = error.ToString();
            report["CompletedUtc"] = DateTimeOffset.UtcNow;
            try { WriteReport(); }
            catch (Exception reportError) { Console.Error.WriteLine($"Could not save failure report: {reportError.Message}"); }
            throw;
        }
    }

    private static T Field<T>(Qwen35QuantizedModel model, string name) =>
        typeof(Qwen35QuantizedModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)?.GetValue(model) is T value
            ? value : throw new MissingFieldException($"Training probe requires diagnostic field {name}.");
    private static string SourcePath([CallerFilePath] string path = "") => path;
    private static string HashText(string text) => Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
    private static string HashFile(string path) { using var stream = File.OpenRead(path); return HashStream(stream); }
    private static string HashStream(Stream stream) { stream.Position = 0; return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
    private sealed record Example(int Record, int PhysicalLine, string SourceSha256, int[] TokenIds, int ResponseStart);
    private sealed record LaneSnapshot(int DeviceIndex, long H2DBytes, long D2HBytes, long AllocatedBytes, long PeakAllocatedBytes,
        long CachedBytes, long RetiredBytes, long AllocationCount, long NativeAllocatedBytes, long NativeReleasedBytes,
        long KernelLaunchCount, long PoolHits, double AllocationMilliseconds, double TransferMilliseconds);
    private static LaneSnapshot Snapshot(ArcExecutionLane lane) => new(lane.DeviceIndex, lane.H2DBytes, lane.D2HBytes,
        lane.AllocatedBytes, lane.PeakAllocatedBytes, lane.CachedBytes, lane.RetiredBytes, lane.AllocationCount,
        lane.NativeAllocatedBytes, lane.NativeReleasedBytes, lane.KernelLaunchCount, lane.PoolHits,
        lane.AllocationMilliseconds, lane.TransferMilliseconds);
    private static object Delta(LaneSnapshot after, LaneSnapshot before) => new
    {
        after.DeviceIndex, H2DBytes = after.H2DBytes - before.H2DBytes, D2HBytes = after.D2HBytes - before.D2HBytes,
        AllocationCount = after.AllocationCount - before.AllocationCount,
        NativeAllocatedBytes = after.NativeAllocatedBytes - before.NativeAllocatedBytes,
        NativeReleasedBytes = after.NativeReleasedBytes - before.NativeReleasedBytes,
        KernelLaunchCount = after.KernelLaunchCount - before.KernelLaunchCount, PoolHits = after.PoolHits - before.PoolHits,
        AllocationMilliseconds = after.AllocationMilliseconds - before.AllocationMilliseconds,
        TransferMilliseconds = after.TransferMilliseconds - before.TransferMilliseconds
    };
}
