using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using NNtrain;
using NNtrain.Arc;

namespace Iq2sProductionBench;

/// <summary>
/// Model A/B with the GUI's inference options. The caller selects this entry
/// before creating a microbenchmark lane; models are loaded sequentially.
/// </summary>
internal static class ModelBenchmark
{
    internal static int Run(IReadOnlyDictionary<string, string> options)
    {
        string Get(string name, string value) => options.TryGetValue(name, out string? found) ? found : value;
        int Number(string name, int fallback, int minimum = 1) => int.TryParse(Get(name, fallback.ToString(CultureInfo.InvariantCulture)), out int value) && value >= minimum
            ? value : throw new ArgumentException($"Invalid {name}");
        string modelPath = Path.GetFullPath(options.TryGetValue("--model", out string? model) ? model : throw new ArgumentException("Model benchmark requires --model."));
        string directory = Path.GetFullPath(Get("--output", $"benchmark-results/iq2-model-{DateTime.UtcNow:yyyyMMdd-HHmmss}"));
        string reportPath = Path.Combine(directory, "model-results.json");
        Directory.CreateDirectory(directory);
        if (File.Exists(reportPath)) throw new IOException($"Refusing to replace {reportPath}");
        int[] lengths = Get("--rows", "128,1024").Split(',').Select(int.Parse).ToArray();
        if (lengths.Length == 0 || lengths.Distinct().Count() != lengths.Length || lengths.Any(value => value < 128))
            throw new ArgumentException("Model prefix lengths must be distinct and at least 128, to exercise tiled forwarding.");
        int[] devices = Get("--devices", "0").Split(',').Select(int.Parse).ToArray();
        if (devices.Length == 0 || devices.Any(value => value < 0) || devices.Distinct().Count() != devices.Length)
            throw new ArgumentException("Invalid distinct device indices.");
        int sampleCount = Number("--samples", 3), warmup = Number("--warmup", 1, 0), chunk = Number("--prefill-chunk", 1024, 128);
        bool profileKernels = !bool.Parse(Get("--model-no-profile", "false"));
        bool ExpectCandidateKernel(int length) => Math.Min(length, chunk) >= 512;
        const string candidateProjectionKernel = "q35l_prefill_xmx_iq2_s_gguf_bslm_k32r";
        static bool AttentionExercised(IReadOnlyDictionary<string, double> kernels) =>
            kernels.GetValueOrDefault("q35a_scores_rows") > 0 && kernels.GetValueOrDefault("q35a_attend_rows") > 0;
        double l2Limit = double.Parse(Get("--model-l2-limit", "0.001"), CultureInfo.InvariantCulture);
        if (!double.IsFinite(l2Limit) || l2Limit <= 0) throw new ArgumentException("Invalid model L2 limit.");
        string seed = options.TryGetValue("--prompt-file", out string? promptFile) ? File.ReadAllText(promptFile)
            : "日本語の説明文です。数値計算では、入力、出力、計算時間、誤差を同じ条件で確認します。データを小さなまとまりに分け、行列の積を順番に求めます。結果を比較して、その理由を簡潔に説明してください。\n";
        if (string.IsNullOrWhiteSpace(seed)) throw new ArgumentException("Prompt seed must not be empty.");
        DateTimeOffset started = DateTimeOffset.UtcNow;
        var modelFile = new FileInfo(modelPath);
        long modelBytes = modelFile.Length;
        DateTime modelTime = modelFile.LastWriteTimeUtc;
        string? modelSha = null;
        var fileHashes = Assemblies();
        var passes = new List<ModelPass>();
        var prompts = new Dictionary<int, int[]>();
        var baseline = new Dictionary<int, float[]>();
        var logitFiles = new Dictionary<string, object>();
        object? deviceInformation = null;
        Qwen35ExecutionOptions GuiOptions(bool tiled) => new()
        {
            LoraTraining = false,
            IQ2TiledForward = tiled,
            IQ2TiledBackward = false,
            InferencePrefillChunkTokens = chunk,
            InferenceBatchMixedAttention = true,
            InferenceBatchTextAttention = tiled,
            CollectKernelTimings = profileKernels,
            InferenceProjectionRows = 4,
            InferenceXmxPrefill = true,
            InferenceXmxPackedPrefill = false,
            InferenceXmxFactoredPrefill = true,
            InferenceXmxGgufBslmPrefill = true,
            InferenceResidentIq2Panels = false,
            InferenceSubgroupRecurrentRms = true,
            InferenceBufferPoolMiB = 512,
            InferenceDeferredReleaseMiB = 256,
            InferenceBatchRecurrent = true,
            ComputeModelFingerprintOnLoad = false
        };
        void Save(bool complete, string? error = null)
        {
            var summary = lengths.Select(length =>
            {
                ModelPass? oldPass = passes.FirstOrDefault(pass => !pass.Tiled && pass.PrefixTokens == length);
                ModelPass? newPass = passes.FirstOrDefault(pass => pass.Tiled && pass.PrefixTokens == length);
                return new { prefixTokens = length, totalPromptTokens = length + 1,
                    wallMedianSpeedup = oldPass is null || newPass is null ? (double?)null : oldPass.WallMilliseconds.Median / newPass.WallMilliseconds.Median,
                    gpuKernelMedianSpeedup = !profileKernels || oldPass is null || newPass is null ? (double?)null : oldPass.GpuKernelMilliseconds!.Median / newPass.GpuKernelMilliseconds!.Median,
                    candidateLogitComparison = newPass?.LogitComparison,
                    candidateKernelExpected = ExpectCandidateKernel(length),
                    candidateKernelExercised = !profileKernels ? (bool?)null : newPass?.Samples.All(sample => sample.KernelMilliseconds.GetValueOrDefault(candidateProjectionKernel) > 0),
                    candidateAttentionExpected = true,
                    candidateAttentionExercised = !profileKernels ? (bool?)null : newPass?.Samples.All(sample => AttentionExercised(sample.KernelMilliseconds)) };
            }).ToArray();
            var report = new
            {
                schemaVersion = 1, complete, error, startedUtc = started, completedUtc = complete ? (DateTimeOffset?)DateTimeOffset.UtcNow : null,
                operation = "Reset -> PrimePromptPrefix(prefix of M tokens) -> ForwardToken(next prompt token, returnLogits=true)",
                purpose = "Full base-model prefill plus next logits: original versus IQ2 row-major BSLM and batched text attention",
                scope = "No LoRA adapter and no vision encoder. Candidate enables IQ2 row-major BSLM plus batched text attention; microbenchmarks isolate IQ2 alone. GUI inference option values otherwise; per-kernel profiling "
                    + (profileKernels ? "enabled." : "disabled to measure GUI wall latency.")
                    + " Prefix checkpoint capture and final logits download included in wall time.",
                timing = new { sampleCount, warmup, profileKernels, requiresSeparateProfileEvidence = !profileKernels,
                    modelLoadExcluded = true, tokenizationExcluded = true,
                    prefixReuse = "Reset before every operation; PrimePromptPrefix must report 0 reused tokens",
                    order = "One original model session then one candidate session; never loaded concurrently",
                    gpuMetric = profileKernels ? "Sum of recorded kernel event times across selected GPUs, not critical-path elapsed time"
                        : "Not measured; GPU time, GPU speedup and kernel-coverage fields are null. Validate the selected route with separate profiled evidence.",
                    hostMetric = "Reset through blocking final logits readback; excludes comparison, serialization, file hashing",
                    phaseMetric = "Contiguous host intervals for Reset, PrimePromptPrefix and the final ForwardToken; their sum equals wall time. No additional phase synchronization is inserted.",
                    configuredPrefixChunkTokens = chunk },
                candidateKernels = new { projection = candidateProjectionKernel, projectionMinimumRows = 512,
                    attention = new[] { "q35a_scores_rows", "q35a_softmax_rows", "q35a_attend_rows" } },
                numericGate = new { l2Limit, top1MustMatch = true, repeatL2Limit = 1e-5,
                    note = "Full vocabulary logit L2 gate is distinct from projection microbenchmark tolerances; changed FP32 order can propagate through 64 layers." },
                model = new { path = modelPath, bytes = modelBytes, lastWriteTimeUtc = modelTime, sha256 = modelSha },
                deviceInformation, options = new { original = GuiOptions(false), candidate = GuiOptions(true) },
                prompt = new { source = promptFile is null ? "Built-in Japanese numerical-computing prose" : Path.GetFullPath(promptFile),
                    seedText = seed, construction = "Repeat UTF-8 seed within a user ChatML prefix, tokenize, and take exactly M+1 tokens; M tokens are primed and the final token obtains logits",
                    tokenIds = prompts },
                measuredAssemblySha256 = fileHashes, logitFiles, summary, passes
            };
            File.WriteAllText(reportPath, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase }));
        }
        try
        {
            var available = ArcDevices.Enumerate();
            if (devices.Any(index => index >= available.Count)) throw new ArgumentException("Selected device is not available.");
            deviceInformation = devices.Select(index => new { available[index].Index, available[index].Name, available[index].DriverVersion,
                available[index].GlobalMemoryBytes, available[index].MaximumAllocationBytes }).ToArray();
            Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(modelPath);
            var builder = new StringBuilder("<|im_start|>user\n");
            int[] tokens = [];
            while (tokens.Length < lengths.Max() + 1)
            {
                for (int i = 0; i < 8; i++) builder.Append(seed);
                tokens = tokenizer.Encode(builder.ToString());
            }
            foreach (int length in lengths) prompts.Add(length, tokens.Take(length + 1).ToArray());
            Save(false);
            foreach (bool tiled in new[] { false, true })
            {
                var loadWatch = Stopwatch.StartNew();
                using (Qwen35QuantizedModel loaded = Qwen35QuantizedModel.Load(modelPath, devices,
                    progress: message => Console.WriteLine($"model {(tiled ? "tiled" : "original")}: {message}"), options: GuiOptions(tiled)))
                {
                    loadWatch.Stop();
                    if (lengths.Any(length => length + 1 >= loaded.Descriptor.ContextLength)) throw new ArgumentException("Requested prefix exceeds model context.");
                    foreach (int length in lengths)
                    {
                        int[] ids = prompts[length], prefix = ids[..^1];
                        float[]? firstLogits = null;
                        var samples = new List<ModelSample>();
                        for (int round = -warmup; round < sampleCount; round++)
                        {
                            var kernelsBefore = profileKernels ? new Dictionary<string, double>(loaded.KernelMilliseconds) : null;
                            long[] uploads = loaded.UploadedBytes.ToArray(), downloads = loaded.DownloadedBytes.ToArray();
                            int[] gcBefore = [GC.CollectionCount(0), GC.CollectionCount(1), GC.CollectionCount(2)];
                            long allocatedBefore = GC.GetTotalAllocatedBytes(precise: false);
                            long operationStart = Stopwatch.GetTimestamp();
                            loaded.Reset();
                            long resetEnd = Stopwatch.GetTimestamp();
                            var primed = loaded.PrimePromptPrefix(prefix);
                            long prefixEnd = Stopwatch.GetTimestamp();
                            float[] logits = loaded.ForwardToken(ids[^1]);
                            long operationEnd = Stopwatch.GetTimestamp();
                            double wallMilliseconds = Stopwatch.GetElapsedTime(operationStart, operationEnd).TotalMilliseconds;
                            double resetMilliseconds = Stopwatch.GetElapsedTime(operationStart, resetEnd).TotalMilliseconds;
                            double primePrefixMilliseconds = Stopwatch.GetElapsedTime(resetEnd, prefixEnd).TotalMilliseconds;
                            double lastTokenMilliseconds = Stopwatch.GetElapsedTime(prefixEnd, operationEnd).TotalMilliseconds;
                            int[] gcDeltas = [GC.CollectionCount(0) - gcBefore[0], GC.CollectionCount(1) - gcBefore[1], GC.CollectionCount(2) - gcBefore[2]];
                            long allocatedBytesDelta = GC.GetTotalAllocatedBytes(precise: false) - allocatedBefore;
                            if (primed.ReusedTokens != 0) throw new InvalidOperationException("Unexpected prompt-cache reuse in model benchmark.");
                            if (logits.Length != loaded.Descriptor.VocabularySize || logits.Any(value => !float.IsFinite(value)))
                                throw new ArithmeticException("Invalid model logits.");
                            var kernelTimes = profileKernels ? loaded.KernelMilliseconds
                                .Where(pair => pair.Value - kernelsBefore!.GetValueOrDefault(pair.Key) > 0)
                                .ToDictionary(pair => pair.Key, pair => pair.Value - kernelsBefore!.GetValueOrDefault(pair.Key))
                                : new Dictionary<string, double>();
                            bool exercised = kernelTimes.GetValueOrDefault(candidateProjectionKernel) > 0;
                            bool expectedKernel = tiled && ExpectCandidateKernel(length);
                            if (profileKernels && exercised != expectedKernel) throw new InvalidOperationException($"Expected tiled route={expectedKernel}, observed={exercised}; hybrid route coverage is incorrect or original was contaminated.");
                            if (profileKernels && AttentionExercised(kernelTimes) != tiled)
                                throw new InvalidOperationException($"Expected batched text attention={tiled}, observed={AttentionExercised(kernelTimes)}.");
                            if (firstLogits is not null)
                            {
                                LogitComparison repeat = Compare(firstLogits, logits);
                                if (repeat.RelativeL2 > 1e-5 || !repeat.Top1Equal) throw new ArithmeticException("Repeated model output did not remain stable after Reset.");
                            }
                            firstLogits ??= logits;
                            if (round < 0) { Console.WriteLine($"{(tiled ? "tiled" : "original")} prefix={length} warmup {round + warmup + 1}/{warmup}: {wallMilliseconds:F2} ms"); continue; }
                            samples.Add(new(round, wallMilliseconds, profileKernels ? kernelTimes.Values.Sum() : null, kernelTimes,
                                primed.Cached, loaded.LiveDeviceBytes.ToArray(), loaded.PeakDeviceBytes.ToArray(),
                                Difference(loaded.UploadedBytes, uploads), Difference(loaded.DownloadedBytes, downloads),
                                resetMilliseconds, primePrefixMilliseconds, lastTokenMilliseconds, gcDeltas, allocatedBytesDelta));
                            Console.WriteLine($"{(tiled ? "tiled" : "original")} prefix={length} sample {round + 1}/{sampleCount}: {wallMilliseconds:F2} ms (reset {resetMilliseconds:F2}, prime {primePrefixMilliseconds:F2}, last {lastTokenMilliseconds:F2}), top1={ArgMax(logits)}");
                        }
                        if (firstLogits is null) throw new InvalidOperationException("No logits were produced.");
                        LogitComparison? comparison = null;
                        if (!tiled) baseline.Add(length, firstLogits);
                        else comparison = Compare(baseline[length], firstLogits);
                        string logitsPath = Path.Combine(directory, $"{(tiled ? "tiled" : "original")}-prefix{length}.logits.f32");
                        using (var stream = new FileStream(logitsPath, FileMode.CreateNew, FileAccess.Write, FileShare.Read))
                            stream.Write(MemoryMarshal.AsBytes(firstLogits.AsSpan()));
                        logitFiles.Add(Path.GetFileName(logitsPath), new { path = logitsPath, sha256 = HashFile(logitsPath), count = firstLogits.Length,
                            format = "Little-endian IEEE754 Float32, vocabulary-index order", top10 = Top(firstLogits, 10) });
                        passes.Add(new(tiled, length, loadWatch.Elapsed.TotalMilliseconds, Stats.Of(samples.Select(sample => sample.WallMilliseconds)),
                            profileKernels ? Stats.Of(samples.Select(sample => sample.GpuKernelMilliseconds!.Value)) : null, comparison, samples));
                        Save(false);
                        if (comparison is not null && (comparison.RelativeL2 > l2Limit || !comparison.Top1Equal))
                            throw new ArithmeticException($"Model comparison failed at prefix {length}: L2={comparison.RelativeL2:E6} (limit {l2Limit:E6}), top1 equal={comparison.Top1Equal}.");
                    }
                }
                GC.Collect(); GC.WaitForPendingFinalizers();
            }
            modelFile.Refresh();
            if (modelFile.Length != modelBytes || modelFile.LastWriteTimeUtc != modelTime) throw new IOException("Model file changed during the benchmark.");
            modelSha = HashFile(modelPath);
            var after = Assemblies();
            if (fileHashes.Count != after.Count || fileHashes.Any(pair => !after.TryGetValue(pair.Key, out string? hash) || hash != pair.Value))
                throw new IOException("Measured assembly changed during model benchmark.");
            Save(true);
            Console.WriteLine($"Model comparison complete: {reportPath}");
            return 0;
        }
        catch (Exception exception)
        {
            Save(false, exception.ToString());
            Console.Error.WriteLine(exception);
            return 1;
        }
    }

    private static long[] Difference(IReadOnlyList<long> current, IReadOnlyList<long> previous) => current.Select((value, index) => value - previous[index]).ToArray();
    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++) if (values[i] > values[best]) best = i;
        return best;
    }
    private static object[] Top(float[] values, int count) => Enumerable.Range(0, values.Length).OrderByDescending(index => values[index])
        .ThenBy(index => index).Take(count).Select(index => (object)new { tokenId = index, value = values[index] }).ToArray();
    private static LogitComparison Compare(float[] reference, float[] actual)
    {
        if (reference.Length != actual.Length) throw new InvalidDataException("Logit lengths differ.");
        double maximum = 0, error2 = 0, reference2 = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double error = (double)actual[i] - reference[i];
            maximum = Math.Max(maximum, Math.Abs(error)); error2 += error * error; reference2 += (double)reference[i] * reference[i];
        }
        int first = ArgMax(reference), second = ArgMax(actual);
        return new(reference.Length, maximum, Math.Sqrt(error2 / Math.Max(reference2, 1e-100)), first, second, first == second);
    }
    private static Dictionary<string, string> Assemblies() => Directory.GetFiles(AppContext.BaseDirectory, "NNtrain*.dll")
        .ToDictionary(path => Path.GetFileName(path)!, HashFile, StringComparer.Ordinal);
    private static string HashFile(string path) { using Stream stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }

    internal sealed record LogitComparison(int Count, double MaxAbsoluteError, double RelativeL2, int OriginalTop1, int CandidateTop1, bool Top1Equal);
    private sealed record ModelSample(int Round, double WallMilliseconds, double? GpuKernelMilliseconds,
        Dictionary<string, double> KernelMilliseconds, bool PrefixCheckpointCached,
        long[] LiveDeviceBytes, long[] PeakDeviceBytes, long[] UploadedBytes, long[] DownloadedBytes,
        double ResetMilliseconds, double PrimePrefixMilliseconds, double LastTokenMilliseconds,
        int[] GcCollectionDeltas, long ManagedAllocatedBytesDelta);
    private sealed record ModelPass(bool Tiled, int PrefixTokens, double LoadMilliseconds, Stats WallMilliseconds,
        Stats? GpuKernelMilliseconds, LogitComparison? LogitComparison, List<ModelSample> Samples);
}
