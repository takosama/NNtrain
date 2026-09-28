using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class QwenGenerationProbe
{
    internal static void Run(string[] args)
    {
        if (args.Length < 2) throw new ArgumentException(
            "Expected model.gguf NEW-result.json [--tokens N --runs N --warmup N " +
            "--prompt TEXT --fast on|off --cache on|off --norm on|off " +
            "--subgroup on|off --subgroup-prefill on|off --profile].");
        string path = Path.GetFullPath(args[0]), resultPath = Path.GetFullPath(args[1]);
        if (File.Exists(resultPath)) throw new IOException("Result must be a new file.");
        int tokens = 8, runs = 2, warmup = 1;
        string prompt = "国会議事堂への行き方を教えて";
        bool fast = true, cache = true, detailed = false, norm = true, subgroup = true, subgroupPrefill = true;
        for (int i = 2; i < args.Length; i++)
        {
            if (args[i] == "--profile") { detailed = true; continue; }
            string option = args[i];
            if (++i >= args.Length) throw new ArgumentException($"Missing {option} value.");
            string value = args[i];
            bool Toggle() => value switch { "on" => true, "off" => false,
                _ => throw new ArgumentException("Expected on or off.") };
            switch (option)
            {
                case "--tokens": tokens = int.Parse(value); break;
                case "--runs": runs = int.Parse(value); break;
                case "--warmup": warmup = int.Parse(value); break;
                case "--prompt": prompt = value; break;
                case "--fast": fast = Toggle(); break;
                case "--cache": cache = Toggle(); break;
                case "--norm": norm = Toggle(); break;
                case "--subgroup": subgroup = Toggle(); break;
                case "--subgroup-prefill": subgroupPrefill = Toggle(); break;
                default: throw new ArgumentException($"Unknown {option}.");
            }
        }
        if (tokens is < 1 or > 4096 || runs is < 1 or > 20 || warmup is < 0 or > 3)
            throw new ArgumentOutOfRangeException(nameof(args));
        Qwen2GgufDescriptor descriptor = Qwen2Gguf.Inspect(path);
        Qwen2GgufTokenizer tokenizer = Qwen2GgufTokenizer.Load(path);
        int[] promptIds = tokenizer.Encode(prompt);
        var options = new ArcExecutionOptions { QwenQuantizedLinearFast = fast,
            QwenInferenceKvCache = cache, QwenRmsNormFast = norm,
            QwenQuantizedLinearSubgroup = subgroup,
            QwenQuantizedSubgroupPrefill = subgroupPrefill, DetailedProfiling = detailed };
        Console.WriteLine($"Qwen {descriptor.LayerCount} layers, {descriptor.EmbeddingLength} width, " +
            $"prompt={promptIds.Length}, generate={tokens}, fast={fast}, cache={cache}");
        var loadWatch = Stopwatch.StartNew();
        using var execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32, options);
        double sessionMs = loadWatch.Elapsed.TotalMilliseconds;
        using Qwen2QuantizedForCausalLM model = Qwen2Gguf.LoadQuantizedModel(path);
        model.to(new TorchDevice(TensorDevice.Arc));
        double loadMs = loadWatch.Elapsed.TotalMilliseconds - sessionMs;
        Console.WriteLine($"Session {sessionMs:F1} ms, load {loadMs:F1} ms");
        ArcExecutionLane lane = Tensor.ArcLane;
        bool callbackTimingValid = model.GetType().GetMethods(
            System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)
            .Single(method => method.Name == "GenerateTokenIds" && method.GetParameters().Length == 7)
            .DeclaringType != typeof(LanguageModel);
        var samples = new List<object>();
        var allIds = new List<int[]>();
        for (int run = 0; run < warmup + runs; run++)
        {
            lane.Synchronize();
            lane.ResetDetailedProfile();
            if (lane.DetailedProfiler is { } profiler) profiler.Phase = "prefill";
            long upload = lane.H2DBytes, download = lane.D2HBytes, launches = lane.KernelLaunchCount;
            var beforeKernels = new Dictionary<string, double>(lane.KernelTimings);
            var timestamps = new List<double>();
            var watch = Stopwatch.StartNew();
            int[] ids = model.GenerateTokenIds(promptIds, tokens, 0f, 1, null, new Random(1), token =>
            {
                timestamps.Add(watch.Elapsed.TotalMilliseconds);
                if (lane.DetailedProfiler is { } p) p.Phase = "decode";
                if (timestamps.Count == 1 || timestamps.Count % 100 == 0)
                    Console.WriteLine($"run {run}: token {timestamps.Count} at {watch.Elapsed.TotalMilliseconds:F1} ms");
            });
            lane.Synchronize();
            watch.Stop();
            int count = ids.Length - promptIds.Length;
            double elapsed = watch.Elapsed.TotalMilliseconds;
            int[] generated = ids.Skip(promptIds.Length).ToArray();
            var sample = new
            {
                Run = run, Warmup = run < warmup, TotalMs = elapsed,
                Tokens = count, TokensPerSecond = count * 1000d / elapsed,
                FirstTokenMs = callbackTimingValid && timestamps.Count > 0 ? timestamps[0] : (double?)null,
                DecodeTokensPerSecond = callbackTimingValid && timestamps.Count > 1
                    ? (count - 1) * 1000d / (timestamps[^1] - timestamps[0]) : (double?)null,
                TokenElapsedMs = timestamps, GeneratedTokenIds = generated,
                GeneratedText = tokenizer.Decode(generated),
                UploadBytes = lane.H2DBytes - upload, DownloadBytes = lane.D2HBytes - download,
                KernelLaunches = lane.KernelLaunchCount - launches,
                lane.AllocatedBytes, lane.PeakAllocatedBytes, lane.CachedBytes, lane.RetiredBytes,
                KernelMs = lane.KernelTimings.ToDictionary(p => p.Key,
                    p => p.Value - beforeKernels.GetValueOrDefault(p.Key)),
                Profile = lane.DetailedProfiler?.Snapshot(),
            };
            samples.Add(sample);
            if (run >= warmup) allIds.Add(generated);
            Console.WriteLine(JsonSerializer.Serialize(new { sample.Run, sample.Warmup,
                sample.TotalMs, sample.TokensPerSecond, sample.FirstTokenMs,
                sample.DecodeTokensPerSecond, sample.AllocatedBytes, sample.DownloadBytes }));
        }
        Directory.CreateDirectory(Path.GetDirectoryName(resultPath)!);
        using FileStream output = new(resultPath, FileMode.CreateNew, FileAccess.Write);
        JsonSerializer.Serialize(output, new
        {
            TimestampUtc = DateTimeOffset.UtcNow, ModelPath = path,
            ModelBytes = new FileInfo(path).Length, ModelSha256 = Hash(path),
            BinarySha256 = new[] { typeof(Tensor).Assembly, typeof(ArcExecutionLane).Assembly,
                typeof(QwenGenerationProbe).Assembly }.ToDictionary(a => a.GetName().Name!, a => Hash(a.Location)),
            descriptor.LayerCount, descriptor.EmbeddingLength, descriptor.HeadCount,
            descriptor.KvHeadCount, descriptor.VocabularySize,
            Prompt = prompt, PromptIds = promptIds, TokensRequested = tokens,
            Options = options, CallbackTimingValid = callbackTimingValid,
            SessionMs = sessionMs, ModelLoadMs = loadMs,
            Deterministic = allIds.All(ids => ids.SequenceEqual(allIds[0])), Samples = samples,
            Notes = "Greedy, EOS disabled for a fixed token count. First warmup includes initial weight upload. " +
                "GPU allocation peaks are session cumulative backend counters, not driver VRAM readings. " +
                "Callbacks run after every sampled token on the optimized and reference generation routes."
        }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static string Hash(string path)
    {
        using var stream = File.OpenRead(path);
        return Convert.ToHexString(SHA256.HashData(stream));
    }
}
