using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace NNtrain.Benchmarks;

internal static class PerformanceBaselineCommand
{
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    internal static int Run(string[] args)
    {
        if (args.Length == 0 || IsHelp(args[0]))
        {
            PrintUsage();
            return args.Length == 0 ? 2 : 0;
        }

        string preset = args[0].ToLowerInvariant();
        string? configurationArgument = ReadOption(args, "--config");
        string? outputArgument = ReadOption(args, "--output");
        string commit = GetCommit();
        bool dirty = IsWorkingTreeDirty();
        IReadOnlyList<BaselineGpu> gpus = QueryGpus();

        BaselineModelConfiguration model;
        string? configurationPath;
        string? configurationHash;
        if (preset == "cpu-smoke")
        {
            model = CreateSmokeConfiguration();
            configurationPath = null;
            configurationHash = null;
        }
        else
        {
            configurationPath = Path.GetFullPath(
                configurationArgument ?? "training.transformer.json");
            model = ReadConfiguration(configurationPath);
            configurationHash = Convert.ToHexString(
                SHA256.HashData(File.ReadAllBytes(configurationPath)))
                .ToLowerInvariant();
        }

        BaselineScenario[] scenarios = CreateScenarios(preset);
        string outputPath = Path.GetFullPath(
            outputArgument ?? CreateDefaultOutputPath(preset));
        Console.WriteLine("NNtrain reproducible training performance baseline");
        Console.WriteLine(
            $"preset={preset}, commit={commit}" + (dirty ? " (dirty)" : ""));
        Console.WriteLine(
            $"precision={model.Precision}, batch={model.Batch}, " +
            $"sequence={model.Sequence}, optimizer=NekoMuon fixed NS5+AdamW");

        var results = new List<BaselineScenarioResult>(scenarios.Length);
        string temporaryDirectory = Path.Combine(
            Path.GetTempPath(),
            $"nntrain-baseline-{Guid.NewGuid():N}");
        Directory.CreateDirectory(temporaryDirectory);
        bool completed = false;
        try
        {
            for (int index = 0; index < scenarios.Length; index++)
            {
                BaselineScenario scenario = scenarios[index];
                Console.WriteLine();
                Console.WriteLine(
                    $"=== {scenario.Name}: warmup={scenario.WarmupSteps}, " +
                    $"steps={scenario.MeasuredSteps}, " +
                    $"repetitions={scenario.Repetitions} ===");
                var job = new BaselineWorkerJob(
                    preset,
                    commit,
                    model,
                    scenario,
                    gpus,
                    configurationPath,
                    configurationHash);
                string jobPath = Path.Combine(
                    temporaryDirectory, $"job-{index}.json");
                string resultPath = Path.Combine(
                    temporaryDirectory, $"result-{index}.json");
                WriteJsonAtomically(jobPath, job);
                LaunchWorker(jobPath, resultPath);
                BaselineScenarioResult result = JsonSerializer.Deserialize<
                    BaselineScenarioResult>(
                        File.ReadAllText(resultPath), JsonOptions)
                    ?? throw new InvalidDataException(
                        $"Worker returned an empty result for '{scenario.Name}'.");
                results.Add(result);
                PrintScenarioSummary(result);
            }

            var document = new PerformanceBaselineDocument(
                PerformanceBaselineSchema.Version,
                DateTimeOffset.UtcNow,
                preset,
                commit,
                dirty,
                CreateHost(gpus),
                results);
            WriteJsonAtomically(outputPath, document);
            completed = true;
            Console.WriteLine();
            Console.WriteLine($"JSON: {outputPath}");
            return 0;
        }
        finally
        {
            if (completed)
            {
                try
                {
                    Directory.Delete(temporaryDirectory, recursive: true);
                }
                catch (Exception exception)
                {
                    Console.Error.WriteLine(
                        $"Could not remove temporary worker artifacts at " +
                        $"{temporaryDirectory}: {exception.Message}");
                }
            }
            else
                Console.Error.WriteLine(
                    $"Failed worker artifacts were retained at {temporaryDirectory}");
        }
    }

    internal static int RunWorker(string jobPath, string resultPath)
    {
        BaselineWorkerJob job = JsonSerializer.Deserialize<BaselineWorkerJob>(
                File.ReadAllText(jobPath), JsonOptions)
            ?? throw new InvalidDataException("Baseline worker job is empty.");
        BaselineScenarioResult result = PerformanceBaselineRunner.Run(job);
        WriteJsonAtomically(resultPath, result);
        return 0;
    }

    private static BaselineScenario[] CreateScenarios(string preset)
        => preset switch
        {
            "compare10" =>
            [
                CpuScenario("cpu-10-step", warmup: 1, steps: 10),
                CudaScenario("1gpu-10-step", [0], warmup: 1, steps: 10),
                CudaScenario("2gpu-10-step", [0, 1], warmup: 1, steps: 10),
            ],
            "cpu10" => [CpuScenario("cpu-10-step", warmup: 1, steps: 10)],
            "gpu1-10" =>
                [CudaScenario("1gpu-10-step", [0], warmup: 1, steps: 10)],
            "gpu2-10" =>
                [CudaScenario("2gpu-10-step", [0, 1], warmup: 1, steps: 10)],
            "official2gpu" =>
            [
                CudaScenario(
                    "2gpu-official-210",
                    [0, 1],
                    warmup: 20,
                    steps: 210,
                    repetitions: 3),
            ],
            "cpu-smoke" =>
                [CpuScenario("cpu-smoke", warmup: 1, steps: 2)],
            _ => throw new ArgumentException(
                $"Unknown baseline preset '{preset}'. Use --help for choices."),
        };

    private static BaselineScenario CpuScenario(
        string name,
        int warmup,
        int steps,
        int repetitions = 1)
        => new(
            name,
            BaselineDeviceKind.Cpu,
            [0],
            warmup,
            steps,
            repetitions,
            CollectPhaseProbe: true);

    private static BaselineScenario CudaScenario(
        string name,
        int[] devices,
        int warmup,
        int steps,
        int repetitions = 1)
        => new(
            name,
            BaselineDeviceKind.Cuda,
            devices,
            warmup,
            steps,
            repetitions,
            CollectPhaseProbe: true);

    private static BaselineModelConfiguration ReadConfiguration(string path)
    {
        using JsonDocument document = JsonDocument.Parse(File.ReadAllText(path));
        JsonElement root = document.RootElement;
        string architecture = root.GetProperty("modelArchitecture").GetString()
            ?? string.Empty;
        if (!string.Equals(
            architecture, "transformer", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Performance baseline requires transformer, got '{architecture}'.");
        }

        JsonElement optimizer = root.GetProperty("optimization")
            .GetProperty("optimizer");
        string optimizerType = optimizer.GetProperty("type").GetString()
            ?? string.Empty;
        if (!string.Equals(
            optimizerType, "nekomuon", StringComparison.OrdinalIgnoreCase))
        {
            throw new InvalidDataException(
                $"Performance baseline requires NekoMuon, got '{optimizerType}'.");
        }

        string depthMode = optimizer.TryGetProperty(
            "nekoMuonNewtonSchulzDepthMode", out JsonElement depthModeElement)
            ? depthModeElement.GetString() ?? string.Empty
            : string.Empty;
        int depth = optimizer.TryGetProperty(
            "nekoMuonNewtonSchulzDepth", out JsonElement depthElement)
            ? (int)depthElement.GetSingle()
            : 0;
        if (!string.Equals(depthMode, "fixed", StringComparison.OrdinalIgnoreCase)
            || depth != 5)
        {
            throw new InvalidDataException(
                "The frozen baseline requires NekoMuon fixed NS5 " +
                $"(configured mode='{depthMode}', depth={depth}).");
        }

        TensorPrecisionMode precision = PrecisionModeConfiguration.Read(root);
        return new BaselineModelConfiguration(
            root.GetProperty("vocabularySize").GetInt32(),
            root.GetProperty("batchSize").GetInt32(),
            root.GetProperty("contextLength").GetInt32(),
            root.GetProperty("modelWidth").GetInt32(),
            root.GetProperty("heads").GetInt32(),
            root.GetProperty("hiddenSize").GetInt32(),
            root.GetProperty("layers").GetInt32(),
            root.GetProperty("seed").GetInt32(),
            root.GetProperty("dropout").GetSingle(),
            root.GetProperty("initializationScale").GetSingle(),
            root.GetProperty("tieWordEmbeddings").GetBoolean(),
            TensorPrecisionModeNames.Format(precision),
            optimizer.GetProperty("learningRate").GetSingle(),
            optimizer.GetProperty("auxiliaryLearningRate").GetSingle(),
            optimizer.GetProperty("weightDecay").GetSingle(),
            optimizer.GetProperty("nekoMuonNewtonSchulzInterval").GetInt32(),
            root.TryGetProperty(
                "adaptiveCudaSharding", out JsonElement adaptive)
                    ? adaptive.GetBoolean()
                    : true,
            root.TryGetProperty(
                "cudaShardEmaAlpha", out JsonElement emaAlpha)
                    ? emaAlpha.GetDouble()
                    : 0.2d,
            root.TryGetProperty(
                "cudaMinimumRelativeShardSize", out JsonElement minimumShard)
                    ? minimumShard.GetDouble()
                    : 0.5d,
            root.TryGetProperty(
                "cudaMaximumBatchAdjustmentPerStep",
                out JsonElement maximumAdjustment)
                    ? maximumAdjustment.GetInt32()
                    : 1);
    }

    private static BaselineModelConfiguration CreateSmokeConfiguration()
        => new(
            Vocabulary: 64,
            Batch: 1,
            Sequence: 4,
            Width: 8,
            Heads: 2,
            Hidden: 16,
            Layers: 1,
            Seed: 1234,
            Dropout: 0.1f,
            InitializationScale: 0.02f,
            TieWordEmbeddings: true,
            Precision: TensorPrecisionModeNames.Mix16_32,
            LearningRate: 0.001f,
            AuxiliaryLearningRate: 0.003f,
            WeightDecay: 0.01f,
            NewtonSchulzInterval: 1,
            AdaptiveCudaSharding: false,
            CudaShardEmaAlpha: 0.2d,
            CudaMinimumRelativeShardSize: 0.5d,
            CudaMaximumBatchAdjustmentPerStep: 1);

    private static void LaunchWorker(string jobPath, string resultPath)
    {
        string processPath = Environment.ProcessPath
            ?? throw new InvalidOperationException(
                "Cannot locate the current benchmark executable.");
        var start = new ProcessStartInfo
        {
            FileName = processPath,
            UseShellExecute = false,
            WorkingDirectory = Environment.CurrentDirectory,
        };
        if (string.Equals(
            Path.GetFileNameWithoutExtension(processPath),
            "dotnet",
            StringComparison.OrdinalIgnoreCase))
        {
            start.ArgumentList.Add(Assembly.GetExecutingAssembly().Location);
        }
        start.ArgumentList.Add("--performance-baseline-worker");
        start.ArgumentList.Add(jobPath);
        start.ArgumentList.Add(resultPath);
        using Process process = Process.Start(start)
            ?? throw new InvalidOperationException("Failed to start baseline worker.");
        process.WaitForExit();
        if (process.ExitCode != 0)
        {
            throw new InvalidOperationException(
                $"Baseline worker exited with code {process.ExitCode}.");
        }
        if (!File.Exists(resultPath))
            throw new InvalidDataException("Baseline worker produced no JSON result.");
    }

    private static BaselineHost CreateHost(IReadOnlyList<BaselineGpu> gpus)
        => new(
            RuntimeInformation.OSDescription,
            RuntimeInformation.FrameworkDescription,
            RuntimeInformation.ProcessArchitecture.ToString(),
            Environment.GetEnvironmentVariable("PROCESSOR_IDENTIFIER")
                ?? "unknown",
            gpus);

    private static IReadOnlyList<BaselineGpu> QueryGpus()
    {
        string? output = RunForOutput(
            "nvidia-smi",
            ["--query-gpu=index,name,compute_cap", "--format=csv,noheader,nounits"],
            timeoutMilliseconds: 5_000);
        if (string.IsNullOrWhiteSpace(output))
            return [];

        var result = new List<BaselineGpu>();
        foreach (string line in output.Split(
            ['\r', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            string[] fields = line.Split(',', 3, StringSplitOptions.TrimEntries);
            if (fields.Length != 3 || !int.TryParse(fields[0], out int index))
                continue;
            string capability = fields[2];
            string? sm = ParseSmArchitecture(capability);
            result.Add(new BaselineGpu(index, fields[1], capability, sm));
        }
        return result.OrderBy(gpu => gpu.Index).ToArray();
    }

    private static string? ParseSmArchitecture(string computeCapability)
    {
        string[] parts = computeCapability.Split('.', 2);
        return parts.Length == 2
            && int.TryParse(parts[0], out int major)
            && int.TryParse(parts[1], out int minor)
                ? $"sm_{major}{minor}"
                : null;
    }

    private static string GetCommit()
    {
        string? environmentCommit = Environment.GetEnvironmentVariable("GITHUB_SHA")
            ?? Environment.GetEnvironmentVariable("BUILD_SOURCEVERSION");
        if (!string.IsNullOrWhiteSpace(environmentCommit))
            return environmentCommit.Trim();
        return RunForOutput("git", ["rev-parse", "HEAD"], 5_000)?.Trim()
            ?? "unknown";
    }

    private static bool IsWorkingTreeDirty()
        => !string.IsNullOrWhiteSpace(RunForOutput(
            "git",
            ["status", "--porcelain", "--untracked-files=no"],
            5_000));

    private static string? RunForOutput(
        string fileName,
        IReadOnlyList<string> arguments,
        int timeoutMilliseconds)
    {
        try
        {
            var start = new ProcessStartInfo
            {
                FileName = fileName,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                UseShellExecute = false,
                CreateNoWindow = true,
                WorkingDirectory = Environment.CurrentDirectory,
            };
            foreach (string argument in arguments)
                start.ArgumentList.Add(argument);
            using Process process = Process.Start(start)!;
            Task<string> standardOutput = process.StandardOutput.ReadToEndAsync();
            Task<string> standardError = process.StandardError.ReadToEndAsync();
            if (!process.WaitForExit(timeoutMilliseconds))
            {
                process.Kill(entireProcessTree: true);
                process.WaitForExit();
                return null;
            }
            string output = standardOutput.GetAwaiter().GetResult();
            _ = standardError.GetAwaiter().GetResult();
            return process.ExitCode == 0 ? output : null;
        }
        catch
        {
            return null;
        }
    }

    private static string? ReadOption(string[] args, string name)
    {
        for (int index = 1; index < args.Length; index++)
        {
            if (!string.Equals(args[index], name, StringComparison.Ordinal))
                continue;
            if (index + 1 >= args.Length)
                throw new ArgumentException($"Missing value after {name}.");
            return args[index + 1];
        }
        return null;
    }

    private static string CreateDefaultOutputPath(string preset)
        => Path.Combine(
            "benchmark-results",
            $"performance-baseline-{preset}-" +
            $"{DateTimeOffset.UtcNow:yyyyMMdd-HHmmss}.json");

    private static void WriteJsonAtomically<T>(string path, T value)
    {
        string fullPath = Path.GetFullPath(path);
        string? directory = Path.GetDirectoryName(fullPath);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string temporaryPath = fullPath + $".{Guid.NewGuid():N}.tmp";
        try
        {
            File.WriteAllText(
                temporaryPath,
                JsonSerializer.Serialize(value, JsonOptions));
            File.Move(temporaryPath, fullPath, overwrite: true);
        }
        finally
        {
            if (File.Exists(temporaryPath))
                File.Delete(temporaryPath);
        }
    }

    private static void PrintScenarioSummary(BaselineScenarioResult result)
    {
        BaselineConditions conditions = result.Conditions;
        string accelerators = conditions.Gpus.Count == 0
            ? "CPU"
            : string.Join(
                ", ",
                conditions.Gpus.Select(gpu =>
                    $"cuda:{gpu.Index} {gpu.Name} " +
                    $"({gpu.SmArchitecture ?? "SM unknown"})"));
        Console.WriteLine(
            $"conditions: device={conditions.Device} [{accelerators}], " +
            $"precision={conditions.Precision}, batch={conditions.Batch}, " +
            $"sequence={conditions.Sequence}, NS=" +
            $"{conditions.NewtonSchulzDepthMode}{conditions.NewtonSchulzSteps}, " +
            $"warmup={conditions.WarmupSteps}, " +
            $"steps={conditions.MeasuredSteps}, " +
            $"repetitions={conditions.Repetitions}, " +
            $"adaptive-shards={conditions.AdaptiveCudaSharding} " +
            $"(ema={conditions.CudaShardEmaAlpha:G}, min=" +
            $"{conditions.CudaMinimumRelativeShardSize:G}, max-step=" +
            $"{conditions.CudaMaximumBatchAdjustmentPerStep})");
        BaselineDistribution step = result.AggregateStep;
        Console.WriteLine(
            $"{conditions.Scenario}: step p50={step.P50:F2} ms, " +
            $"p95={step.P95:F2} ms, mean={step.Mean:F2} ms " +
            $"({step.Count} measured steps)");
        BaselineStepMeasurement[] measured = result.Runs
            .SelectMany(run => run.Measurements).ToArray();
        Console.WriteLine(
            "normal-path p50: " +
            $"forward+backward=" +
            $"{BaselineDistribution.From(measured.Select(value => value.ForwardBackwardMilliseconds)).P50:F2} ms, " +
            $"clip={BaselineDistribution.From(measured.Select(value => value.ClipMilliseconds)).P50:F2} ms, " +
            $"optimizer={BaselineDistribution.From(measured.Select(value => value.OptimizerMilliseconds)).P50:F2} ms, " +
            $"managed-allocation=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.ManagedAllocationBytes)).P50:N0} B, " +
            $"native-allocation=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.NativeAllocationBytes)).P50:N0} B");
        foreach (BaselineRunResult run in result.Runs)
        {
            if (run.FinalShardBatchSizes.Count > 0)
            {
                Console.WriteLine(
                    $"run {run.Repetition} final GPU shards = [" +
                    $"{string.Join(',', run.FinalShardBatchSizes)}]");
            }
        }
        if (result.PhaseProbe is { } probe)
        {
            Console.WriteLine(
                "diagnostic phase probe: " +
                $"forward={Format(probe.ForwardMilliseconds)}, " +
                $"backward={Format(probe.BackwardMilliseconds)}, " +
                $"reduce-wait={Format(probe.ReduceWaitMilliseconds)}, " +
                $"clip={Format(probe.ClipMilliseconds)}, " +
                $"optimizer={Format(probe.OptimizerMilliseconds)}, " +
                $"transfer={Format(probe.TransferMilliseconds)}");
        }
    }

    private static string Format(double? milliseconds)
        => milliseconds.HasValue ? $"{milliseconds.Value:F2} ms" : "not isolated";

    private static bool IsHelp(string argument)
        => string.Equals(argument, "--help", StringComparison.OrdinalIgnoreCase)
            || string.Equals(argument, "-h", StringComparison.OrdinalIgnoreCase)
            || string.Equals(argument, "help", StringComparison.OrdinalIgnoreCase);

    private static void PrintUsage()
    {
        Console.WriteLine("Usage:");
        Console.WriteLine(
            "  --performance-baseline compare10 [--config PATH] [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline cpu10|gpu1-10|gpu2-10 " +
            "[--config PATH] [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline official2gpu " +
            "[--config PATH] [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline cpu-smoke [--output PATH]");
        Console.WriteLine();
        Console.WriteLine(
            "compare10 measures CPU, one GPU, and two GPUs for 10 steps each " +
            "after one warmup step. official2gpu performs 20 warmup steps and " +
            "210 measured steps, three times. Each scenario runs in an isolated " +
            "worker process, and each repetition uses a fresh model, optimizer, " +
            "and explicitly owned data-parallel engine.");
    }
}
