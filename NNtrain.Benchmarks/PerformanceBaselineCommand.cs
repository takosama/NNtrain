using System.Diagnostics;
using System.Globalization;
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

        BaselineCommandOptions options = ParseOptions(args);
        string preset = options.Preset;
        BaselineScenario[] scenarios = CreateScenarios(preset);
        string commit = GetCommit();
        bool dirty = IsWorkingTreeDirty();

        BaselineModelConfiguration model;
        string? configurationPath;
        string? configurationHash;
        if (preset is "cpu-smoke" or "gpu-smoke" or "soak-smoke"
            or "soak-failure-smoke")
        {
            model = CreateSmokeConfiguration();
            if (preset is "gpu-smoke" or "soak-smoke"
                or "soak-failure-smoke")
                model = model with { Batch = 2 };
            if (preset is "soak-smoke" or "soak-failure-smoke")
            {
                model = model with
                {
                    Precision = TensorPrecisionModeNames.Mix8_32,
                    Bfp8BlockSize =
                        Bfp8QuantizationDescriptor.DefaultBlockSize,
                };
            }
            configurationPath = null;
            configurationHash = null;
        }
        else
        {
            configurationPath = Path.GetFullPath(
                options.ConfigurationPath ?? "training.transformer.json");
            model = ReadConfiguration(
                configurationPath,
                requireConfiguredFixedNs5: preset != "official2gpu");
            configurationHash = Convert.ToHexString(
                SHA256.HashData(File.ReadAllBytes(configurationPath)))
                .ToLowerInvariant();
        }

        BaselineConfigurationResolution resolution =
            ResolveEffectiveConfiguration(options, model);
        model = resolution.Model;
        IReadOnlyList<BaselineEffectiveOverride> effectiveOverrides =
            resolution.EffectiveOverrides;
        IReadOnlyList<BaselineGpu> gpus = QueryGpus();
        string outputPath = Path.GetFullPath(
            options.OutputPath ?? CreateDefaultOutputPath(preset));
        Console.WriteLine("NNtrain reproducible training performance baseline");
        Console.WriteLine(
            $"preset={preset}, commit={commit}" + (dirty ? " (dirty)" : ""));
        Console.WriteLine(
            $"precision={model.Precision}, batch={model.Batch}, " +
            $"sequence={model.Sequence}, bfp8-block=" +
            (string.Equals(
                model.Precision,
                TensorPrecisionModeNames.Mix8_32,
                StringComparison.Ordinal)
                    ? model.Bfp8BlockSize.ToString(CultureInfo.InvariantCulture)
                    : "n/a") + ", " +
            "optimizer=NekoMuon fixed NS5+AdamW");
        PrintEffectiveOverrides(effectiveOverrides);

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
                    configurationHash,
                    effectiveOverrides);
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
            bool gateFailure = results.Any(result =>
                result.Validation is { Passed: false });
            if (gateFailure)
            {
                Console.Error.WriteLine(
                    "One or more required performance/soak gates failed.");
            }
            return gateFailure ? 3 : 0;
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

    internal static BaselineScenario[] CreateScenarios(string preset)
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
                    repetitions: 3,
                    performanceGate:
                        PerformanceBaselineGatePolicy.OfficialTwoGpu),
            ],
            "soak2100" =>
            [
                CudaScenario(
                    "2gpu-soak-2100",
                    [0, 1],
                    warmup: 20,
                    steps: 2100,
                    soak: new BaselineSoakConfiguration(
                        TotalCommittedSteps: 2100,
                        PerformanceWarmupSteps: 20,
                        TrendWindowSteps: 100,
                        GenerationStep: 2000,
                        GenerationTokens: 1,
                        RestartStep: 1050,
                        MaximumPostWarmupVramGrowthBytes:
                            256L * 1024L * 1024L,
                        MaximumLastToFirstP50Ratio: 1.05d)),
            ],
            "soak-smoke" =>
            [
                CudaScenario(
                    "2gpu-soak-smoke",
                    [0, 1],
                    warmup: 2,
                    steps: 12,
                    soak: new BaselineSoakConfiguration(
                        TotalCommittedSteps: 12,
                        PerformanceWarmupSteps: 2,
                        TrendWindowSteps: 3,
                        GenerationStep: 8,
                        GenerationTokens: 1,
                        RestartStep: 6,
                        MaximumPostWarmupVramGrowthBytes:
                            256L * 1024L * 1024L,
                        MaximumLastToFirstP50Ratio: 5d)),
            ],
            "soak-failure-smoke" =>
            [
                CudaScenario(
                    "2gpu-soak-failure-smoke",
                    [0, 1],
                    warmup: 2,
                    steps: 7,
                    soak: new BaselineSoakConfiguration(
                        TotalCommittedSteps: 7,
                        PerformanceWarmupSteps: 2,
                        TrendWindowSteps: 2,
                        GenerationStep: 4,
                        GenerationTokens: 1,
                        RestartStep: 6,
                        MaximumPostWarmupVramGrowthBytes:
                            256L * 1024L * 1024L,
                        MaximumLastToFirstP50Ratio: 5d,
                        InjectCheckpointFailureAfterFirstArtifact: true)),
            ],
            "cpu-smoke" =>
                [CpuScenario("cpu-smoke", warmup: 1, steps: 2)],
            "gpu-smoke" =>
                [CudaScenario("2gpu-smoke", [0, 1], warmup: 1, steps: 2)],
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
        int repetitions = 1,
        BaselineSoakConfiguration? soak = null,
        BaselinePerformanceGateConfiguration? performanceGate = null)
        => new(
            name,
            BaselineDeviceKind.Cuda,
            devices,
            warmup,
            steps,
            repetitions,
            CollectPhaseProbe: true,
            Soak: soak,
            PerformanceGate: performanceGate);

    internal static BaselineCommandOptions ParseOptions(string[] args)
    {
        ArgumentNullException.ThrowIfNull(args);
        if (args.Length == 0)
            throw new ArgumentException("A performance baseline preset is required.");

        string preset = args[0].ToLowerInvariant();
        string? configurationPath = null;
        string? outputPath = null;
        TensorPrecisionMode? precision = null;
        int? bfp8BlockSize = null;
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int index = 1; index < args.Length; index++)
        {
            string option = args[index];
            if (option is not ("--config" or "--output" or "--precision"
                or "--bfp8-block-size"))
            {
                throw new ArgumentException(
                    $"Unknown performance baseline option '{option}'.");
            }
            if (!seen.Add(option))
            {
                throw new ArgumentException(
                    $"Performance baseline option '{option}' was specified twice.");
            }
            if (++index >= args.Length)
                throw new ArgumentException($"Missing value after {option}.");
            string value = args[index];
            switch (option)
            {
                case "--config":
                    configurationPath = value;
                    break;
                case "--output":
                    outputPath = value;
                    break;
                case "--precision":
                    try
                    {
                        precision = TensorPrecisionModeNames.Parse(value);
                    }
                    catch (ArgumentException exception)
                    {
                        throw new ArgumentException(
                            $"Invalid --precision value '{value}': " +
                            exception.Message,
                            nameof(args),
                            exception);
                    }
                    break;
                case "--bfp8-block-size":
                    if (!int.TryParse(
                            value,
                            NumberStyles.Integer,
                            CultureInfo.InvariantCulture,
                            out int parsedBlockSize)
                        || parsedBlockSize <= 0)
                    {
                        throw new ArgumentException(
                            "--bfp8-block-size requires a positive integer.");
                    }
                    bfp8BlockSize = parsedBlockSize;
                    break;
            }
        }

        bool supportsPrecisionOverride = preset is "compare10" or "cpu10"
            or "gpu1-10" or "gpu2-10";
        if ((precision.HasValue || bfp8BlockSize.HasValue)
            && !supportsPrecisionOverride)
        {
            throw new ArgumentException(
                $"Preset '{preset}' does not accept --precision or " +
                "--bfp8-block-size. official2gpu always uses its frozen " +
                "mix16_32 contract.");
        }
        if (bfp8BlockSize.HasValue
            && precision.HasValue
            && precision.Value != TensorPrecisionMode.Mix8_32)
        {
            throw new ArgumentException(
                "--bfp8-block-size is only valid with --precision mix8_32.");
        }
        return new BaselineCommandOptions(
            preset,
            configurationPath,
            outputPath,
            precision,
            bfp8BlockSize);
    }

    internal static BaselineConfigurationResolution ResolveEffectiveConfiguration(
        BaselineCommandOptions options,
        BaselineModelConfiguration configured)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(configured);
        var overrides = new List<BaselineEffectiveOverride>();
        if (options.Preset == "official2gpu")
        {
            const string reason =
                "official2gpu frozen contract; input JSON remains read-only";
            AddOverride(
                overrides,
                "batchSize",
                configured.Batch,
                PerformanceBaselineGatePolicy.OfficialBatch,
                reason);
            AddOverride(
                overrides,
                "contextLength",
                configured.Sequence,
                PerformanceBaselineGatePolicy.OfficialSequence,
                reason);
            AddOverride(
                overrides,
                "precisionMode",
                configured.Precision,
                PerformanceBaselineGatePolicy.OfficialPrecision,
                reason);
            AddOverride(
                overrides,
                "optimization.optimizer.type",
                "nekomuon",
                "nekomuon",
                reason);
            AddOverride(
                overrides,
                "optimization.optimizer.nekoMuonNewtonSchulzDepthMode",
                configured.NewtonSchulzDepthMode,
                PerformanceBaselineGatePolicy.OfficialNewtonSchulzDepthMode,
                reason);
            AddOverride(
                overrides,
                "optimization.optimizer.nekoMuonNewtonSchulzDepth",
                configured.NewtonSchulzDepth,
                PerformanceBaselineGatePolicy.OfficialNewtonSchulzSteps,
                reason);
            return new BaselineConfigurationResolution(
                configured with
                {
                    Batch = PerformanceBaselineGatePolicy.OfficialBatch,
                    Sequence = PerformanceBaselineGatePolicy.OfficialSequence,
                    Precision = PerformanceBaselineGatePolicy.OfficialPrecision,
                    NewtonSchulzDepthMode = PerformanceBaselineGatePolicy
                        .OfficialNewtonSchulzDepthMode,
                    NewtonSchulzDepth = PerformanceBaselineGatePolicy
                        .OfficialNewtonSchulzSteps,
                },
                overrides);
        }

        BaselineModelConfiguration effective = configured;
        const string cliReason =
            "command-line benchmark override; input JSON remains read-only";
        if (options.Precision.HasValue)
        {
            string effectivePrecision = TensorPrecisionModeNames.Format(
                options.Precision.Value);
            AddOverride(
                overrides,
                "precisionMode",
                configured.Precision,
                effectivePrecision,
                cliReason);
            effective = effective with { Precision = effectivePrecision };
        }
        if (options.Bfp8BlockSize.HasValue)
        {
            TensorPrecisionMode effectivePrecision =
                TensorPrecisionModeNames.Parse(effective.Precision);
            if (effectivePrecision != TensorPrecisionMode.Mix8_32)
            {
                throw new ArgumentException(
                    "--bfp8-block-size is only valid when the effective " +
                    "precision is mix8_32.");
            }
            AddOverride(
                overrides,
                "bfp8BlockSize",
                configured.Bfp8BlockSize,
                options.Bfp8BlockSize.Value,
                cliReason);
            effective = effective with
            {
                Bfp8BlockSize = options.Bfp8BlockSize.Value,
            };
        }
        return new BaselineConfigurationResolution(effective, overrides);
    }

    private static void AddOverride<T>(
        ICollection<BaselineEffectiveOverride> overrides,
        string setting,
        T configured,
        T effective,
        string reason)
    {
        string configuredValue = Convert.ToString(
            configured, CultureInfo.InvariantCulture) ?? string.Empty;
        string effectiveValue = Convert.ToString(
            effective, CultureInfo.InvariantCulture) ?? string.Empty;
        overrides.Add(new BaselineEffectiveOverride(
            setting,
            configuredValue,
            effectiveValue,
            !string.Equals(
                configuredValue,
                effectiveValue,
                StringComparison.OrdinalIgnoreCase),
            reason));
    }

    private static BaselineModelConfiguration ReadConfiguration(
        string path,
        bool requireConfiguredFixedNs5)
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
        if (requireConfiguredFixedNs5
            && (!string.Equals(
                    depthMode,
                    PerformanceBaselineGatePolicy
                        .OfficialNewtonSchulzDepthMode,
                    StringComparison.OrdinalIgnoreCase)
                || depth != PerformanceBaselineGatePolicy
                    .OfficialNewtonSchulzSteps))
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
            depthMode,
            depth,
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
                    : 1,
            root.TryGetProperty(
                "cudaGraphCacheBudgetMiB",
                out JsonElement graphCacheBudget)
                    ? graphCacheBudget.GetInt32()
                    : 512,
            root.TryGetProperty(
                "bfp8BlockSize",
                out JsonElement bfp8BlockSize)
                    ? bfp8BlockSize.GetInt32()
                    : Bfp8QuantizationDescriptor.DefaultBlockSize);
    }

    internal static BaselineModelConfiguration CreateSmokeConfiguration()
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
            NewtonSchulzDepthMode:
                PerformanceBaselineGatePolicy.OfficialNewtonSchulzDepthMode,
            NewtonSchulzDepth:
                PerformanceBaselineGatePolicy.OfficialNewtonSchulzSteps,
            NewtonSchulzInterval: 1,
            AdaptiveCudaSharding: false,
            CudaShardEmaAlpha: 0.2d,
            CudaMinimumRelativeShardSize: 0.5d,
            CudaMaximumBatchAdjustmentPerStep: 1,
            CudaGraphCacheBudgetMiB: 512,
            Bfp8BlockSize: Bfp8QuantizationDescriptor.DefaultBlockSize);

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
            $"{conditions.CudaMaximumBatchAdjustmentPerStep}, graph-cache=" +
            $"{conditions.CudaGraphCacheBudgetMiB} MiB), " +
            $"training-plan={conditions.TrainingExecutionPlan}");
        PrintEffectiveOverrides(conditions.EffectiveOverrides);
        if (conditions.MaximumAllowedStepP50Milliseconds is double maximum)
        {
            Console.WriteLine(
                $"performance gate: " +
                $"{conditions.PerformanceGateStatistic}, " +
                $"frozen={conditions.FrozenBaselineStepP50Milliseconds:F3} ms, " +
                $"maximum-ratio={conditions.MaximumBaselineRatio:P0}, " +
                $"required <= {maximum:F3} ms");
        }
        BaselineDistribution step = result.AggregateStep;
        Console.WriteLine(
            $"{conditions.Scenario}: step p50={step.P50:F2} ms, " +
            $"p95={step.P95:F2} ms, mean={step.Mean:F2} ms " +
            $"({step.Count} measured steps)");
        BaselineStepMeasurement[] measured = result.Runs
            .SelectMany(run => run.Measurements)
            .Where(value => !value.IsWarmup)
            .ToArray();
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
        Console.WriteLine(
            "normal-path telemetry p50: " +
            $"H2D=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.HostToDeviceBytes)).P50:N0} B/" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.HostToDeviceCopyCount)).P50:N0} copies, " +
            $"D2H=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.DeviceToHostBytes)).P50:N0} B/" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.DeviceToHostCopyCount)).P50:N0} copies, " +
            $"gradient-collective H2D=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.GradientCollectiveHostToDeviceBytes)).P50:N0} B/" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.GradientCollectiveHostToDeviceCopyCount)).P50:N0} copies, " +
            $"D2H=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.GradientCollectiveDeviceToHostBytes)).P50:N0} B/" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.GradientCollectiveDeviceToHostCopyCount)).P50:N0} copies, " +
            $"cuda malloc/free=" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.NativeAllocationCount)).P50:N0}/" +
            $"{BaselineDistribution.From(measured.Select(value => (double)value.NativeFreeCount)).P50:N0}");
        foreach (BaselineRunResult run in result.Runs)
        {
            if (run.TrainingGraph is { } graph)
            {
                Console.WriteLine(
                    $"run {run.Repetition} CUDA Graph: " +
                    $"capture={graph.CaptureCount} " +
                    $"(measured +{graph.MeasuredCaptureCount}), " +
                    $"replay={graph.ReplayCount} " +
                    $"(measured +{graph.MeasuredReplayCount}), " +
                    $"fallback={graph.FallbackCount} " +
                    $"(measured +{graph.MeasuredFallbackCount}), " +
                    $"compiled={graph.CachedCompiledPlanCount}, " +
                    $"pinned={graph.GraphPinnedBytes:N0} B, " +
                    $"post-replay ready-events=" +
                    $"{graph.MeasuredReadyEventRecordCount} / " +
                    $"{graph.MeasuredReadyEventRecordMilliseconds:F3} ms, " +
                    $"measured-path=" +
                    (graph.MeasuredIntervalFullyCompiledReplay
                        ? "compiled replay"
                        : "not fully compiled replay"));
            }
            if (run.FinalShardBatchSizes.Count > 0)
            {
                Console.WriteLine(
                    $"run {run.Repetition} final GPU shards = [" +
                    $"{string.Join(',', run.FinalShardBatchSizes)}]");
            }
            foreach (BaselineDeviceMemorySummary memory in run.DeviceMemory)
            {
                Console.WriteLine(
                    $"run {run.Repetition} cuda:{memory.Device} VRAM: " +
                    $"start={memory.StartUsedBytes / 1048576d:F1} MiB, " +
                    $"peak={memory.PeakUsedBytes / 1048576d:F1} MiB, " +
                    $"end={memory.EndUsedBytes / 1048576d:F1} MiB, " +
                    $"growth={memory.PeakGrowthBytes / 1048576d:F1} MiB");
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
        if (result.Validation is { } validation)
        {
            Console.WriteLine(
                $"validation ({validation.Scope}): " +
                (validation.Passed ? "PASS" : "FAIL"));
            foreach (BaselineGateResult gate in validation.Gates)
            {
                string status = gate.Passed switch
                {
                    true => "PASS",
                    false => "FAIL",
                    null => "NOT AVAILABLE",
                };
                Console.WriteLine(
                    $"  {status} {gate.Name}: actual={gate.Actual}; " +
                    $"required={gate.Required}" +
                    (string.IsNullOrWhiteSpace(gate.Detail)
                        ? string.Empty
                        : $"; {gate.Detail}"));
            }
        }
    }

    private static string Format(double? milliseconds)
        => milliseconds.HasValue ? $"{milliseconds.Value:F2} ms" : "not isolated";

    private static void PrintEffectiveOverrides(
        IReadOnlyList<BaselineEffectiveOverride> overrides)
    {
        if (overrides.Count == 0)
            return;
        Console.WriteLine(
            "effective overrides (the input JSON was not modified):");
        foreach (BaselineEffectiveOverride value in overrides)
        {
            Console.WriteLine(
                $"  {value.Setting}: configured={value.ConfiguredValue}, " +
                $"effective={value.EffectiveValue}, " +
                $"changed={value.Changed}; {value.Reason}");
        }
    }

    private static bool IsHelp(string argument)
        => string.Equals(argument, "--help", StringComparison.OrdinalIgnoreCase)
            || string.Equals(argument, "-h", StringComparison.OrdinalIgnoreCase)
            || string.Equals(argument, "help", StringComparison.OrdinalIgnoreCase);

    private static void PrintUsage()
    {
        Console.WriteLine("Usage:");
        Console.WriteLine(
            "  --performance-baseline compare10 [--config PATH] " +
            "[--precision MODE] [--bfp8-block-size N] [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline cpu10|gpu1-10|gpu2-10 " +
            "[--config PATH] [--precision MODE] [--bfp8-block-size N] " +
            "[--output PATH]");
        Console.WriteLine(
            "  --performance-baseline official2gpu " +
            "[--config PATH] [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline soak2100 " +
            "[--config PATH] [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline soak-smoke [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline soak-failure-smoke [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline cpu-smoke [--output PATH]");
        Console.WriteLine(
            "  --performance-baseline gpu-smoke [--output PATH]");
        Console.WriteLine();
        Console.WriteLine(
            "compare10 measures CPU, one GPU, and two GPUs for 10 steps each " +
            "after one warmup step. official2gpu performs 20 warmup steps and " +
            "210 measured steps, three times, then gates the median of the " +
            "three run p50 values at <= 380.384 ms (80% of the frozen " +
            "475.480 ms baseline). Each scenario runs in an isolated " +
            "worker process, and each repetition uses a fresh model, optimizer, " +
            "and explicitly owned data-parallel engine.");
        Console.WriteLine(
            "compare10/cpu10/gpu1-10/gpu2-10 accept precision modes " +
            "float32, bfloat16, mix16_32 (fp16_32 alias), bfp8, and " +
            "mix8_32. --bfp8-block-size is valid only for effective " +
            "mix8_32. official2gpu does not silently ignore overrides: it " +
            "rejects those options and pins batch 72, sequence 512, mix16_32, and " +
            "NekoMuon fixed NS5 without modifying the input JSON.");
        Console.WriteLine(
            "soak2100 commits exactly 2100 two-GPU steps. It excludes the " +
            "first 20 from trend windows, writes a Wiki v8 streaming checkpoint " +
            "after committed step 1050, disposes the full model/optimizer/DP " +
            "fixture, restores into a fresh fixture, verifies checkpoint, " +
            "JSONL, and HTML continuity, and runs a one-token generation event " +
            "after committed step 2000. It never modifies the input JSON. " +
            "soak-smoke exercises the same full restart with a tiny model.");
    }
}

internal sealed record BaselineCommandOptions(
    string Preset,
    string? ConfigurationPath,
    string? OutputPath,
    TensorPrecisionMode? Precision,
    int? Bfp8BlockSize);

internal sealed record BaselineConfigurationResolution(
    BaselineModelConfiguration Model,
    IReadOnlyList<BaselineEffectiveOverride> EffectiveOverrides);
