using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text.Json;
using Iq2sProductionBench;
using NNtrain;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    string arg = args[i];
    if (!arg.StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Unexpected argument {arg}");
    int equal = arg.IndexOf('=');
    if (equal >= 0) options.Add(arg[..equal], arg[(equal + 1)..]);
    else if (arg is "--fallback" or "--stress" or "--hash-model" or "--transpose" or "--model-forward" or "--model-no-profile" or "--row-major") options.Add(arg, "true");
    else options.Add(arg, ++i < args.Length ? args[i] : throw new ArgumentException($"Missing value for {arg}"));
}
string[] known = ["--output", "--model", "--tensors", "--rows", "--input", "--output-width", "--tiles", "--modes",
    "--panel-columns", "--samples", "--warmup", "--device", "--cpu-points", "--fallback", "--stress", "--hash-model", "--repo", "--transpose", "--scratch-mib",
    "--model-forward", "--model-no-profile", "--devices", "--prefill-chunk", "--model-l2-limit", "--prompt-file", "--row-major", "--bslm-k"];
foreach (string name in options.Keys) if (!known.Contains(name)) throw new ArgumentException($"Unknown option {name}");
if (options.ContainsKey("--model-forward"))
{
    Environment.ExitCode = ModelBenchmark.Run(options);
    return;
}
string Option(string name, string fallback) => options.GetValueOrDefault(name, fallback);
int Number(string name, int fallback, int minimum = 1) => int.TryParse(Option(name, fallback.ToString(CultureInfo.InvariantCulture)), out int value) && value >= minimum
    ? value : throw new ArgumentException($"Invalid {name}");
bool Flag(string name) => bool.Parse(Option(name, "false"));
string outputDirectory = Path.GetFullPath(Option("--output", $"benchmark-results/iq2s-production-{DateTime.UtcNow:yyyyMMdd-HHmmss}"));
Directory.CreateDirectory(outputDirectory);
string resultPath = Path.Combine(outputDirectory, "results.json");
if (File.Exists(resultPath)) throw new IOException($"Refusing to replace {resultPath}");
string repo = Path.GetFullPath(Option("--repo", Environment.CurrentDirectory));
int[] rowCounts = Option("--rows", "130").Split(',').Select(int.Parse).ToArray();
if (rowCounts.Length == 0 || rowCounts.Any(rows => rows <= 0)) throw new ArgumentException("Rows must be positive.");
int samples = Number("--samples", 9), warmup = Number("--warmup", 3, 0), device = Number("--device", 0, 0);
int cpuPoints = Number("--cpu-points", 1024, 4), panelColumns = Number("--panel-columns", 0, 0);
bool fallback = Flag("--fallback"), stress = Flag("--stress");
bool transpose = Flag("--transpose");
int scratchMiB = Number("--scratch-mib", 128);
string[] modes = Option("--modes", "inference,training,tiled").Split(',');
if (modes.Any(mode => mode is not ("inference" or "training" or "tiled")) || modes.Distinct().Count() != modes.Length)
    throw new ArgumentException("Modes must be distinct inference,training,tiled.");
var tileShapes = Option("--tiles", "16x32").Split(',').Select(text =>
{
    int[] dimensions = text.Split('x').Select(int.Parse).ToArray();
    if (dimensions.Length != 2 || dimensions.Any(value => value <= 0)) throw new ArgumentException("Tiles use MxN syntax.");
    return (Rows: dimensions[0], Columns: dimensions[1]);
}).ToArray();
var jsonOptions = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
DateTimeOffset started = DateTimeOffset.UtcNow;
var records = new List<object>();
var inputSources = new List<InputSource>();
string? model = options.TryGetValue("--model", out string? modelValue) ? Path.GetFullPath(modelValue) : null;
DateTime? modelModified = null;
long? modelBytes = null;
object? gpu = null;
string? modelSha256 = null;
Dictionary<string, string> measuredFiles = Fingerprints();
Dictionary<string, string> sourceFiles = Directory.Exists(Path.Combine(repo, "NNtrain.Arc"))
    ? Directory.GetFiles(Path.Combine(repo, "NNtrain.Arc", "Kernels"), "qwen35*.cl")
        .Concat(Directory.GetFiles(Path.Combine(repo, "NNtrain.Arc"), "*Iq2*.cs"))
        .Concat(Directory.GetFiles(Path.Combine(repo, "NNtrain.Core", "Modules"), "Qwen35Gpu*.cs"))
        .Concat(new[] { Path.Combine(repo, "NNtrain.Core", "Modules", "Qwen35ExecutionOptions.cs"),
            Path.Combine(repo, "NNtrain.Core", "Modules", "Qwen35QuantizedModel.cs"),
            Path.Combine(repo, "NNtrain.Core", "Modules", "Qwen35QuantizedModel.Prefill.cs") })
        .Concat(Directory.GetFiles(Path.Combine(repo, "tools", "iq2s-production-bench"), "*.cs"))
        .ToDictionary(path => Path.GetRelativePath(repo, path), HashFile) : [];

void Save(bool complete, string? error = null)
{
    object report = new
    {
        schemaVersion = 1, complete, error, startedUtc = started, completedUtc = complete ? (DateTimeOffset?)DateTimeOffset.UtcNow : null,
        commandArguments = args, device = gpu, measuredFileSha256 = measuredFiles, sourceFileSha256 = sourceFiles,
        direction = transpose ? "transpose" : "forward",
        operation = transpose ? "dX[M,K] = dY_FP32[M,N] * W_IQ2S[N,K]; FP32 output"
            : "Y[M,N] = X_FP32[M,K] * W_IQ2S[N,K]^T + bias_FP32[N]; FP32 output",
        scope = "Production projection operations with actual unquantized FP32 activations; not complete model forward/training throughput.",
        timing = new { samples, warmup, order = "Round robin; reverse variant order on odd rounds", gpuMetric = "Sum of OpenCL kernel event durations, excluding device-to-device copies",
            wallMetric = "Host Stopwatch from projection allocation/dispatch through synchronization, excluding profile dictionary copies",
            included = "Each call packs activation and weights required by its route, including temporary allocation/pool reuse and device-to-device row chunk copies in host wall timing; input/bias/encoded weights and output remain resident",
            excluded = "Program compilation, initial input uploads, CPU references, output readback, model disk reads, source/file hashing" },
        validation = new { cpuPointsRequested = cpuPoints, cpuTolerance = "abs <= max(1e-7, 2e-6 * sum(abs(products),abs(bias)))", allOutputL2Limit = 5e-5,
            outputGuards = "Every validation poisons output and two trailing guards with NaN", fallback,
            originalReference = transpose ? "Original training vec8 rows8 transpose plus FP32 split reducer; independent CPU double dot uses FP32-decoded IQ2 weights"
                : "Original inference GGUF BSLM with high/residual input packing; independent CPU double dot uses FP32-decoded IQ2 weights" },
        model = model is null ? null : new { path = model, bytes = modelBytes, lastWriteTimeUtc = modelModified, sha256 = modelSha256 },
        cases = records
    };
    File.WriteAllText(resultPath, JsonSerializer.Serialize(report, jsonOptions));
}

try
{
    if (model is null)
    {
        int input = Number("--input", 512), output = Number("--output-width", 65);
        if (input % 256 != 0) throw new ArgumentException("Input width must be a multiple of 256.");
        byte[] payload = ReferenceData.Synthetic(input, output, 831008 + input + output, stress);
        inputSources.Add(new("synthetic", input, output, payload, new { kind = "Deterministic valid synthetic IQ2_S weight blocks", seed = 831008 + input + output, stress, payloadSha256 = HashBytes(payload) }));
    }
    else
    {
        var fileInfo = new FileInfo(model); modelModified = fileInfo.LastWriteTimeUtc; modelBytes = fileInfo.Length;
        using var reader = new GgufReader(model);
        foreach (string requested in Option("--tensors", "attn_gate,ffn_gate,ffn_down").Split(','))
        {
            GgufTensorInfo tensor = requested.Contains('.') ? reader.GetTensor(requested)
                : reader.Tensors.FirstOrDefault(tensor => tensor.Type == 22 && tensor.Shape.Count == 2 && tensor.Name.EndsWith($".{requested}.weight", StringComparison.Ordinal))
                    ?? throw new InvalidDataException($"No IQ2_S tensor with suffix {requested}.");
            if (tensor.Type != 22 || tensor.Shape.Count != 2 || tensor.Shape[0] % 256 != 0)
                throw new InvalidDataException($"{tensor.Name} is not a two-dimensional IQ2_S matrix.");
            int input = checked((int)tensor.Shape[0]), output = checked((int)tensor.Shape[1]);
            byte[] payload = reader.ReadTensorBytes(tensor, checked(input / 256 * output * 82));
            inputSources.Add(new(tensor.Name, input, output, payload, new { kind = "Complete unmodified GGUF IQ2_S tensor", tensor = tensor.Name,
                type = tensor.Type, shape = tensor.Shape, tensor.Offset, bytes = payload.Length, payloadSha256 = HashBytes(payload) }));
        }
    }
    using var lane = new ArcExecutionLane(device, new()
    {
        Qwen35InferenceKernelsOnly = true, Qwen35TrainingKernels = true, Qwen35GgufBslmPrefill = true,
        CollectKernelTimings = true, BufferPoolBytes = 512L * 1024 * 1024, QueuedKernelLimit = 4096,
        BatchDispatch = true, ExperimentalOptimizationKernels = false, CacheKernelArguments = true, CacheProgramBinary = true,
        Qwen35NativeHalfScale = true
    });
    gpu = new { lane.Device.Index, lane.Device.Name, lane.Device.DriverVersion, lane.Device.GlobalMemoryBytes,
        lane.Device.MaximumAllocationBytes, lane.Device.MinimumSubgroupSize, lane.Device.SupportsXmx };
    Save(false);
    foreach (InputSource source in inputSources)
    foreach (int rows in rowCounts)
    {
        if (transpose)
        {
            if (source.Output % 16 != 0) throw new ArgumentException("Tiled transpose requires output-width divisible by 16; use e.g. 80 for synthetic edge checks.");
            records.Add(TransposeBenchmark.Run(lane, source, rows, samples, warmup, cpuPoints, tileShapes, panelColumns, scratchMiB, fallback));
            Save(false);
            continue;
        }
        int inputWidth = source.Input, outputWidth = source.Output, count = checked(rows * outputWidth);
        var random = new Random(771008 + rows + inputWidth + outputWidth);
        float[] values = Enumerable.Range(0, checked(rows * inputWidth)).Select(_ => random.NextSingle() * 4 - 2).ToArray();
        if (fallback) { values[0] = 100000; values[Math.Min(values.Length - 1, 8)] = -100000; }
        float[] biases = Enumerable.Range(0, outputWidth).Select(index => (index % 9 - 4) * .03f).ToArray();
        var reference = ReferenceData.Reference(values, source.Payload, biases, rows, inputWidth, outputWidth, cpuPoints);
        using ArcBuffer x = lane.Upload(values), weight = lane.UploadRaw(source.Payload), bias = lane.Upload(biases), output = lane.Allocate(count + 2);
        lane.Synchronize();
        Console.WriteLine($"case {source.Name}: rows={rows}, input={inputWidth}, output={outputWidth}, fallback={fallback}");
        void DirectInference(string kernel)
        {
            int elements = checked((rows + 7) / 8 * 8 * inputWidth);
            using ArcBuffer high = lane.AllocateBytes(checked(elements * 2)), low = lane.AllocateBytes(checked(elements * 2)), status = lane.Allocate(1);
            lane.Run("q35a_zero", 1, 0, status, 1);
            lane.Run(kernel.EndsWith('r') ? "q35l_prefill_xmx_pack_input_f16x2_rowmajor"
                : "q35l_prefill_xmx_pack_input_f16x2", elements, 0, x, high, low, status, rows, inputWidth);
            lane.Run2D(kernel, ((long)outputWidth + 63) / 64 * 16,
                ((long)rows + 127) / 128 * 16, 16, 16, high, low, weight, bias, output, rows, inputWidth, outputWidth, x, status);
        }
        void OriginalInference() => DirectInference("q35l_prefill_xmx_iq2_s_gguf_bslm");
        void OriginalTraining() => lane.Run("q35t_linear_iq2_s_rows4", ((((long)rows + 3) / 4 * outputWidth + 1) / 2) * 32,
            32, x, weight, bias, output, rows, inputWidth, outputWidth);
        var variants = new List<Variant>();
        if (modes.Contains("inference")) variants.Add(new("original-inference-gguf-bslm", OriginalInference));
        if (modes.Contains("training")) variants.Add(new("original-training-fp32-rows4", OriginalTraining));
        if (modes.Contains("tiled")) foreach (var tile in tileShapes)
            variants.Add(new($"tiled-{tile.Rows}x{tile.Columns}-panel{panelColumns}", () =>
                lane.Iq2TiledForward(x, weight, bias, output, rows, inputWidth, outputWidth, tile.Rows, tile.Columns, panelColumns)));
        if (Flag("--row-major"))
            variants.Add(new($"tiled-rowmajor-16x32-panel{panelColumns}", () =>
                lane.Iq2TiledForward(x, weight, bias, output, rows, inputWidth, outputWidth,
                    16, 32, panelColumns, rowMajor: true)));
        if (options.TryGetValue("--bslm-k", out string? bslmChoices))
            foreach (string blockK in bslmChoices.Split(','))
            {
                if (blockK is not ("64" or "128" or "32c" or "64c" or "128c" or "32r" or "64r" or "64n"))
                    throw new ArgumentException("Invalid BSLM K candidate.");
                variants.Add(new($"inference-bslm-k{blockK}", () =>
                    DirectInference($"q35l_prefill_xmx_iq2_s_gguf_bslm_k{blockK}")));
            }
        if (variants.Count == 0) throw new ArgumentException("No benchmark variants selected.");
        float[] poison = Enumerable.Repeat(float.NaN, count + 2).ToArray();
        float[] ReadCheckedOutput(Action run)
        {
            lane.WriteRaw(output, poison); lane.Synchronize(); run(); lane.Synchronize();
            var result = new float[count + 2]; lane.Read(output, result); return result;
        }
        float[] original = ReadCheckedOutput(OriginalInference);
        var originalValidation = ReferenceData.Validate(original, count, reference, null);
        float[]? scalarFallback = fallback ? ReadCheckedOutput(() => lane.Run("q35l_iq2_s_reference", count, 0,
            x, weight, bias, output, rows, inputWidth, outputWidth)) : null;
        foreach (Variant variant in variants)
        {
            float[] result = ReadCheckedOutput(variant.Run);
            variant.Validation = ReferenceData.Validate(result, count, reference, original,
                requireBitwise: variant.Name.StartsWith("inference-bslm-", StringComparison.Ordinal));
            if (scalarFallback is not null) variant.ScalarFallbackComparison = ReferenceData.Validate(result, count, reference, scalarFallback);
            Console.WriteLine($" validated {variant.Name}: CPU L2={variant.Validation.RelativeL2VsDouble:E4}, original L2={variant.Validation.RelativeL2VsOriginal:E4}");
        }
        for (int round = 0; round < warmup; round++) foreach (Variant variant in round % 2 == 0 ? variants : variants.AsEnumerable().Reverse())
        { variant.Run(); lane.Synchronize(); }
        for (int round = 0; round < samples; round++)
        {
            foreach (Variant variant in round % 2 == 0 ? variants : variants.AsEnumerable().Reverse())
            {
                var kernelBefore = new Dictionary<string, double>(lane.KernelTimings);
                double gpuBefore = lane.KernelMilliseconds;
                long allocations = lane.AllocationCount, poolHits = lane.PoolHits, launches = lane.KernelLaunchCount;
                long beforeH2D = lane.H2DBytes, beforeD2H = lane.D2HBytes;
                var watch = Stopwatch.StartNew(); variant.Run(); lane.Synchronize(); watch.Stop();
                double gpuMilliseconds = lane.KernelMilliseconds - gpuBefore;
                var byKernel = lane.KernelTimings.Where(pair => pair.Value - kernelBefore.GetValueOrDefault(pair.Key) > 0)
                    .ToDictionary(pair => pair.Key, pair => pair.Value - kernelBefore.GetValueOrDefault(pair.Key));
                if (lane.H2DBytes != beforeH2D || lane.D2HBytes != beforeD2H) throw new InvalidOperationException("A measured route performed host transfer.");
                if (!double.IsFinite(gpuMilliseconds) || gpuMilliseconds <= 0) throw new ArithmeticException("Missing positive GPU event duration.");
                variant.Samples.Add(new(round, gpuMilliseconds, watch.Elapsed.TotalMilliseconds, byKernel,
                    lane.AllocationCount - allocations, lane.PoolHits - poolHits, lane.KernelLaunchCount - launches));
            }
        }
        foreach (Variant variant in variants)
        {
            // Revalidate after all timed samples, outside the measured interval.
            float[] result = ReadCheckedOutput(variant.Run);
            variant.PostTimingValidation = ReferenceData.Validate(result, count, reference, original,
                requireBitwise: variant.Name.StartsWith("inference-bslm-", StringComparison.Ordinal));
        }
        var summaries = variants.Select(variant => new { variant.Name, variant.Validation, variant.PostTimingValidation, variant.ScalarFallbackComparison,
            gpuMilliseconds = Stats.Of(variant.Samples.Select(sample => sample.GpuMilliseconds)),
            wallMilliseconds = Stats.Of(variant.Samples.Select(sample => sample.WallMilliseconds)), samples = variant.Samples }).ToArray();
        records.Add(new { name = source.Name, rows, inputWidth, outputWidth, source = source.Provenance,
            activation = new { type = "FP32", distribution = "Deterministic uniform [-2,2), optional two ±100000 range-fallback values", seed = 771008 + rows + inputWidth + outputWidth,
                sha256 = HashBytes(System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan())), additionalQuantization = false },
            bias = "Nonzero FP32, (column % 9 - 4) * 0.03", originalValidation,
            peakLaneAllocatedBytes = lane.PeakAllocatedBytes, variants = summaries });
        Save(false);
        foreach (var summary in summaries) Console.WriteLine($" {summary.Name}: GPU median {summary.gpuMilliseconds.Median:F6} ms, wall median {summary.wallMilliseconds.Median:F6} ms");
    }
    if (model is not null)
    {
        var file = new FileInfo(model);
        if (file.Length != modelBytes || file.LastWriteTimeUtc != modelModified) throw new IOException("Model file changed during benchmark.");
        if (Flag("--hash-model")) modelSha256 = HashFile(model);
    }
    var after = Fingerprints();
    if (measuredFiles.Count != after.Count || measuredFiles.Any(pair => !after.TryGetValue(pair.Key, out string? hash) || hash != pair.Value))
        throw new IOException("A measured assembly or CPU lookup table changed during benchmark.");
    Save(true);
    Console.WriteLine($"Complete: {resultPath}");
}
catch (Exception exception)
{
    Save(false, exception.ToString());
    Console.Error.WriteLine(exception);
    Environment.ExitCode = 1;
}

static string HashFile(string path) { using Stream stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
static string HashBytes(ReadOnlySpan<byte> data) => Convert.ToHexString(SHA256.HashData(data)).ToLowerInvariant();
static Dictionary<string, string> Fingerprints() => Directory.GetFiles(AppContext.BaseDirectory, "NNtrain*.dll")
    .Concat([Path.Combine(AppContext.BaseDirectory, "qwen35_iq.cl")])
    .ToDictionary(path => Path.GetFileName(path)!, HashFile, StringComparer.Ordinal);

internal sealed record InputSource(string Name, int Input, int Output, byte[] Payload, object Provenance);
internal sealed class Variant(string name, Action run)
{
    public string Name { get; } = name;
    public Action Run { get; } = run;
    public ReferenceData.Validation? Validation { get; set; }
    public ReferenceData.Validation? PostTimingValidation { get; set; }
    public ReferenceData.Validation? ScalarFallbackComparison { get; set; }
    public List<Sample> Samples { get; } = [];
}
internal sealed record Sample(int Round, double GpuMilliseconds, double WallMilliseconds,
    Dictionary<string, double> KernelMilliseconds, long LogicalAllocations, long PoolHits, long KernelLaunches);
internal sealed record Stats(int Count, double Median, double Mean, double StandardDeviation, double Minimum, double Maximum)
{
    public static Stats Of(IEnumerable<double> numbers)
    {
        double[] values = numbers.Order().ToArray();
        double mean = values.Average();
        return new(values.Length, values.Length % 2 == 0 ? (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2 : values[values.Length / 2],
            mean, Math.Sqrt(values.Select(value => (value - mean) * (value - mean)).Average()), values[0], values[^1]);
    }
}
