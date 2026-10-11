using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using Iq2sProductionBench;
using NNtrain;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

// Actual GGUF IQ2_S weights, one FP32 query row, and complete output checks.
// This executable is independent of the production model's routing options.
var cli = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    string name = args[i];
    if (!name.StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"Unexpected argument {name}");
    int equal = name.IndexOf('=');
    if (equal >= 0) cli.Add(name[..equal], name[(equal + 1)..]);
    else if (name == "--hash-model") cli.Add(name, "true");
    else cli.Add(name, ++i < args.Length ? args[i] : throw new ArgumentException($"Missing value for {name}"));
}
string[] known = ["--model", "--output", "--tensors", "--device", "--samples", "--warmup", "--iterations",
    "--pack-samples", "--pack-warmup", "--profile", "--workgroup", "--cpu-points", "--variants", "--repo", "--hash-model"];
if (cli.Keys.Any(key => !known.Contains(key))) throw new ArgumentException("Unknown command-line option.");
string Option(string key, string fallback) => cli.GetValueOrDefault(key, fallback);
string Required(string key) => cli.GetValueOrDefault(key) ?? throw new ArgumentException($"Missing {key}");
int Number(string key, int fallback, int minimum = 1) => int.TryParse(Option(key, fallback.ToString(CultureInfo.InvariantCulture)), out int value) && value >= minimum
    ? value : throw new ArgumentException($"Invalid {key}");
string modelPath = Path.GetFullPath(Required("--model"));
string directory = Path.GetFullPath(Required("--output"));
string repo = Path.GetFullPath(Option("--repo", Environment.CurrentDirectory));
string resultPath = Path.Combine(directory, "results.json");
if (File.Exists(resultPath)) throw new IOException($"Refusing to overwrite {resultPath}");
int deviceIndex = Number("--device", 0, 0), samples = Number("--samples", 9), warmup = Number("--warmup", 3, 0);
int iterations = Number("--iterations", 16), packSamples = Number("--pack-samples", 9), packWarmup = Number("--pack-warmup", 3, 0);
int cpuPoints = Number("--cpu-points", 256, 4), workgroup = Number("--workgroup", 32);
int profileValue = Number("--profile", 0, 0);
if (profileValue > 1 || workgroup is not (32 or 64 or 128)) throw new ArgumentException("Use --profile 0/1 and --workgroup 32/64/128.");
bool profile = profileValue != 0, hashModel = bool.Parse(Option("--hash-model", "false"));
string[] candidates = Option("--variants", "packed82-sg16,packed82-pair,signed-lookup-pair").Split(',', StringSplitOptions.RemoveEmptyEntries);
if (candidates.Length == 0 || candidates.Distinct().Count() != candidates.Length
    || candidates.Any(name => name is not ("packed82-sg16" or "packed82-pair" or "signed-lookup-pair")))
    throw new ArgumentException("--variants accepts distinct packed82-sg16,packed82-pair,signed-lookup-pair.");
string[] tensors = Option("--tensors", "attn_gate,ffn_gate,ffn_down").Split(',', StringSplitOptions.RemoveEmptyEntries);
if (tensors.Length == 0 || tensors.Distinct().Count() != tensors.Length) throw new ArgumentException("Specify distinct tensor names or suffixes.");
Directory.CreateDirectory(directory);
DateTimeOffset started = DateTimeOffset.UtcNow;
var records = new List<object>();
var json = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
Dictionary<string, string> measuredFiles = Fingerprints();
Dictionary<string, string> sourceFiles = SourceFingerprints(repo);
var info = new FileInfo(modelPath);
long modelBytes = info.Length;
DateTime modelModified = info.LastWriteTimeUtc;
string? modelSha256 = null;
object? device = null;
object? signedLookupPreparation = null;
const string PackKernel = "q35l_iq2_s_pack82";
const string LookupBuildKernel = "q35l_iq2_s_build_signed_lookup";
const string LookupDecodeKernel = "q35l_iq2_s_sg16_pair_signed_lookup";

void Save(bool complete, string? error = null) => File.WriteAllText(resultPath, JsonSerializer.Serialize(new
{
    schemaVersion = 1, complete, error, startedUtc = started,
    completedUtc = complete ? (DateTimeOffset?)DateTimeOffset.UtcNow : null,
    commandArguments = args,
    operation = "Single-token FP32 x complete IQ2_S GGUF tensor -> FP32 projection, with nonzero FP32 bias",
    scope = "One resident projection only. This is not full-model generation tok/s. Packing is reported separately from repeated decode.",
    model = new { path = modelPath, bytes = modelBytes, lastWriteTimeUtc = modelModified, sha256 = modelSha256 },
    device, measuredFileSha256 = measuredFiles, sourceFileSha256 = sourceFiles, signedLookupPreparation,
    layout = "Exactly 82 bytes per original 256-value block in one allocation: all 64-byte ushort low-index/sign planes, all 16-byte pre-folded high-index/scale planes, then all 2-byte raw FP16 coefficients. No weight expansion or requantization.",
    timing = new { profile, samples, warmup, iterations, packSamples, packWarmup, workgroup,
        order = "Round robin; reverse route order on alternate warmup/sample rounds",
        decodeWall = "Per-call batch average: dispatch iterations then synchronize once; input, encoded/packed weight, bias, output already resident",
        decodeGpu = profile ? "Per-call sum of kernel event durations; separate diagnostic profile mode" : "Not collected; use a separate --profile 1 run for kernel evidence",
        excludedFromDecode = "Compilation, initial allocation/upload, weight packing, CPU reference, poison/readback validation, model reads and hashing",
        packing = "First preparation includes allocation+pack+sync after kernel handles are created; repeated pack statistics use the existing output allocation. CPU pack is one allocation+conversion measurement.",
        comparison = "Only compare primary wall times from unprofiled runs; profile wall times are diagnostics." },
    validation = new { allOutputsBitwiseEqualRequired = true, cpuPointsRequested = cpuPoints,
        cpuTolerance = "max(1e-7, 2e-6 * sum(abs(products),abs(bias)))", guards = "Two NaN output guards",
        packedBytes = "GPU packed bytes must equal independent CPU packing, and CPU unpacking must reproduce every original GGUF byte",
        productionBaseline = "q35l_iq2_s_sg16_pair", rawSingleBaseline = "q35l_iq2_s_sg16" },
    cases = records
}, json));

try
{
    Save(false);
    using var reader = new GgufReader(modelPath);
    GgufTensorInfo[] selected = tensors.Select(requested => requested.Contains('.') ? reader.GetTensor(requested)
        : reader.Tensors.FirstOrDefault(tensor => tensor.Type == 22 && tensor.Shape.Count == 2
            && tensor.Name.EndsWith($".{requested}.weight", StringComparison.Ordinal))
            ?? throw new InvalidDataException($"No IQ2_S tensor with suffix {requested}.")).ToArray();
    if (selected.Select(t => t.Name).Distinct().Count() != selected.Length) throw new ArgumentException("Tensor selections overlap.");
    if (selected.Any(t => t.Type != 22 || t.Shape.Count != 2 || t.Shape[0] == 0 || t.Shape[0] % 256 != 0 || t.Shape[1] == 0))
        throw new InvalidDataException("Each selected tensor must be a complete, nonempty IQ2_S matrix.");
    using var lane = new ArcExecutionLane(deviceIndex, new()
    {
        Qwen35InferenceKernelsOnly = true, CollectKernelTimings = profile,
        CacheKernelArguments = true, CacheProgramBinary = true, BatchDispatch = true,
        Qwen35NativeHalfScale = true, Qwen35ProjectionWorkgroupSize = workgroup,
        BufferPoolBytes = 0, DeferredReleaseBytes = 0, QueuedKernelLimit = 4096
    });
    if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16
        || !lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups"))
        throw new NotSupportedException("Intel Arc SG16 is required.");
    device = new { lane.Device.Index, lane.Device.Name, lane.Device.DriverVersion, lane.Device.GlobalMemoryBytes,
        lane.Device.MaximumAllocationBytes, lane.Device.MinimumSubgroupSize, lane.Device.SupportsXmx };
    foreach (string kernel in new[] { PackKernel, "q35l_iq2_s_sg16", "q35l_iq2_s_sg16_pair",
        "q35l_iq2_s_packed82_sg16", "q35l_iq2_s_packed82_sg16_pair", LookupBuildKernel, LookupDecodeKernel }) _ = lane.GetKernelResources(kernel);
    lane.Synchronize();
    long lookupStart = Stopwatch.GetTimestamp();
    using ArcBuffer? signedLookup = candidates.Contains("signed-lookup-pair") ? lane.AllocateBytes(131072) : null;
    byte[]? lookupOriginal = null;
    if (signedLookup is not null)
    {
        void BuildLookup() => lane.Run(LookupBuildKernel, 16384, 0, signedLookup);
        BuildLookup(); lane.Synchronize();
        double firstPreparationMilliseconds = Stopwatch.GetElapsedTime(lookupStart).TotalMilliseconds;
        lookupOriginal = new byte[131072];
        lane.ReadRaw(signedLookup, lookupOriginal);
        for (int round = 0; round < packWarmup; round++) { BuildLookup(); lane.Synchronize(); }
        var lookupMeasurements = new List<Sample>();
        for (int round = 0; round < packSamples; round++) lookupMeasurements.Add(Measure(lane, BuildLookup, LookupBuildKernel, round, 1, profile));
        var rebuilt = new byte[131072]; lane.ReadRaw(signedLookup, rebuilt);
        if (!rebuilt.AsSpan().SequenceEqual(lookupOriginal)) throw new ArithmeticException("Repeated lookup initialization changed table bytes.");
        signedLookupPreparation = new
        {
            bytes = signedLookup.ByteLength, firstPreparationMilliseconds,
            repeatedWallMilliseconds = Stats.Of(lookupMeasurements.Select(s => s.WallMilliseconds)),
            repeatedGpuMilliseconds = profile ? Stats.Of(lookupMeasurements.Select(s => s.GpuMilliseconds!.Value)) : null,
            sha256 = HashBytes(lookupOriginal), samples = lookupMeasurements,
            scope = "One shared 128 KiB lookup allocation for all tensors; original 82-byte GGUF weights remain unchanged"
        };
    }
    Save(false);
    foreach (GgufTensorInfo tensor in selected)
    {
        int inputWidth = checked((int)tensor.Shape[0]), outputWidth = checked((int)tensor.Shape[1]);
        int blockCount = checked(inputWidth / 256 * outputWidth), bytes = checked(blockCount * 82);
        byte[] raw = reader.ReadTensorBytes(tensor, bytes);
        long cpuStart = Stopwatch.GetTimestamp();
        byte[] expectedPacked = Pack82(raw);
        double cpuPackMilliseconds = Stopwatch.GetElapsedTime(cpuStart).TotalMilliseconds;
        if (!Unpack82(expectedPacked).AsSpan().SequenceEqual(raw)) throw new ArithmeticException("CPU packing did not preserve all GGUF bytes.");
        int seed = 801_008 + inputWidth + outputWidth;
        var random = new Random(seed);
        float[] input = Enumerable.Range(0, inputWidth).Select(_ => random.NextSingle() * 2 - 1).ToArray();
        float[] biases = Enumerable.Range(0, outputWidth).Select(i => (i % 9 - 4) * .03f).ToArray();
        ReferenceData.Point[] points = ReferenceData.Reference(input, raw, biases, 1, inputWidth, outputWidth, cpuPoints);
        using ArcBuffer x = lane.Upload(input), encoded = lane.UploadRaw(raw), bias = lane.Upload(biases);
        using ArcBuffer result = lane.Allocate(checked(outputWidth + 2));
        lane.Synchronize();
        long liveBeforePack = lane.AllocatedBytes, requestedBeforePack = lane.RequestedBytes;
        long preparationStart = Stopwatch.GetTimestamp();
        using ArcBuffer packed = lane.AllocateBytes(bytes);
        void Pack() => lane.Run(PackKernel, (long)blockCount * 16, 0, encoded, packed, blockCount);
        Pack(); lane.Synchronize();
        double firstPreparationMilliseconds = Stopwatch.GetElapsedTime(preparationStart).TotalMilliseconds;
        long extraLiveBytes = lane.AllocatedBytes - liveBeforePack;
        if (extraLiveBytes != bytes || lane.RequestedBytes - requestedBeforePack != bytes)
            throw new InvalidOperationException("Unexpected packed-weight allocation size.");
        var packedReadback = new byte[bytes];
        lane.ReadRaw(packed, packedReadback);
        if (!packedReadback.AsSpan().SequenceEqual(expectedPacked)) throw new ArithmeticException("GPU packing differs from independent CPU layout.");
        if (!Unpack82(packedReadback).AsSpan().SequenceEqual(raw)) throw new ArithmeticException("GPU packing lost original GGUF bytes.");
        for (int round = 0; round < packWarmup; round++) { Pack(); lane.Synchronize(); }
        var packMeasurements = new List<Sample>();
        for (int round = 0; round < packSamples; round++) packMeasurements.Add(Measure(lane, Pack, PackKernel, round, 1, profile));
        lane.ReadRaw(packed, packedReadback);
        if (!packedReadback.AsSpan().SequenceEqual(expectedPacked)) throw new ArithmeticException("Repeated packing changed packed bytes.");

        void Run(string kernel, ArcBuffer weights, bool pair)
        {
            long groups = pair ? ((long)outputWidth + 1) / 2 : outputWidth;
            int group = pair ? 32 : workgroup;
            long global = (groups + group / 16 - 1) / (group / 16) * group;
            lane.Run(kernel, global, group, x, weights, bias, result, 1, inputWidth, outputWidth);
        }
        var routes = new List<Route>
        {
            new("original-pair", "q35l_iq2_s_sg16_pair", () => Run("q35l_iq2_s_sg16_pair", encoded, true)),
            new("original-sg16", "q35l_iq2_s_sg16", () => Run("q35l_iq2_s_sg16", encoded, false))
        };
        foreach (string candidate in candidates)
        {
            if (candidate == "signed-lookup-pair")
            {
                long lookupGlobal = (((long)outputWidth + 1) / 2 + 1) / 2 * 32;
                routes.Add(new(candidate, LookupDecodeKernel, () => lane.Run(LookupDecodeKernel, lookupGlobal, 32,
                    x, encoded, bias, result, 1, inputWidth, outputWidth, signedLookup!)));
                continue;
            }
            bool pair = candidate == "packed82-pair";
            string kernel = pair ? "q35l_iq2_s_packed82_sg16_pair" : "q35l_iq2_s_packed82_sg16";
            routes.Add(new(candidate, kernel, () => Run(kernel, packed, pair)));
        }
        float[] poison = Enumerable.Repeat(float.NaN, outputWidth + 2).ToArray();
        float[] ReadOutput(Action action)
        {
            lane.Write(result, poison); lane.Synchronize();
            action(); lane.Synchronize();
            var values = new float[outputWidth + 2]; lane.Read(result, values); return values;
        }
        float[] baseline = ReadOutput(routes[0].Run);
        ReferenceData.Validation baselineValidation = ReferenceData.Validate(baseline, outputWidth, points, null);
        foreach (Route route in routes)
            route.Validation = ReferenceData.Validate(ReadOutput(route.Run), outputWidth, points, baseline, requireBitwise: true);
        Console.WriteLine($"Validated all bits: {tensor.Name} K={inputWidth} N={outputWidth}; raw={bytes:N0}B packed={packed.ByteLength:N0}B");
        for (int round = 0; round < warmup; round++)
            foreach (Route route in (round & 1) == 0 ? routes : routes.AsEnumerable().Reverse())
            {
                for (int iteration = 0; iteration < iterations; iteration++) route.Run();
                lane.Synchronize();
            }
        for (int round = 0; round < samples; round++)
            foreach (Route route in (round & 1) == 0 ? routes : routes.AsEnumerable().Reverse())
                route.Samples.Add(Measure(lane, route.Run, route.Kernel, round, iterations, profile));
        foreach (Route route in routes)
            route.PostTimingValidation = ReferenceData.Validate(ReadOutput(route.Run), outputWidth, points, baseline, requireBitwise: true);
        lane.ReadRaw(packed, packedReadback);
        if (!packedReadback.AsSpan().SequenceEqual(expectedPacked)) throw new ArithmeticException("Decode changed packed weight bytes.");
        if (signedLookup is not null)
        {
            var afterLookup = new byte[131072]; lane.ReadRaw(signedLookup, afterLookup);
            if (!afterLookup.AsSpan().SequenceEqual(lookupOriginal)) throw new ArithmeticException("Decode changed signed lookup table bytes.");
        }
        double baselineMedian = Stats.Of(routes[0].Samples.Select(s => s.WallMilliseconds)).Median;
        double singleMedian = Stats.Of(routes[1].Samples.Select(s => s.WallMilliseconds)).Median;
        var summaries = routes.Select(route => new
        {
            route.Name, route.Kernel, route.Validation, route.PostTimingValidation,
            weightStorage = route.Name.StartsWith("packed82", StringComparison.Ordinal) ? "Lossless packed82" : "Original GGUF IQ2_S",
            additionalSharedLookupBytes = route.Name == "signed-lookup-pair" ? 131072 : 0,
            wallMilliseconds = Stats.Of(route.Samples.Select(s => s.WallMilliseconds)),
            gpuMilliseconds = profile ? Stats.Of(route.Samples.Select(s => s.GpuMilliseconds!.Value)) : null,
            versusProductionPairWallSpeedup = baselineMedian / Stats.Of(route.Samples.Select(s => s.WallMilliseconds)).Median,
            versusOriginalSingleWallSpeedup = singleMedian / Stats.Of(route.Samples.Select(s => s.WallMilliseconds)).Median,
            samples = route.Samples
        }).ToArray();
        records.Add(new
        {
            name = tensor.Name, rows = 1, inputWidth, outputWidth,
            source = new { kind = "Complete unmodified GGUF IQ2_S tensor", type = tensor.Type, shape = tensor.Shape,
                tensor.Offset, bytes, payloadSha256 = HashBytes(raw) },
            activation = new { type = "FP32", seed, distribution = "Deterministic uniform [-1,1)",
                sha256 = HashBytes(MemoryMarshal.AsBytes(input.AsSpan())), additionalQuantization = false },
            bias = new { type = "FP32", expression = "(column % 9 - 4) * 0.03", sha256 = HashBytes(MemoryMarshal.AsBytes(biases.AsSpan())) },
            packing = new { layout = "packed82", cpuPackMilliseconds, firstPreparationMilliseconds,
                repeatedWallMilliseconds = Stats.Of(packMeasurements.Select(s => s.WallMilliseconds)),
                repeatedGpuMilliseconds = profile ? Stats.Of(packMeasurements.Select(s => s.GpuMilliseconds!.Value)) : null,
                packedSha256 = HashBytes(expectedPacked), allPackedBytesMatchIndependentCpu = true,
                allRawBytesRecovered = true, packedWeightsUnchangedAfterDecode = true, samples = packMeasurements },
            memory = new { rawWeightBytes = bytes, packedWeightBytes = packed.ByteLength,
                packedToRawRatio = (double)packed.ByteLength / bytes,
                additionalResidentBytesWhileKeepingRaw = extraLiveBytes,
                replacementWeightBytesDelta = packed.ByteLength - bytes,
                simultaneouslyResidentWeightBytes = encoded.ByteLength + packed.ByteLength,
                sharedLookupBytes = signedLookup?.ByteLength ?? 0,
                liveBytesWithAllOperands = lane.AllocatedBytes,
                note = "Both raw and packed weights stay resident for the comparison. Replacing raw storage saves that duplicate only if the model no longer needs raw weights for another route." },
            baselineValidation, variants = summaries
        });
        Save(false);
        foreach (var summary in summaries)
            Console.WriteLine($" {summary.Name}: wall {summary.wallMilliseconds.Median:F6} ms, vs original pair {summary.versusProductionPairWallSpeedup:F3}x");
    }
    info.Refresh();
    if (info.Length != modelBytes || info.LastWriteTimeUtc != modelModified) throw new IOException("Model changed while benchmarking.");
    if (hashModel) modelSha256 = HashFile(modelPath);
    if (!Same(measuredFiles, Fingerprints()) || !Same(sourceFiles, SourceFingerprints(repo)))
        throw new IOException("A measured binary or source changed while benchmarking.");
    Save(true);
    Console.WriteLine($"Complete: {resultPath}");
}
catch (Exception exception)
{
    Save(false, exception.ToString());
    Console.Error.WriteLine(exception);
    Environment.ExitCode = 1;
}

static Sample Measure(ArcExecutionLane lane, Action action, string kernel, int round, int iterations, bool profile)
{
    lane.Synchronize();
    Dictionary<string, double>? before = profile ? new(lane.KernelTimings) : null;
    double gpu = lane.KernelMilliseconds;
    long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes, requested = lane.RequestedBytes;
    long allocated = lane.AllocationCount, launches = lane.KernelLaunchCount;
    long start = Stopwatch.GetTimestamp();
    for (int i = 0; i < iterations; i++) action();
    lane.Synchronize();
    double batchWall = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
    if (uploaded != lane.H2DBytes || downloaded != lane.D2HBytes || requested != lane.RequestedBytes || allocated != lane.AllocationCount)
        throw new InvalidOperationException("A resident measurement transferred host data or allocated a buffer.");
    if (lane.KernelLaunchCount - launches != iterations) throw new InvalidOperationException("Unexpected measured kernel launch count.");
    Dictionary<string, double> kernels = profile ? lane.KernelTimings.Where(p => p.Value - before!.GetValueOrDefault(p.Key) > 0)
        .ToDictionary(p => p.Key, p => (p.Value - before!.GetValueOrDefault(p.Key)) / iterations) : [];
    double? gpuPerCall = profile ? (lane.KernelMilliseconds - gpu) / iterations : null;
    if (profile && (gpuPerCall is not > 0 || !kernels.TryGetValue(kernel, out double duration) || duration <= 0))
        throw new InvalidOperationException("Required kernel has no positive profiling evidence.");
    return new(round, iterations, batchWall / iterations, batchWall, gpuPerCall, kernels, 0, 0, lane.KernelLaunchCount - launches);
}

static byte[] Pack82(byte[] raw)
{
    if (raw.Length == 0 || raw.Length % 82 != 0) throw new ArgumentException("Invalid IQ2_S payload.");
    int blocks = raw.Length / 82;
    var packed = new byte[raw.Length];
    for (int b = 0; b < blocks; b++)
    {
        int source = b * 82;
        for (int octet = 0; octet < 32; octet++)
            BinaryPrimitives.WriteUInt16LittleEndian(packed.AsSpan(b * 64 + octet * 2, 2),
                (ushort)(raw[source + 2 + octet] | raw[source + 34 + octet] << 8));
        for (int pair = 0; pair < 16; pair++)
        {
            int group = pair >> 1, shift = (pair & 1) * 4;
            int high = (raw[source + 66 + group] >> shift) & 15;
            int scale = (raw[source + 74 + group] >> shift) & 15;
            packed[blocks * 64 + b * 16 + pair] = (byte)(high | scale << 4);
        }
        packed[blocks * 80 + b * 2] = raw[source];
        packed[blocks * 80 + b * 2 + 1] = raw[source + 1];
    }
    return packed;
}

static byte[] Unpack82(byte[] packed)
{
    int blocks = packed.Length / 82;
    var raw = new byte[packed.Length];
    for (int b = 0; b < blocks; b++)
    {
        int target = b * 82;
        for (int octet = 0; octet < 32; octet++)
        {
            ushort code = BinaryPrimitives.ReadUInt16LittleEndian(packed.AsSpan(b * 64 + octet * 2, 2));
            raw[target + 2 + octet] = (byte)code;
            raw[target + 34 + octet] = (byte)(code >> 8);
        }
        for (int group = 0; group < 8; group++)
        {
            int even = packed[blocks * 64 + b * 16 + group * 2];
            int odd = packed[blocks * 64 + b * 16 + group * 2 + 1];
            raw[target + 66 + group] = (byte)((even & 15) | (odd & 15) << 4);
            raw[target + 74 + group] = (byte)((even >> 4) | (odd & 240));
        }
        raw[target] = packed[blocks * 80 + b * 2];
        raw[target + 1] = packed[blocks * 80 + b * 2 + 1];
    }
    return raw;
}

static string HashFile(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
static string HashBytes(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
static Dictionary<string, string> Fingerprints() => Directory.GetFiles(AppContext.BaseDirectory, "NNtrain*.dll")
    .Concat([Path.Combine(AppContext.BaseDirectory, "qwen35_iq.cl")])
    .ToDictionary(path => Path.GetFileName(path)!, HashFile, StringComparer.Ordinal);
static Dictionary<string, string> SourceFingerprints(string repo)
{
    string kernels = Path.Combine(repo, "NNtrain.Arc", "Kernels");
    string core = Path.Combine(repo, "NNtrain.Core", "Modules");
    return Directory.GetFiles(kernels, "qwen35_iq*.cl")
        .Concat([Path.Combine(kernels, "qwen35_projection_pair.cl")])
        .Concat(Directory.GetFiles(core, "Qwen35*.cs"))
        .Concat(Directory.GetFiles(Path.Combine(repo, "tools", "iq2s-packed-decode-bench"), "*.*")
            .Where(path => Path.GetExtension(path) is ".cs" or ".csproj"))
        .Concat([Path.Combine(repo, "tools", "iq2s-production-bench", "ReferenceData.cs")])
        .ToDictionary(path => Path.GetRelativePath(repo, path), HashFile, StringComparer.Ordinal);
}
static bool Same(Dictionary<string, string> before, Dictionary<string, string> after)
    => before.Count == after.Count && before.All(p => after.TryGetValue(p.Key, out string? value) && value == p.Value);

internal sealed class Route(string name, string kernel, Action run)
{
    internal string Name { get; } = name;
    internal string Kernel { get; } = kernel;
    internal Action Run { get; } = run;
    internal ReferenceData.Validation? Validation { get; set; }
    internal ReferenceData.Validation? PostTimingValidation { get; set; }
    internal List<Sample> Samples { get; } = [];
}
internal sealed record Sample(int Round, int Iterations, double WallMilliseconds, double BatchWallMilliseconds,
    double? GpuMilliseconds, Dictionary<string, double> KernelMilliseconds,
    long HostTransferBytes, long LogicalAllocationBytes, long KernelLaunches);
internal sealed record Stats(int Count, double Median, double Mean, double StandardDeviation, double Minimum, double Maximum)
{
    internal static Stats Of(IEnumerable<double> numbers)
    {
        double[] values = numbers.Order().ToArray();
        if (values.Length == 0 || values.Any(value => !double.IsFinite(value) || value <= 0))
            throw new ArithmeticException("Timings must be finite and positive.");
        double mean = values.Average();
        return new(values.Length, values.Length % 2 == 0 ? (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2 : values[values.Length / 2],
            mean, Math.Sqrt(values.Average(value => (value - mean) * (value - mean))), values[0], values[^1]);
    }
}
