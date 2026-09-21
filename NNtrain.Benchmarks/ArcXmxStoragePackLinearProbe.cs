using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcXmxStoragePackLinearProbe
{
    private readonly record struct Plan(string Suffix, int Rows, int Depth, int Workgroup, bool VectorOnly = false);
    private static readonly Plan[] Plans =
    [
        new("reference", 0, 0, 0),
        new("c32", 32, 32, 256), new("c64", 32, 64, 256),
        new("c128", 32, 128, 256), new("r16c128", 16, 128, 256),
        new("c64w128", 32, 64, 128), new("vec4", 0, 0, 0, true)
    ];
    private sealed class Fixture : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly string Format;
        internal readonly int M, K, Block, Offset, GateOffset, PanelElements;
        internal readonly bool Gated;
        internal readonly ArcBuffer Source, Scales, Gate;
        internal Fixture(ArcExecutionLane lane, string format, int m, int k, int block, int offset, bool gated,
            bool exceptional, Array? sourceOverride = null)
        {
            Lane = lane; Format = format; M = m; K = k; Block = block; Offset = offset; Gated = gated;
            GateOffset = 5; PanelElements = checked(((m + 7) / 8) * ((k + 15) / 16) * 128);
            int length = checked(offset + m * k + 8);
            Source = lane.UploadRaw(sourceOverride ?? BuildSource(format, length, exceptional));
            float[] scales = Enumerable.Range(0, (length + block - 1) / block)
                .Select(i => exceptional ? ExceptionalScale(i) : (i % 13 + 1) / 8192f).ToArray();
            Scales = lane.Upload(scales);
            float[] gate = Enumerable.Range(0, GateOffset + m * k + 8).Select(i => !gated ? 1f : (i % 7) switch
            {
                0 => -1f, 1 => 0f, 2 => -0f, 3 => float.NaN, 4 => float.PositiveInfinity,
                5 => float.NegativeInfinity, _ => 1f
            }).ToArray();
            Gate = lane.Upload(gate);
        }
        internal string Kernel(Plan plan) => plan.Suffix == "reference"
            ? $"xmx_storage_pack_a_{Format}" : $"xmx_storage_pack_a_linear_{plan.Suffix}_{Format}";
        internal void Dispatch(Plan plan, ArcBuffer output)
        {
            object[] args = [Source, Scales, Gate, output, M, K, 0, Block, Gated ? 1 : 0, Offset, GateOffset];
            if (plan.Suffix == "reference") Lane.Run(Kernel(plan), PanelElements, 0, args);
            else if (plan.VectorOnly) Lane.Run(Kernel(plan), PanelElements / 4, 0, args);
            else Lane.Run2D(Kernel(plan), ((M + (long)plan.Rows - 1) / plan.Rows) * plan.Workgroup,
                (K + (long)plan.Depth - 1) / plan.Depth, plan.Workgroup, 1, args);
        }
        internal ushort[] Validate(Plan plan, ushort[]? expected)
        {
            using var output = Lane.UploadRaw(Enumerable.Repeat((ushort)0x1234, PanelElements).ToArray());
            long h2d = Lane.H2DBytes, d2h = Lane.D2HBytes;
            Dispatch(plan, output);
            if (Lane.H2DBytes != h2d || Lane.D2HBytes != d2h) throw new InvalidOperationException("Packing transferred data to/from the host.");
            ushort[] packed = new ushort[PanelElements]; Lane.ReadRaw(output, packed);
            if (expected is not null)
                for (int i = 0; i < packed.Length; i++)
                    if (packed[i] != expected[i]) throw new InvalidOperationException($"{Kernel(plan)} {M}x{K}, block={Block}, offset={Offset}, gated={Gated}, panel[{i}]: {expected[i]:x4} != {packed[i]:x4}.");
            return packed;
        }
        public void Dispose() { Gate.Dispose(); Scales.Dispose(); Source.Dispose(); }
    }

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane();
        if (!lane.Options.XmxMatrices || !lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("Storage packing candidates require SG16 Intel XMX.");
        int validationCases = 0;
        foreach (string format in new[] { "f32", "bf16", "bfp8" })
        foreach (var (m, k) in new[] { (9, 17), (33, 65), (67, 129), (137, 67) })
        foreach (int offset in new[] { 0, 7 })
        foreach (bool gated in new[] { false, true })
        foreach (int block in format == "bfp8" ? new[] { 3, 32, 128 } : new[] { 32 })
        {
            using var fixture = new Fixture(lane, format, m, k, block, offset, gated, exceptional: true);
            ushort[] expected = fixture.Validate(Plans[0], null);
            foreach (var plan in Plans.Skip(1)) fixture.Validate(plan, expected);
            validationCases++;
        }
        // Exhaustively cover every BF16 bit pattern (including signaling NaNs)
        // and FP32 RNE ties around all 65536 BF16 prefixes.
        using (var fixture = new Fixture(lane, "bf16", 256, 256, 32, 0, false, exceptional: false,
            sourceOverride: Enumerable.Range(0, 65536).Select(i => (ushort)i).ToArray()))
        {
            ushort[] expected = fixture.Validate(Plans[0], null);
            foreach (var plan in Plans.Skip(1)) fixture.Validate(plan, expected);
            validationCases++;
        }
        int[] tails = [0, 1, 0x7fff, 0x8000, 0x8001, 0xffff];
        int[] patterns = Enumerable.Range(0, 65536).SelectMany(prefix => tails.Select(tail => unchecked((prefix << 16) | tail))).ToArray();
        using (var fixture = new Fixture(lane, "f32", 256, 1536, 32, 0, false, exceptional: false, sourceOverride: patterns))
        {
            ushort[] expected = fixture.Validate(Plans[0], null);
            foreach (var plan in Plans.Skip(1)) fixture.Validate(plan, expected);
            validationCases++;
        }
        Console.WriteLine($"Storage A packing: {validationCases} fixtures passed full-panel exact checks for all candidates (native dtypes, RNE ties, NaNs, infinities, signed zero, gates, tails, offsets and scale boundaries).");

        var results = new List<object>();
        foreach (string format in new[] { "f32", "bf16", "bfp8" })
        foreach (var (m, k, gated) in new[] { (16384, 512, false), (16384, 1536, false), (16384, 1536, true), (512, 11500, false), (512, 512, false) })
        {
            using var fixture = new Fixture(lane, format, m, k, 32, 0, gated, exceptional: false);
            ushort[] expected = fixture.Validate(Plans[0], null);
            foreach (var plan in Plans)
            {
                fixture.Validate(plan, expected);
                var resources = lane.GetKernelResources(fixture.Kernel(plan));
                using var output = lane.AllocateBytes(checked(fixture.PanelElements * 2));
                var wall = new List<double>(); var gpu = new List<double>();
                long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
                for (int sample = 0; sample < 13; sample++)
                {
                    double before = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp();
                    fixture.Dispatch(plan, output); lane.Synchronize();
                    if (sample >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - before); }
                }
                if (lane.H2DBytes != h2d || lane.D2HBytes != d2h) throw new InvalidOperationException("Packing benchmark crossed the host/device boundary.");
                double wallMedian = Median(wall), gpuMedian = Median(gpu);
                int bytesPerSource = format == "f32" ? 4 : format == "bf16" ? 2 : 1;
                long logicalBytes = (long)m * k * bytesPerSource + (long)fixture.PanelElements * 2
                    + (gated ? (long)m * k * 4 : 0) + (format == "bfp8" ? ((long)m * k + 31) / 32 * 4 : 0);
                Console.WriteLine($"{format} {m}x{k} gated={gated} {plan.Suffix}: GPU={gpuMedian:F4} ms, wall={wallMedian:F4} ms, logical={logicalBytes / gpuMedian / 1e6:F1} GB/s, spill={resources.SpillMemoryBytes}");
                results.Add(new { Format = format, M = m, K = k, Gated = gated, Candidate = plan.Suffix,
                    Kernel = fixture.Kernel(plan), plan.Rows, plan.Depth, plan.Workgroup,
                    P50Ms = wallMedian, GpuP50Ms = gpuMedian, Samples = wall, GpuSamples = gpu,
                    LogicalBytes = logicalBytes, LogicalGBps = logicalBytes / gpuMedian / 1e6,
                    Resources = resources, FullPanelExact = true, Device = lane.Device.Name, lane.Device.DriverVersion });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow, ValidationCases = validationCases,
            Note = "Nontransposed A storage->BF16 DPAS panels. Whole-panel ushort equality, including native F32/BF16/BFP8, FP32->BF16 RNE and exceptional patterns, original BFP8 scale origin, gates and padding. 3 warmup+10 measured kernel dispatches. Allocation/upload/validation readback outside timing; no persistent cache or extra device workspace except reported SLM. Logical bandwidth is bytes required by semantics, not measured physical bus traffic.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static Array BuildSource(string format, int length, bool exceptional)
    {
        int[] special = [0, unchecked((int)0x80000000), 1, 0x007fffff, 0x00800000,
            0x3f807fff, 0x3f808000, 0x3f818000, 0x7f7fffff, 0x7f800000,
            unchecked((int)0xff800000), 0x7f800001, 0x7fc01234, unchecked((int)0xff812345)];
        int Bits(int i) => exceptional && i % 3 == 0 ? special[(i / 3) % special.Length]
            : BitConverter.SingleToInt32Bits((i % 251 - 125) / 127f);
        return format switch
        {
            "f32" => Enumerable.Range(0, length).Select(Bits).ToArray(),
            "bf16" => Enumerable.Range(0, length).Select(i => exceptional ? (ushort)(i * 109) : RoundBf16(Bits(i))).ToArray(),
            "bfp8" => Enumerable.Range(0, length).Select(i => unchecked((sbyte)(i % 256 - 128))).ToArray(),
            _ => throw new ArgumentOutOfRangeException(nameof(format))
        };
    }
    private static ushort RoundBf16(int bits)
    {
        uint value = unchecked((uint)bits);
        if ((value & 0x7f800000u) != 0x7f800000u) value += 0x7fffu + ((value >> 16) & 1u);
        else if ((value & 0x007fffffu) != 0) value |= 0x00400000u;
        return (ushort)(value >> 16);
    }
    private static float ExceptionalScale(int i) => (i % 9) switch
    {
        0 => 1f / 127f, 1 => 1e-38f, 2 => float.PositiveInfinity, 3 => float.NegativeInfinity,
        4 => BitConverter.Int32BitsToSingle(0x7f800001), 5 => -0f, 6 => 1e30f, 7 => .33333334f, _ => 1f
    };
    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2;
    }
}
