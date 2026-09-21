using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

/// <summary>QK-only experiment; GPU packing, allocation and release are timed.</summary>
internal static class ArcAttentionQkDirectProbe
{
    private enum Candidate { Reference, Direct128x64, Direct128x32, Direct256x32 }

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane();
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("SG16 Intel XMX is required.");
        Validate(lane);
        var results = new List<object>();
        foreach (int batch in new[] { 16, 64 })
        {
            const int heads = 16, d = 32, sequence = 1024, tileHeads = 8;
            const int width = heads * d;
            using var qkv = lane.Upload(Values(checked(batch * sequence * width * 3)));
            using var scores = lane.Allocate(checked(tileHeads * sequence * sequence));
            foreach (bool causal in new[] { true, false })
            foreach (Candidate candidate in Enum.GetValues<Candidate>())
            {
                var (kernel, _, _) = Plan(candidate);
                var resources = lane.GetKernelResources(kernel);
                var wall = new List<double>(); var gpu = new List<double>(); var packing = new List<double>();
                var compute = new List<double>(); var allocations = new List<long>();
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                for (int repeat = 0; repeat < 13; repeat++)
                {
                    double gpuStart = lane.KernelMilliseconds;
                    double packStart = lane.KernelTimings.GetValueOrDefault("attention_qk_pack_combined_candidate");
                    double computeStart = lane.KernelTimings.GetValueOrDefault(kernel);
                    long allocationStart = lane.AllocationCount, start = Stopwatch.GetTimestamp();
                    for (int first = 0; first < batch * heads; first += tileHeads)
                        Dispatch(lane, candidate, qkv, scores, sequence, width, heads, first,
                            Math.Min(tileHeads, batch * heads - first), false, causal);
                    lane.Synchronize();
                    if (repeat >= 3)
                    {
                        wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                        gpu.Add(lane.KernelMilliseconds - gpuStart);
                        packing.Add(lane.KernelTimings.GetValueOrDefault("attention_qk_pack_combined_candidate") - packStart);
                        compute.Add(lane.KernelTimings.GetValueOrDefault(kernel) - computeStart);
                        allocations.Add(lane.AllocationCount - allocationStart);
                    }
                }
                if (uploads != lane.H2DBytes || downloads != lane.D2HBytes)
                    throw new InvalidOperationException("QK benchmark crossed the host/device boundary.");
                Console.WriteLine($"QK {candidate}, B{batch} T{sequence} D{d} H{heads} tile{tileHeads}, causal={causal}: "
                    + $"{Median(wall):F4}ms wall, {Median(gpu):F4}ms GPU, pack={Median(packing):F4}ms, "
                    + $"compute={Median(compute):F4}ms, SLM={resources.LocalMemoryBytes}, spill={resources.SpillMemoryBytes}");
                results.Add(new { Candidate = candidate.ToString(), Batch = batch, Heads = heads, Sequence = sequence,
                    HeadWidth = d, TileHeads = tileHeads, Causal = causal, Kernel = kernel,
                    PanelBytesPerHeadTile = candidate == Candidate.Reference ? 0 : tileHeads * sequence * d * 4,
                    P50Ms = Median(wall), GpuP50Ms = Median(gpu), PackGpuP50Ms = Median(packing),
                    ComputeGpuP50Ms = Median(compute), Samples = wall, GpuSamples = gpu, PackGpuSamples = packing,
                    ComputeGpuSamples = compute, Allocations = allocations, Resources = resources,
                    Device = lane.Device.Name, lane.Device.DriverVersion });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow,
            Note = "Experimental QK only. Same BF16 RNE, padded32 ordered DPAS and 128-row causal publication as attention_xmx_panel. "
                + "Per-head-tile device pack, allocation and release included. 3 warmup +10 samples across ALL B16/B64 heads. "
                + "Full-array validation first includes T/D tails, heads crossing batches, two launches, accumulation, non-finites and signed zero. No production dispatch changed.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static double Median(List<double> values)
    {
        var sorted = values.Order().ToArray();
        return (sorted[4] + sorted[5]) / 2;
    }

    private static (string Kernel, int Rows, int Columns) Plan(Candidate candidate) => candidate switch {
        Candidate.Reference => ("attention_xmx_panel", 128, 64),
        Candidate.Direct128x64 => ("attention_qk_direct_128x64_candidate", 128, 64),
        Candidate.Direct128x32 => ("attention_qk_direct_128x32_candidate", 128, 32),
        Candidate.Direct256x32 => ("attention_qk_direct_256x32_candidate", 256, 32),
        _ => throw new ArgumentOutOfRangeException(nameof(candidate))
    };

    private static void Dispatch(ArcExecutionLane lane, Candidate candidate, ArcBuffer qkv, ArcBuffer scores,
        int sequence, int width, int heads, int first, int count, bool add, bool causal)
    {
        int d = width / heads;
        if (candidate == Candidate.Reference)
        {
            lane.Run3D("attention_xmx_panel", ((sequence + 63L) / 64) * 16, ((sequence + 127L) / 128) * 16,
                count, 16, 16, 1, qkv, qkv, scores, sequence, sequence, d,
                3 * width, 1, 0, sequence * 3 * width, d, 0,
                1, 3 * width, 0, sequence * 3 * width, d, width,
                sequence, 1, sequence * sequence, 0, 0, 0, heads, first, add ? 1 : 0, causal ? 1 : 0);
            return;
        }
        int paddedRows = checked(((sequence + 15) / 16) * 16), depth = checked(((d + 31) / 32) * 32);
        int elements = checked(count * paddedRows * depth);
        // Both buffers remain local to this QK call. Subsequent layers/updates
        // cannot accidentally reuse panels from a previous QKV activation.
        using var q = lane.AllocateBytes(checked(elements * 2));
        using var k = lane.AllocateBytes(checked(elements * 2));
        lane.Run("attention_qk_pack_combined_candidate", elements / 2, 256, qkv, q, k,
            sequence, width, heads, first, count);
        var (kernel, rows, columns) = Plan(candidate);
        lane.Run3D(kernel, ((sequence + columns - 1L) / columns) * 16, ((sequence + rows - 1L) / rows) * 16,
            count, 16, 16, 1, q, k, scores, sequence, d, add ? 1 : 0, causal ? 1 : 0);
    }

    private static void Validate(ArcExecutionLane lane)
    {
        foreach (var (sequence, d) in new[] { (65, 7), (137, 17), (137, 32), (137, 35), (128, 64), (1024, 32) })
        foreach (bool causal in new[] { false, true })
        foreach (bool add in new[] { false, true })
            Compare(lane, sequence, d, causal, add, poison: false);
        Compare(lane, 137, 17, true, true, poison: true);
        Compare(lane, 65, 7, false, false, poison: true);
        Console.WriteLine("QK full-array validation passed: sequence/head-width tails, partial heads crossing batches, "
            + "causal publication, two launches, accumulation, signed zero and non-finite classification.");
    }

    private static void Compare(ArcExecutionLane lane, int sequence, int d, bool causal, bool add, bool poison)
    {
        const int batch = 2, heads = 3, first = 1, count = 4;
        int width = heads * d;
        float[] values = Values(checked(batch * sequence * width * 3));
        if (poison)
        {
            values[d] = BitConverter.Int32BitsToSingle(unchecked((int)0x7f800001));
            values[3 * width + width + d + 1] = float.PositiveInfinity;
            values[6 * width + d + 2] = float.NegativeInfinity;
        }
        float[] initial = Enumerable.Range(0, count * sequence * sequence)
            .Select(i => i % 19 == 0 ? -0f : (i % 13 - 6) * .0037f).ToArray();
        using var input = lane.Upload(values);
        float[]? expected = null;
        foreach (Candidate candidate in Enum.GetValues<Candidate>())
        {
            using var output = lane.Upload(initial);
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
                Dispatch(lane, candidate, input, output, sequence, width, heads, first, count, add, causal);
            if (uploads != lane.H2DBytes || downloads != lane.D2HBytes)
                throw new InvalidOperationException("QK validation encountered an implicit transfer.");
            float[] actual = new float[initial.Length]; lane.Read(output, actual);
            if (expected is null) { expected = actual; continue; }
            for (int i = 0; i < actual.Length; i++)
            {
                bool equal = float.IsNaN(expected[i]) ? float.IsNaN(actual[i])
                    : BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]);
                if (!equal) throw new InvalidOperationException($"QK {candidate}: T{sequence} D{d}, causal={causal}, "
                    + $"add={add}, poison={poison}, element{i}: {expected[i]} != {actual[i]}.");
            }
        }
    }

    private static float[] Values(int length)
    {
        var values = new float[length];
        for (int i = 0; i < length; i++)
            values[i] = i % 101 == 0 ? -0f : MathF.ScaleB((i % 31 - 15) * .00173f, i % 5 - 2);
        return values;
    }
}
