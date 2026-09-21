using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

/// <summary>Experimental GEMM probes; GPU packing/allocation is included.</summary>
internal static class ArcXmxNextProbe
{
    private enum Candidate { Current, Retile, DirectWide, DirectNarrow }

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane();
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("The experimental XMX layouts require SG16 Intel XMX.");
        Validate(lane);
        var results = new List<object>();
        foreach (Candidate candidate in Enum.GetValues<Candidate>())
        foreach (var (m, n, k, ta, tb) in new[] {
            (16384,1536,512,false,true), (16384,512,1536,false,true),
            (16384,512,1536,false,false), (16384,1536,512,false,false),
            (1536,512,16384,true,false), (512,1536,16384,true,false),
            (512,512,11500,false,false), (11500,512,512,true,false),
            (512,11500,512,false,true), (512,512,512,false,false) })
        {
            float[] av = Enumerable.Range(0, m * k).Select(i => (i % 17 - 8) / 32f).ToArray();
            float[] bv = Enumerable.Range(0, k * n).Select(i => (i % 19 - 9) / 32f).ToArray();
            using var a = lane.Upload(av); using var b = lane.Upload(bv);
            using var c = lane.Upload(new float[m * n]);
            var resources = lane.GetKernelResources(Plan(candidate, m, n, ta, tb).Kernel);
            Console.WriteLine($"{candidate} {m}x{n}x{k} SLM={resources.LocalMemoryBytes} spill={resources.SpillMemoryBytes}");
            var wall = new List<double>(); var gpu = new List<double>();
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int i = 0; i < 13; i++)
            {
                double gpuStart = lane.KernelMilliseconds;
                long start = Stopwatch.GetTimestamp();
                Gemm(lane, candidate, a, b, c, m, n, k, ta, tb, ta);
                lane.Synchronize();
                if (i >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - gpuStart); }
            }
            if (h2d != lane.H2DBytes || d2h != lane.D2HBytes)
                throw new InvalidOperationException("GEMM crossed the host/device boundary.");
            foreach (int row in new[] { 0, m / 2, m - 1 })
            foreach (int col in new[] { 0, n / 2, n - 1 })
            {
                float expected = 0;
                for (int j = 0; j < k; j++) expected += av[ta ? j * m + row : row * k + j] * bv[tb ? col * k + j : j * n + col];
                if (ta) expected *= 13;
                float[] actual = new float[1]; lane.ReadFloatRange(c, row * n + col, actual);
                if (expected != actual[0]) throw new InvalidOperationException($"{candidate}: {m}x{n}x{k} at {row},{col}: {expected} != {actual[0]}.");
            }
            double Median(List<double> values) { var sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2; }
            Console.WriteLine($"{candidate} {m}x{n}x{k} ta={ta} tb={tb}: {Median(wall):F4} ms wall, {Median(gpu):F4} ms GPU (packing included)");
            results.Add(new { Candidate = candidate.ToString(), M = m, N = n, K = k, Ta = ta, Tb = tb,
                P50Ms = Median(wall), GpuP50Ms = Median(gpu), Samples = wall, GpuSamples = gpu, Resources = resources,
                Device = lane.Device.Name, lane.Device.DriverVersion });
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow,
            Note = "BF16 operand RNE / FP32 accumulation; 3 warmup + 10 samples. Candidate Direct includes BOTH packing kernels and scratch lifetime. CPU sampling and full-array validation outside timing.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }

    private static (string Kernel, int Rows, int Columns) Plan(Candidate candidate, int m, int n, bool ta, bool tb)
    {
        if (candidate == Candidate.Current) return ArcMuonMath.SelectXmxPlan(ArcXmxGemmMode.Auto, 16, m, n, ta, tb);
        if (candidate == Candidate.Retile) return ($"gemm_xmx_retile_{(ta ? 't' : 'n')}{(tb ? 't' : 'n')}", 256, 128);
        return candidate == Candidate.DirectWide ? ("gemm_xmx_direct_wide", 256, 128) : ("gemm_xmx_direct_narrow", 128, 64);
    }

    private static void Gemm(ArcExecutionLane lane, Candidate candidate, ArcBuffer a, ArcBuffer b, ArcBuffer output,
        int m, int n, int k, bool ta, bool tb, bool accumulate, ArcBuffer? bias = null, bool relu = false,
        ArcBuffer? gate = null, int gateOperand = 0)
    {
        if (candidate == Candidate.Current)
        {
            ArcMuonMath.Gemm(lane, a, b, output, m, n, k, ta, tb, 3, accumulate, bias, relu, gate, gateOperand);
            return;
        }
        ArcBuffer? packedA = null, packedB = null;
        try
        {
            if (candidate is Candidate.DirectWide or Candidate.DirectNarrow)
            {
                int aElements = checked(((m + 7) / 8) * ((k + 15) / 16) * 128);
                int bElements = checked(((n + 15) / 16) * ((k + 15) / 16) * 128);
                packedA = lane.AllocateBytes(checked(aElements * 2)); packedB = lane.Allocate(bElements);
                lane.Run("xmx_next_pack_a", aElements, 0, a, gate ?? a, packedA, m, k, ta ? 1 : 0, gateOperand == 1 ? 1 : 0);
                lane.Run("xmx_next_pack_b", bElements, 0, b, gate ?? b, packedB, n, k, tb ? 1 : 0, gateOperand == 2 ? 1 : 0);
            }
            var (kernel, rows, columns) = Plan(candidate, m, n, ta, tb);
            long gx = ((n + columns - 1L) / columns) * 16, gy = ((m + rows - 1L) / rows) * (rows / 8);
            int slices = (k + 2047) / 2048;
            if (ta && !tb && accumulate && bias is null && !relu && k >= 4096 && m >= 64 && n >= 64
                && (long)m * n * slices * 4 <= 64 * 1024 * 1024)
            {
                using var partials = lane.Allocate(checked(m * n * slices));
                lane.Run3D(kernel, gx, gy, slices, 16, rows / 8, 1, packedA ?? a, packedB ?? b,
                    partials, a, gate ?? a, m, n, k, ta ? 1 : 0, tb ? 1 : 0, 3, 0, 0, 0, gateOperand, 0, 2048);
                lane.Run("gemm_split_finish", checked(m * n), 0, partials, output, checked(m * n), slices);
                return;
            }
            for (int start = 0; start < k; start += 2048)
                lane.Run2D(kernel, gx, gy, 16, rows / 8, packedA ?? a, packedB ?? b, output, bias ?? a,
                    gate ?? a, m, n, k, ta ? 1 : 0, tb ? 1 : 0, 3, accumulate || start != 0 ? 1 : 0,
                    bias is not null && start == 0 ? 1 : 0, relu && start + 2048 >= k ? 1 : 0, gateOperand, start, Math.Min(2048, k - start));
        }
        finally { packedA?.Dispose(); packedB?.Dispose(); }
    }

    private static void Validate(ArcExecutionLane lane)
    {
        foreach (var (m, n, k) in new[] { (137, 131, 67), (67, 71, 4101), (257, 513, 35) })
        foreach (bool ta in new[] { false, true }) foreach (bool tb in new[] { false, true })
        foreach (int gateOperand in new[] { 0, 1, 2 })
        {
            using var a = lane.Upload(Enumerable.Range(0, m * k).Select(i => (i % 17 - 8) / 32f).ToArray());
            using var b = lane.Upload(Enumerable.Range(0, k * n).Select(i => (i % 19 - 9) / 32f).ToArray());
            using var bias = lane.Upload(Enumerable.Range(0, n).Select(i => (i % 7 - 3) / 32f).ToArray());
            float[]? expected = null;
            foreach (Candidate candidate in Enum.GetValues<Candidate>())
            {
                using var c = lane.Upload(Enumerable.Repeat(.125f, m * n).ToArray());
                long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
                for (int step = 0; step < 2; step++) Gemm(lane, candidate, a, b, c, m, n, k, ta, tb, true,
                    gateOperand == 0 ? null : bias, gateOperand != 0, gateOperand == 2 ? b : a, gateOperand);
                if (lane.H2DBytes != h2d || lane.D2HBytes != d2h) throw new InvalidOperationException("Unexpected GEMM transfer.");
                float[] actual = new float[m * n]; lane.Read(c, actual);
                if (expected is null) expected = actual;
                else for (int i = 0; i < actual.Length; i++)
                    if (actual[i] != expected[i]) throw new InvalidOperationException($"{candidate} validation failed {m}x{n}x{k}, ta={ta},tb={tb},gate={gateOperand}, element={i}: {actual[i]} != {expected[i]}.");
            }
        }
        Console.WriteLine("Full-array tail/transpose/gate/bias/ReLU/accumulation/split-K validation passed for all candidates.");
    }
}
