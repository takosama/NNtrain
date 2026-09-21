using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcXmxTuningProbe
{
    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        var results = new List<object>();
        foreach (var mode in new[] { ArcXmxGemmMode.Legacy, ArcXmxGemmMode.Narrow, ArcXmxGemmMode.Wide, ArcXmxGemmMode.Auto })
        {
            using var lane = new ArcExecutionLane(options: new() { XmxGemmMode = mode, DirectXmxMatrices = false });
            if (!lane.Device.SupportsXmx) throw new NotSupportedException("Intel XMX is required.");
            Console.WriteLine($"Subgroup local block I/O: {lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io")}");
            foreach (var (m, n, k, ta, tb) in new[] {
                (16384,1536,512,false,true), (16384,512,1536,false,true),
                (16384,512,1536,false,false), (16384,1536,512,false,false),
                (1536,512,16384,true,false), (512,1536,16384,true,false),
                (512,512,11500,false,false), (11500,512,512,true,false),
                (512,11500,512,false,true), (512,512,512,false,false) })
            {
                var plan = ArcMuonMath.SelectXmxPlan(mode, lane.Device.MinimumSubgroupSize, m, n, ta, tb);
                var resources = lane.GetKernelResources(plan.Kernel);
                Console.WriteLine($"Mode {mode}: local={resources.LocalMemoryBytes}, private={resources.PrivateMemoryBytes}, spill={resources.SpillMemoryBytes} bytes");
                float[] av = Enumerable.Range(0, m * k).Select(i => (i % 17 - 8) / 32f).ToArray();
                float[] bv = Enumerable.Range(0, k * n).Select(i => (i % 19 - 9) / 32f).ToArray();
                using var a = lane.Upload(av); using var b = lane.Upload(bv);
                using var c = lane.Upload(new float[m * n]);
                var wall = new List<double>(); var gpu = new List<double>();
                for (int i = 0; i < 13; i++)
                {
                    double gpuStart = lane.KernelMilliseconds;
                    long start = Stopwatch.GetTimestamp();
                    ArcMuonMath.Gemm(lane, a, b, c, m, n, k, ta, tb, bf16: 3, accumulate: ta);
                    lane.Synchronize();
                    if (i >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - gpuStart); }
                }
                // Dyadic operands keep these samples exactly representable. Validate
                // outside measured dispatches, including repeated accumulated dW.
                foreach (int row in new[] { 0, m / 2, m - 1 })
                foreach (int col in new[] { 0, n / 2, n - 1 })
                {
                    float expected = 0;
                    for (int j = 0; j < k; j++) expected += av[ta ? j * m + row : row * k + j] * bv[tb ? col * k + j : j * n + col];
                    if (ta) expected *= 13;
                    float[] actual = new float[1]; lane.ReadFloatRange(c, row * n + col, actual);
                    if (expected != actual[0]) throw new InvalidOperationException($"Mode {mode} failed {m}x{n}x{k} at {row},{col}: {expected} vs {actual[0]}.");
                }
                double Median(List<double> values) { var sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2; }
                double p50 = Median(wall), gpuP50 = Median(gpu);
                Console.WriteLine($"XMX mode={mode} {m}x{n}x{k} ta={ta} tb={tb}: wall={p50:F4}, gpu={gpuP50:F4} ms, {2d*m*n*k/gpuP50/1e9:F3} TFLOP/s");
                results.Add(new { Mode = mode.ToString(), M = m, N = n, K = k, Ta = ta, Tb = tb, P50Ms = p50,
                    GpuP50Ms = gpuP50, Samples = wall, GpuSamples = gpu, Resources = resources, Device = lane.Device.Name, lane.Device.DriverVersion });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow,
            Note = "BF16 exact dyadic operands; FP32 accumulation; 3 warmup+10 measured dispatches. Validation copies outside timing. Split workspace allocation included. Fresh native session per variant.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }
}
