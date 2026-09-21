using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcSerialGemmProbe
{
    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Output already exists.");
        using var lane = new ArcExecutionLane();
        var results = new List<object>();
        foreach (var shape in new[] { (1536, 512, 16384), (512, 1536, 16384), (137, 71, 4101) })
        {
            (int m, int n, int k) = shape;
            using var av = lane.Upload(Enumerable.Range(0, m * k).Select(i => (i % 31 - 15) * .0031f).ToArray());
            using var bv = lane.Upload(Enumerable.Range(0, n * k).Select(i => (i % 37 - 18) * .0047f).ToArray());
            using var a = new ArcXmxStorageOperand(lane, av, null, TensorDType.Float32, m * k);
            using var b = new ArcXmxStorageOperand(lane, bv, null, TensorDType.Float32, n * k);
            using var pa = a.PackA(m, k, true); using var pb = b.PackB(n, k, false);
            float[] initial = Enumerable.Range(0, m * n).Select(i => (i % 13 - 6) * .0123f).ToArray();
            using var seed = lane.Upload(initial); using var output = lane.Allocate(m * n);
            int[]? expected = null;
            foreach (int tile in new[] { 256, 128 })
                foreach (int chunk in new[] { 2048, 4096, 8192, 16384 })
                {
                    string kernel = tile == 256 ? "gemm_xmx_streamed_block_16x32_wg16" : "gemm_xmx_streamed_block_8x32_wg16";
                    void Execute()
                    {
                        for (int start = 0; start < k; start += chunk)
                            lane.Run2D(kernel, ((n + 31L) / 32) * 16, ((m + tile - 1L) / tile) * 16, 16, 16,
                                pa, pb, output, pa, m, n, k, 1, 0, 0, start, Math.Min(chunk, k - start), 0);
                    }
                    lane.CopyBytes(seed, output, 0, 0, m * n * 4); Execute(); Execute();
                    float[] actual = new float[m * n]; lane.Read(output, actual);
                    int[] bits = actual.Select(BitConverter.SingleToInt32Bits).ToArray();
                    if (expected is null) expected = bits;
                    else if (!expected.SequenceEqual(bits)) throw new InvalidOperationException($"Not bit-exact: {shape}, tile {tile}, chunk {chunk}.");
                    var wall = new List<double>(); var gpu = new List<double>();
                    for (int i = 0; i < 13; i++)
                    {
                        lane.CopyBytes(seed, output, 0, 0, m * n * 4); lane.Synchronize();
                        double before = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp();
                        Execute(); lane.Synchronize();
                        if (i >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - before); }
                    }
                    double Median(List<double> x) => (x.Order().ElementAt(4) + x.Order().ElementAt(5)) * .5;
                    Console.WriteLine($"{shape} tile={tile} K-chunk={chunk}: wall={Median(wall):F4}, GPU={Median(gpu):F4} ms; bit-exact");
                    results.Add(new { M = m, N = n, K = k, Tile = tile, Chunk = chunk, WallP50Ms = Median(wall), GpuP50Ms = Median(gpu), Wall = wall, Gpu = gpu, Resources = lane.GetKernelResources(kernel) });
                }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { lane.Device, Note = "Same packed BF16 operands, FP32 accumulation; all outputs bit-exact after two accumulations against K2048/tile256. 3 warmup + 10 timed dispatch groups; packing and seed reset excluded.", Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }
}
