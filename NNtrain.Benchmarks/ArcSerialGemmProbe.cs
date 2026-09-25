using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcSerialGemmProbe
{
    internal static void RunExpanded(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Output already exists.");
        using var lane = new ArcExecutionLane();
        var results = new List<object>();
        foreach (var (m, n, k, transposed, candidate, candidateRows, candidateColumns) in new[]
        {
            (1536, 512, 16384, true, "gemm_xmx_streamed_block_32x32_wg16", 512, 32),
            (512, 1536, 16384, true, "gemm_xmx_streamed_block_32x32_wg16", 512, 32),
            (16384, 512, 1536, false, "gemm_xmx_streamed_block_16x64_wg16", 256, 64)
        })
        {
            float[] av = new float[checked(m * k)], bv = new float[checked(n * k)];
            for (int i = 0; i < av.Length; i++) av[i] = (i % 31 - 15) * .0031f;
            for (int i = 0; i < bv.Length; i++) bv[i] = (i % 37 - 18) * .0047f;
            using var aValues = lane.Upload(av);
            using var bValues = lane.Upload(bv);
            using var a = new ArcXmxStorageOperand(lane, aValues, null, TensorDType.Float32, av.Length);
            using var b = new ArcXmxStorageOperand(lane, bValues, null, TensorDType.Float32, bv.Length);
            using var packedA = a.PackA(m, k, transposed);
            using var packedB = b.PackB(n, k, false);
            // Production dW uses parallel K slices in one 3D launch, whereas
            // dX uses a row-tiled 2D launch. Probe the same dispatch geometry.
            int outputLength = checked(m * n * (transposed ? (k + 2047) / 2048 : 1));
            float[] initial = new float[outputLength];
            for (int i = 0; i < initial.Length; i++) initial[i] = (i % 13 - 6) * .0123f;
            using var seed = lane.Upload(initial);
            using var output = lane.Allocate(initial.Length);
            int[]? expected = null;
            foreach (var (kernel, rows, columns) in new[]
            {
                ("gemm_xmx_streamed_block_16x32_wg16", 256, 32),
                (candidate, candidateRows, candidateColumns)
            })
            {
                void Execute()
                {
                    if (transposed)
                    {
                        lane.Run3D(kernel, ((n + columns - 1L) / columns) * 16,
                            ((m + rows - 1L) / rows) * 16, (k + 2047L) / 2048,
                            16, 16, 1, packedA, packedB, output, packedA,
                            m, n, k, 0, 0, 0, 0, 2048, 0);
                        return;
                    }
                    for (int start = 0; start < k; start += 2048)
                        lane.Run2D(kernel, ((n + columns - 1L) / columns) * 16,
                            ((m + rows - 1L) / rows) * 16, 16, 16,
                            packedA, packedB, output, packedA, m, n, k, 1, 0, 0,
                            start, Math.Min(2048, k - start), 0);
                }
                void Reset() { lane.CopyBytes(seed, output, 0, 0, initial.Length * 4); lane.Synchronize(); }
                Reset(); Execute(); Execute();
                int[] bits = new int[initial.Length];
                float[] actual = new float[initial.Length];
                lane.Read(output, actual);
                for (int i = 0; i < bits.Length; i++) bits[i] = BitConverter.SingleToInt32Bits(actual[i]);
                if (expected is null) expected = bits;
                else if (!expected.SequenceEqual(bits))
                    throw new InvalidOperationException($"Streamed GEMM is not bitwise equal: {m}x{n}x{k}, {kernel}.");
                var gpu = new List<double>(5);
                var wall = new List<double>(5);
                for (int sample = 0; sample < 7; sample++)
                {
                    Reset();
                    double before = lane.KernelMilliseconds;
                    long start = Stopwatch.GetTimestamp();
                    Execute(); Execute(); lane.Synchronize();
                    if (sample >= 2)
                    {
                        gpu.Add(lane.KernelMilliseconds - before);
                        wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                    }
                }
                static double P50(List<double> values) => values.Order().ElementAt(values.Count / 2);
                var resources = lane.GetKernelResources(kernel);
                Console.WriteLine($"streamed {m}x{n}x{k} {kernel}: GPU {P50(gpu):F3} ms, wall {P50(wall):F3} ms, spill {resources.SpillMemoryBytes} bytes; bitwise equal");
                results.Add(new { M = m, N = n, K = k, TransposeA = transposed, Kernel = kernel,
                    TileRows = rows, TileColumns = columns, GpuP50Ms = P50(gpu), WallP50Ms = P50(wall),
                    GpuSamplesMs = gpu, WallSamplesMs = wall, BitwiseEqual = true, Resources = resources });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { lane.Device, Warmup = 2, Measured = 5, KChunk = 2048,
            Accumulations = 2, Note = "Resident packed BF16 inputs; dW uses production parallel K slices and dX uses a row-tiled launch; reset and pack excluded from timed samples.", Results = results },
            new JsonSerializerOptions { WriteIndented = true });
    }

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
