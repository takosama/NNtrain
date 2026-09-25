using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcWeightGradientTuneProbe
{
    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Use a new probe result path.");
        using var lane = new ArcExecutionLane();
        var results = new List<object>();
        var tiles = new (string Kernel, int Rows, int Columns, int LocalRows)[] {
            ("gemm_xmx_direct_block_32x32_wg16", 512, 32, 16),
            ("gemm_xmx_direct_block_32x32_wg8", 256, 32, 8),
            ("gemm_xmx_direct_block_32x32_wg4", 128, 32, 4),
            ("gemm_xmx_direct_block_16x64_wg16", 256, 64, 16),
            ("gemm_xmx_direct_block_16x64_wg8", 128, 64, 8),
            ("gemm_xmx_direct_block_16x64_wg4", 64, 64, 4),
            ("gemm_xmx_direct_block_8x64_wg8", 64, 64, 8),
        };
        foreach (var (m, n, k) in new[] { (1536, 512, 16384), (512, 1536, 16384), (512, 512, 16384) })
        {
            float[] left = new float[m * k], right = new float[n * k];
            for (int i = 0; i < left.Length; i++) left[i] = (i % 31 - 15) * .0031f;
            for (int i = 0; i < right.Length; i++) right[i] = (i % 37 - 18) * .0047f;
            using var av = lane.Upload(left); using var bv = lane.Upload(right);
            using var a = new ArcXmxStorageOperand(lane, av, null, TensorDType.Float32, left.Length);
            using var b = new ArcXmxStorageOperand(lane, bv, null, TensorDType.Float32, right.Length);
            using var pa = a.PackA(m, k, true); using var pb = b.PackB(n, k, false);
            float[] seedValues = Enumerable.Range(0, m * n).Select(i => (i % 13 - 6) * .0123f).ToArray();
            using var seed = lane.Upload(seedValues); using var output = lane.Allocate(m * n);
            float[]? reference = null;
            foreach (int split in new[] { 2048, 1024, 4096, 8192 })
            foreach (var tile in tiles)
            {
                int slices = (k + split - 1) / split;
                using var partials = lane.Allocate(checked(m * n * slices));
                void Execute()
                {
                    lane.Run3D(tile.Kernel, ((n + tile.Columns - 1L) / tile.Columns) * 16,
                        ((m + tile.Rows - 1L) / tile.Rows) * tile.LocalRows, slices,
                        16, tile.LocalRows, 1, pa, pb, partials, pa, pa,
                        m, n, k, 1, 0, 3, 0, 0, 0, 0, 0, split);
                    lane.Run("gemm_split_finish", m * n, 0, partials, output, m * n, slices);
                }
                void Reset() { lane.CopyBytes(seed, output, 0, 0, m * n * 4); lane.Synchronize(); }
                Reset(); Execute(); Execute();
                float[] actual = new float[m * n]; lane.Read(output, actual);
                reference ??= (float[])actual.Clone();
                double error = 0, norm = 0; bool bitwise = true;
                for (int i = 0; i < actual.Length; i++)
                {
                    if (!float.IsFinite(actual[i])) throw new ArithmeticException("Non-finite weight gradient.");
                    double delta = actual[i] - reference[i]; error += delta * delta;
                    norm += (double)reference[i] * reference[i];
                    bitwise &= BitConverter.SingleToInt32Bits(actual[i]) == BitConverter.SingleToInt32Bits(reference[i]);
                }
                double rms = Math.Sqrt(error / Math.Max(norm, 1e-30));
                if (rms > .001 || split == 2048 && !bitwise)
                    throw new ArithmeticException($"dW mismatch {m}x{n}: {tile.Kernel}, split={split}, RMS={rms}.");
                var gpu = new List<double>(); var wall = new List<double>();
                for (int sample = 0; sample < 10; sample++)
                {
                    Reset(); double before = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp();
                    Execute(); lane.Synchronize();
                    if (sample >= 3) { gpu.Add(lane.KernelMilliseconds - before); wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); }
                }
                double gpuP50 = gpu.Order().ElementAt(3), wallP50 = wall.Order().ElementAt(3);
                Console.WriteLine($"dW {m}x{n}x{k} {tile.Kernel} split={split}: GPU={gpuP50:F4}, wall={wallP50:F4} ms, RMS={rms:G4}");
                results.Add(new { M=m, N=n, K=k, tile.Kernel, tile.Rows, tile.Columns, tile.LocalRows,
                    SplitK=split, GpuP50Ms=gpuP50, WallP50Ms=wallP50, RelativeRms=rms, BitwiseEqual=bitwise,
                    PartialBytes=(long)m*n*slices*4, GpuSamplesMs=gpu, WallSamplesMs=wall,
                    Resources=lane.GetKernelResources(tile.Kernel) });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { lane.Device, Warmup=3, Measured=7,
            Note="Resident BF16 panels, production dW parallel-slice GEMM plus finish; packing/reset excluded. Accuracy compares two accumulated calls against split2048/32x32wg16. GPU 0 only, not full-step throughput.", Results=results },
            new JsonSerializerOptions { WriteIndented=true });
    }
}
