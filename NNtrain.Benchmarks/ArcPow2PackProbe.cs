using System.Diagnostics;
using System.Numerics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcPow2PackProbe
{
    internal static void Run(string path)
    {
        if (File.Exists(path)) throw new IOException("New artifact required.");
        using var lane = new ArcExecutionLane();
        int validations = 0;
        var rows = new List<object>();
        foreach (int block in new[] { 1, 2, 4, 32, 128, 1024 })
        foreach (int offset in new[] { 0, 7 })
        foreach (bool transpose in new[] { false, true })
        foreach (bool right in new[] { false, true })
        {
            Compare(33, 65, block, offset, transpose, right, true, true, false);
            validations++;
        }
        Console.WriteLine($"Pow2 packing: {validations} exact full-panel cases passed (tails, offsets, gates, exceptional scales).");
        foreach (var shape in new[] { (Outer: 512, K: 16384), (Outer: 1536, K: 16384), (Outer: 16384, K: 512), (Outer: 16384, K: 1536) })
        foreach (bool right in new[] { false, true })
        foreach (bool transpose in new[] { false, true })
            Compare(shape.Outer, shape.K, 32, 0, transpose, right, false, false, true);
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Validations = validations, Device = lane.Device.Name, lane.Device.DriverVersion,
            Note = "Resident BFP8->BF16 panels, unchanged layout/RNE. 3 warmup + 10 measurements. Full-panel equality before timing. Allocation/transfer outside kernel timing.", Results = rows }, new JsonSerializerOptions { WriteIndented = true });

        void Compare(int outer, int k, int block, int offset, bool transpose, bool right, bool gated, bool exceptional, bool benchmark)
        {
            int length = checked(outer * k + offset);
            using var source = lane.UploadRaw(Enumerable.Range(0, length).Select(i => (sbyte)(i % 255 - 127)).ToArray());
            using var scales = lane.Upload(Enumerable.Range(0, (length + block - 1) / block).Select(i => exceptional
                ? (i % 7) switch { 0 => float.NaN, 1 => float.PositiveInfinity, 2 => float.NegativeInfinity, 3 => -0f, 4 => 1e-38f, _ => .001f }
                : (i % 19 + 1) / 8192f).ToArray());
            using var gate = lane.Upload(gated ? Enumerable.Range(0, outer * k).Select(i => (float)(i % 3 - 1)).ToArray() : [1f]);
            int elements = ((outer + (right ? 15 : 7)) / (right ? 16 : 8)) * ((k + 15) / 16) * 128;
            using var output = lane.AllocateBytes(elements * (right ? 4 : 2));
            string kernel = $"xmx_storage_pack_{(right ? 'b' : 'a')}_{(transpose ? "transpose_" : !right ? "linear_vec4_" : "")}bfp8";
            ushort[]? expected = null;
            foreach (bool shift in new[] { false, true })
            {
                int address = shift ? ~BitOperations.Log2((uint)block) : block;
                void Dispatch()
                {
                    object[] args = [source, scales, gate, output, outer, k, transpose ? 1 : 0, address, gated ? 1 : 0, offset, 0];
                    if (transpose) lane.Run2D(kernel, ((outer + 31L) / 32) * 256, (k + 31L) / 32, 256, 1, args);
                    else lane.Run(kernel, right ? elements : elements / 4, 0, args);
                }
                Dispatch(); var actual = new ushort[elements * (right ? 2 : 1)]; lane.ReadRaw(output, actual);
                if (expected is null) expected = actual;
                else if (!expected.SequenceEqual(actual)) throw new InvalidOperationException($"Pow2 panel mismatch: {kernel} {outer}x{k} block{block} offset{offset}.");
                if (!benchmark) continue;
                var gpu = new List<double>(); var wall = new List<double>();
                long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
                for (int i = 0; i < 13; i++)
                {
                    double before = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp(); Dispatch(); lane.Synchronize();
                    if (i >= 3) { gpu.Add(lane.KernelMilliseconds - before); wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); }
                }
                if (h2d != lane.H2DBytes || d2h != lane.D2HBytes) throw new InvalidOperationException("Unexpected transfer.");
                double Median(List<double> v) => (v.Order().ElementAt(4) + v.Order().ElementAt(5)) / 2;
                Console.WriteLine($"{kernel} {outer}x{k} shift={shift}: GPU {Median(gpu):F4} ms, wall {Median(wall):F4} ms");
                rows.Add(new { Kernel = kernel, Outer = outer, K = k, Transpose = transpose, Right = right, Shift = shift,
                    GpuP50Ms = Median(gpu), P50Ms = Median(wall), GpuSamples = gpu, Samples = wall });
            }
        }
    }
}
