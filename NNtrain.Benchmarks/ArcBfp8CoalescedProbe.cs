using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcBfp8CoalescedProbe
{
    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Output already exists.");
        using var lane = new ArcExecutionLane();
        int validated = 0;
        foreach (int block in new[] { 32, 64, 128, 256 })
            foreach (int length in new[] { 1, 31, 32, 33, 257, 8191, 16384 })
                foreach (bool exceptional in new[] { false, true })
                {
                    float[] input = Enumerable.Range(0, length).Select(i => exceptional ? i % 17 switch
                    {
                        0 => float.NaN,
                        1 => float.PositiveInfinity,
                        2 => float.NegativeInfinity,
                        3 => float.Epsilon,
                        4 => -float.Epsilon,
                        5 => float.MaxValue,
                        6 => -float.MaxValue,
                        7 => -0f,
                        _ => i * .0123f
                    }
                        : MathF.ScaleB((i % 31 - 15) * .0019f, i % 21 - 10)).ToArray();
                    Validate(lane, input, block); validated++;
                    // Zero and underflow-only blocks exercise distinct scale/status paths.
                    Validate(lane, new float[length], block); validated++;
                    Validate(lane, Enumerable.Repeat(float.Epsilon, length).ToArray(), block); validated++;
                }
        Console.WriteLine($"BFP8 codec bit-exact: {validated} cases (payload/scales/status, finite/nonfinite/zero/underflow/tails).");
        var results = new List<object>();
        foreach (int block in new[] { 32, 128 })
            foreach (int count in new[] { 65536 * 512, 65536 * 1536, 512 * 1536 })
            {
                using var source = lane.Upload(Enumerable.Range(0, count).Select(i => (i % 101 - 50) * .0013f).ToArray());
                int groups = (count + block - 1) / block;
                using var output = lane.AllocateBytes(count); using var scales = lane.Allocate(groups); using var status = lane.UploadRaw(new int[1]);
                foreach (int variant in block == 32 ? new[] { 0, 1, 2, 3, 4, 5 } : new[] { 0, 1 })
                {
                    bool coalesced = variant == 1;
                    var wall = new List<double>(); var gpu = new List<double>();
                    string kernel = variant >= 3 ? $"resident_bfp8_quad{1 << (variant - 2)}_32" : variant == 2 ? "resident_bfp8_private32" : coalesced ? $"resident_bfp8_sg16_{block}" : "resident_bfp8";
                    for (int i = 0; i < 13; i++)
                    {
                        lane.Synchronize(); double before = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp();
                        lane.Run(kernel, variant >= 3 ? groups * (1L << (variant - 2)) : coalesced ? groups * 16L : groups, coalesced || variant >= 3 ? 256 : 0, source, output, scales, status, count, block); lane.Synchronize();
                        if (i >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - before); }
                    }
                    double Median(List<double> x) => (x.Order().ElementAt(4) + x.Order().ElementAt(5)) * .5;
                    Console.WriteLine($"{kernel} n={count}: GPU={Median(gpu):F4} wall={Median(wall):F4} ms");
                    results.Add(new { Count = count, Block = block, Kernel = kernel, GpuP50Ms = Median(gpu), WallP50Ms = Median(wall), Gpu = gpu, Wall = wall, Resources = lane.GetKernelResources(kernel) });
                }
            }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!); using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { lane.Device, ValidationCases = validated, Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }

    internal static void Validate(ArcExecutionLane lane, float[] input, int block)
    {
        int n = input.Length, groups = (n + block - 1) / block;
        using var source = lane.Upload(input);
        sbyte[]? expected = null; int[]? expectedScales = null; int? expectedStatus = null;
        foreach (int variant in block == 32 ? new[] { 0, 1, 2, 3, 4, 5 } : new[] { 0, 1 })
        {
            bool coalesced = variant == 1;
            using var output = lane.AllocateBytes(n); using var scales = lane.Allocate(groups); using var status = lane.UploadRaw(new int[1]);
            string kernel = variant >= 3 ? $"resident_bfp8_quad{1 << (variant - 2)}_32" : variant == 2 ? "resident_bfp8_private32" : coalesced ? $"resident_bfp8_sg16_{block}" : "resident_bfp8";
            lane.Run(kernel, variant >= 3 ? groups * (1L << (variant - 2)) : coalesced ? groups * 16L : groups, coalesced || variant >= 3 ? 256 : 0, source, output, scales, status, n, block);
            var bytes = new sbyte[n]; var scale = new float[groups]; var flags = new int[1];
            lane.ReadRaw(output, bytes); lane.Read(scales, scale); lane.ReadRaw(status, flags);
            int[] bits = scale.Select(BitConverter.SingleToInt32Bits).ToArray();
            if (expected is null) { expected = bytes; expectedScales = bits; expectedStatus = flags[0]; }
            else if (!expected.SequenceEqual(bytes) || !expectedScales!.SequenceEqual(bits) || expectedStatus != flags[0])
                throw new InvalidOperationException($"Codec mismatch n={n},block={block}.");
        }
    }
}
