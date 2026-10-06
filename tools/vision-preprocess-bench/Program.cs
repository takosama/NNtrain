using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using NNtrain;

var results = new List<object>();
foreach ((int width, int height) in new[] { (4096, 3072), (768, 768), (256, 256) })
{
    byte[] rgb = new byte[width * height * 3];
    new Random(42).NextBytes(rgb);
    var expected = Qwen35VisionPreprocessor.Prepare(rgb, width, height, 16, 2, 768 * 768, 1);
    string expectedHash = Convert.ToHexStringLower(SHA256.HashData(MemoryMarshal.AsBytes(expected.Patches.AsSpan())));
    foreach (int parallelism in new[] { 1, 2, 4, 8 })
    {
        for (int warmup = 0; warmup < 2; warmup++)
            _ = Qwen35VisionPreprocessor.Prepare(rgb, width, height, 16, 2, 768 * 768, parallelism);
        var samples = new List<double>();
        var allocations = new List<long>();
        bool bitwiseMatch = true;
        for (int repeat = 0; repeat < 6; repeat++)
        {
            long allocatedBefore = GC.GetTotalAllocatedBytes(precise: true);
            var timer = Stopwatch.StartNew();
            var input = Qwen35VisionPreprocessor.Prepare(rgb, width, height, 16, 2, 768 * 768, parallelism);
            samples.Add(timer.Elapsed.TotalMilliseconds);
            allocations.Add(GC.GetTotalAllocatedBytes(precise: true) - allocatedBefore);
            bitwiseMatch &= expectedHash == Convert.ToHexStringLower(
                SHA256.HashData(MemoryMarshal.AsBytes(input.Patches.AsSpan())));
        }
        double[] ordered = samples.Order().ToArray();
        results.Add(new { width, height, parallelism, samples_ms = samples,
            median_ms = (ordered[2] + ordered[3]) / 2, mean_allocated_bytes = allocations.Average(),
            patch_sha256 = expectedHash, bitwise_match = bitwiseMatch });
    }
}
Console.WriteLine(JsonSerializer.Serialize(new { cpu_threads = Environment.ProcessorCount, results },
    new JsonSerializerOptions { WriteIndented = true }));
