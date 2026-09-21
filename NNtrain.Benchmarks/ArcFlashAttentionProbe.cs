using System.Diagnostics;
using System.Text.Json;
using System.Security.Cryptography;
using NNtrain.Arc;

namespace NNtrain.Benchmarks;

internal static class ArcFlashAttentionProbe
{
    internal static void Run(string path, int? onlyBatch = null, int? onlyStage = null, bool bulkQk = false)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        if (onlyBatch is <= 0 or > 64 || onlyStage is < 0 or > 5)
            throw new ArgumentOutOfRangeException(nameof(onlyStage), "Batch must be 1..64 and stage 0..5.");
        var results = new List<object>();
        foreach (int batch in onlyBatch.HasValue ? new[] { onlyBatch.Value } : new[] { 2, 16, 64 })
        foreach (int stage in onlyStage.HasValue ? new[] { onlyStage.Value } : new[] { 0, 1, 2, 3, 4, 5 })
        {
            const int sequence = 1024, heads = 16, d = 32, width = heads * d;
            using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix16_32, options: new() {
                BulkAttentionQkPanels = bulkQk, PipelineEventCollection = true,
                FlashAttention = stage is > 0 and < 4, FlashAttentionXmxProducts = stage > 1, FlashAttentionAsyncCopy = stage == 3,
                XmxAttentionProducts = stage >= 4, DirectXmxAttentionProducts = stage == 5 });
            var lane = Tensor.ArcLane;
            var input = new Tensor(Enumerable.Range(0, batch * sequence * width * 3)
                .Select(i => MathF.Sin(i * .017f) * .2f).ToArray(), [batch, sequence, width * 3]);
            input.ConvertStorageInPlace(TensorDType.BFloat16);
            using var seed = lane.Upload(Enumerable.Repeat(.01f, batch * sequence * width).ToArray());
            var wall = new List<double>();var forward = new List<double>();var backward = new List<double>();
            var gpu = new List<double>();var allocations = new List<long>();var launches = new List<long>();
            long uploadStart = 0, downloadStart = 0;
            for (int repeat = 0; repeat < 8; repeat++)
            {
                if (repeat == 3) { uploadStart = lane.H2DBytes; downloadStart = lane.D2HBytes; }
                double beforeGpu = lane.KernelMilliseconds;
                long beforeAllocation = lane.AllocationCount, beforeLaunch = lane.KernelLaunchCount;
                long start = Stopwatch.GetTimestamp();
                var output = input.FusedMultiHeadAttention(heads, true);
                lane.Synchronize();
                double f = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                // Reuse a resident seed; no activation-sized host gradient copy.
                AutogradEngine.BackwardArcSeed(output, seed);
                lane.Synchronize();
                double elapsed = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                if (repeat >= 3) { wall.Add(elapsed); forward.Add(f);backward.Add(elapsed-f);
                    gpu.Add(lane.KernelMilliseconds-beforeGpu);allocations.Add(lane.AllocationCount-beforeAllocation);
                    launches.Add(lane.KernelLaunchCount-beforeLaunch); }
            }
            double Median(List<double> v) => v.Order().ElementAt(v.Count / 2);
            Console.WriteLine($"Flash stage{stage} B{batch}/H{heads}/T{sequence}/D{d}: total={Median(wall):F3} ms, forward={Median(forward):F3}, backward={Median(backward):F3}, GPU={Median(gpu):F3}, peak={lane.PeakAllocatedBytes/1048576.0:F1} MiB");
            results.Add(new { Stage = stage, Batch = batch, Sequence = sequence, Heads = heads, HeadWidth = d,
                P50Ms = Median(wall), ForwardP50Ms = Median(forward), BackwardP50Ms = Median(backward), GpuP50Ms = Median(gpu),
                Samples = wall, ForwardSamples = forward, BackwardSamples = backward, GpuSamples = gpu, Allocations = allocations, Launches = launches,
                PeakBytes = lane.PeakAllocatedBytes, H2DBytes = lane.H2DBytes-uploadStart, D2HBytes = lane.D2HBytes-downloadStart,
                Kernels = new Dictionary<string, double>(lane.KernelTimings),
                Resources = lane.KernelTimings.Keys.Where(k => k.StartsWith("attention_flash_") || k.StartsWith("attention_xmx_products_") || k.StartsWith("attention_products_"))
                    .ToDictionary(k => k, k => lane.GetKernelResources(k)),
                Options = lane.Options, Device = lane.Device.Name, lane.Device.DriverVersion });
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        string Hash(string filePath) { using var stream = File.OpenRead(filePath); return Convert.ToHexString(SHA256.HashData(stream)); }
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow,
            ArcBinarySha256 = Hash(typeof(ArcExecutionLane).Assembly.Location), CoreBinarySha256 = Hash(typeof(Tensor).Assembly.Location),
            Conditions = "Attention forward+backward including pack, raw-output replay, allocations and synchronization. Resident FP32 gradient seed. Causal, mix16_32, 3 warmup + 5 samples. All heads. Stage0 existing, 1 online softmax+FP32 products, 2 compensated XMX products, 3 asynchronous collective ping-pong, 4 materialized compensated XMX products.", Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }
}
