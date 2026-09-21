using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcAttentionDpDsProbe
{
    private const string Candidate = "attention_dp_ds_register_d32_candidate";

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane();
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("This bounded candidate requires Intel XMX with SG16.");
        int checks = 0;
        foreach (int sequence in new[] { 1, 31, 65, 137, 1024 })
        foreach (int causal in new[] { 0, 1, 2 })
        {
            // Cross the batch boundary and leave guard heads before/after the
            // selected range; compare every output, including untouched tails.
            using var fixture = new Fixture(lane, sequence, heads: 3, first: 1, count: 4, causal);
            AssertExact(fixture.Read(false), fixture.Read(true), $"T{sequence}, causal={causal}");
            checks++;
        }
        Console.WriteLine($"dP+dS: {checks} full-array exact checks passed.");
        var results = new List<object>();
        // Current backward uses a cache-sized four-head score tile.
        using var production = new Fixture(lane, 1024, heads: 16, first: 0, count: 4, causal: 2);
        AssertExact(production.Read(false), production.Read(true), "production T1024 D32 H16 count4");
        foreach (bool fused in new[] { false, true })
        {
            using var output = lane.Upload(production.Initial);
            var wall = new List<double>(); var gpu = new List<double>();
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int repeat = 0; repeat < 13; repeat++)
            {
                double beforeGpu = lane.KernelMilliseconds;
                long started = Stopwatch.GetTimestamp();
                production.Dispatch(fused, output); lane.Synchronize();
                if (repeat >= 3)
                {
                    wall.Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
                    gpu.Add(lane.KernelMilliseconds - beforeGpu);
                }
            }
            if (h2d != lane.H2DBytes || d2h != lane.D2HBytes)
                throw new InvalidOperationException("Timed dP+dS performed a host transfer.");
            var resources = lane.GetKernelResources(fused ? Candidate : "attention_fp32_dp_d32_aligned");
            Console.WriteLine($"dP+dS {(fused ? Candidate : "separate tuned dP + derivatives")}: wall={Median(wall):F4} ms, GPU={Median(gpu):F4} ms, SLM={resources.LocalMemoryBytes}, spill={resources.SpillMemoryBytes}");
            results.Add(new {
                Fused = fused, Kernel = fused ? Candidate : "attention_fp32_dp_d32_aligned + attention_derivatives",
                WallP50Ms = Median(wall), GpuP50Ms = Median(gpu), WallSamples = wall, GpuSamples = gpu,
                Resources = resources, ExactChecks = checks + 1, GlobalScratchBytes = 0,
                Device = lane.Device.Name, lane.Device.DriverVersion,
            });
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new {
            Timestamp = DateTimeOffset.UtcNow, Sequence = 1024, HeadWidth = 32, Heads = 16, Count = 4, CausalMode = 2,
            Note = "Bounded SG16 register dP+dS fusion. Original FP32 channel-FMA and 64-lane reduction order. Full-array bit comparison at T1/31/65/137/1024, causal0/1/2, first-head/batch offsets and two repeated dispatches. 3 warmup + 10 measured. No packing or host transfers inside timings.",
            Results = results,
        }, new JsonSerializerOptions { WriteIndented = true });
    }

    private sealed class Fixture : IDisposable
    {
        private readonly ArcExecutionLane _lane;
        private readonly int _sequence, _width, _heads, _first, _count, _causal;
        private readonly ArcBuffer _qkv, _dy, _p;
        internal readonly float[] Initial;

        internal Fixture(ArcExecutionLane lane, int sequence, int heads, int first, int count, int causal)
        {
            _lane = lane; _sequence = sequence; _heads = heads; _first = first; _count = count; _causal = causal;
            _width = heads * 32;
            int batch = (first + count + heads - 1) / heads;
            _qkv = lane.Upload(Enumerable.Range(0, batch * sequence * _width * 3)
                .Select(i => MathF.Sin(i * .013f) * .13f).ToArray());
            _dy = lane.Upload(Enumerable.Range(0, batch * sequence * _width)
                .Select(i => MathF.Cos(i * .017f) * .03f).ToArray());
            float[] probabilities = new float[count * sequence * sequence];
            for (int g = 0; g < count; g++)
            for (int query = 0; query < sequence; query++)
            for (int key = 0; key < sequence; key++)
                probabilities[(g * sequence + query) * sequence + key] = causal != 0 && key > query
                    ? float.NaN : (.5f + .4f * MathF.Sin(key * .021f + query * .019f + g)) / (causal != 0 ? query + 1 : sequence);
            _p = lane.Upload(probabilities);
            Initial = Enumerable.Repeat(.125f, probabilities.Length + 67).ToArray();
        }

        internal void Dispatch(bool fused, ArcBuffer output)
        {
            if (fused)
                _lane.Run2D(Candidate, ((_sequence + 15L) / 16) * 256, _count, 256, 1,
                    _qkv, _dy, _p, output, _sequence, _width, _heads, _first, _causal);
            else
            {
                string kernel = _sequence % 64 == 0 ? "attention_fp32_dp_d32_aligned" : "attention_fp32_dp_d32_special";
                // For mode2, skip only wholly irrelevant 64-key tiles. The
                // dP baseline publishes 32-diagonal subtiles; derivatives zero
                // the rest of the 64-diagonal tile before any dQ/dK consumer.
                _lane.Run3D(kernel, ((_sequence + 63L) / 64) * 16, ((_sequence + 63L) / 64) * 16,
                    _count, 16, 16, 1, _dy, _qkv, output, _sequence, _sequence, 32,
                    _width, 1, 0, _sequence * _width, 32, 0,
                    1, 3 * _width, 0, _sequence * 3 * _width, 32, 2 * _width,
                    _sequence, 1, _sequence * _sequence, 0, 0, 0,
                    _heads, _first, 0, _causal == 2 ? 1 : 0);
                _lane.Run("attention_derivatives", _count * _sequence * 64L, 64,
                    _p, output, _sequence, _width, _heads, _causal);
            }
        }

        internal float[] Read(bool fused)
        {
            using var output = _lane.Upload(Initial);
            long h2d = _lane.H2DBytes, d2h = _lane.D2HBytes;
            Dispatch(fused, output); Dispatch(fused, output);
            if (_lane.H2DBytes != h2d || _lane.D2HBytes != d2h)
                throw new InvalidOperationException("Validation dP+dS performed a host transfer.");
            float[] result = new float[Initial.Length]; _lane.Read(output, result); return result;
        }

        public void Dispose() { _p.Dispose(); _dy.Dispose(); _qkv.Dispose(); }
    }

    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray();
        return (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
    }

    private static void AssertExact(float[] expected, float[] actual, string description)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException($"Length mismatch: {description}.");
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                throw new InvalidOperationException($"FP32 mismatch {description}, index {i}: {expected[i]:R} versus {actual[i]:R}.");
    }
}
