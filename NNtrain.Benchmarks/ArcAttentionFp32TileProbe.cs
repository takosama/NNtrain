using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcAttentionFp32TileProbe
{
    private readonly record struct Plan(string Kernel, int Rows, int Columns, int LocalRows, bool Dense,
        string? Operation = null, bool D32Only = false, bool AlignedSequence = false);
    private readonly record struct Matrix(ArcBuffer Buffer, int Row, int Column, int Group, int Batch, int Head, int Offset)
    {
        internal Matrix Transpose() => this with { Row = Column, Column = Row };
    }
    private sealed class Fixture : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly int Sequence, Width, Heads, First, Count, D;
        internal readonly bool Causal;
        internal readonly string Operation;
        internal readonly ArcBuffer Input, Dy, Scores;
        internal readonly float[] Initial;
        internal Fixture(ArcExecutionLane lane, string operation, int sequence, int d, int heads, int first, int count, bool causal)
        {
            Lane = lane; Operation = operation; Sequence = sequence; D = d; Heads = heads;
            Width = d * heads; First = first; Count = count; Causal = causal;
            int batch = (first + count + heads - 1) / heads;
            Input = lane.Upload(Enumerable.Range(0, batch * sequence * Width * 3).Select(i => MathF.Sin(i * .013f) * .13f).ToArray());
            Dy = lane.Upload(Enumerable.Range(0, batch * sequence * Width).Select(i => MathF.Cos(i * .017f) * .03f).ToArray());
            float[] scores = Enumerable.Range(0, count * sequence * sequence).Select(i => MathF.Sin(i * .021f) * .013f).ToArray();
            if (causal)
                for (int g = 0; g < count; g++)
                for (int q = 0; q < sequence; q++)
                for (int key = q + 1; key < sequence; key++) scores[(g * sequence + q) * sequence + key] = 0;
            Scores = lane.Upload(scores);
            int outputElements = operation == "dp" ? count * sequence * sequence
                : batch * sequence * Width * (operation == "pv" ? 1 : 3);
            Initial = Enumerable.Repeat(.125f, outputElements).ToArray();
        }
        internal Plan Reference => Operation switch
        {
            "dp" => new("attention_gemm", 32, 32, 16, true),
            "pv" => new("attention_gemm_narrow", 64, 32, 16, false),
            _ => new("attention_gemm_compact_unrolled", 32, 32, 16, false)
        };
        internal bool Accumulate => Operation is "dq" or "dk" or "dv";
        internal (int M, int N, int K) Shape => Operation == "dp" ? (Sequence, Sequence, D) : (Sequence, D, Sequence);
        internal void Dispatch(Plan plan, ArcBuffer output)
        {
            Matrix Qkv(ArcBuffer x, int component) => new(x, 3 * Width, 1, 0, Sequence * 3 * Width, D, component * Width);
            Matrix Y(ArcBuffer x) => new(x, Width, 1, 0, Sequence * Width, D, 0);
            Matrix P(ArcBuffer x) => new(x, Sequence, 1, Sequence * Sequence, 0, 0, 0);
            var (a, b, c, mode) = Operation switch
            {
                "dp" => (Y(Dy), Qkv(Input, 2).Transpose(), P(output), 1),
                "pv" => (P(Scores), Qkv(Input, 2), Y(output), 2),
                "dq" => (P(Scores), Qkv(Input, 1), Qkv(output, 0), 2),
                "dk" => (P(Scores).Transpose(), Qkv(Input, 0), Qkv(output, 1), 3),
                "dv" => (P(Scores).Transpose(), Y(Dy), Qkv(output, 2), 3),
                _ => throw new InvalidOperationException(Operation)
            };
            var (m, n, k) = Shape;
            Lane.Run3D(plan.Kernel, ((n + (long)plan.Columns - 1) / plan.Columns) * 16,
                ((m + (long)plan.Rows - 1) / plan.Rows) * plan.LocalRows, Count, 16, plan.LocalRows, 1,
                a.Buffer, b.Buffer, c.Buffer, m, n, k,
                a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset,
                b.Row, b.Column, b.Group, b.Batch, b.Head, b.Offset,
                c.Row, c.Column, c.Group, c.Batch, c.Head, c.Offset,
                Heads, First, Accumulate ? 1 : 0, Causal ? mode : 0);
        }
        internal float[] ExecuteForValidation(Plan plan)
        {
            using var output = Lane.Upload(Initial);
            long up = Lane.H2DBytes, down = Lane.D2HBytes;
            Dispatch(plan, output); Dispatch(plan, output);
            if (Lane.H2DBytes != up || Lane.D2HBytes != down)
                throw new InvalidOperationException($"{plan.Kernel} moved data during the measured operator.");
            float[] values = new float[Initial.Length]; Lane.Read(output, values); return values;
        }
        public void Dispose() { Scores.Dispose(); Dy.Dispose(); Input.Dispose(); }
    }

    private static readonly Plan[] Candidates =
    [
        new("attention_fp32_dp_64x64x32", 64, 64, 16, true),
        new("attention_fp32_dp_64x64x16", 64, 64, 16, true),
        new("attention_fp32_dp_32x64x32", 32, 64, 16, true),
        new("attention_fp32_dp_64x64x32_w128", 64, 64, 8, true),
        new("attention_fp32_n_32x32x64", 32, 32, 16, false),
        new("attention_fp32_n_64x32x32", 64, 32, 16, false),
        new("attention_fp32_n_64x32x64", 64, 32, 16, false),
        new("attention_fp32_n_128x32x32", 128, 32, 16, false),
        new("attention_fp32_n_128x32x64", 128, 32, 16, false),
        new("attention_fp32_n_256x32x16", 256, 32, 16, false),
        new("attention_fp32_n_64x32x32_w128", 64, 32, 8, false),
        new("attention_fp32_n_32x32x64_w128", 32, 32, 8, false),
        new("attention_fp32_dp_d32_special", 64, 64, 16, true, "dp", true),
        new("attention_fp32_dp_d32_aligned", 64, 64, 16, true, "dp", true, true),
        new("attention_fp32_dp_d32_block_slm", 64, 64, 16, true, "dp", true, true),
        new("attention_fp32_pv_d32_special", 64, 32, 16, false, "pv", true),
        new("attention_fp32_pv_d32_aligned", 64, 32, 16, false, "pv", true, true),
        new("attention_fp32_dq_d32_special", 64, 32, 16, false, "dq", true),
        new("attention_fp32_dq_d32_aligned", 64, 32, 16, false, "dq", true, true),
        new("attention_fp32_pv_d32_block_slm", 64, 32, 16, false, "pv", true, true),
        new("attention_fp32_dq_d32_block_slm", 64, 32, 16, false, "dq", true, true)
    ];

    internal static void Run(string path, bool specializationOnly = false)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = true });
        var results = new List<object>();
        var validated = new Dictionary<string, int>();
        foreach (string operation in specializationOnly ? new[] { "dp", "pv", "dq" } : new[] { "dp", "pv", "dq", "dk", "dv" })
        {
            Plan[] candidates = Candidates.Where(p => p.Dense == (operation == "dp") && (p.Operation is null || p.Operation == operation)
                && (!specializationOnly || p.D32Only || p.Kernel is "attention_fp32_dp_64x64x16" or "attention_fp32_n_64x32x32")).ToArray();
            foreach (bool causal in new[] { false, true })
            foreach (var (sequence, d) in new[] { (65, 31), (137, 65), (137, 32), (128, 32) })
            {
                using var fixture = new Fixture(lane, operation, sequence, d, heads: 3, first: 1, count: 4, causal);
                float[] expected = fixture.ExecuteForValidation(fixture.Reference);
                foreach (var plan in candidates)
                {
                    if ((plan.D32Only && d != 32) || (plan.AlignedSequence && sequence % 64 != 0)) continue;
                    AssertExact(expected, fixture.ExecuteForValidation(plan), $"{operation}, T{sequence}, D{d}, causal={causal}, {plan.Kernel}");
                    validated[plan.Kernel] = validated.GetValueOrDefault(plan.Kernel) + 1;
                }
            }
            // Eight simultaneous heads reproduce the existing 64 MiB two-score
            // workspace budget at sequence 1024. Input remains interleaved.
            using var production = new Fixture(lane, operation, 1024, 32, heads: 16, first: 0, count: 8, causal: true);
            float[] reference = production.ExecuteForValidation(production.Reference);
            foreach (var plan in new[] { production.Reference }.Concat(candidates))
            {
                AssertExact(reference, production.ExecuteForValidation(plan), $"production {operation}, {plan.Kernel}");
                var resources = lane.GetKernelResources(plan.Kernel);
                using var output = lane.Upload(production.Initial);
                var wall = new List<double>(); var gpu = new List<double>();
                for (int step = 0; step < 13; step++)
                {
                    double beforeGpu = lane.KernelMilliseconds;
                    long start = Stopwatch.GetTimestamp();
                    production.Dispatch(plan, output); lane.Synchronize();
                    if (step >= 3)
                    {
                        wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                        gpu.Add(lane.KernelMilliseconds - beforeGpu);
                    }
                }
                double wallMedian = Median(wall), gpuMedian = Median(gpu);
                Console.WriteLine($"{operation} {plan.Kernel}: wall={wallMedian:F4} ms, GPU={gpuMedian:F4} ms, SLM={resources.LocalMemoryBytes}, spill={resources.SpillMemoryBytes}, exact checks={validated.GetValueOrDefault(plan.Kernel) + 1}");
                results.Add(new
                {
                    Operation = operation, plan.Kernel, plan.Rows, plan.Columns, plan.LocalRows,
                    P50Ms = wallMedian, GpuP50Ms = gpuMedian, Samples = wall, GpuSamples = gpu,
                    Resources = resources, ExactTailChecks = validated.GetValueOrDefault(plan.Kernel), plan.D32Only, plan.AlignedSequence,
                    ExactProductionCheck = true, PackingBytes = 0, PackingMilliseconds = 0,
                    production.Sequence, HeadWidth = production.D, production.Count, production.Causal,
                    Accumulation = production.Accumulate, Device = lane.Device.Name, lane.Device.DriverVersion
                });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new
        {
            Timestamp = DateTimeOffset.UtcNow,
            Note = "FP32 FMA in original increasing-K order. Full-array bitwise correctness before timing, causal/noncausal and row/channel tails, two accumulated calls. 3 warmup + 10 measured dispatches, eight heads at T1024 D32. No packing. Upload and validation readback outside timing.",
            Results = results
        }, new JsonSerializerOptions { WriteIndented = true });
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
