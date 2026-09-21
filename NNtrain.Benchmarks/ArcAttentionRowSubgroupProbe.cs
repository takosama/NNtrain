using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcAttentionRowSubgroupProbe
{
    private readonly record struct Plan(string Name, int Workgroup)
    {
        internal bool Reference => Workgroup == 0;
        internal string ProbabilityKernel => Reference ? "attention_probabilities" : $"attention_probabilities_row_sg16_w{Workgroup}";
        internal string DerivativeKernel => Reference ? "attention_derivatives" : $"attention_derivatives_row_sg16_w{Workgroup}";
    }
    private static readonly Plan[] Plans = [new("reference", 0), new("row-sg16-w64", 64), new("row-sg16-w128", 128)];

    private sealed class Fixture : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly int Sequence, HeadWidth, Causal, First, Count, Rows, Elements;
        internal readonly ArcBuffer SourceScores, SourceDp, Scores, Recomputed, Dp, Stats, Values, Gradient;
        internal Fixture(ArcExecutionLane lane, int sequence, int headWidth, int causal, int count = 5, int first = 2)
        {
            Lane = lane; Sequence = sequence; HeadWidth = headWidth; Causal = causal;
            First = first; Count = count; Rows = count * sequence; Elements = Rows * sequence;
            float[] scores = Enumerable.Range(0, Elements).Select(i => MathF.Sin(i * .017f) * 20f).ToArray();
            float[] dp = Enumerable.Range(0, Elements).Select(i => MathF.ScaleB(MathF.Cos(i * .071f), i % 17 - 8)).ToArray();
            if (causal != 0)
                for (int g = 0; g < count; g++)
                for (int q = 0; q < sequence; q++)
                for (int key = q + 1; key < sequence; key++)
                {
                    int i = (g * sequence + q) * sequence + key;
                    // Causal masks must ignore these signaling NaNs. Values
                    // outside the published64-key region must remain untouched.
                    scores[i] = BitConverter.Int32BitsToSingle(0x7f801234);
                    dp[i] = BitConverter.Int32BitsToSingle(unchecked((int)0xff812345));
                }
            SourceScores = lane.Upload(scores); SourceDp = lane.Upload(dp);
            Scores = lane.Allocate(Elements); Recomputed = lane.Allocate(Elements); Dp = lane.Allocate(Elements);
            Stats = lane.Upload(Enumerable.Repeat(-.125f, (first + count + 1) * sequence * 2).ToArray());
            Values = lane.Upload(Enumerable.Range(0, Rows * headWidth).Select(i => MathF.Cos(i * .031f) * .003f).ToArray());
            Gradient = lane.Upload(Enumerable.Repeat(.125f, Rows * headWidth).ToArray());
        }
        internal void Probability(Plan plan, ArcBuffer output, bool saved)
        {
            if (plan.Reference)
                Lane.Run(plan.ProbabilityKernel, Rows * 64L, 64, output, Stats, Sequence, HeadWidth * 3, 3, First, Causal, saved ? 1 : 0);
            else
                Lane.Run(plan.ProbabilityKernel, ((Rows + (long)(plan.Workgroup / 16) - 1) / (plan.Workgroup / 16)) * plan.Workgroup,
                    plan.Workgroup, output, Stats, Sequence, HeadWidth * 3, 3, First, Causal, saved ? 1 : 0, Rows);
        }
        internal void Derivative(Plan plan)
        {
            if (plan.Reference)
                Lane.Run(plan.DerivativeKernel, Rows * 64L, 64, Scores, Dp, Sequence, HeadWidth * 3, 3, Causal);
            else
                Lane.Run(plan.DerivativeKernel, ((Rows + (long)(plan.Workgroup / 16) - 1) / (plan.Workgroup / 16)) * plan.Workgroup,
                    plan.Workgroup, Scores, Dp, Sequence, HeadWidth * 3, 3, Causal, Rows);
        }
        internal void AccumulateGradient()
        {
            Lane.Run3D("attention_gemm_compact_unrolled", ((HeadWidth + 31L) / 32) * 16, ((Sequence + 31L) / 32) * 16, Count, 16, 16, 1,
                Dp, Values, Gradient, Sequence, HeadWidth, Sequence,
                Sequence, 1, Sequence * Sequence, 0, 0, 0,
                HeadWidth, 1, Sequence * HeadWidth, 0, 0, 0,
                HeadWidth, 1, Sequence * HeadWidth, 0, 0, 0,
                1, 0, 1, Causal != 0 ? 2 : 0);
        }
        internal void ResetScores() => Lane.CopyBytes(SourceScores, Scores, 0, 0, Elements * 4);
        internal void ResetDp() => Lane.CopyBytes(SourceDp, Dp, 0, 0, Elements * 4);
        internal float[][] Validate(Plan plan)
        {
            long h2d = Lane.H2DBytes, d2h = Lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                ResetScores(); ResetDp();
                Lane.CopyBytes(SourceScores, Recomputed, 0, 0, Elements * 4);
                Probability(plan, Scores, false); Probability(plan, Recomputed, true);
                Derivative(plan); AccumulateGradient();
            }
            if (Lane.H2DBytes != h2d || Lane.D2HBytes != d2h) throw new InvalidOperationException("Row subgroup operation moved data to/from host.");
            ArcBuffer[] buffers = [Scores, Recomputed, Stats, Dp, Gradient];
            int[] lengths = [Elements, Elements, (First + Count + 1) * Sequence * 2, Elements, Rows * HeadWidth];
            var result = lengths.Select(length => new float[length]).ToArray();
            for (int i = 0; i < buffers.Length; i++) Lane.Read(buffers[i], result[i]);
            AssertBits(result[0], result[1], "fresh versus saved statistics");
            return result;
        }
        public void Dispose()
        {
            Gradient.Dispose(); Values.Dispose(); Stats.Dispose(); Dp.Dispose(); Recomputed.Dispose(); Scores.Dispose();
            SourceDp.Dispose(); SourceScores.Dispose();
        }
    }

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane();
        if (!lane.Options.XmxMatrices || !lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("The subgroup candidate requires SG16 Intel XMX compilation capabilities.");
        int validated = 0;
        foreach (int sequence in new[] { 1, 31, 65, 137, 1024 })
        foreach (int causal in new[] { 0, 1, 2 })
        {
            int headWidth = sequence is 31 or 137 ? 5 : 32;
            float[][]? expected = null;
            foreach (var plan in Plans)
            {
                using var fixture = new Fixture(lane, sequence, headWidth, causal);
                float[][] result = fixture.Validate(plan);
                if (expected is null) expected = result;
                else for (int i = 0; i < result.Length; i++) AssertBits(expected[i], result[i], $"{plan.Name}, T{sequence}, D{headWidth}, causal={causal}, array={i}");
            }
            validated++;
        }
        Console.WriteLine($"Row-SG16: {validated} fixtures passed full-array exact fresh/saved probabilities, stats, derivatives and twice-accumulated gradients; masked signaling NaNs and last-workgroup tails included.");

        var results = new List<object>();
        foreach (var plan in Plans)
        foreach (string operation in new[] { "probability", "saved-probability", "derivative" })
        {
            using var fixture = new Fixture(lane, 1024, 32, 2, count: 8, first: 0);
            fixture.ResetScores(); fixture.Probability(Plans[0], fixture.Scores, false); lane.Synchronize();
            var wall = new List<double>(); var gpu = new List<double>();
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int sample = 0; sample < 13; sample++)
            {
                // Input restoration is device-to-device and intentionally
                // drained BEFORE timing; all variants see identical inputs.
                if (operation == "derivative") fixture.ResetDp(); else fixture.ResetScores();
                lane.Synchronize();
                double beforeGpu = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp();
                if (operation == "derivative") fixture.Derivative(plan);
                else fixture.Probability(plan, fixture.Scores, operation == "saved-probability");
                lane.Synchronize();
                if (sample >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - beforeGpu); }
            }
            if (lane.H2DBytes != h2d || lane.D2HBytes != d2h) throw new InvalidOperationException("Benchmark crossed the host/device boundary.");
            var resources = lane.GetKernelResources(operation == "derivative" ? plan.DerivativeKernel : plan.ProbabilityKernel);
            double gpuMedian = Median(gpu), wallMedian = Median(wall);
            Console.WriteLine($"{operation} {plan.Name}: GPU={gpuMedian:F4} ms, wall={wallMedian:F4} ms, SLM={resources.LocalMemoryBytes}, spill={resources.SpillMemoryBytes}");
            results.Add(new { Operation = operation, Candidate = plan.Name, plan.Workgroup,
                P50Ms = wallMedian, GpuP50Ms = gpuMedian, Samples = wall, GpuSamples = gpu, Resources = resources,
                Sequence = 1024, HeadWidth = 32, Count = 8, Causal = 2, Device = lane.Device.Name, lane.Device.DriverVersion });
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow, ValidationCases = validated,
            Note = "One SG16 per row, four virtual-lane private partial streams and original 64-lane binary reduction order. No SLM/barriers or approximate math. Full bitwise array checks include fresh/saved probabilities/statistics, FP32 derivatives, two downstream gradient accumulations, causal modes 0/1/2, masked signaling NaNs and partial workgroups. 3 warmup+10 measured dispatches. Device-to-device input restoration, allocation, uploads and validation readback excluded from timing.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }
    private static void AssertBits(float[] expected, float[] actual, string description)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException($"Length mismatch: {description}.");
        for (int i = 0; i < actual.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                throw new InvalidOperationException($"Bitwise mismatch {description}, index {i}: {expected[i]:R}/0x{BitConverter.SingleToInt32Bits(expected[i]):x8} versus {actual[i]:R}/0x{BitConverter.SingleToInt32Bits(actual[i]):x8}.");
    }
    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2;
    }
}
