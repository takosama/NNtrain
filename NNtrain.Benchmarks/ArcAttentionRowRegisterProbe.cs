using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcAttentionRowRegisterProbe
{
    private readonly record struct Plan(bool Cached, bool ExactSubgroup = false)
    {
        internal string Name => ExactSubgroup ? "exact-subgroup-cache" : Cached ? "row-register-cache" : "reference";
        internal string Probability(int sequence) => ExactSubgroup ? $"attention_probabilities_exact_sg_{sequence}" : Cached ? $"attention_probabilities_register_{sequence}" : "attention_probabilities";
        internal string Derivative(int sequence) => ExactSubgroup ? $"attention_derivatives_exact_sg_{sequence}" : Cached ? $"attention_derivatives_register_{sequence}" : "attention_derivatives";
    }
    private static readonly Plan[] Plans = [new(false), new(true), new(true, true)];

    private sealed class Fixture : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly int Sequence, HeadWidth, Causal, First, Count, Rows, Elements;
        internal readonly ArcBuffer SourceScores, SourceDp, Scores, Recomputed, Dp, Stats, Values, Gradient;
        internal Fixture(ArcExecutionLane lane, int sequence, int headWidth, int causal, int count = 5, int first = 2)
        {
            if (sequence is not (512 or 1024)) throw new ArgumentOutOfRangeException(nameof(sequence));
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
                    // Never read masked signaling NaNs; outside published64
                    // both score and derivative payloads must remain unchanged.
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
            => Lane.Run(plan.Probability(Sequence), Rows * 64L, 64, output, Stats,
                Sequence, HeadWidth * 3, 3, First, Causal, saved ? 1 : 0);
        internal void Derivative(Plan plan)
            => Lane.Run(plan.Derivative(Sequence), Rows * 64L, 64, Scores, Dp, Sequence, HeadWidth * 3, 3, Causal);
        internal void AccumulateGradient()
        {
            Lane.Run3D("attention_gemm_compact_unrolled", ((HeadWidth + 31L) / 32) * 16,
                ((Sequence + 31L) / 32) * 16, Count, 16, 16, 1,
                Dp, Values, Gradient, Sequence, HeadWidth, Sequence,
                Sequence, 1, Sequence * Sequence, 0, 0, 0,
                HeadWidth, 1, Sequence * HeadWidth, 0, 0, 0,
                HeadWidth, 1, Sequence * HeadWidth, 0, 0, 0, 1, 0, 1, Causal != 0 ? 2 : 0);
        }
        internal void ResetScores() => Lane.CopyBytes(SourceScores, Scores, 0, 0, Elements * 4);
        internal void ResetDp() => Lane.CopyBytes(SourceDp, Dp, 0, 0, Elements * 4);
        internal float[][] Validate(Plan plan)
        {
            long uploads = Lane.H2DBytes, downloads = Lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                ResetScores(); ResetDp();
                Lane.CopyBytes(SourceScores, Recomputed, 0, 0, Elements * 4);
                Probability(plan, Scores, false); Probability(plan, Recomputed, true);
                Derivative(plan); AccumulateGradient();
            }
            if (Lane.H2DBytes != uploads || Lane.D2HBytes != downloads)
                throw new InvalidOperationException("Register-cache operation moved data to/from host.");
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
        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = true });
        if (!lane.Device.SupportsXmx || lane.Device.MinimumSubgroupSize != 16)
            throw new NotSupportedException("The downstream validation GEMM requires SG16 Intel XMX.");
        int validated = 0;
        foreach (int sequence in new[] { 512, 1024 })
        foreach (int headWidth in new[] { 5, 32 })
        foreach (int causal in new[] { 0, 1, 2 })
        {
            float[][]? expected = null;
            foreach (var plan in Plans)
            {
                using var fixture = new Fixture(lane, sequence, headWidth, causal);
                float[][] result = fixture.Validate(plan);
                if (expected is null) expected = result;
                else for (int i = 0; i < result.Length; i++)
                    AssertBits(expected[i], result[i], $"{plan.Name}, T{sequence}, D{headWidth}, causal={causal}, array={i}");
            }
            validated++;
        }
        Console.WriteLine($"Row-register cache: {validated} fixtures passed bitwise fresh/saved probabilities, "
            + "stats, derivatives and twice-accumulated gradients; masked signaling NaNs and untouched regions included.");

        var results = new List<object>();
        foreach (int sequence in new[] { 512, 1024 })
        foreach (int causal in new[] { 0, 2 })
        foreach (var plan in Plans)
        foreach (string operation in new[] { "probability", "saved-probability", "derivative" })
        {
            using var fixture = new Fixture(lane, sequence, 32, causal, count: 8, first: 0);
            fixture.ResetScores(); fixture.Probability(Plans[0], fixture.Scores, false); lane.Synchronize();
            var wall = new List<double>(); var gpu = new List<double>();
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            for (int sample = 0; sample < 13; sample++)
            {
                // Restore the same input on-device, drained before timing.
                if (operation == "derivative") fixture.ResetDp(); else fixture.ResetScores();
                lane.Synchronize();
                double beforeGpu = lane.KernelMilliseconds; long start = Stopwatch.GetTimestamp();
                if (operation == "derivative") fixture.Derivative(plan);
                else fixture.Probability(plan, fixture.Scores, operation == "saved-probability");
                lane.Synchronize();
                if (sample >= 3) { wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds); gpu.Add(lane.KernelMilliseconds - beforeGpu); }
            }
            if (lane.H2DBytes != uploads || lane.D2HBytes != downloads)
                throw new InvalidOperationException("Register-cache benchmark crossed the host/device boundary.");
            string kernel = operation == "derivative" ? plan.Derivative(sequence) : plan.Probability(sequence);
            var resources = lane.GetKernelResources(kernel);
            Console.WriteLine($"{operation} {plan.Name} T{sequence} causal={causal}: GPU={Median(gpu):F4} ms, "
                + $"wall={Median(wall):F4} ms, SLM={resources.LocalMemoryBytes}, spill={resources.SpillMemoryBytes}");
            results.Add(new { Operation = operation, Candidate = plan.Name, Kernel = kernel,
                P50Ms = Median(wall), GpuP50Ms = Median(gpu), Samples = wall, GpuSamples = gpu, Resources = resources,
                Sequence = sequence, HeadWidth = 32, Count = 8, Causal = causal, Device = lane.Device.Name, lane.Device.DriverVersion });
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow, ValidationCases = validated,
            Note = "Exact original 64-lane workgroup, modulo64 ordered FP32 FMA and reduction tree, SLM256B; "
                + "per-lane private caches are statically unrolled8/16 floats for probability and 2x8/16 for derivative. "
                + "Fresh probability reduces score traffic from 3 reads/2 writes to1 read/1 write, derivative from4 reads to2. "
                + "Full bitwise checks include fresh/saved probabilities/statistics, derivatives, two downstream gradient accumulations "
                + "and causal modes0/1/2 with masked signaling NaNs. 3warmup+10 measured dispatches; input restoration, "
                + "allocation, uploads and validation readback excluded. No normal dispatch changed.", Results = results },
            new JsonSerializerOptions { WriteIndented = true });
    }

    private static void AssertBits(float[] expected, float[] actual, string description)
    {
        if (expected.Length != actual.Length) throw new InvalidOperationException($"Length mismatch: {description}.");
        for (int i = 0; i < actual.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                throw new InvalidOperationException($"Bitwise mismatch {description}, index{i}: "
                    + $"{expected[i]:R}/0x{BitConverter.SingleToInt32Bits(expected[i]):x8} versus "
                    + $"{actual[i]:R}/0x{BitConverter.SingleToInt32Bits(actual[i]):x8}.");
    }
    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2;
    }
}
