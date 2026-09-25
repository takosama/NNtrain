using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

// Diagnostic bounds for the production T2048 derivative, not dispatch options.
internal static class ArcAttentionDerivativeDiagnosticProbe
{
    private const int Sequence = 2048;
    private const int Heads = 16;
    private const int Width = 512;
    private const int Causal = 2;
    private const int ResidentHeads = 2;
    private const int Warmup = 3;
    private const int Samples = 10;
    private const string Production = "attention_derivatives";
    private const string Streaming = "attention_derivative_streaming_diagnostic_2048";
    private const string Reduction = "attention_derivative_reduction_diagnostic_2048";

    private sealed record Timing(double GpuP50Ms, double WallP50Ms, double[] GpuMs, double[] WallMs);

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = true });
        int rows = ResidentHeads * Sequence;
        int elements = checked(rows * Sequence);
        float[] probabilities = new float[elements];
        float[] gradient = new float[elements];
        for (int row = 0; row < rows; ++row)
        {
            int q = row % Sequence, begin = row * Sequence;
            float probability = 1f / (q + 1);
            for (int key = 0; key < Sequence; ++key)
            {
                int i = begin + key;
                probabilities[i] = key <= q ? probability : 0f;
                gradient[i] = ((i * 31L & 2047) - 1024) * .0001f;
            }
        }
        float[] smallProbabilities = new float[256];
        float[] smallGradient = new float[256];
        for (int i = 0; i < 256; ++i)
        {
            smallProbabilities[i] = ((i * 13 + 17) & 255) * .00001f + .0001f;
            smallGradient[i] = ((i * 29 + 7) & 255) * .0002f - .025f;
        }

        using var p = lane.Upload(probabilities);
        using var sourceDp = lane.Upload(gradient);
        using var productionDp = lane.Upload(gradient);
        using var streamingDp = lane.Upload(gradient);
        using var checksums = lane.AllocateBytes(checked(rows * sizeof(uint)));
        using var smallP = lane.Upload(smallProbabilities);
        using var smallDp = lane.Upload(smallGradient);
        using var sink = lane.Allocate(rows * 64);

        void Reset(ArcBuffer target)
        {
            lane.CopyBytes(sourceDp, target, 0, 0, checked(elements * sizeof(float)));
            lane.Synchronize(); // D2D reset is deliberately outside the timing interval.
        }
        void RunProduction() => lane.Run(Production, rows * 64L, 64, p, productionDp,
            Sequence, Width, Heads, Causal);
        void RunStreaming() => lane.Run(Streaming, rows * 64L, 64, p, streamingDp,
            checksums, Sequence, Width, Heads, Causal);
        void RunReduction() => lane.Run(Reduction, rows * 64L, 64, smallP, smallDp,
            sink, Sequence, Width, Heads, Causal, 17);

        // Compile every kernel and check obvious mistakes before sampling.
        var resources = new[] { Production, Streaming, Reduction }
            .ToDictionary(name => name, lane.GetKernelResources);
        RunProduction();
        RunStreaming();
        RunReduction();
        lane.Synchronize();
        var productionRow = new float[64];
        var streamingRow = new float[64];
        var reductionRow = new float[64];
        lane.ReadFloatRange(productionDp, 3 * Sequence, productionRow);
        lane.ReadFloatRange(streamingDp, 3 * Sequence, streamingRow);
        lane.ReadFloatRange(sink, 3 * 64, reductionRow);
        for (int key = 0; key < 64; ++key)
        {
            if (!float.IsFinite(productionRow[key]) || !float.IsFinite(streamingRow[key])
                || !float.IsFinite(reductionRow[key]))
                throw new InvalidOperationException($"Nonfinite derivative diagnostic at key {key}.");
            if (key < 4)
            {
                int i = 3 * Sequence + key;
                float expected = probabilities[i] + gradient[i];
                if (Math.Abs(streamingRow[key] - expected) > 1e-6f)
                    throw new InvalidOperationException($"Streaming access diagnostic failed at key {key}.");
            }
            else if (productionRow[key] != 0f || streamingRow[key] != 0f)
                throw new InvalidOperationException($"Causal block-tail diagnostic failed at key {key}.");
        }

        var gpu = new Dictionary<string, List<double>>();
        var wall = new Dictionary<string, List<double>>();
        foreach (string name in resources.Keys)
        {
            gpu[name] = new List<double>(Samples);
            wall[name] = new List<double>(Samples);
        }
        long h2dBefore = lane.H2DBytes, d2hBefore = lane.D2HBytes;
        for (int round = 0; round < Warmup + Samples; ++round)
        {
            // Rotate order to reduce thermal/drift bias; every input remains resident.
            (string Name, ArcBuffer? ResetTarget, Action Dispatch)[] work =
            [
                (Production, productionDp, RunProduction),
                (Streaming, streamingDp, RunStreaming),
                (Reduction, null, RunReduction),
            ];
            for (int offset = 0; offset < work.Length; ++offset)
            {
                var item = work[(round + offset) % work.Length];
                if (item.ResetTarget is not null) Reset(item.ResetTarget);
                double before = lane.KernelMilliseconds;
                long started = Stopwatch.GetTimestamp();
                item.Dispatch();
                lane.Synchronize();
                if (round >= Warmup)
                {
                    gpu[item.Name].Add(lane.KernelMilliseconds - before);
                    wall[item.Name].Add(Stopwatch.GetElapsedTime(started).TotalMilliseconds);
                }
            }
        }
        if (lane.H2DBytes != h2dBefore || lane.D2HBytes != d2hBefore)
            throw new InvalidOperationException("Timed derivative diagnostic crossed the host/device boundary.");

        static double Median(List<double> values)
        {
            double[] sorted = values.Order().ToArray();
            return (sorted[4] + sorted[5]) * .5;
        }
        var results = resources.Keys.Select(name => new
        {
            Kernel = name, Resources = resources[name],
            Timing = new Timing(Median(gpu[name]), Median(wall[name]), gpu[name].ToArray(), wall[name].ToArray()),
        }).ToArray();
        foreach (var result in results)
            Console.WriteLine($"{result.Kernel}: GPU={result.Timing.GpuP50Ms:F4} ms, "
                + $"wall={result.Timing.WallP50Ms:F4} ms, SLM={result.Resources.LocalMemoryBytes}, "
                + $"private={result.Resources.PrivateMemoryBytes}, spill={result.Resources.SpillMemoryBytes}");

        long valid = (long)ResidentHeads * Sequence * (Sequence + 1) / 2;
        long padded = (long)ResidentHeads * (Sequence / 64) * (64 * 63 / 2);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new
        {
            Timestamp = DateTimeOffset.UtcNow, lane.Device.Name, lane.Device.DriverVersion,
            Sequence, Heads, Width, Causal, ResidentHeads, Warmup, Samples,
            ValidEntries = valid, LogicalDerivativeBytes = valid * 20 + padded * sizeof(float),
            ExternalTrafficLowerBoundBytes = valid * 12 + padded * sizeof(float),
            Note = "Streaming and reduction variants are non-equivalent diagnostic bounds, not replacements. "
                + "Logical bytes count the production kernel's two P/dP reads and one dP write per valid entry; "
                + "external lower bound assumes cache serves the second reads. Neither is measured DRAM traffic. "
                + "Three warmup and ten measured rounds, rotating order. GPU-resident inputs; D2D reset excluded. "
                + "No H2D/D2H within timed rounds; full-update acceptance remains mandatory.",
            Results = results,
        }, new JsonSerializerOptions { WriteIndented = true });
    }
}
