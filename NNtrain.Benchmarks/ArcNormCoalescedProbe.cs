using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Benchmarks;

internal static class ArcNormCoalescedProbe
{
    private enum Mode { Reference, Columns, Subgroup64, Subgroup128 }
    private sealed class Fixture : IDisposable
    {
        internal readonly ArcExecutionLane Lane;
        internal readonly int Rows, Width, Elements;
        internal readonly ArcBuffer X, Gamma, Beta, Dy, Stats, Output, Accumulated;
        internal Fixture(ArcExecutionLane lane, int rows, int width, string storage)
        {
            Lane = lane; Rows = rows; Width = width; Elements = checked(rows * width);
            float Value(int i)
            {
                float value = i % 37 == 0 ? 1000f : MathF.Sin(i * .019f) * .3f;
                if (storage == "bf16")
                {
                    uint bits = BitConverter.SingleToUInt32Bits(value);
                    bits = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
                    return BitConverter.UInt32BitsToSingle(bits);
                }
                return storage == "bfp8" ? (i % 255 - 127) * MathF.ScaleB(1f / 127f, (i / 32) % 7 - 3) : value;
            }
            X = lane.Upload(Enumerable.Range(0, Elements).Select(Value).ToArray());
            Dy = lane.Upload(Enumerable.Range(0, Elements).Select(i => MathF.ScaleB(MathF.Cos(i * .011f), -(i % 9))).ToArray());
            Gamma = lane.Upload(Enumerable.Range(0, width).Select(i => i % 9 == 0 ? -0f : .5f + MathF.Sin(i * .077f) * .8f).ToArray());
            Beta = lane.Upload(Enumerable.Range(0, width).Select(i => MathF.Cos(i * .027f) * .03f).ToArray());
            Stats = lane.Allocate(rows * 2); Output = lane.Allocate(Elements);
            Accumulated = lane.Upload(Enumerable.Repeat(.125f, Elements).ToArray());
            // Backward candidates always consume exactly the original forward
            // statistics, isolating their arithmetic from any forward change.
            Forward(Mode.Reference); lane.Synchronize();
        }
        private void Transpose(ArcBuffer input, ArcBuffer output, int rows, int columns)
            => Lane.Run2D("norm_dx_unpack_columns_candidate", ((columns + 31L) / 32) * 16, ((rows + 31L) / 32) * 16,
                16, 16, input, output, rows, columns);
        internal void Forward(Mode mode)
        {
            if (mode == Mode.Reference)
                Lane.Run("norm", Rows, 0, X, Gamma, Beta, Output, Stats, Rows, Width, 1e-5f);
            else if (mode == Mode.Columns)
            {
                using var column = Lane.Allocate(Elements);
                Transpose(X, column, Rows, Width);
                Lane.Run("norm_columns_serial_candidate", Rows, 0, column, Gamma, Beta, Stats, Rows, Width, 1e-5f);
                Transpose(column, Output, Width, Rows);
            }
            else
            {
                int group = mode == Mode.Subgroup64 ? 64 : 128;
                Lane.Run($"norm_row_sg16_w{group}_candidate", ((Rows + (long)(group / 16) - 1) / (group / 16)) * group,
                    group, X, Gamma, Beta, Output, Stats, Rows, Width, 1e-5f);
            }
        }
        internal void Backward(Mode mode)
        {
            if (mode == Mode.Reference)
                Lane.Run("norm_dx_set", Rows, 0, X, Gamma, Dy, Stats, Output, Rows, Width);
            else if (mode == Mode.Columns)
            {
                using var xc = Lane.Allocate(Elements); using var dyc = Lane.Allocate(Elements);
                Lane.Run2D("norm_dx_pack_columns_candidate", ((Width + 31L) / 32) * 16, ((Rows + 31L) / 32) * 16,
                    16, 16, X, Dy, xc, dyc, Rows, Width);
                Lane.Run("norm_dx_columns_serial_candidate", Rows, 0, xc, Gamma, dyc, Stats, Rows, Width);
                Transpose(xc, Output, Width, Rows);
            }
            else
            {
                int group = mode == Mode.Subgroup64 ? 64 : 128;
                Lane.Run($"norm_dx_row_sg16_w{group}_candidate", ((Rows + (long)(group / 16) - 1) / (group / 16)) * group,
                    group, X, Gamma, Dy, Stats, Output, Rows, Width);
            }
        }
        internal (float[] Output, float[] Stats, float[] Accumulated) Validate(Mode mode, bool forward)
        {
            long up = Lane.H2DBytes, down = Lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                if (forward) Forward(mode); else Backward(mode);
                Lane.Run("copy_scale", Elements, 0, Output, Accumulated, Elements, 1f, 1);
            }
            if (Lane.H2DBytes != up || Lane.D2HBytes != down) throw new InvalidOperationException("Norm candidate crossed the host/device boundary.");
            float[] output = new float[Elements], stats = new float[Rows * 2], accumulated = new float[Elements];
            Lane.Read(Output, output); Lane.Read(Stats, stats); Lane.Read(Accumulated, accumulated);
            return (output, stats, accumulated);
        }
        public void Dispose()
        {
            Accumulated.Dispose(); Output.Dispose(); Stats.Dispose(); Beta.Dispose(); Gamma.Dispose(); Dy.Dispose(); X.Dispose();
        }
    }

    internal static void Run(string path)
    {
        path = Path.GetFullPath(path);
        if (File.Exists(path)) throw new IOException("Probe output must be a new file.");
        using var lane = new ArcExecutionLane();
        bool subgroup = lane.Options.XmxMatrices && lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16;
        Mode[] modes = subgroup ? Enum.GetValues<Mode>() : [Mode.Reference, Mode.Columns];
        int validated = 0;
        foreach (bool forward in new[] { true, false })
        foreach (string storage in new[] { "f32", "bf16", "bfp8" })
        foreach (var (rows, width) in new[] { (1, 1), (31, 37), (137, 65), (257, 512), (65, 513) })
        {
            (float[] Output, float[] Stats, float[] Accumulated)? expected = null;
            foreach (var mode in modes)
            {
                using var fixture = new Fixture(lane, rows, width, storage);
                var actual = fixture.Validate(mode, forward);
                if (expected is null) expected = actual;
                else
                {
                    string condition = $"{mode}, forward={forward}, {storage}, {rows}x{width}";
                    AssertExact(expected.Value.Stats, actual.Stats, condition + " stats");
                    AssertExact(expected.Value.Output, actual.Output, condition + " output");
                    AssertExact(expected.Value.Accumulated, actual.Accumulated, condition + " twice accumulated");
                }
            }
            validated++;
        }
        Console.WriteLine($"Strict coalesced norm: {validated} full-array bitwise forward/backward fixtures passed, including row/channel tails, decoded BF16/BFP8 values, signed zero gamma and two accumulated outputs.");

        var results = new List<object>();
        foreach (var (rows, width) in new[] { (16384, 512), (65536, 512), (16384, 1536) })
        foreach (bool forward in new[] { true, false })
        {
            using var fixture = new Fixture(lane, rows, width, "f32");
            // Validate the complete production-sized output before timings.
            if (forward) fixture.Forward(Mode.Reference); else fixture.Backward(Mode.Reference);
            float[] expected = new float[fixture.Elements]; lane.Read(fixture.Output, expected);
            float[] expectedStats = new float[rows * 2]; lane.Read(fixture.Stats, expectedStats);
            foreach (var mode in modes)
            {
                if (forward) fixture.Forward(mode); else fixture.Backward(mode);
                float[] actual = new float[fixture.Elements]; lane.Read(fixture.Output, actual);
                AssertExact(expected, actual, $"production {rows}x{width}, forward={forward}, {mode}");
                float[] actualStats = new float[rows * 2]; lane.Read(fixture.Stats, actualStats);
                AssertExact(expectedStats, actualStats, $"production stats {rows}x{width}, forward={forward}, {mode}");
                var wall = new List<double>(); var gpu = new List<double>(); var transpose = new List<double>();
                double TransposeTime() => lane.KernelTimings.Where(p => p.Key is "norm_dx_pack_columns_candidate" or "norm_dx_unpack_columns_candidate").Sum(p => p.Value);
                long up = lane.H2DBytes, down = lane.D2HBytes;
                for (int sample = 0; sample < 13; sample++)
                {
                    double before = lane.KernelMilliseconds, transBefore = TransposeTime(); long start = Stopwatch.GetTimestamp();
                    if (forward) fixture.Forward(mode); else fixture.Backward(mode);
                    lane.Synchronize();
                    if (sample >= 3)
                    {
                        wall.Add(Stopwatch.GetElapsedTime(start).TotalMilliseconds);
                        gpu.Add(lane.KernelMilliseconds - before); transpose.Add(TransposeTime() - transBefore);
                    }
                }
                if (lane.H2DBytes != up || lane.D2HBytes != down) throw new InvalidOperationException("Benchmark transferred tensor data.");
                string kernel = mode switch
                {
                    Mode.Reference => forward ? "norm" : "norm_dx_set",
                    Mode.Columns => forward ? "norm_columns_serial_candidate" : "norm_dx_columns_serial_candidate",
                    Mode.Subgroup64 => forward ? "norm_row_sg16_w64_candidate" : "norm_dx_row_sg16_w64_candidate",
                    _ => forward ? "norm_row_sg16_w128_candidate" : "norm_dx_row_sg16_w128_candidate"
                };
                var resources = lane.GetKernelResources(kernel);
                long scratchBytes = mode == Mode.Columns ? (long)fixture.Elements * 4 * (forward ? 1 : 2) : 0;
                Console.WriteLine($"norm {(forward ? "forward" : "backward")} {rows}x{width} {mode}: wall={Median(wall):F3} ms, GPU(total)={Median(gpu):F3} ms, transpose={Median(transpose):F3} ms, scratch={scratchBytes / 1048576.0:F1} MiB, spill={resources.SpillMemoryBytes}");
                results.Add(new { Mode = mode.ToString(), Forward = forward, Rows = rows, Width = width,
                    P50Ms = Median(wall), GpuP50Ms = Median(gpu), TransposeP50Ms = Median(transpose),
                    Samples = wall, GpuSamples = gpu, TransposeSamples = transpose, ScratchBytes = scratchBytes,
                    Resources = resources, FullArrayExact = true, Device = lane.Device.Name, lane.Device.DriverVersion });
            }
        }
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var file = new FileStream(path, FileMode.CreateNew);
        JsonSerializer.Serialize(file, new { Timestamp = DateTimeOffset.UtcNow, ValidationCases = validated,
            Note = "Strict original serial channel order and FP32 expressions. Column-major route includes all transposes, temporary allocations and releases in wall time and ALL three kernels in GPU time. SG16 route has no SLM/barriers/scratch. 3 warmup+10 samples, initial source upload and validation readback excluded. Whole-array bit checks cover outputs/statistics/twice accumulated outputs and production matrices. Native source tensors contain FP32 values or exact decoded BF16/BFP8 values; no arithmetic precision reduction.",
            Results = results }, new JsonSerializerOptions { WriteIndented = true });
    }
    private static void AssertExact(float[] expected, float[] actual, string condition)
    {
        for (int i = 0; i < expected.Length; i++)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                throw new InvalidOperationException($"Strict norm mismatch {condition}, index {i}: {expected[i]:R}/0x{BitConverter.SingleToInt32Bits(expected[i]):x8} versus {actual[i]:R}/0x{BitConverter.SingleToInt32Bits(actual[i]):x8}.");
    }
    private static double Median(List<double> values)
    {
        double[] sorted = values.Order().ToArray(); return (sorted[4] + sorted[5]) / 2;
    }
}
