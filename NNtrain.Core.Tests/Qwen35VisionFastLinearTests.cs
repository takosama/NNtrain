using System.Diagnostics;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35VisionFastLinearTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1, 3, 2, false)]
    [InlineData(63, 33, 65, true)]
    [InlineData(67, 4304, 1152, false)]
    [InlineData(130, 1152, 3456, true)]
    public void CoalescedRegisterTilePreservesLegacyProjection(int rows, int inputWidth,
        int outputWidth, bool gelu)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        var random = new Random(29005 + inputWidth);
        float[] source = Enumerable.Range(0, rows * inputWidth)
            .Select(_ => (random.NextSingle() * 2f - 1f) * .6f).ToArray();
        ushort[] weights = Enumerable.Range(0, inputWidth * outputWidth)
            .Select(_ => BitConverter.HalfToUInt16Bits((Half)((random.NextSingle() * 2f - 1f) * .04f)))
            .ToArray();
        float[] biases = Enumerable.Range(0, outputWidth)
            .Select(_ => (random.NextSingle() * 2f - 1f) * .1f).ToArray();
        using var lane = CreateLane();
        using ArcBuffer input = lane.Upload(source);
        using ArcBuffer weight = lane.AllocateBytes(weights.Length * sizeof(ushort));
        lane.WriteRaw(weight, weights);
        using ArcBuffer bias = lane.Upload(biases);
        using ArcBuffer expected = lane.Allocate(rows * outputWidth);
        using ArcBuffer actual = lane.Allocate(rows * outputWidth);
        lane.Run2D("q35v_linear_f16", Round16(outputWidth), Round16(rows), 16, 16,
            input, weight, bias, expected, rows, inputWidth, outputWidth, gelu ? 1 : 0);
        RunFast(lane, input, weight, bias, actual, rows, inputWidth, outputWidth, gelu);
        var referenceValues = new float[rows * outputWidth];
        var actualValues = new float[referenceValues.Length];
        lane.Read(expected, referenceValues);
        lane.Read(actual, actualValues);
        float maximumError = 0;
        for (int i = 0; i < actualValues.Length; ++i)
        {
            Assert.True(float.IsFinite(actualValues[i]));
            maximumError = Math.Max(maximumError, Math.Abs(referenceValues[i] - actualValues[i]));
        }
        output.WriteLine($"{rows} x {inputWidth} x {outputWidth}: maximum error {maximumError:G9}");
        Assert.InRange(maximumError, 0, 2e-6f);

        // An independently accumulated CPU spot check catches accidental shared
        // indexing mistakes, including widths that do not fill the last tile.
        foreach ((int row, int col) in new[] { (0, 0), (rows - 1, outputWidth - 1) })
        {
            float sum = 0;
            for (int k = 0; k < inputWidth; ++k)
                sum = MathF.FusedMultiplyAdd(source[row * inputWidth + k],
                    (float)BitConverter.UInt16BitsToHalf(weights[col * inputWidth + k]), sum);
            sum += biases[col];
            if (gelu) sum = .5f * sum * (1f + MathF.Tanh(.7978845608028654f
                * (sum + .044715f * sum * sum * sum)));
            Assert.InRange(Math.Abs(actualValues[row * outputWidth + col] - sum), 0, 2e-5f);
        }
    }

    [Fact]
    public void ProjectionMicrobenchmarkWhenRequested()
    {
        Assert.SkipWhen(Environment.GetEnvironmentVariable("NNTRAIN_VISION_LINEAR_BENCHMARK") != "1",
            "Set NNTRAIN_VISION_LINEAR_BENCHMARK=1 to run the vision projection benchmark.");
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using var lane = CreateLane();
        foreach ((int rows, int k, int n) in new[]
        {
            (64, 1152, 3456), (256, 1152, 4304), (256, 4304, 1152),
            (576, 4608, 4608), (576, 4608, 5120)
        })
        {
            using ArcBuffer input = lane.Upload(Enumerable.Repeat(.05f, rows * k).ToArray());
            ushort[] halves = Enumerable.Repeat(BitConverter.HalfToUInt16Bits((Half).02f), k * n).ToArray();
            using ArcBuffer weight = lane.AllocateBytes(halves.Length * sizeof(ushort));
            lane.WriteRaw(weight, halves);
            using ArcBuffer bias = lane.Upload(new float[n]);
            using ArcBuffer result = lane.Allocate(rows * n);
            void Baseline() => lane.Run2D("q35v_linear_f16", Round16(n), Round16(rows), 16, 16,
                input, weight, bias, result, rows, k, n, 0);
            void Fast() => RunFast(lane, input, weight, bias, result, rows, k, n, false);
            Baseline(); Fast(); lane.Synchronize();
            double oldMilliseconds = Measure(lane, Baseline);
            double newMilliseconds = Measure(lane, Fast);
            double xmxMilliseconds = double.NaN;
            double packedMilliseconds = double.NaN, packAllMilliseconds = double.NaN;
            if (CanUseXmx(lane))
            {
                void Xmx() => RunXmx(lane, input, weight, bias, result, rows, k, n, false);
                Xmx(); lane.Synchronize();
                xmxMilliseconds = Measure(lane, Xmx);
                using var panels = new PackedOperands(lane, rows, k, n);
                void PackAll()
                {
                    panels.Pack(lane, input, weight);
                    panels.Run(lane, bias, result, false);
                }
                PackAll(); lane.Synchronize();
                packedMilliseconds = Measure(lane, () => panels.Run(lane, bias, result, false));
                packAllMilliseconds = Measure(lane, PackAll);
            }
            output.WriteLine($"{lane.Device.Name}: M={rows} K={k} N={n}; "
                + $"legacy={oldMilliseconds:F3} ms, tiled={newMilliseconds:F3} ms, "
                + $"speedup={oldMilliseconds / newMilliseconds:F2}x, "
                + $"xmx={xmxMilliseconds:F3} ms, xmx speedup={oldMilliseconds / xmxMilliseconds:F2}x, "
                + $"packed={packedMilliseconds:F3} ms, packed with pack={packAllMilliseconds:F3} ms");
        }
    }

    [Theory]
    [InlineData(1, 3, 2, false)]
    [InlineData(67, 33, 65, true)]
    [InlineData(130, 4304, 1152, false)]
    [InlineData(130, 1152, 3456, true)]
    public void XmxResidualExpansionPreservesProjectionPrecision(int rows, int k, int n, bool gelu)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using var lane = CreateLane();
        Assert.SkipWhen(!CanUseXmx(lane), "F16 XMX with SG16 is required.");
        var random = new Random(7105 + k);
        float[] values = Enumerable.Range(0, rows * k)
            .Select(_ => (random.NextSingle() * 2f - 1f) * 2.7f).ToArray();
        ushort[] halves = Enumerable.Range(0, k * n)
            .Select(_ => BitConverter.HalfToUInt16Bits((Half)((random.NextSingle() * 2f - 1f) * .035f)))
            .ToArray();
        using ArcBuffer input = lane.Upload(values);
        using ArcBuffer weight = lane.AllocateBytes(halves.Length * sizeof(ushort));
        lane.WriteRaw(weight, halves);
        using ArcBuffer bias = lane.Upload(Enumerable.Range(0, n)
            .Select(_ => random.NextSingle() * .1f - .05f).ToArray());
        using ArcBuffer reference = lane.Allocate(rows * n);
        using ArcBuffer result = lane.Allocate(rows * n);
        lane.Run2D("q35v_linear_f16", Round16(n), Round16(rows), 16, 16,
            input, weight, bias, reference, rows, k, n, gelu ? 1 : 0);
        RunXmx(lane, input, weight, bias, result, rows, k, n, gelu);
        var expected = new float[rows * n];
        var actual = new float[expected.Length];
        lane.Read(reference, expected);
        lane.Read(result, actual);
        using var panels = new PackedOperands(lane, rows, k, n);
        panels.Pack(lane, input, weight);
        panels.Run(lane, bias, result, gelu);
        var packed = new float[actual.Length];
        lane.Read(result, packed);
        Assert.Equal(actual, packed);
        double errorSquared = 0, magnitudeSquared = 0;
        float maximumError = 0;
        for (int i = 0; i < actual.Length; ++i)
        {
            Assert.True(float.IsFinite(actual[i]));
            double error = actual[i] - expected[i];
            maximumError = Math.Max(maximumError, (float)Math.Abs(error));
            errorSquared += error * error;
            magnitudeSquared += expected[i] * (double)expected[i];
        }
        double relativeError = Math.Sqrt(errorSquared / magnitudeSquared);
        output.WriteLine($"XMX M={rows} K={k} N={n}; maxerror={maximumError:G9}, relativeL2={relativeError:G9}");
        Assert.InRange(maximumError, 0, 3e-5f);
        Assert.InRange(relativeError, 0, 3e-6);
    }

    private static double Measure(ArcExecutionLane lane, Action run)
    {
        const int repeats = 3;
        var timer = Stopwatch.StartNew();
        for (int i = 0; i < repeats; ++i) run();
        lane.Synchronize();
        return timer.Elapsed.TotalMilliseconds / repeats;
    }

    private static ArcExecutionLane CreateLane() => new(options: new()
    {
        Qwen35InferenceKernelsOnly = true,
        CacheProgramBinary = false
    });

    private static void RunFast(ArcExecutionLane lane, ArcBuffer input, ArcBuffer weight,
        ArcBuffer bias, ArcBuffer result, int rows, int k, int n, bool gelu)
        => lane.Run2D("q35v_linear_f16_fast", ((n + 63L) / 64) * 16,
            ((rows + 63L) / 64) * 16, 16, 16,
            input, weight, bias, result, rows, k, n, gelu ? 1 : 0);

    private static long Round16(int value) => ((value + 15L) / 16) * 16;

    private static bool CanUseXmx(ArcExecutionLane lane)
        => lane.Device.SupportsXmx && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_khr_fp16");

    private static void RunXmx(ArcExecutionLane lane, ArcBuffer input, ArcBuffer weight,
        ArcBuffer bias, ArcBuffer result, int rows, int k, int n, bool gelu)
        => lane.Run2D("q35v_linear_f16_xmx", ((n + 63L) / 64) * 16,
            ((rows + 127L) / 128) * 16, 16, 16,
            input, weight, bias, result, rows, k, n, gelu ? 1 : 0);

    private sealed class PackedOperands : IDisposable
    {
        private readonly int _rows, _k, _n;
        private readonly ArcBuffer _high, _low, _weight;
        private readonly int _aElements, _bElements;

        public PackedOperands(ArcExecutionLane lane, int rows, int k, int n)
        {
            _rows = rows; _k = k; _n = n;
            _aElements = ((rows + 7) / 8) * ((k + 15) / 16) * 128;
            _bElements = ((n + 15) / 16) * ((k + 15) / 16) * 128;
            _high = lane.AllocateBytes(_aElements * sizeof(ushort));
            _low = lane.AllocateBytes(_aElements * sizeof(ushort));
            _weight = lane.AllocateBytes(_bElements * sizeof(uint));
        }

        public void Pack(ArcExecutionLane lane, ArcBuffer input, ArcBuffer weight)
        {
            lane.Run("q35v_linear_pack_a_f16", _aElements, 0,
                input, _high, _low, _rows, _k);
            lane.Run("q35v_linear_pack_b_f16", _bElements, 0, weight, _weight, _k, _n);
        }

        public void Run(ArcExecutionLane lane, ArcBuffer bias, ArcBuffer result, bool gelu)
            => lane.Run2D("q35v_linear_f16_xmx_packed", ((_n + 63L) / 64) * 16,
                ((_rows + 127L) / 128) * 16, 16, 16,
                _high, _low, _weight, bias, result, _rows, _k, _n, gelu ? 1 : 0);

        public void Dispose()
        {
            _high.Dispose(); _low.Dispose(); _weight.Dispose();
        }
    }
}
