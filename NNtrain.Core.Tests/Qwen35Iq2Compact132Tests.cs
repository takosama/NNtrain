using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2Compact132Tests(ITestOutputHelper output)
{
    private const string Pair2 = "q35l_iq2_s_sg16_pair", Compact132 = "q35c_iq2_pair132";

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Compact132RetainsPair2BitsForRowsTailsAndPackedScales(bool nativeHalf)
    {
        RequireDevice();
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true, ExperimentalOptimizationKernels = true,
            Qwen35NativeHalfScale = nativeHalf
        });
        foreach (var shape in new[] { (M: 0, K: 256, N: 5), (M: 1, K: 256, N: 0),
            (M: 1, K: 256, N: 1), (M: 2, K: 512, N: 6), (M: 3, K: 512, N: 7),
            (M: 1, K: 5120, N: 33), (M: 2, K: 17408, N: 9) })
        {
            int count = shape.M * shape.N;
            var random = new Random(shape.K + shape.N + 427);
            float[] values = Enumerable.Range(0, shape.M * shape.K)
                .Select(i => i % 17 == 0 ? -0f : (random.NextSingle() * 2 - 1) * .03f).ToArray();
            float[] biasValues = Enumerable.Range(0, shape.N).Select(i => (i % 7 - 3) * .03f).ToArray();
            byte[] payload = Weights(shape.K, shape.N, stress: true);
            using ArcBuffer x = lane.Upload(values), w = lane.UploadRaw(payload), bias = lane.Upload(biasValues);
            using ArcBuffer compact = lane.Upload(Enumerable.Repeat(float.NaN, payload.Length / 82 * 33 + 5).ToArray());
            Pack(lane, w, compact, payload.Length / 82);
            AssertPacked(lane, compact, payload);
            using ArcBuffer previous = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
            Run(lane, false, x, w, bias, previous, shape.M, shape.K, shape.N);
            Run(lane, true, x, compact, bias, candidate, shape.M, shape.K, shape.N);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            Assert.Equal(allocated, lane.AllocatedBytes);
            Assert.Equal(Math.Max(4, payload.Length), w.ByteLength);
            AssertOutput(lane, previous, candidate, count);
            AssertBits(values, Read(lane, x, values.Length));
            AssertBits(biasValues, Read(lane, bias, biasValues.Length));
        }
    }

    [Fact]
    public void OptionalPair2VersusCompact132Benchmark()
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_IQ2_COMPACT132_BENCHMARK") != "1") return;
        RequireDevice();
        foreach (bool profile in new[] { false, true })
        {
            using var lane = new ArcExecutionLane(0, new()
            {
                Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true, ExperimentalOptimizationKernels = true,
                CacheKernelArguments = true, CollectKernelTimings = profile, Qwen35NativeHalfScale = true
            });
            foreach (var shape in new[] { (K: 5120, N: 6144), (K: 5120, N: 17408), (K: 17408, N: 5120) })
            {
                var random = new Random(shape.K + shape.N + 427);
                using ArcBuffer x = lane.Upload(Enumerable.Range(0, shape.K).Select(_ => random.NextSingle() * 2 - 1).ToArray());
                byte[] raw = Weights(shape.K, shape.N, stress: false);
                using ArcBuffer w = lane.UploadRaw(raw);
                using ArcBuffer compact = lane.AllocateBytes(raw.Length / 82 * 132);
                var packTiming = Pack(lane, w, compact, raw.Length / 82);
                using ArcBuffer bias = lane.Upload(Enumerable.Range(0, shape.N).Select(i => (i % 7 - 3) * .03f).ToArray());
                using ArcBuffer previous = lane.Upload(Enumerable.Repeat(float.NaN, shape.N + 5).ToArray());
                using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, shape.N + 5).ToArray());
                const int iterations = 16, warmup = 3, samples = 9;
                var measurements = new List<Sample>();
                for (int round = -warmup; round < samples; round++)
                {
                    TestContext.Current.CancellationToken.ThrowIfCancellationRequested();
                    foreach (bool compact132 in (round & 1) == 0 ? new[] { false, true } : new[] { true, false })
                    {
                        lane.Synchronize();
                        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                        double gpu = lane.KernelMilliseconds;
                        long started = Stopwatch.GetTimestamp();
                        for (int iteration = 0; iteration < iterations; iteration++)
                            Run(lane, compact132, x, compact132 ? compact : w, bias, compact132 ? candidate : previous, 1, shape.K, shape.N);
                        lane.Synchronize();
                        double wall = Stopwatch.GetElapsedTime(started).TotalMilliseconds / iterations;
                        Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
                        if (round >= 0) measurements.Add(new(round, compact132, wall,
                            profile ? (lane.KernelMilliseconds - gpu) / iterations : null));
                    }
                    AssertOutput(lane, previous, candidate, shape.N);
                }
                object Stats(IEnumerable<double> values)
                {
                    double[] ordered = values.Order().ToArray(); double mean = ordered.Average();
                    return new { count = ordered.Length, median = ordered[ordered.Length / 2], mean,
                        standardDeviation = Math.Sqrt(ordered.Average(value => (value - mean) * (value - mean))),
                        minimum = ordered[0], maximum = ordered[^1] };
                }
                var old = measurements.Where(item => !item.Compact132).ToArray();
                var current = measurements.Where(item => item.Compact132).ToArray();
                output.WriteLine(JsonSerializer.Serialize(new { benchmark = "IQ2_S decode pair2 versus compact132", profile,
                    rows = 1, inputWidth = shape.K, outputWidth = shape.N,
                    device = new { lane.Device.Index, lane.Device.Name, lane.Device.DriverVersion },
                    scope = "Synthetic valid IQ2_S projection only, resident FP32 input; no host transfer, packing, allocation, or full-model generation in decode timing; one-time GPU packing is measured separately.",
                    complete = true, allOutputsBitwiseEqual = true, guardsIntact = true, nativeHalfScale = true, iterations, warmup, samples,
                    hostWallMedianSpeedup = old.Select(item => item.Wall).Order().ElementAt(samples / 2)
                        / current.Select(item => item.Wall).Order().ElementAt(samples / 2),
                    pair2Wall = Stats(old.Select(item => item.Wall)), compact132Wall = Stats(current.Select(item => item.Wall)),
                    pair2Gpu = profile ? Stats(old.Select(item => item.Gpu!.Value)) : null,
                    compact132Gpu = profile ? Stats(current.Select(item => item.Gpu!.Value)) : null,
                    rawWeightBytes = raw.Length, compactWeightBytes = raw.Length / 82 * 132, packTiming,
                    rawSamples = measurements }));
            }
        }
    }

    private sealed record Sample(int Round, bool Compact132, double Wall, double? Gpu);
    private sealed record Packing(double WallMilliseconds, double? GpuMilliseconds,
        long InputBytes, long OutputBytes, string Scope);

    private static Packing Pack(ArcExecutionLane lane, ArcBuffer raw, ArcBuffer compact, int blocks)
    {
        lane.Synchronize();
        double beforeGpu = lane.KernelMilliseconds;
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        long started = Stopwatch.GetTimestamp();
        lane.Run("q35c_iq2_pack132", Math.Max(32L, (long)blocks * 32), 32, raw, compact, blocks);
        lane.Synchronize();
        double wall = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
        Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
        return new(wall, lane.Options.CollectKernelTimings ? lane.KernelMilliseconds - beforeGpu : null,
            (long)blocks * 82, (long)blocks * 132,
            "One-time GPU recoding including dispatch and synchronization, excluding allocation/upload; one observation, not steady-state decode latency.");
    }

    private static void AssertPacked(ArcExecutionLane lane, ArcBuffer compact, byte[] raw)
    {
        int blocks = raw.Length / 82, count = blocks * 33;
        float[] words = Read(lane, compact, count + 5);
        for (int block = 0; block < blocks; block++)
        {
            uint coefficient = (uint)raw[block * 82] | ((uint)raw[block * 82 + 1] << 8);
            Assert.Equal(coefficient, BitConverter.SingleToUInt32Bits(words[block * 33 + 32]));
            for (int octet = 0; octet < 32; octet++)
            {
                uint descriptor = BitConverter.SingleToUInt32Bits(words[block * 33 + octet]);
                Assert.Equal(0u, descriptor >> 28);
                uint scale = (uint)(raw[block * 82 + 74 + octet / 4] >> ((octet % 4 / 2) * 4)) & 15;
                Assert.Equal(scale, (descriptor >> 24) & 15);
                int signs = raw[block * 82 + 34 + octet];
                for (int component = 0; component < 8; component++)
                {
                    uint code = (descriptor >> (component * 3)) & 7;
                    Assert.InRange(code & 3, 0u, 2u);
                    Assert.Equal((uint)(signs >> component) & 1, code >> 2);
                }
            }
        }
        for (int i = count; i < count + 5; i++) Assert.True(float.IsNaN(words[i]));
    }

    private static void RequireDevice()
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups"), "Intel SG16 is required.");
    }

    private static void Run(ArcExecutionLane lane, bool compact132, ArcBuffer x, ArcBuffer weight,
        ArcBuffer bias, ArcBuffer output, int rows, int input, int columns)
    {
        const int outputsPerSubgroup = 2;
        long groups = (long)rows * ((columns + outputsPerSubgroup - 1) / outputsPerSubgroup);
        lane.Run(compact132 ? Compact132 : Pair2, Math.Max(32L, (groups + 1) / 2 * 32), 32,
            x, weight, bias, output, rows, input, columns);
    }

    private static byte[] Weights(int input, int output, bool stress)
    {
        byte[] payload = new byte[input / 256 * output * 82];
        new Random(input + output + 881).NextBytes(payload);
        ushort[] scales = stress ? [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff,
            0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff] : [0x1400, 0x1800, 0x1000, 0x9400];
        for (int block = 0; block < payload.Length / 82; block++)
        {
            ushort bits = scales[block % scales.Length];
            payload[block * 82] = (byte)bits; payload[block * 82 + 1] = (byte)(bits >> 8);
            if (stress)
                for (int group = 0; group < 8; group++)
                    payload[block * 82 + 74 + group] = (byte)(((block * 8 + group) % 16)
                        | (((block * 8 + group + 8) % 16) << 4));
        }
        return payload;
    }

    private static void AssertOutput(ArcExecutionLane lane, ArcBuffer previous, ArcBuffer candidate, int count)
    {
        float[] expected = Read(lane, previous, count + 5), actual = Read(lane, candidate, count + 5);
        AssertBits(expected[..count], actual[..count]);
        Assert.All(actual[..count], value => Assert.True(float.IsFinite(value)));
        for (int i = count; i < count + 5; i++) Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]));
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var values = new float[count];
        if (count > 0) lane.Read(buffer, values);
        return values;
    }

    private static void AssertBits(float[] expected, float[] actual)
        => Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));
}
