using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2DecodeLevel3Tests(ITestOutputHelper output)
{
    private const string Pair2 = "q35l_iq2_s_sg16_pair", Level3 = "q35l_iq2_s_sg16_levels3";

    [Fact]
    public void Level3RemainsOptIn() => Assert.False(new Qwen35ExecutionOptions().InferenceIq2DecodeLevel3);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Level3RetainsPair2BitsForRowsTailsAndPackedScales(bool nativeHalf)
    {
        RequireDevice();
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true,
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
            using ArcBuffer previous = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocated = lane.AllocatedBytes;
            Run(lane, false, x, w, bias, previous, shape.M, shape.K, shape.N);
            Run(lane, true, x, w, bias, candidate, shape.M, shape.K, shape.N);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            Assert.Equal(allocated, lane.AllocatedBytes);
            Assert.Equal(Math.Max(4, payload.Length), w.ByteLength);
            AssertOutput(lane, previous, candidate, count);
            AssertBits(values, Read(lane, x, values.Length));
            AssertBits(biasValues, Read(lane, bias, biasValues.Length));
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ModelDecodeAndLoraFallbackRetainLogitsAndTokens(bool lora, bool fusedLora)
    {
        RequireDevice();
        using TemporaryQwenGguf file = Qwen35ResidentModelTests.CreateFixture(
            tiedOutput: false, contextLength: 48, iq2Qkv: true, iq2Gate: true);
        var options = new Qwen35ExecutionOptions
        {
            InferencePairedProjectionTypes = 5, InferenceFusedLora = fusedLora,
            InferencePairedLoraProjection = true, CollectKernelTimings = true
        };
        using Qwen35QuantizedModel previous = Qwen35QuantizedModel.Load(file.Path, [0], options: options);
        using Qwen35QuantizedModel candidate = Qwen35QuantizedModel.Load(file.Path, [0],
            options: options with { InferenceIq2DecodeLevel3 = true });
        if (lora)
        {
            var settings = new Qwen35LoraOptions { Rank = 2, Alpha = 4, IncludeOutput = true, Seed = 53 };
            previous.AttachLora(settings); candidate.AttachLora(settings);
            var random = new Random(117);
            foreach (string name in previous.LoraMatrices.Keys)
            {
                float[][] state = previous.LoraMatrices[name].ReadState();
                foreach (int part in new[] { 0, 1 })
                    for (int i = 0; i < state[part].Length; i++) state[part][i] = random.NextSingle() * .1f - .05f;
                previous.LoraMatrices[name].RestoreState(state, training: false);
                candidate.LoraMatrices[name].RestoreState(state, training: false);
            }
        }
        for (int i = 0; i < 13; i++)
            AssertBits(previous.ForwardToken((i * 3 + 1) % 4, i % 4 != 0),
                candidate.ForwardToken((i * 3 + 1) % 4, i % 4 != 0));
        Assert.DoesNotContain(Level3, previous.KernelMilliseconds.Keys);
        if (!lora || !fusedLora) Assert.Contains(Level3, candidate.KernelMilliseconds.Keys);
        if (lora && fusedLora) Assert.Contains(Pair2 + "_lora", candidate.KernelMilliseconds.Keys);
        Assert.Equal(previous.ResidentStateBytes, candidate.ResidentStateBytes);
        Assert.Equal(previous.GenerateTokenIds([1, 3, 0, 2], 8), candidate.GenerateTokenIds([1, 3, 0, 2], 8));
        AssertBits(previous.ForwardToken(2), candidate.ForwardToken(2));
    }

    [Fact]
    public void OptionalPair2VersusLevel3Benchmark()
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_IQ2_DECODE_LEVEL3_BENCHMARK") != "1") return;
        RequireDevice();
        foreach (bool profile in new[] { false, true })
        {
            using var lane = new ArcExecutionLane(0, new()
            {
                Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true,
                CacheKernelArguments = true, CollectKernelTimings = profile, Qwen35NativeHalfScale = true
            });
            foreach (var shape in new[] { (K: 5120, N: 6144), (K: 5120, N: 17408), (K: 17408, N: 5120) })
            {
                var random = new Random(shape.K + shape.N + 427);
                using ArcBuffer x = lane.Upload(Enumerable.Range(0, shape.K).Select(_ => random.NextSingle() * 2 - 1).ToArray());
                using ArcBuffer w = lane.UploadRaw(Weights(shape.K, shape.N, stress: false));
                using ArcBuffer bias = lane.Upload(Enumerable.Range(0, shape.N).Select(i => (i % 7 - 3) * .03f).ToArray());
                using ArcBuffer previous = lane.Upload(Enumerable.Repeat(float.NaN, shape.N + 5).ToArray());
                using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, shape.N + 5).ToArray());
                const int iterations = 16, warmup = 3, samples = 9;
                var measurements = new List<Sample>();
                for (int round = -warmup; round < samples; round++)
                {
                    TestContext.Current.CancellationToken.ThrowIfCancellationRequested();
                    foreach (bool level3 in (round & 1) == 0 ? new[] { false, true } : new[] { true, false })
                    {
                        lane.Synchronize();
                        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                        double gpu = lane.KernelMilliseconds;
                        long started = Stopwatch.GetTimestamp();
                        for (int iteration = 0; iteration < iterations; iteration++)
                            Run(lane, level3, x, w, bias, level3 ? candidate : previous, 1, shape.K, shape.N);
                        lane.Synchronize();
                        double wall = Stopwatch.GetElapsedTime(started).TotalMilliseconds / iterations;
                        Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
                        if (round >= 0) measurements.Add(new(round, level3, wall,
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
                var old = measurements.Where(item => !item.Level3).ToArray();
                var current = measurements.Where(item => item.Level3).ToArray();
                output.WriteLine(JsonSerializer.Serialize(new { benchmark = "IQ2_S decode pair2 versus level3", profile,
                    rows = 1, inputWidth = shape.K, outputWidth = shape.N,
                    device = new { lane.Device.Index, lane.Device.Name, lane.Device.DriverVersion },
                    scope = "Synthetic valid IQ2_S projection only, resident FP32 input; no host transfer, packing, allocation, or full-model generation in timing.",
                    complete = true, allOutputsBitwiseEqual = true, guardsIntact = true, nativeHalfScale = true, iterations, warmup, samples,
                    hostWallMedianSpeedup = old.Select(item => item.Wall).Order().ElementAt(samples / 2)
                        / current.Select(item => item.Wall).Order().ElementAt(samples / 2),
                    pair2Wall = Stats(old.Select(item => item.Wall)), level3Wall = Stats(current.Select(item => item.Wall)),
                    pair2Gpu = profile ? Stats(old.Select(item => item.Gpu!.Value)) : null,
                    level3Gpu = profile ? Stats(current.Select(item => item.Gpu!.Value)) : null,
                    rawSamples = measurements }));
            }
        }
    }

    private sealed record Sample(int Round, bool Level3, double Wall, double? Gpu);

    private static void RequireDevice()
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups"), "Intel SG16 is required.");
    }

    private static void Run(ArcExecutionLane lane, bool level3, ArcBuffer x, ArcBuffer weight,
        ArcBuffer bias, ArcBuffer output, int rows, int input, int columns)
    {
        const int outputsPerSubgroup = 2;
        long groups = (long)rows * ((columns + outputsPerSubgroup - 1) / outputsPerSubgroup);
        lane.Run(level3 ? Level3 : Pair2, Math.Max(32L, (groups + 1) / 2 * 32), 32,
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
