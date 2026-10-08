using System.Diagnostics;
using System.Text.Json;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35DeltaDecodeCachedTests(ITestOutputHelper output)
{
    private const string CandidateKernel = "q35d_recurrent_gated_rmsnorm_rows128_subgroup";

    [Theory]
    [InlineData(16, 48, 128, 4, true, true)]
    [InlineData(2, 4, 128, 4, false, true)]
    [InlineData(3, 6, 128, 1, true, true)]
    [InlineData(2, 4, 128, 4, true, false)]
    [InlineData(2, 4, 7, 4, false, true)]
    public void CachedSingleTokenRetainsEveryOutputAndBothStatesAcrossUpdates(
        int keyHeads, int valueHeads, int width, int convKernel, bool parallelNorm, bool subgroupEnabled)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true,
            Qwen35ParallelDeltaNorm = parallelNorm, Qwen35DeltaSubgroupRms = subgroupEnabled
        });
        using var data = new DeltaData(lane, keyHeads, valueHeads, width, convKernel);
        bool expectCandidate = width == 128 && subgroupEnabled && HasSubgroup(lane);
        for (int token = 0; token < 6; token++)
        {
            data.WriteInputs(token);
            using ArcBuffer reference = data.Step(cacheState: false);
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            long launches = lane.KernelLaunchCount, allocated = lane.AllocatedBytes;
            using (ArcBuffer candidate = data.Step(cacheState: true))
            {
                Assert.Equal(uploads, lane.H2DBytes);
                Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(width == 128 ? 3L : 4L, lane.KernelLaunchCount - launches);
                Assert.Equal(allocated + data.Values * sizeof(float), lane.AllocatedBytes);
                AssertBits(Read(lane, reference, data.Values), Read(lane, candidate, data.Values));
                data.AssertStatesEqual();
            }
            Assert.Equal(allocated, lane.AllocatedBytes);
        }
        Assert.Equal(expectCandidate, lane.KernelTimings.ContainsKey(CandidateKernel));
        Assert.Contains(width == 128 ? "q35d_recurrent_gated_rmsnorm_fused128" : "q35d_recurrent",
            lane.KernelTimings.Keys);
    }

    [Fact]
    public void OptionalSingleTokenBenchmark()
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_DELTA_DECODE_BENCHMARK") != "1") return;
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups"), "SG16 Intel subgroups are required.");
        // The no-profile pass supplies primary host latency. A separate event
        // pass measures GPU work and proves the selected recurrent dispatch.
        foreach (bool profile in new[] { false, true })
        {
            using var lane = new ArcExecutionLane(0, new()
            {
                Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true,
                Qwen35ParallelDeltaNorm = true, Qwen35DeltaSubgroupRms = true,
                CollectKernelTimings = profile, CacheKernelArguments = true,
                BufferPoolBytes = 512L * 1024 * 1024
            });
            using var data = new DeltaData(lane, 16, 48, 128, 4);
            data.WriteInputs(0);
            const int updates = 32, warmup = 3, samples = 9;
            var measurements = new List<Measurement>();
            Measurement Measure(bool cached, int round)
            {
                data.ResetState(cached);
                lane.Synchronize();
                var before = new Dictionary<string, double>(lane.KernelTimings);
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                long started = Stopwatch.GetTimestamp();
                ArcBuffer? last = null;
                try
                {
                    for (int token = 0; token < updates; token++)
                    {
                        last?.Dispose();
                        last = data.Step(cached);
                    }
                    lane.Synchronize();
                    double wall = Stopwatch.GetElapsedTime(started).TotalMilliseconds / updates;
                    Assert.Equal(uploads, lane.H2DBytes);
                    Assert.Equal(downloads, lane.D2HBytes);
                    var kernels = lane.KernelTimings.ToDictionary(pair => pair.Key,
                        pair => (pair.Value - before.GetValueOrDefault(pair.Key)) / updates)
                        .Where(pair => pair.Value > 0).ToDictionary(pair => pair.Key, pair => pair.Value);
                    if (profile) Assert.Equal(cached, kernels.ContainsKey(CandidateKernel));
                    float[] result = Read(lane, last!, data.Values);
                    Assert.All(result, value => Assert.True(float.IsFinite(value)));
                    return new(round, cached, wall, profile ? kernels.Values.Sum() : null, kernels, result);
                }
                finally { last?.Dispose(); }
            }
            for (int round = -warmup; round < samples; round++)
            {
                TestContext.Current.CancellationToken.ThrowIfCancellationRequested();
                bool reverse = (round & 1) != 0;
                Measurement first = Measure(reverse, round), second = Measure(!reverse, round);
                AssertBits(first.Output, second.Output);
                data.AssertStatesEqual();
                if (round >= 0) { measurements.Add(first); measurements.Add(second); }
            }
            object Stats(IEnumerable<double> source)
            {
                double[] values = source.Order().ToArray();
                double mean = values.Average();
                return new { count = values.Length, median = values[values.Length / 2], mean,
                    standardDeviation = Math.Sqrt(values.Average(value => (value - mean) * (value - mean))),
                    minimum = values[0], maximum = values[^1] };
            }
            var original = measurements.Where(item => !item.Cached).OrderBy(item => item.Round).ToArray();
            var candidate = measurements.Where(item => item.Cached).OrderBy(item => item.Round).ToArray();
            double Median(IEnumerable<double> values) => values.Order().ElementAt(samples / 2);
            output.WriteLine(JsonSerializer.Serialize(new
            {
                benchmark = "DeltaStepFused single-token cached-state comparison", profile,
                keyHeads = 16, valueHeads = 48, headWidth = 128, convKernel = 4,
                updatesPerSample = updates, warmup, samples, seed = DeltaData.Seed,
                device = new { lane.Device.Index, lane.Device.Name, lane.Device.DriverVersion },
                scope = "One DeltaNet update, including unchanged convolution/normalization and temporary buffers; excludes state reset, transfers and readback. 32 sequential resident updates per timing sample. Not a full-model tokens/sec result.",
                complete = true, outputsAndStatesBitwiseEqual = true,
                hostWallMedianSpeedup = Median(original.Select(item => item.WallMilliseconds)) / Median(candidate.Select(item => item.WallMilliseconds)),
                originalWallPerUpdate = Stats(original.Select(item => item.WallMilliseconds)),
                candidateWallPerUpdate = Stats(candidate.Select(item => item.WallMilliseconds)),
                originalGpuPerUpdate = profile ? Stats(original.Select(item => item.GpuMilliseconds!.Value)) : null,
                candidateGpuPerUpdate = profile ? Stats(candidate.Select(item => item.GpuMilliseconds!.Value)) : null,
                rawSamples = measurements.Select(item => new { item.Round, item.Cached, item.WallMilliseconds,
                    item.GpuMilliseconds, item.KernelMilliseconds })
            }));
        }
    }

    private sealed record Measurement(int Round, bool Cached, double WallMilliseconds,
        double? GpuMilliseconds, Dictionary<string, double> KernelMilliseconds, float[] Output);

    private static bool HasSubgroup(ArcExecutionLane lane) => lane.Device.MinimumSubgroupSize == 16
        && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var values = new float[count];
        if (count > 0) lane.Read(buffer, values);
        return values;
    }

    private static void AssertBits(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int index = 0; index < expected.Length; index++)
            Assert.True(float.IsFinite(expected[index]) && float.IsFinite(actual[index])
                && BitConverter.SingleToInt32Bits(expected[index]) == BitConverter.SingleToInt32Bits(actual[index]),
                $"Mismatch at {index}: expected {expected[index]:R}, actual {actual[index]:R}.");
    }

    private sealed class DeltaData : IDisposable
    {
        internal const int Seed = 328771;
        private readonly ArcExecutionLane lane;
        private readonly int keyHeads, valueHeads, width, convKernel, channels;
        private readonly List<ArcBuffer> owned = [];
        private readonly float[] initialHistory, initialState;
        private readonly ArcBuffer qkv, gate, alpha, beta, weights, dt, a, norm;
        private readonly ArcBuffer referenceHistory, referenceState, candidateHistory, candidateState;
        internal int Values { get; }

        internal DeltaData(ArcExecutionLane lane, int keyHeads, int valueHeads, int width, int convKernel)
        {
            this.lane = lane; this.keyHeads = keyHeads; this.valueHeads = valueHeads;
            this.width = width; this.convKernel = convKernel;
            Values = valueHeads * width; channels = (2 * keyHeads + valueHeads) * width;
            var random = new Random(Seed + keyHeads + width + convKernel);
            float[] RandomValues(int count, float scale) => Enumerable.Range(0, count)
                .Select(_ => (float)(2 * random.NextDouble() - 1) * scale).ToArray();
            ArcBuffer Upload(float[] values) { ArcBuffer buffer = lane.Upload(values); owned.Add(buffer); return buffer; }
            weights = Upload(RandomValues(channels * convKernel, .6f));
            dt = Upload(RandomValues(valueHeads, .5f));
            a = Upload(RandomValues(valueHeads, 1).Select(value => -MathF.Abs(value) - .15f).ToArray());
            norm = Upload(RandomValues(width, .4f).Select(value => 1 + value).ToArray());
            initialHistory = RandomValues(channels * (convKernel - 1), .6f);
            initialState = RandomValues(Values * width, .25f);
            referenceHistory = Upload(initialHistory); candidateHistory = Upload(initialHistory);
            referenceState = Upload(initialState); candidateState = Upload(initialState);
            qkv = Upload(new float[channels]); gate = Upload(new float[Values]);
            alpha = Upload(new float[valueHeads]); beta = Upload(new float[valueHeads]);
        }

        internal void WriteInputs(int token)
        {
            var random = new Random(Seed + token * 7919);
            float[] ValuesOf(int count) => Enumerable.Range(0, count).Select(_ => (float)(2 * random.NextDouble() - 1)).ToArray();
            float[] qkvValues = ValuesOf(channels), gateValues = ValuesOf(Values);
            float[] alphaValues = ValuesOf(valueHeads), betaValues = ValuesOf(valueHeads);
            alphaValues[0] = betaValues[0] = (token & 1) == 0 ? -30 : 30;
            gateValues[0] = -80; gateValues[width] = 80;
            lane.Write(qkv, qkvValues); lane.Write(gate, gateValues);
            lane.Write(alpha, alphaValues); lane.Write(beta, betaValues);
        }

        internal ArcBuffer Step(bool cacheState) => Qwen35Gpu.DeltaStepFused(lane,
            qkv, gate, alpha, beta, weights, dt, a, norm,
            cacheState ? candidateHistory : referenceHistory, cacheState ? candidateState : referenceState,
            keyHeads, valueHeads, width, convKernel, 1e-6f, cacheState);

        internal void ResetState(bool cached)
        {
            if (initialHistory.Length > 0) lane.Write(cached ? candidateHistory : referenceHistory, initialHistory);
            lane.Write(cached ? candidateState : referenceState, initialState);
        }

        internal void AssertStatesEqual()
        {
            AssertBits(Read(lane, referenceHistory, initialHistory.Length), Read(lane, candidateHistory, initialHistory.Length));
            AssertBits(Read(lane, referenceState, initialState.Length), Read(lane, candidateState, initialState.Length));
        }

        public void Dispose()
        {
            foreach (ArcBuffer buffer in owned) buffer.Dispose();
        }
    }
}
