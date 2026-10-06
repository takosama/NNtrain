using System.Diagnostics;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuDeltaRowsTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1, 2, 4, 1, false)]
    [InlineData(2, 4, 4, 7, false)]
    [InlineData(2, 4, 4, 16, true)]
    [InlineData(3, 6, 1, 17, true)]
    [InlineData(2, 4, 4, 129, true)]
    public void ChunkMatchesEverySequentialOutputAndFinalStateExactly(
        int keyHeads, int valueHeads, int convKernel, int rows, bool parallelNorm)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            CacheProgramBinary = true,
            Qwen35ParallelDeltaNorm = parallelNorm
        });
        Compare(lane, keyHeads, valueHeads, convKernel, rows);
    }

    private static void Compare(ArcExecutionLane lane, int keyHeads, int valueHeads, int convKernel, int rows)
    {
        const int width = 128;
        const float eps = 1e-6f;
        int values = valueHeads * width, channels = (2 * keyHeads + valueHeads) * width;
        var random = new Random(38167 + rows + convKernel);
        float[] RandomValues(int count, float scale = 1f)
            => Enumerable.Range(0, count).Select(_ => (float)(2 * random.NextDouble() - 1) * scale).ToArray();
        float[] weights = RandomValues(channels * convKernel, 0.6f);
        float[] dt = RandomValues(valueHeads, 0.5f);
        float[] a = RandomValues(valueHeads).Select(x => -MathF.Abs(x) - 0.15f).ToArray();
        float[] norm = RandomValues(width, 0.4f).Select(x => x + 1f).ToArray();
        float[] history = RandomValues(channels * (convKernel - 1), 0.6f);
        float[] state = RandomValues(values * width, 0.25f);
        float[] qkv = RandomValues(rows * channels), gate = RandomValues(rows * values);
        float[] alpha = RandomValues(rows * valueHeads), beta = RandomValues(rows * valueHeads);
        // Include saturated activations without changing the state-order test.
        alpha[0] = -30f; beta[0] = -30f; gate[0] = -80f;
        if (rows > 1) { alpha[valueHeads] = 30f; beta[valueHeads] = 30f; gate[values] = 80f; }
        using ArcBuffer weightsGpu = lane.Upload(weights), dtGpu = lane.Upload(dt);
        using ArcBuffer aGpu = lane.Upload(a), normGpu = lane.Upload(norm);
        using ArcBuffer candidateHistory = lane.Upload(history), candidateState = lane.Upload(state);
        using ArcBuffer referenceHistory = lane.Upload(history), referenceState = lane.Upload(state);
        using ArcBuffer qkvGpu = lane.Upload(qkv), gateGpu = lane.Upload(gate);
        using ArcBuffer alphaGpu = lane.Upload(alpha), betaGpu = lane.Upload(beta);
        var expected = new float[rows * values];
        for (int row = 0; row < rows; row++)
        {
            using ArcBuffer rowQkv = lane.Upload(qkv.AsSpan(row * channels, channels).ToArray());
            using ArcBuffer rowGate = lane.Upload(gate.AsSpan(row * values, values).ToArray());
            using ArcBuffer rowAlpha = lane.Upload(alpha.AsSpan(row * valueHeads, valueHeads).ToArray());
            using ArcBuffer rowBeta = lane.Upload(beta.AsSpan(row * valueHeads, valueHeads).ToArray());
            using ArcBuffer output = Qwen35Gpu.DeltaStepFused(lane, rowQkv, rowGate, rowAlpha,
                rowBeta, weightsGpu, dtGpu, aGpu, normGpu, referenceHistory, referenceState,
                keyHeads, valueHeads, width, convKernel, eps);
            Read(lane, output, values).CopyTo(expected, row * values);
        }
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes, launches = lane.KernelLaunchCount;
        using ArcBuffer candidate = Qwen35Gpu.DeltaRowsFused(lane, qkvGpu, gateGpu, alphaGpu,
            betaGpu, weightsGpu, dtGpu, aGpu, normGpu, candidateHistory, candidateState,
            keyHeads, valueHeads, width, convKernel, eps, rows,
            cancellationToken: TestContext.Current.CancellationToken, unrollState: false, subgroupRms: false);
        Assert.Equal(uploads, lane.H2DBytes);
        Assert.Equal(downloads, lane.D2HBytes);
        Assert.Equal(2L + (rows + 127) / 128, lane.KernelLaunchCount - launches);
        AssertBitsEqual(expected, Read(lane, candidate, expected.Length));
        AssertBitsEqual(Read(lane, referenceState, state.Length), Read(lane, candidateState, state.Length));
        AssertBitsEqual(Read(lane, referenceHistory, history.Length), Read(lane, candidateHistory, history.Length));
        if (lane.Options.Qwen35DeltaSubgroupRms && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups"))
        {
            lane.Write(candidateHistory, history);
            lane.Write(candidateState, state);
            using ArcBuffer subgroup = Qwen35Gpu.DeltaRowsFused(lane, qkvGpu, gateGpu, alphaGpu,
                betaGpu, weightsGpu, dtGpu, aGpu, normGpu, candidateHistory, candidateState,
                keyHeads, valueHeads, width, convKernel, eps, rows,
                cancellationToken: TestContext.Current.CancellationToken, subgroupRms: true);
            AssertBitsEqual(expected, Read(lane, subgroup, expected.Length));
            AssertBitsEqual(Read(lane, referenceState, state.Length), Read(lane, candidateState, state.Length));
            AssertBitsEqual(Read(lane, referenceHistory, history.Length), Read(lane, candidateHistory, history.Length));
        }
        lane.Write(candidateHistory, history);
        lane.Write(candidateState, state);
        using ArcBuffer unrolled = Qwen35Gpu.DeltaRowsFused(lane, qkvGpu, gateGpu, alphaGpu,
            betaGpu, weightsGpu, dtGpu, aGpu, normGpu, candidateHistory, candidateState,
            keyHeads, valueHeads, width, convKernel, eps, rows,
            cancellationToken: TestContext.Current.CancellationToken, unrollState: true, subgroupRms: false);
        AssertBitsEqual(expected, Read(lane, unrolled, expected.Length));
        AssertBitsEqual(Read(lane, referenceState, state.Length), Read(lane, candidateState, state.Length));
        AssertBitsEqual(Read(lane, referenceHistory, history.Length), Read(lane, candidateHistory, history.Length));
    }

    [Fact]
    public void FullModelDimensionsRetainExactOutputsAcrossManyRows()
        => ChunkMatchesEverySequentialOutputAndFinalStateExactly(16, 48, 4, 64, true);

    [Fact]
    public void FullModelDimensionsRetainExactOutputsAcrossDispatchBoundary()
        => ChunkMatchesEverySequentialOutputAndFinalStateExactly(16, 48, 4, 129, true);

    [Fact]
    public void SubgroupCandidateRetainsEveryOutputAndStateForAllSevenShapes()
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups"),
            "SG16 Intel subgroups are required.");
        foreach (bool parallelNorm in new[] { false, true })
        {
            using var lane = new ArcExecutionLane(0, new()
            {
                Qwen35InferenceKernelsOnly = true, Qwen35DeltaSubgroupRms = true,
                Qwen35ParallelDeltaNorm = parallelNorm, CacheProgramBinary = true
            });
            if (!parallelNorm)
            {
                Compare(lane, 1, 2, 4, 1);
                Compare(lane, 2, 4, 4, 7);
            }
            else
            {
                Compare(lane, 2, 4, 4, 16);
                Compare(lane, 3, 6, 1, 17);
                Compare(lane, 2, 4, 4, 129);
                Compare(lane, 16, 48, 4, 64);
                Compare(lane, 16, 48, 4, 129);
            }
        }
    }

    [Fact]
    public void OptionalBenchmark()
    {
        if (Environment.GetEnvironmentVariable("NNTRAIN_DELTA_ROWS_BENCHMARK") != "1") return;
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35ParallelDeltaNorm = true,
            Qwen35DeltaSubgroupRms = true,
            CacheProgramBinary = true,
            CollectKernelTimings = false, CacheKernelArguments = true
        });
        const int keyHeads = 16, valueHeads = 48, width = 128, rows = 256, convKernel = 4;
        int channels = (2 * keyHeads + valueHeads) * width, values = valueHeads * width;
        float[] Filled(int length, float value) => Enumerable.Repeat(value, length).ToArray();
        using ArcBuffer qkv = lane.Upload(Filled(rows * channels, .15f));
        using ArcBuffer gate = lane.Upload(Filled(rows * values, .3f));
        using ArcBuffer alpha = lane.Upload(Filled(rows * valueHeads, .2f));
        using ArcBuffer beta = lane.Upload(Filled(rows * valueHeads, .1f));
        using ArcBuffer weights = lane.Upload(Filled(channels * convKernel, .1f));
        using ArcBuffer dt = lane.Upload(Filled(valueHeads, .01f));
        using ArcBuffer a = lane.Upload(Filled(valueHeads, -.8f));
        using ArcBuffer norm = lane.Upload(Filled(width, 1f));
        using ArcBuffer history = lane.Upload(Filled(channels * (convKernel - 1), 0f));
        using ArcBuffer state = lane.Upload(Filled(values * width, 0f));
        using ArcBuffer rowQkv = lane.Allocate(channels), rowGate = lane.Allocate(values);
        using ArcBuffer rowAlpha = lane.Allocate(valueHeads), rowBeta = lane.Allocate(valueHeads);
        void Sequential()
        {
            for (int row = 0; row < rows; row++)
            {
                lane.CopyBytes(qkv, rowQkv, row * channels * 4, 0, channels * 4);
                lane.CopyBytes(gate, rowGate, row * values * 4, 0, values * 4);
                lane.CopyBytes(alpha, rowAlpha, row * valueHeads * 4, 0, valueHeads * 4);
                lane.CopyBytes(beta, rowBeta, row * valueHeads * 4, 0, valueHeads * 4);
                using ArcBuffer output = Qwen35Gpu.DeltaStepFused(lane, rowQkv, rowGate,
                    rowAlpha, rowBeta, weights, dt, a, norm, history, state,
                    keyHeads, valueHeads, width, convKernel, 1e-6f);
            }
        }
        void Batched(bool cached, bool unrolled = false, bool subgroupRms = false)
        {
            using ArcBuffer output = Qwen35Gpu.DeltaRowsFused(lane, qkv, gate, alpha, beta,
                weights, dt, a, norm, history, state, keyHeads, valueHeads, width,
                convKernel, 1e-6f, rows, cached, TestContext.Current.CancellationToken, unrolled, subgroupRms);
        }
        double Time(Action action)
        {
            lane.Synchronize(); var clock = Stopwatch.StartNew(); action(); lane.Synchronize();
            return clock.Elapsed.TotalMilliseconds;
        }
        Sequential(); Batched(false); Batched(true); Batched(true, true); Batched(true, true, true); lane.Synchronize();
        double sequential = Time(Sequential), batched = Time(() => Batched(false));
        double cached = Time(() => Batched(true));
        double unrolled = Time(() => Batched(true, true));
        double subgroup = Time(() => Batched(true, true, true));
        output.WriteLine($"DeltaRows exact full dimensions: sequential {sequential:F3} ms, batched {batched:F3} ms, cached {cached:F3} ms, unrolled {unrolled:F3} ms, subgroup {subgroup:F3} ms, speedup {sequential / subgroup:F2}x");
        output.WriteLine($"cached resources: {lane.GetKernelResources("q35d_recurrent_gated_rmsnorm_rows128_cached")}");
        output.WriteLine($"unrolled resources: {lane.GetKernelResources("q35d_recurrent_gated_rmsnorm_rows128_unrolled")}");
        output.WriteLine($"subgroup resources: {lane.GetKernelResources("q35d_recurrent_gated_rmsnorm_rows128_subgroup")}");
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var result = new float[count];
        if (count > 0) lane.Read(buffer, result);
        return result;
    }

    private static void AssertBitsEqual(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"Mismatch at {i}: expected {expected[i]:R}, actual {actual[i]:R}.");
    }
}
