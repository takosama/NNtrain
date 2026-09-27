using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferenceDeltaNormTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void CooperativeNormalizationMatchesSerialAndLeavesValueChannelsUntouched(bool zero)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        const int keyHeads = 16, valueHeads = 48, width = 128;
        const float eps = 1e-6f;
        int keySize = keyHeads * width;
        var random = new Random(31153);
        float[] mixed = Enumerable.Range(0, (2 * keyHeads + valueHeads) * width)
            .Select(_ => zero ? 0f : (float)(2 * random.NextDouble() - 1)).ToArray();
        // Very small and zero heads exercise epsilon-dominated normalization.
        if (!zero)
            for (int i = 0; i < width; i++)
            {
                mixed[i] *= 1e-9f;
                mixed[keySize + i] = 0f;
            }
        using ArcBuffer reference = lane.Upload(mixed), candidate = lane.Upload(mixed);
        long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        lane.Run("q35d_normalize_qk", keyHeads, 0, reference, keyHeads, width, eps);
        lane.Run("q35d_normalize_qk_coop128", keyHeads * 128, 128, candidate, keyHeads, eps);
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded, lane.D2HBytes);
        float[] expected = Read(lane, reference, mixed.Length);
        float[] actual = Read(lane, candidate, mixed.Length);
        AssertClose(expected, actual, 2e-7f, 2e-6f);
        Assert.Equal(mixed.Skip(2 * keySize), actual.Skip(2 * keySize));
        if (zero) Assert.All(actual, value => Assert.Equal(0f, value));
    }

    [Theory]
    [InlineData(16, 48, false)]
    [InlineData(16, 48, true)]
    [InlineData(1, 2, false)]
    public void FusedParallelNormalizationMatchesReferenceOutputAndPersistentState(
        int keyHeads, int valueHeads, bool zero)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true,
            Qwen35ParallelDeltaNorm = true
        });
        const int width = 128, convKernel = 4;
        const float eps = 1e-6f;
        int values = valueHeads * width, channels = (2 * keyHeads + valueHeads) * width;
        var random = new Random(72311 + keyHeads);
        float[] RandomValues(int count, float scale = 1f)
            => Enumerable.Range(0, count)
                .Select(_ => zero ? 0f : (float)(2 * random.NextDouble() - 1) * scale).ToArray();
        float[] weights = RandomValues(channels * convKernel, 0.6f);
        float[] dt = RandomValues(valueHeads, 0.5f);
        float[] a = RandomValues(valueHeads).Select(value => -MathF.Abs(value) - 0.15f).ToArray();
        float[] norm = RandomValues(width, 0.4f).Select(value => value + 1f).ToArray();
        float[] history = RandomValues(channels * (convKernel - 1), 0.6f);
        float[] state = RandomValues(values * width, 0.25f);
        using ArcBuffer weightsGpu = lane.Upload(weights), dtGpu = lane.Upload(dt);
        using ArcBuffer aGpu = lane.Upload(a), normGpu = lane.Upload(norm);
        using ArcBuffer candidateHistory = lane.Upload(history), candidateState = lane.Upload(state);
        using ArcBuffer referenceHistory = lane.Upload(history), referenceState = lane.Upload(state);
        for (int token = 0; token < 3; token++)
        {
            float[] qkv = RandomValues(channels), gate = RandomValues(values);
            float[] alpha = RandomValues(valueHeads), beta = RandomValues(valueHeads);
            if (!zero)
            {
                alpha[0] = token == 0 ? -30f : 30f;
                beta[0] = token == 0 ? -30f : 30f;
                gate[0] = -80f;
                gate[width] = 80f;
            }
            using ArcBuffer qkvGpu = lane.Upload(qkv), gateGpu = lane.Upload(gate);
            using ArcBuffer alphaGpu = lane.Upload(alpha), betaGpu = lane.Upload(beta);
            float[] cpuExpected = Qwen35Math.DeltaStep(qkv, gate, alpha, beta,
                weights, dt, a, norm, history, state, keyHeads, valueHeads, width, convKernel, eps);
            long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
            long allocated = lane.AllocatedBytes, launches = lane.KernelLaunchCount;
            using (ArcBuffer candidate = Qwen35Gpu.DeltaStepFused(lane, qkvGpu, gateGpu,
                alphaGpu, betaGpu, weightsGpu, dtGpu, aGpu, normGpu, candidateHistory,
                candidateState, keyHeads, valueHeads, width, convKernel, eps))
            {
                Assert.Equal(uploaded, lane.H2DBytes);
                Assert.Equal(downloaded, lane.D2HBytes);
                Assert.Equal(3L, lane.KernelLaunchCount - launches);
                Assert.Equal(allocated + values * sizeof(float), lane.AllocatedBytes);
                using ArcBuffer reference = Qwen35Gpu.DeltaStep(lane, qkvGpu, gateGpu,
                    alphaGpu, betaGpu, weightsGpu, dtGpu, aGpu, normGpu, referenceHistory,
                    referenceState, keyHeads, valueHeads, width, convKernel, eps);
                float[] actual = Read(lane, candidate, values);
                float[] actualState = Read(lane, candidateState, state.Length);
                AssertClose(cpuExpected, actual, 3e-5f, 1e-4f);
                AssertClose(state, actualState, 3e-5f, 1e-4f);
                AssertClose(Read(lane, reference, values), actual, 3e-6f, 1e-5f);
                AssertClose(Read(lane, referenceState, state.Length), actualState, 3e-7f, 1e-5f);
                Assert.Equal(Read(lane, referenceHistory, history.Length),
                    Read(lane, candidateHistory, history.Length));
                Assert.Equal(history, Read(lane, candidateHistory, history.Length));
                if (zero)
                {
                    Assert.All(actual, value => Assert.Equal(0f, value));
                    Assert.All(actualState, value => Assert.Equal(0f, value));
                }
            }
            Assert.Equal(allocated, lane.AllocatedBytes);
        }
        Assert.Contains("q35d_normalize_qk_coop128", lane.KernelTimings.Keys);
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var result = new float[count];
        lane.Read(buffer, result);
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, float absolute, float relative)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(expected[i] - actual[i])
                <= absolute + relative * MathF.Abs(expected[i]),
                $"Mismatch at {i}: expected {expected[i]:R}, actual {actual[i]:R}.");
    }
}
