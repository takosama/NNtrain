using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuDeltaFusedTests
{
    [Theory]
    [InlineData(2, 4, 128, 4)]
    [InlineData(3, 6, 128, 1)]
    [InlineData(2, 4, 7, 4)]
    public void FusedCandidateMatchesCpuAndGenericAcrossTwoTokensWithNonzeroState(
        int keyHeads, int valueHeads, int width, int convKernel)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        const float eps = 1e-6f;
        int values = valueHeads * width, channels = (2 * keyHeads + valueHeads) * width;
        var random = new Random(8237 + width + convKernel);
        float[] RandomValues(int count, float scale = 1f)
            => Enumerable.Range(0, count).Select(_ => (float)(2 * random.NextDouble() - 1) * scale).ToArray();
        float[] weights = RandomValues(channels * convKernel, 0.6f);
        float[] dt = RandomValues(valueHeads, 0.5f);
        float[] a = RandomValues(valueHeads).Select(x => -MathF.Abs(x) - 0.15f).ToArray();
        float[] norm = RandomValues(width, 0.4f).Select(x => x + 1f).ToArray();
        float[] history = RandomValues(channels * (convKernel - 1), 0.6f);
        float[] state = RandomValues(values * width, 0.25f);
        using ArcBuffer weightsGpu = lane.Upload(weights), dtGpu = lane.Upload(dt);
        using ArcBuffer aGpu = lane.Upload(a), normGpu = lane.Upload(norm);
        using ArcBuffer historyGpu = lane.Upload(history), stateGpu = lane.Upload(state);
        using ArcBuffer genericHistory = lane.Upload(history), genericState = lane.Upload(state);
        for (int token = 0; token < 2; ++token)
        {
            float[] qkv = RandomValues(channels), gate = RandomValues(values);
            float[] alpha = RandomValues(valueHeads), beta = RandomValues(valueHeads);
            alpha[0] = token == 0 ? -30f : 30f;
            beta[0] = token == 0 ? -30f : 30f;
            gate[0] = -80f; gate[width] = 80f;
            using ArcBuffer qkvGpu = lane.Upload(qkv), gateGpu = lane.Upload(gate);
            using ArcBuffer alphaGpu = lane.Upload(alpha), betaGpu = lane.Upload(beta);
            float[] expected = Qwen35Math.DeltaStep(qkv, gate, alpha, beta,
                weights, dt, a, norm, history, state, keyHeads, valueHeads, width, convKernel, eps);
            long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
            long allocated = lane.AllocatedBytes, launches = lane.KernelLaunchCount;
            using (ArcBuffer output = Qwen35Gpu.DeltaStepFused(lane, qkvGpu, gateGpu, alphaGpu,
                betaGpu, weightsGpu, dtGpu, aGpu, normGpu, historyGpu, stateGpu,
                keyHeads, valueHeads, width, convKernel, eps))
            {
                Assert.Equal(uploaded, lane.H2DBytes);
                Assert.Equal(downloaded, lane.D2HBytes);
                Assert.Equal(width == 128 ? 3L : 4L, lane.KernelLaunchCount - launches);
                Assert.Equal(allocated + values * sizeof(float), lane.AllocatedBytes);
                float[] actual = Read(lane, output, values);
                AssertClose(expected, actual, 3e-5f, 1e-4f);
                Assert.Equal(history, Read(lane, historyGpu, history.Length));
                float[] actualState = Read(lane, stateGpu, state.Length);
                AssertClose(state, actualState, 3e-5f, 1e-4f);
                using ArcBuffer generic = Qwen35Gpu.DeltaStep(lane, qkvGpu, gateGpu, alphaGpu,
                    betaGpu, weightsGpu, dtGpu, aGpu, normGpu, genericHistory, genericState,
                    keyHeads, valueHeads, width, convKernel, eps);
                // Compare against the existing GPU path with a tighter bound than
                // the CPU/GPU allowance, including every persistent state component.
                AssertClose(Read(lane, generic, values), actual, 2e-6f, 2e-6f);
                AssertClose(Read(lane, genericState, state.Length), actualState, 2e-7f, 2e-6f);
            }
            Assert.Equal(allocated, lane.AllocatedBytes);
        }
        if (width == 128)
            Assert.Contains("q35d_recurrent_gated_rmsnorm_fused128", lane.KernelTimings.Keys);
        else
            Assert.DoesNotContain("q35d_recurrent_gated_rmsnorm_fused128", lane.KernelTimings.Keys);
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
        for (int i = 0; i < expected.Length; ++i)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(expected[i] - actual[i])
                <= absolute + relative * MathF.Abs(expected[i]),
                $"Mismatch at {i}: expected {expected[i]:R}, actual {actual[i]:R}.");
    }
}
