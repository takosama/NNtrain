using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuDeltaTests
{
    [Theory]
    [InlineData(2, 4, 7, 4)]
    [InlineData(1, 2, 128, 4)]
    [InlineData(2, 4, 1, 1)]
    public void DeviceDeltaMatchesCpuAcrossTokensWithHistoryTiledHeadsAndReset(
        int keyHeads, int valueHeads, int headWidth, int convKernel)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane();
        const float eps = 1e-6f;
        int valueSize = valueHeads * headWidth;
        int channels = (2 * keyHeads + valueHeads) * headWidth;
        var random = new Random(59137 + headWidth);
        float[] Values(int count, float scale = 1f)
            => Enumerable.Range(0, count).Select(_ => ((float)random.NextDouble() * 2f - 1f) * scale).ToArray();

        float[] weights = Values(channels * convKernel, 0.7f);
        float[] dt = Values(valueHeads, 0.3f);
        float[] a = Values(valueHeads).Select(value => -MathF.Abs(value) - 0.1f).ToArray();
        float[] norm = Values(headWidth, 0.4f).Select(value => value + 1f).ToArray();
        float[] convCpu = Values(channels * (convKernel - 1), 0.5f);
        float[] stateCpu = Values(valueSize * headWidth, 0.2f);
        using ArcBuffer weightsGpu = lane.Upload(weights);
        using ArcBuffer dtGpu = lane.Upload(dt);
        using ArcBuffer aGpu = lane.Upload(a);
        using ArcBuffer normGpu = lane.Upload(norm);
        using ArcBuffer convGpu = lane.Upload(convCpu);
        using ArcBuffer stateGpu = lane.Upload(stateCpu);

        var tokens = Enumerable.Range(0, 3).Select(_ => (
            Qkv: Values(channels), Gate: Values(valueSize),
            Alpha: Values(valueHeads), Beta: Values(valueHeads))).ToArray();
        // Cover stable activation branches as well as ordinary positive/negative values.
        tokens[0].Alpha[0] = -30f;
        tokens[1].Alpha[0] = 30f;
        tokens[0].Beta[0] = -30f;
        tokens[1].Beta[0] = 30f;

        for (int pass = 0; pass < 2; pass++)
        {
            if (pass == 1)
            {
                // Reset persistent memory on the GPU, without uploading host state.
                lane.Run("resident_zero", convCpu.Length, 0, convGpu, convCpu.Length);
                lane.Run("resident_zero", stateCpu.Length, 0, stateGpu, stateCpu.Length);
                Array.Clear(convCpu);
                Array.Clear(stateCpu);
            }
            foreach (var token in tokens)
            {
                using ArcBuffer qkvGpu = lane.Upload(token.Qkv);
                using ArcBuffer gateGpu = lane.Upload(token.Gate);
                using ArcBuffer alphaGpu = lane.Upload(token.Alpha);
                using ArcBuffer betaGpu = lane.Upload(token.Beta);
                long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
                long allocated = lane.AllocatedBytes;
                float[] expected = Qwen35Math.DeltaStep(
                    token.Qkv, token.Gate, token.Alpha, token.Beta, weights, dt, a, norm,
                    convCpu, stateCpu, keyHeads, valueHeads, headWidth, convKernel, eps);

                using (ArcBuffer actualGpu = Qwen35Gpu.DeltaStep(
                    lane, qkvGpu, gateGpu, alphaGpu, betaGpu, weightsGpu, dtGpu, aGpu, normGpu,
                    convGpu, stateGpu, keyHeads, valueHeads, headWidth, convKernel, eps))
                {
                    // Numerical work and persistent state updates cannot stage arrays on the host.
                    Assert.Equal(uploaded, lane.H2DBytes);
                    Assert.Equal(downloaded, lane.D2HBytes);
                    Assert.Equal(allocated + valueSize * sizeof(float), lane.AllocatedBytes);
                    AssertClose(expected, Read(lane, actualGpu, valueSize));
                    AssertClose(convCpu, Read(lane, convGpu, convCpu.Length));
                    AssertClose(stateCpu, Read(lane, stateGpu, stateCpu.Length));
                }
                Assert.Equal(allocated, lane.AllocatedBytes);
                Assert.True(qkvGpu.IsAlive && gateGpu.IsAlive && alphaGpu.IsAlive && betaGpu.IsAlive);
                Assert.True(convGpu.IsAlive && stateGpu.IsAlive && weightsGpu.IsAlive && normGpu.IsAlive);
            }
        }

        foreach (string kernel in new[] { "q35d_convolution", "q35d_normalize_qk", "q35d_recurrent", "q35d_gated_rmsnorm" })
            Assert.Contains(kernel, lane.KernelTimings.Keys);
    }

    [Fact]
    public void InvalidDeltaBufferIsRejectedBeforeDispatchOrStateMutation()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane();
        using ArcBuffer qkv = lane.Upload(new float[5]); // Shape requires six values.
        using ArcBuffer gate = lane.Upload(new float[2]);
        using ArcBuffer head = lane.Upload(new float[1]);
        using ArcBuffer weights = lane.Upload(new float[6]);
        using ArcBuffer norm = lane.Upload([1f, 1f]);
        using ArcBuffer conv = lane.Allocate(0);
        float[] initial = [1f, 2f, 3f, 4f];
        using ArcBuffer state = lane.Upload(initial);
        long launches = lane.KernelLaunchCount, allocated = lane.AllocatedBytes;
        Assert.Throws<ArgumentException>(() => Qwen35Gpu.DeltaStep(
            lane, qkv, gate, head, head, weights, head, head, norm, conv, state,
            1, 1, 2, 1, 1e-6f));
        Assert.Equal(launches, lane.KernelLaunchCount);
        Assert.Equal(allocated, lane.AllocatedBytes);
        Assert.Equal(initial, Read(lane, state, initial.Length));
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int length)
    {
        float[] values = new float[length];
        lane.Read(buffer, values);
        return values;
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"Non-finite result at {i}.");
            float tolerance = 3e-5f + MathF.Abs(expected[i]) * 1e-4f;
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, tolerance);
        }
    }
}
