using System.Runtime.InteropServices;
using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcDkvSlmQueryMajorCandidateTests
{
    [Fact]
    public void T2048CausalD32CandidatesPreserveAllGradientBitsAcrossHeadsAndTwoAccumulations()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroup_local_block_io"),
            "SG16 XMX with subgroup local block I/O is required.");

        const int batch = 2, heads = 3, d = 32, sequence = 2048;
        const int width = heads * d, first = 2, count = 2;
        // first=2,count=2 crosses from the final head of batch 0 to batch 1.
        float[] qkv = Enumerable.Range(0, batch * sequence * 3 * width)
            .Select(i => ((i * 17 % 127) - 63) * .0013f).ToArray();
        float[] dy = Enumerable.Range(0, batch * sequence * width)
            .Select(i => ((i * 23 % 109) - 54) * .0007f).ToArray();
        float[] initial = Enumerable.Range(0, qkv.Length)
            .Select(i => .001f + (i % 17) * .00013f).ToArray();
        var p = new float[count * sequence * sequence];
        var ds = new float[p.Length];
        for (int head = 0; head < count; ++head)
        for (int query = 0; query < sequence; ++query)
        for (int key = 0; key < sequence; ++key)
        {
            int index = (head * sequence + query) * sequence + key;
            if (key <= query)
            {
                p[index] = ((key + head) % 13 + 1f) / sequence;
                ds[index] = ((query * 7 + key * 3 + head) % 31 - 15) * .00011f;
            }
            else if (key >= ((query / 32) + 1) * 32)
                p[index] = ds[index] = float.NaN;
            // Within the current 32-key causal tile, masked scores are zero.
        }

        using var lane = new ArcExecutionLane(options: new() { ExperimentalOptimizationKernels = true });
        using var qkvBuffer = lane.Upload(qkv);
        using var dyBuffer = lane.Upload(dy);
        using var pBuffer = lane.Upload(p);
        using var dsBuffer = lane.Upload(ds);

        float[] Run(string kernel)
        {
            using var dxBuffer = lane.Upload(initial);
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; ++repeat)
                lane.Run3D(kernel, 16, (sequence / 32L) * 16, count, 16, 16, 1,
                    qkvBuffer, dyBuffer, pBuffer, dsBuffer, dxBuffer,
                    sequence, width, heads, first, 1);
            lane.Synchronize();
            Assert.Equal(h2d, lane.H2DBytes);
            Assert.Equal(d2h, lane.D2HBytes);
            var output = new float[initial.Length];
            lane.Read(dxBuffer, output);
            return output;
        }

        float[] expected = Run("attention_dkv_block_slm_causal");
        Assert.NotEqual(BitConverter.SingleToInt32Bits(initial[width + (first % heads) * d]),
            BitConverter.SingleToInt32Bits(expected[width + (first % heads) * d]));
        foreach (string kernel in new[] { "attention_dkv_slm_qmajor32_causal", "attention_dkv_slm_qmajor33_causal" })
        {
            float[] actual = Run(kernel);
            var expectedBits = MemoryMarshal.Cast<float, int>(expected.AsSpan());
            var actualBits = MemoryMarshal.Cast<float, int>(actual.AsSpan());
            Assert.Equal(expectedBits.Length, actualBits.Length);
            for (int i = 0; i < expectedBits.Length; ++i)
                if (expectedBits[i] != actualBits[i])
                    Assert.Fail($"{kernel} differs from the existing block-SLM kernel at element {i}: "
                        + $"0x{expectedBits[i]:X8} != 0x{actualBits[i]:X8}.");
        }
    }
}
