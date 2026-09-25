using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcAttentionDpDs2048CandidateTests
{
    [Theory]
    [InlineData(0, 129)]
    [InlineData(2, 137)]
    [InlineData(1, 2048)]
    [InlineData(2, 2048)]
    [InlineData(0, 2048)]
    public void TwoPassDerivativeMatchesFp32ReferenceBits(int causal, int sequence)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16,
            "The two-pass candidate requires Intel XMX and 16-wide subgroups.");

        const int heads = 2, first = 1, count = 2, batch = 2, d = 32;
        const int width = heads * d;
        float[] qkv = Enumerable.Range(0, batch * sequence * 3 * width)
            .Select(i => MathF.Sin(i * .013f) * .13f).ToArray();
        float[] dy = Enumerable.Range(0, batch * sequence * width)
            .Select(i => MathF.Cos(i * .017f) * .03f).ToArray();
        float[] probabilities = new float[count * sequence * sequence];
        for (int g = 0; g < count; g++)
        for (int q = 0; q < sequence; q++)
        for (int k = 0; k < sequence; k++)
            probabilities[(g * sequence + q) * sequence + k] = causal != 0 && k > q
                ? float.NaN : ((k + g) % 31 + 1f) / sequence;

        float[][] Run(bool candidate)
        {
            using var lane = new ArcExecutionLane();
            using var input = lane.Upload(qkv); using var gradient = lane.Upload(dy);
            using var p = lane.Upload(probabilities);
            using var ds = lane.Upload(Enumerable.Repeat(-.125f, probabilities.Length).ToArray());
            float[][] results = new float[2][];
            for (int repeat = 0; repeat < results.Length; repeat++)
            {
                long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
                if (candidate)
                    lane.Run3D("attention_dp_ds_two_pass_d32_2048_candidate",
                        ((sequence + 15L) / 16) * 256, count, 1, 256, 1, 1,
                        input, gradient, p, ds, sequence, width, heads, first, causal);
                else
                {
                    bool aligned = sequence % 64 == 0;
                    string gemm = aligned ? "attention_fp32_dp_d32_aligned" : "attention_gemm_compact";
                    int tile = aligned ? 64 : 32;
                    lane.Run3D(gemm,
                        ((sequence + tile - 1L) / tile) * 16,
                        ((sequence + tile - 1L) / tile) * 16,
                        count, 16, 16, 1,
                        gradient, input, ds, sequence, sequence, d,
                        width, 1, 0, sequence * width, d, 0,
                        1, 3 * width, 0, sequence * 3 * width, d, 2 * width,
                        sequence, 1, sequence * sequence, 0, 0, 0,
                        heads, first, 0, causal == 0 ? 0 : 1);
                    lane.Run("attention_derivatives", count * sequence * 64L, 64,
                        p, ds, sequence, width, heads, causal);
                }
                Assert.Equal(h2d, lane.H2DBytes);
                Assert.Equal(d2h, lane.D2HBytes);
                results[repeat] = new float[probabilities.Length];
                lane.Read(ds, results[repeat]);
            }
            return results;
        }

        float[][] expected = Run(false), actual = Run(true);
        for (int repeat = 0; repeat < expected.Length; repeat++)
            Assert.Equal(expected[repeat].Select(BitConverter.SingleToInt32Bits),
                actual[repeat].Select(BitConverter.SingleToInt32Bits));
    }
}
