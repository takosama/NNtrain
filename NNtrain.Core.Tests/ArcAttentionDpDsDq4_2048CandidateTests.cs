using NNtrain;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

public sealed class ArcAttentionDpDsDq4_2048CandidateTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FourRowFusionMatchesTunedFp32BackwardAndPublishesDkvDerivative(bool causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int sequence = 2048, d = 32, heads = 3, first = 2, count = 2, batch = 2;
        const int width = heads * d;
        int qkvLength = batch * sequence * 3 * width;
        int scoresLength = count * sequence * sequence;
        float[] qkv = Enumerable.Range(0, qkvLength)
            .Select(i => MathF.Sin(i * 0.013f) * 0.13f).ToArray();
        float[] dy = Enumerable.Range(0, batch * sequence * width)
            .Select(i => MathF.Cos(i * 0.017f) * 0.03f).ToArray();
        float[] probabilities = new float[scoresLength];
        for (int group = 0; group < count; group++)
        for (int query = 0; query < sequence; query++)
        for (int key = 0; key < sequence; key++)
            probabilities[(group * sequence + query) * sequence + key] = causal && key > query
                ? float.NaN : ((key + group) % 7 + 1f) / sequence;

        (float[] Gradient, float[] Derivative) Run(bool fused)
        {
            using var lane = new ArcExecutionLane();
            using var input = lane.Upload(qkv);
            using var gradient = lane.Upload(dy);
            using var p = lane.Upload(probabilities);
            using var ds = lane.Upload(new float[scoresLength]);
            using var dx = lane.Upload(Enumerable.Repeat(0.001f, qkvLength).ToArray());
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                if (fused)
                    lane.Run3D("attention_dp_ds_dq4_t2048_candidate", 32, sequence, count, 32, 4, 1,
                        input, gradient, p, ds, dx, sequence, width, heads, first,
                        causal ? 2 : 0, new LocalMemory(4 * sequence * sizeof(float)));
                else
                {
                    // The production T=2048 path uses these tuned kernels.
                    // Their fixed layout parameters are retained in the ABI.
                    lane.Run3D("attention_fp32_dp_d32_aligned", (sequence / 64) * 16L,
                        (sequence / 64) * 16L, count, 16, 16, 1,
                        gradient, input, ds, sequence, sequence, d,
                        width, 1, 0, sequence * width, d, 0,
                        3 * width, 1, 0, sequence * 3 * width, d, 2 * width,
                        sequence, 1, sequence * sequence, 0, 0, 0,
                        heads, first, 0, causal ? 1 : 0);
                    lane.Run("attention_derivatives", count * sequence * 64L, 64,
                        p, ds, sequence, width, heads, causal ? 2 : 0);
                    lane.Run3D("attention_fp32_dq_d32_aligned", 16, (sequence / 64) * 16L,
                        count, 16, 16, 1,
                        ds, input, dx, sequence, d, sequence,
                        sequence, 1, sequence * sequence, 0, 0, 0,
                        3 * width, 1, 0, sequence * 3 * width, d, width,
                        3 * width, 1, 0, sequence * 3 * width, d, 0,
                        heads, first, 1, causal ? 2 : 0);
                }
            }
            Assert.Equal(h2d, lane.H2DBytes);
            Assert.Equal(d2h, lane.D2HBytes);
            float[] result = new float[qkvLength], derivative = new float[scoresLength];
            lane.Read(dx, result);
            lane.Read(ds, derivative);
            return (result, derivative);
        }

        var expected = Run(false);
        var actual = Run(true);
        AssertBitwiseEqual(expected.Gradient, actual.Gradient, "dQ");
        AssertBitwiseEqual(expected.Derivative, actual.Derivative, "dS");
    }

    private static void AssertBitwiseEqual(float[] expected, float[] actual, string name)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            int expectedBits = BitConverter.SingleToInt32Bits(expected[i]);
            int actualBits = BitConverter.SingleToInt32Bits(actual[i]);
            if (expectedBits != actualBits)
                Assert.Fail($"{name}[{i}]: expected {expected[i]:R} ({expectedBits:X8}), " +
                    $"actual {actual[i]:R} ({actualBits:X8})");
        }
    }
}
