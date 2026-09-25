using NNtrain;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

public sealed class ArcAttentionProbPvFusedT2048Tests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FusedFourRowKernelMatchesProbabilityStatsAndPvBits(bool causal)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int sequence = 2048, width = 512, heads = 16, first = 15, count = 2;
        var score = new float[count * sequence * sequence];
        for (int g = 0; g < count; ++g)
        for (int q = 0; q < sequence; ++q)
        for (int key = 0; key < sequence; ++key)
            score[(g * sequence + q) * sequence + key] = causal && key > q
                ? float.NaN : ((g * 17 + q * 3 + key * 7) % 107 - 53) * .0005f;
        var qkv = new float[2 * sequence * 3 * width];
        for (int i = 0; i < qkv.Length; ++i)
            qkv[i] = (i % 97 - 48) * .0002f;
        var initial = Enumerable.Repeat(-.125f, 2 * sequence * width).ToArray();
        int statsLength = (first + count) * sequence * 2;

        using var lane = new ArcExecutionLane();
        using ArcBuffer input = lane.Upload(qkv);
        using ArcBuffer raw = lane.Upload(score);
        using ArcBuffer referenceScores = lane.Upload(score);
        using ArcBuffer referenceOutput = lane.Upload(initial);
        using ArcBuffer candidateOutput = lane.Upload(initial);
        using ArcBuffer referenceStats = lane.Allocate(statsLength);
        using ArcBuffer candidateStats = lane.Allocate(statsLength);
        lane.Run("resident_zero", statsLength, 0, referenceStats, statsLength);
        lane.Run("resident_zero", statsLength, 0, candidateStats, statsLength);

        lane.Run("attention_probabilities", count * sequence * 64L, 64,
            referenceScores, referenceStats, sequence, width, heads, first,
            causal ? 2 : 0, 0);
        lane.Run3D("attention_fp32_pv_d32_aligned", 16, sequence / 64L * 16,
            count, 16, 16, 1, referenceScores, input, referenceOutput,
            sequence, 32, sequence,
            sequence, 1, sequence * sequence, 0, 0, 0,
            3 * width, 1, 0, sequence * 3 * width, 32, 2 * width,
            width, 1, 0, sequence * width, 32, 0,
            heads, first, 0, causal ? 2 : 0);

        lane.Run3D("attention_prob_pv_fused_t2048_r4_candidate",
            32, sequence / 4L * 4, count, 32, 4, 1,
            input, raw, candidateOutput, candidateStats,
            sequence, width, heads, first, causal ? 1 : 0);

        var expectedOutput = new float[initial.Length];
        var actualOutput = new float[initial.Length];
        var expectedStats = new float[statsLength];
        var actualStats = new float[statsLength];
        lane.Read(referenceOutput, expectedOutput);
        lane.Read(candidateOutput, actualOutput);
        lane.Read(referenceStats, expectedStats);
        lane.Read(candidateStats, actualStats);
        AssertSameBits(expectedOutput, actualOutput, "PV", causal);
        AssertSameBits(expectedStats, actualStats, "softmax stats", causal);
    }

    private static void AssertSameBits(float[] expected, float[] actual, string operation, bool causal)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"{operation} causal={causal}, element {i}: "
                    + $"{expected[i]:R} versus {actual[i]:R}.");
    }
}
