using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcAttentionSubgroupCandidateTests
{
    [Theory]
    [InlineData(0, 1, 1)]
    [InlineData(1, 3, 5)]
    [InlineData(2, 31, 31)]
    [InlineData(0, 64, 32)]
    [InlineData(1, 65, 65)]
    [InlineData(2, 137, 32)]
    [InlineData(0, 513, 31)]
    [InlineData(1, 1024, 32)]
    [InlineData(2, 1025, 32)]
    public void SubgroupTreesMatchOriginalBitsForFreshSavedStatisticsAndDerivatives(int causal, int sequence, int headWidth)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        var device = ArcDevices.Enumerate()[0];
        Assert.SkipWhen(!device.SupportsXmx || device.MinimumSubgroupSize != 16, "The SG16 candidate requires an Intel XMX device with 16-wide subgroups.");
        const int heads = 3, first = 1, count = 4, totalHeads = 6;
        int width = heads * headWidth, elements = count * sequence * sequence;
        float[] scores = Enumerable.Range(0, elements).Select(i => MathF.Sin(i * .017f) * 60f).ToArray();
        float[] dp = Enumerable.Range(0, elements).Select(i => MathF.ScaleB(MathF.Cos(i * .071f), i % 17 - 8)).ToArray();

        (float[] Probabilities, float[] Recomputed, float[] Statistics, float[] Derivatives) Run(bool candidate)
        {
            using var lane = new ArcExecutionLane();
            using var p = lane.Upload(scores); using var recomputed = lane.Upload(scores);
            using var stats = lane.Upload(Enumerable.Repeat(-.125f, totalHeads * sequence * 2).ToArray());
            using var derivatives = lane.Upload(dp);
            string softmax = candidate ? "attention_probabilities_subgroup_candidate" : "attention_probabilities";
            string backward = candidate ? "attention_derivatives_subgroup_candidate" : "attention_derivatives";
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            lane.Run(softmax, count * sequence * 64L, 64, p, stats, sequence, width, heads, first, causal, 0);
            lane.Run(softmax, count * sequence * 64L, 64, recomputed, stats, sequence, width, heads, first, causal, 1);
            lane.Run(backward, count * sequence * 64L, 64, p, derivatives, sequence, width, heads, causal);
            Assert.Equal(h2d, lane.H2DBytes); Assert.Equal(d2h, lane.D2HBytes);
            var probabilities = new float[elements]; var recomputedValues = new float[elements];
            var statistics = new float[totalHeads * sequence * 2]; var derivativeValues = new float[elements];
            lane.Read(p, probabilities); lane.Read(recomputed, recomputedValues);
            lane.Read(stats, statistics); lane.Read(derivatives, derivativeValues);
            Assert.Equal(probabilities, recomputedValues);
            return (probabilities, recomputedValues, statistics, derivativeValues);
        }

        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Statistics, actual.Statistics);
        Assert.Equal(expected.Probabilities, actual.Probabilities);
        Assert.Equal(expected.Recomputed, actual.Recomputed);
        Assert.Equal(expected.Derivatives, actual.Derivatives);
    }
}
