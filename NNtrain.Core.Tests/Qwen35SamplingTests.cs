using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35SamplingTests
{
    [Fact]
    public void GreedyTemperatureOrSingleCandidateUsesLowestIdOnTies()
    {
        float[] logits = [2f, 2f, 1f];
        Assert.Equal(0, Qwen35QuantizedModel.SampleLogits(logits, 0f, 0.95f, 3, new Random(1)));
        Assert.Equal(0, Qwen35QuantizedModel.SampleLogits(logits, 0.6f, 0.95f, 1, new Random(1)));
    }

    [Fact]
    public void TopKIsAppliedBeforeTopPAndTiesPreferLowerTokenIds()
    {
        float[] equalLogits = [0f, 0f, 0f, 0f];
        // Top-k=2 retains IDs 0 and 1. Top-p=0.4 of their normalized
        // distribution retains only ID 0. Filtering in the opposite order
        // would retain two IDs from the full four-token distribution.
        for (int seed = 0; seed < 20; seed++)
            Assert.Equal(0, Qwen35QuantizedModel.SampleLogits(
                equalLogits, 0.6f, 0.4f, 2, new Random(seed)));

        int[] sampled = Enumerable.Range(0, 100)
            .Select(seed => Qwen35QuantizedModel.SampleLogits(
                equalLogits, 0.6f, 1f, 2, new Random(seed))).ToArray();
        Assert.All(sampled, id => Assert.InRange(id, 0, 1));
        Assert.Contains(0, sampled);
        Assert.Contains(1, sampled);
    }

    [Fact]
    public void SamplingIsReproducibleWithSeedAndTopKMayExceedVocabulary()
    {
        float[] logits = [0f, 1f, 1.5f, 2f];
        Random first = new(174);
        Random second = new(174);
        int[] firstSequence = Enumerable.Range(0, 50)
            .Select(_ => Qwen35QuantizedModel.SampleLogits(logits, 0.6f, 0.95f, 20, first))
            .ToArray();
        int[] secondSequence = Enumerable.Range(0, 50)
            .Select(_ => Qwen35QuantizedModel.SampleLogits(logits, 0.6f, 0.95f, 20, second))
            .ToArray();
        Assert.Equal(firstSequence, secondSequence);
        Assert.All(firstSequence, id => Assert.InRange(id, 0, logits.Length - 1));
    }

    [Fact]
    public void TemperatureChangesProbabilityBeforeNucleusFiltering()
    {
        float[] logits = [0f, 1f];
        Assert.Equal(0, Qwen35QuantizedModel.SampleLogits(
            logits, 1f, 1f, 2, new FixedRandom(0.8)));
        Assert.Equal(1, Qwen35QuantizedModel.SampleLogits(
            logits, 0.5f, 1f, 2, new FixedRandom(0.8)));
    }

    [Theory]
    [InlineData(float.NaN, 0.95f, 20)]
    [InlineData(float.PositiveInfinity, 0.95f, 20)]
    [InlineData(-0.1f, 0.95f, 20)]
    [InlineData(0.6f, float.NaN, 20)]
    [InlineData(0.6f, 0f, 20)]
    [InlineData(0.6f, 1.1f, 20)]
    [InlineData(0.6f, 0.95f, 0)]
    public void InvalidSamplingSettingsAreRejected(float temperature, float topP, int topK)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.SampleLogits(
            [0f, 1f], temperature, topP, topK, new Random(1)));
    }

    [Fact]
    public void NonFiniteLogitsAreRejectedEvenOutsideTopK()
    {
        Assert.Throws<ArithmeticException>(() => Qwen35QuantizedModel.SampleLogits(
            [10f, 1f, float.NaN], 0.6f, 0.95f, 2, new Random(1)));
        Assert.Throws<ArithmeticException>(() => Qwen35QuantizedModel.SampleLogits(
            [10f, 1f, float.PositiveInfinity], 0f, 1f, 1, new Random(1)));
    }

    private sealed class FixedRandom(double value) : Random
    {
        public override double NextDouble() => value;
    }
}
