using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35MathTests
{
    [Fact]
    public void RmsNormUsesEffectiveGgufWeightsSeparatelyForEachHead()
    {
        float[] result = Qwen35Math.RmsNorm([3f, 4f, -3f, -4f], [2f, 0.5f], 0f, 2);
        AssertClose([1.6970562748f, 0.5656854249f, -1.6970562748f, -0.5656854249f], result);
        AssertClose([0f, 0f], Qwen35Math.RmsNorm([0f, 0f], [1f, 1f], 1e-6f));
    }

    [Fact]
    public void PartialRopeUsesSplitHalfPairsAndPreservesUnrotatedDimensions()
    {
        float[] input = [1f, 2f, 3f, 4f, 5f, 6f, -1f, -2f, -3f, -4f, -5f, -6f];
        Qwen35Math.Rope(input, heads: 2, headWidth: 6, ropeDims: 4, position: 1, theta: 100f);
        // Pairs (0,2) and (1,3) have angles 1 and 0.1; components 4 and 5 pass through.
        float[] expected = [-1.9841106486f, 1.59067466397f, 2.4623779024f, 4.1796834944f, 5f, 6f,
            1.9841106486f, -1.59067466397f, -2.4623779024f, -4.1796834944f, -5f, -6f];
        AssertClose(expected, input);
    }

    [Fact]
    public void ActivationsRemainFiniteForLargePositiveAndNegativeInputs()
    {
        Assert.Equal(1f, Qwen35Math.Sigmoid(100f));
        Assert.InRange(Qwen35Math.Sigmoid(-100f), 0f, 1e-40f);
        Assert.Equal(100f, Qwen35Math.Softplus(100f));
        Assert.InRange(Qwen35Math.Softplus(-100f), 0f, 1e-40f);
        Assert.Equal(0f, Qwen35Math.Silu(0f));
        Assert.InRange(Qwen35Math.Softplus(0f), 0.6931471f, 0.6931473f);
    }

    [Fact]
    public void DeltaStepMatchesIndependentTwoTokenMatrixCalculation()
    {
        float[] state = new float[4];
        float[] conv = [];
        float[] norm = [1.2f, 0.8f];
        float[] first = Qwen35Math.DeltaStep(
            [1f, -0.5f, 0.3f, 2f, 1.5f, -0.7f], [0.4f, -1.1f], [-0.2f], [0.6f],
            [1f, 1f, 1f, 1f, 1f, 1f], [0.1f], [-1.3f], norm, conv, state, 1, 1, 2, 1, 1e-6f);
        // Golden values are from a separate float64 evaluation of
        // S'=exp(A*softplus(alpha+dt))*S; S=S'+k*(sigmoid(beta)*(v-k^T*S')); y=q^T*S.
        AssertClose([0.0770927622f, -0.0146010958f, 0.7880461225f, -0.1492531409f], state);
        AssertClose([-0.3992525695f, -0.0578293904f], first);

        float[] second = Qwen35Math.DeltaStep(
            [-0.9f, 0.7f, 1.1f, 0.8f, -0.2f, 1.7f], [-0.3f, 0.9f], [0.4f], [-0.8f],
            [1f, 1f, 1f, 1f, 1f, 1f], [0.1f], [-1.3f], norm, conv, state, 1, 1, 2, 1, 1e-6f);
        AssertClose([-0.0379494335f, 0.3732115207f, 0.1822127381f, 0.2102989472f], state);
        AssertClose([-0.2166253358f, 0.0096976129f], second);
    }

    [Fact]
    public void DeltaConvolutionUsesOldestToNewestTapsAndStoresUnactivatedInputs()
    {
        float[] convState = new float[6];
        float[] state = new float[1];
        float[] currentValues = [2f, 7f, 11f];
        float[] expectedStates = [9.9995366663f, 40.9999803527f, 80.0000018710f];
        for (int token = 0; token < currentValues.Length; ++token)
        {
            _ = Qwen35Math.DeltaStep(
                [1f, 1f, currentValues[token]], [1f], [0f], [100f],
                [0f, 0f, 1f, 0f, 0f, 1f, 2f, 3f, 5f], [0f], [0f], [1f],
                convState, state, 1, 1, 1, 3, 1e-6f);
            Assert.InRange(MathF.Abs(state[0] - expectedStates[token]), 0f, 3e-5f);
        }
        Assert.Equal([1f, 1f, 1f, 1f, 7f, 11f], convState);
    }

    [Fact]
    public void DeltaStepMapsGgufValueHeadsToTiledKeyHeads()
    {
        float[] state = new float[4];
        _ = Qwen35Math.DeltaStep(
            [1f, 1f, 1f, -1f, 2f, 2f, 2f, 2f], [1f, 1f, 1f, 1f],
            [0f, 0f, 0f, 0f], [0f, 0f, 0f, 0f], [1f, 1f, 1f, 1f, 1f, 1f, 1f, 1f],
            [0f, 0f, 0f, 0f], [-1f, -1f, -1f, -1f], [1f], [], state, 2, 4, 1, 1, 1e-6f);
        Assert.True(state[0] > 0f);
        Assert.True(state[1] < 0f);
        Assert.Equal(state[0], state[2]);
        Assert.Equal(state[1], state[3]);
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i)
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, 1e-6f);
    }
}
