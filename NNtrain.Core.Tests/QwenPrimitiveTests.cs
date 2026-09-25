using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenPrimitiveTests
{
    [Fact]
    public void RmsNormMatchesReference()
    {
        var x = new Tensor([1f, 2f, 3f, 4f], [2, 2]);
        var w = new Tensor([2f, 0.5f], [2]);
        Tensor y = x.RmsNormLastDim(w, 1e-6f);

        float r0 = 1f / MathF.Sqrt((1f + 4f) / 2f + 1e-6f);
        float r1 = 1f / MathF.Sqrt((9f + 16f) / 2f + 1e-6f);
        AssertClose([2f * r0, 1f * r0, 6f * r1, 2f * r1], y.Data);
    }

    [Fact]
    public void SiluMultiplyMatchesReference()
    {
        var gate = new Tensor([-1f, 0f, 2f], [3]);
        var up = new Tensor([3f, 4f, -2f], [3]);
        Tensor y = gate.SiluMultiply(up);

        float Silu(float x) => x / (1f + MathF.Exp(-x));
        AssertClose([Silu(-1f) * 3f, 0f, Silu(2f) * -2f], y.Data);
    }

    [Fact]
    public void ZeroQuantBlocksDecodeToZero()
    {
        Assert.All(GgufQ4K.Dequantize(new byte[GgufQ4K.BlockBytes], 256),
            value => Assert.Equal(0f, value));
        Assert.All(GgufQ6K.Dequantize(new byte[GgufQ6K.BlockBytes], 256),
            value => Assert.Equal(0f, value));
    }

    private static void AssertClose(IReadOnlyList<float> expected, IReadOnlyList<float> actual)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; ++i)
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0f, 1e-5f);
    }
}
