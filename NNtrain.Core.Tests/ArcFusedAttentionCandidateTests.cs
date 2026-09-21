using NNtrain;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

public sealed class ArcFusedAttentionCandidateTests
{
    private readonly record struct Matrix(ArcBuffer Buffer, int Row, int Column, int Group, int Batch, int Head, int Offset)
    {
        internal Matrix Transpose() => this with { Row = Column, Column = Row };
    }

    [Theory]
    [InlineData(false, 5, 65)]
    [InlineData(true, 5, 65)]
    [InlineData(false, 31, 137)]
    [InlineData(true, 31, 137)]
    [InlineData(false, 32, 513)]
    [InlineData(true, 32, 1025)]
    public void ForwardFusionRetainsExactSoftmaxStatisticsAndFp32Output(bool causal, int d, int sequence)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 2, heads = 3, first = 1, count = 4;
        int width = heads * d;
        float[] qkv = Enumerable.Range(0, batch * sequence * width * 3).Select(i => MathF.Sin(i * .013f) * .13f).ToArray();
        float[] scores = Enumerable.Range(0, count * sequence * sequence).Select(i => MathF.Sin(i * .017f) * 1.3f).ToArray();
        if (causal)
            for (int h = 0; h < count; h++)
            for (int q = 0; q < sequence; q++)
            for (int k = q + 1; k < sequence; k++) scores[(h * sequence + q) * sequence + k] = float.NaN;
        (float[] Value, float[] Stats) Run(bool fused)
        {
            using var lane = new ArcExecutionLane();
            using var input = lane.Upload(qkv); using var p = lane.Upload(scores);
            using var stats = lane.Upload(new float[batch * heads * sequence * 2]);
            using var output = lane.Upload(Enumerable.Repeat(.125f, batch * sequence * width).ToArray());
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            if (fused)
                lane.Run3D("attention_prob_pv_fused_candidate", 32, ((sequence + 7L) / 8) * 8, count, 32, 8, 1,
                    input, p, output, stats, sequence, width, heads, first, causal ? 1 : 0, new LocalMemory(8 * sequence * 4));
            else
            {
                lane.Run("attention_probabilities", count * sequence * 64L, 64, p, stats, sequence, width, heads, first, causal ? 2 : 0, 0);
                lane.Run3D("attention_gemm_compact", ((d + 31L) / 32) * 16, ((sequence + 31L) / 32) * 16, count, 16, 16, 1,
                    p, input, output, sequence, d, sequence,
                    sequence, 1, sequence * sequence, 0, 0, 0,
                    3 * width, 1, 0, sequence * 3 * width, d, 2 * width,
                    width, 1, 0, sequence * width, d, 0,
                    heads, first, 0, causal ? 2 : 0);
            }
            Assert.Equal(h2d, lane.H2DBytes); Assert.Equal(d2h, lane.D2HBytes);
            float[] value = new float[batch * sequence * width], statistics = new float[batch * heads * sequence * 2];
            lane.Read(output, value); lane.Read(stats, statistics);
            return (value, statistics);
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Stats, actual.Stats);
        Assert.Equal(expected.Value, actual.Value);
    }

    [Theory]
    [InlineData(false, 5, 65)]
    [InlineData(true, 5, 65)]
    [InlineData(false, 31, 137)]
    [InlineData(true, 31, 137)]
    [InlineData(false, 32, 513)]
    [InlineData(true, 32, 513)]
    [InlineData(true, 32, 1025)]
    [InlineData(false, 65, 67)]
    [InlineData(true, 65, 67)]
    public void FusionPreservesFp32ProductsAndAccumulationWithoutTransfers(bool causal, int d, int sequence)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int batch = 2, heads = 3, first = 1, count = 4;
        int width = heads * d;
        float[] qkv = Enumerable.Range(0, batch * sequence * width * 3).Select(i => MathF.Sin(i * .013f) * .13f).ToArray();
        float[] dy = Enumerable.Range(0, batch * sequence * width).Select(i => MathF.Cos(i * .017f) * .03f).ToArray();
        float[] probabilities = new float[count * sequence * sequence];
        for (int h = 0; h < count; h++)
        for (int q = 0; q < sequence; q++)
        for (int k = 0; k < (causal ? q + 1 : sequence); k++)
            probabilities[(h * sequence + q) * sequence + k] = ((k + h) % 5 + 1f) / sequence;

        (float[] Grad, float[] Derivative) Run(bool fuseDq, bool fuseDkv)
        {
            using var lane = new ArcExecutionLane();
            using var input = lane.Upload(qkv); using var gradient = lane.Upload(dy);
            using var p = lane.Upload(probabilities);
            using var ds = lane.Upload(new float[probabilities.Length]);
            using var dx = lane.Upload(Enumerable.Repeat(.001f, qkv.Length).ToArray());
            Matrix Qkv(ArcBuffer x, int component) => new(x, 3 * width, 1, 0, sequence * 3 * width, d, component * width);
            Matrix Activation(ArcBuffer x) => new(x, width, 1, 0, sequence * width, d, 0);
            Matrix Scores(ArcBuffer x) => new(x, sequence, 1, sequence * sequence, 0, 0, 0);
            void Gemm(Matrix a, Matrix b, Matrix c, int m, int n, int k, bool add, int causalMode)
            {
                lane.Run3D("attention_gemm_compact", ((n + 31L) / 32) * 16, ((m + 31L) / 32) * 16, count, 16, 16, 1,
                    a.Buffer, b.Buffer, c.Buffer, m, n, k,
                    a.Row, a.Column, a.Group, a.Batch, a.Head, a.Offset,
                    b.Row, b.Column, b.Group, b.Batch, b.Head, b.Offset,
                    c.Row, c.Column, c.Group, c.Batch, c.Head, c.Offset,
                    heads, first, add ? 1 : 0, causal ? causalMode : 0);
            }
            long h2d = lane.H2DBytes, d2h = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                if (fuseDq)
                    lane.Run3D("attention_dp_ds_dq_fused_candidate", 32, ((sequence + 7L) / 8) * 8, count, 32, 8, 1,
                        input, gradient, p, ds, dx, sequence, width, heads, first, causal ? 1 : 0, new LocalMemory(8 * sequence * 4));
                else
                {
                    Gemm(Activation(gradient), Qkv(input, 2).Transpose(), Scores(ds), sequence, sequence, d, false, 1);
                    lane.Run("attention_derivatives", count * sequence * 64L, 64, p, ds, sequence, width, heads, causal ? 2 : 0);
                    Gemm(Scores(ds), Qkv(input, 1), Qkv(dx, 0), sequence, d, sequence, true, 2);
                }
                if (fuseDkv)
                    lane.Run3D("attention_dkv_fused_candidate", ((d + 31L) / 32) * 16, ((sequence + 31L) / 32) * 16, count, 16, 16, 1,
                        input, gradient, p, ds, dx, sequence, width, heads, first, causal ? 1 : 0);
                else
                {
                    Gemm(Scores(ds).Transpose(), Qkv(input, 0), Qkv(dx, 1), sequence, d, sequence, true, 3);
                    Gemm(Scores(p).Transpose(), Activation(gradient), Qkv(dx, 2), sequence, d, sequence, true, 3);
                }
            }
            Assert.Equal(h2d, lane.H2DBytes); Assert.Equal(d2h, lane.D2HBytes);
            float[] result = new float[qkv.Length], derivatives = new float[probabilities.Length];
            lane.Read(dx, result); lane.Read(ds, derivatives);
            return (result, derivatives);
        }

        var expected = Run(false, false);
        var dkv = Run(false, true);
        Assert.Equal(expected.Grad, dkv.Grad);
        Assert.Equal(expected.Derivative, dkv.Derivative);
        if (d <= 32)
        {
            var dq = Run(true, false);
            Assert.Equal(expected.Grad, dq.Grad);
            Assert.Equal(expected.Derivative, dq.Derivative);
            var both = Run(true, true);
            Assert.Equal(expected.Grad, both.Grad);
            Assert.Equal(expected.Derivative, both.Derivative);
        }
    }
}
