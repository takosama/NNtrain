using NNtrain;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

public sealed class ArcNormCandidateTests
{
    [Theory]
    [InlineData(false, 0f)]
    [InlineData(true, 0f)]
    [InlineData(false, .1f)]
    [InlineData(true, .1f)]
    [InlineData(false, .75f)]
    [InlineData(true, .75f)]
    public void CoalescedGradientConsumerFusionIsBitExactWithAliasesAndSignedZeros(bool alias, float probability)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int length = 1031;
        const uint seed = 913579;
        float negativeZero = BitConverter.Int32BitsToSingle(unchecked((int)0x80000000));
        float[] values = Enumerable.Range(0, length).Select(i => (i % 8) switch {
            0 => 0f, 1 => negativeZero, 2 => float.Epsilon, 3 => -float.Epsilon,
            4 => .03125f, 5 => -.25f, _ => MathF.Sin(i * .037f) * .1f }).ToArray();
        float[] initialDx = Enumerable.Range(0, length).Select(i => (i % 4) switch {
            0 => 0f, 1 => negativeZero, 2 => .03125f, _ => -.015625f }).ToArray();
        float[] initialBranch = Enumerable.Range(0, length).Select(i => (i % 4) switch {
            0 => negativeZero, 1 => 0f, 2 => -.03125f, _ => .015625f }).ToArray();
        uint threshold = (uint)(probability * (uint.MaxValue + 1d));
        float scale = 1f / (1f - probability);
        using var lane = new ArcExecutionLane();
        using var scratch = lane.Upload(values);
        (int[] Dx, int[] Branch) Run(bool fused)
        {
            using var dx = lane.Upload(initialDx);
            using var other = lane.Upload(initialBranch);
            ArcBuffer branch = alias ? dx : other;
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                if (fused)
                    lane.Run("norm_residual_back_accumulate", length, 0, scratch, dx, branch,
                        length, seed, threshold, scale);
                else
                {
                    lane.Run("copy_scale", length, 0, scratch, dx, length, 1f, 1);
                    lane.Run("dropout_back", length, 0, scratch, branch, length, seed, threshold, scale);
                }
            }
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            var dxValues = new float[length]; var branchValues = new float[length];
            lane.Read(dx, dxValues); lane.Read(branch, branchValues);
            return (dxValues.Select(BitConverter.SingleToInt32Bits).ToArray(),
                branchValues.Select(BitConverter.SingleToInt32Bits).ToArray());
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Dx, actual.Dx);
        Assert.Equal(expected.Branch, actual.Branch);
    }

    [Theory]
    [InlineData(17, 37, false, 0f, false)]
    [InlineData(17, 37, true, .1f, false)]
    [InlineData(512, 128, true, .1f, false)]
    [InlineData(513, 512, true, .1f, false)]
    [InlineData(129, 1536, true, .25f, false)]
    [InlineData(7, 2048, false, 0f, false)]
    [InlineData(1, 1, true, .1f, false)]
    [InlineData(17, 37, true, .1f, true)]
    [InlineData(513, 512, true, .1f, true)]
    public void StrictFusedNormPreservesExactForwardAndAccumulatedGradients(
        int rows, int width, bool residual, float probability, bool aliasedResidual)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        int length = checked(rows * width), groups = (rows + 255) / 256;
        const uint seed = 123456789;
        uint threshold = (uint)(probability * (uint.MaxValue + 1d));
        float scale = 1f / (1f - probability);
        using var lane = new ArcExecutionLane();
        using var x = lane.Upload(Enumerable.Range(0, length).Select(i => MathF.Sin(i * .017f) * .7f).ToArray());
        using var branch = lane.Upload(Enumerable.Range(0, length).Select(i => MathF.Cos(i * .011f) * .13f).ToArray());
        ArcBuffer branchInput = aliasedResidual ? x : branch;
        using var gamma = lane.Upload(Enumerable.Range(0, width).Select(i => 1f + MathF.Sin(i * .037f) * .3f).ToArray());
        using var beta = lane.Upload(Enumerable.Range(0, width).Select(i => MathF.Cos(i * .031f) * .1f).ToArray());
        using var dy = lane.Upload(Enumerable.Range(0, length).Select(i => MathF.Cos(i * .031f) * .001f).ToArray());

        (float[] Y, float[] Stats, float[] Dx, float[] Dr, float[] Dg, float[] Db) Run(bool fused)
        {
            using var y = lane.Allocate(length);
            using var stats = lane.Allocate(rows * 2);
            using var merged = lane.Allocate(length);
            using var scratch = lane.Allocate(length);
            using var parts = lane.Allocate(groups * width * 2);
            using var dx = lane.Upload(Enumerable.Repeat(.03125f, length).ToArray());
            using var dr = lane.Upload(Enumerable.Repeat(-.015625f, length).ToArray());
            ArcBuffer branchGradient = aliasedResidual ? dx : dr;
            using var dg = lane.Upload(Enumerable.Repeat(.0078125f, width).ToArray());
            using var db = lane.Upload(Enumerable.Repeat(.00390625f, width).ToArray());
            long up = lane.H2DBytes, down = lane.D2HBytes;
            ArcBuffer input = x;
            if (fused)
                lane.Run("norm_residual_serial", rows, 0, x, branchInput, gamma, beta,
                    y, stats, rows, width, 1e-5f, seed, threshold, scale, residual ? 1 : 0);
            else
            {
                if (residual)
                {
                    lane.Run("dropout", length, 0, branchInput, x, merged, length, seed, threshold, scale, 1);
                    input = merged;
                }
                lane.Run("norm", rows, 0, input, gamma, beta, y, stats, rows, width, 1e-5f);
            }
            for (int repeat = 0; repeat < 2; repeat++)
            {
                if (fused)
                {
                    lane.Run("norm_residual_dx_serial", rows, 0, x, branchInput, gamma,
                        dy, stats, dx, branchGradient, rows, width, seed, threshold, scale, residual ? 1 : 0);
                    lane.Run2D("norm_residual_parameter_parts", ((width + 31L) / 32) * 32,
                        groups * 8L, 32, 8, dy, x, branchInput, stats, parts, rows, width,
                        seed, threshold, scale, residual ? 1 : 0);
                }
                else
                {
                    lane.Run("norm_dx_set", rows, 0, input, gamma, dy, stats, scratch, rows, width);
                    lane.Run("copy_scale", length, 0, scratch, dx, length, 1f, 1);
                    if (residual) lane.Run("dropout_back", length, 0, scratch, branchGradient, length, seed, threshold, scale);
                    lane.Run2D("gradient_rows", ((width + 31L) / 32) * 32,
                        groups * 8L, 32, 8, dy, input, stats, parts, rows, width, 0, 1);
                }
                lane.Run("gradient_rows_finish", width, 0, parts, db, dg, groups, width, 1);
            }
            Assert.Equal(up, lane.H2DBytes);
            Assert.Equal(down, lane.D2HBytes);
            float[] Read(ArcBuffer buffer, int count) { var values = new float[count]; lane.Read(buffer, values); return values; }
            return (Read(y, length), Read(stats, rows * 2), Read(dx, length), Read(dr, length), Read(dg, width), Read(db, width));
        }
        var expected = Run(false); var actual = Run(true);
        Assert.Equal(expected.Y, actual.Y);
        Assert.Equal(expected.Stats, actual.Stats);
        Assert.All(actual.Stats, value => Assert.True(float.IsFinite(value)));
        for (int row = 0; row < rows; row++) Assert.True(actual.Stats[2 * row + 1] > 0);
        Assert.Equal(expected.Dx, actual.Dx); Assert.Equal(expected.Dr, actual.Dr);
        Assert.Equal(expected.Dg, actual.Dg); Assert.Equal(expected.Db, actual.Db);
        if (!residual) Assert.All(actual.Dr, value => Assert.Equal(-.015625f, value));
    }

    [Fact]
    public void FusedNormParameterReductionKeepsExistingReductionOrder()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        const int rows = 1027, width = 37, groups = (rows + 255) / 256, length = rows * width;
        const uint seed = 35791, threshold = 429496729;
        const float scale = 1f / .9f;
        using var lane = new ArcExecutionLane();
        using var x = lane.Upload(Enumerable.Range(0, length).Select(i => MathF.Sin(i * .017f)).ToArray());
        using var branch = lane.Upload(Enumerable.Range(0, length).Select(i => MathF.Cos(i * .023f) * .3f).ToArray());
        using var dy = lane.Upload(Enumerable.Range(0, length).Select(i => MathF.Cos(i * .031f) * .001f).ToArray());
        using var stats = lane.Upload(Enumerable.Range(0, rows * 2).Select(i => i % 2 == 0 ? .125f : 1.25f).ToArray());
        using var merged = lane.Allocate(length);
        using var oldParts = lane.Allocate(groups * width * 2);
        using var newParts = lane.Allocate(groups * width * 2);
        lane.Run("dropout", length, 0, branch, x, merged, length, seed, threshold, scale, 1);
        lane.Run2D("gradient_rows", ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
            dy, merged, stats, oldParts, rows, width, 0, 1);
        lane.Run2D("norm_residual_parameter_parts", ((width + 31L) / 32) * 32, groups * 8L, 32, 8,
            dy, x, branch, stats, newParts, rows, width, seed, threshold, scale, 1);
        var expected = new float[groups * width * 2]; var actual = new float[expected.Length];
        lane.Read(oldParts, expected); lane.Read(newParts, actual);
        Assert.Equal(expected, actual);
    }
}
