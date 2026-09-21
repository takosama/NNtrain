using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcStreamedWeightWorkspaceTests
{
    [Theory]
    [InlineData(1536, 512)]
    [InlineData(512, 1536)]
    public void FullShapeRatherThanPanelShapeControlsPartialBudget(int m, int n)
    {
        var old = ArcXmxStorageOperand.PlanStreamed(m, n, 65536, true, false, true, false, false, true)!.Value;
        var candidate = ArcXmxStorageOperand.PlanStreamed(m, n, 65536, true, false, true, false, false, true,
            partialBudget: 96L * 1048576)!.Value;
        Assert.False(old.ParallelSlices); Assert.True(candidate.ParallelSlices);
        Assert.Equal(old.TileLength, candidate.TileLength); Assert.Equal(old.PeakPanelBytes, candidate.PeakPanelBytes);
        Assert.False(ArcXmxStorageOperand.PlanStreamed(m, n, 65536, true, false, true, false, false, true,
            partialBudget: 95L * 1048576)!.Value.ParallelSlices);
    }

    [Theory]
    [InlineData(1536, 512)]
    [InlineData(512, 1536)]
    public void ProductionWeightGradientMatchesSerialAtExistingTolerance(int m, int n)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16, "SG16 XMX required.");
        const int k = 65536;
        float[] av = Enumerable.Range(0, m * k).Select(i => MathF.Sin(i * .173f) * .037f).ToArray();
        float[] bv = Enumerable.Range(0, n * k).Select(i => MathF.Cos(i * .139f) * .031f).ToArray();
        float[] Run(int budget)
        {
            using var lane = new ArcExecutionLane(options: new() { StreamedWeightGradientWorkspaceMiB = budget });
            using var a = lane.Upload(av); using var b = lane.Upload(bv);
            using var left = new ArcXmxStorageOperand(lane, a, null, TensorDType.Float32, av.Length);
            using var right = new ArcXmxStorageOperand(lane, b, null, TensorDType.Float32, bv.Length);
            using var output = lane.Upload(Enumerable.Repeat(.001f, m * n).ToArray());
            long d2h = lane.D2HBytes, h2d = lane.H2DBytes, retained = lane.AllocatedBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Assert.True(ArcXmxStorageOperand.TryGemmStreamed(lane, left, right, output, m, n, k, ta: true, accumulate: true));
                lane.Synchronize(); Assert.Equal(retained, lane.AllocatedBytes); Assert.Equal(0, lane.RetiredBytes);
            }
            Assert.Equal(d2h, lane.D2HBytes); Assert.Equal(h2d, lane.H2DBytes);
            float[] result = new float[m * n]; lane.Read(output, result); return result;
        }
        var expected = Run(64); var actual = Run(96);
        for (int i = 0; i < expected.Length; i++) { Assert.True(float.IsFinite(actual[i])); Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0, 1e-4f); }
    }
}
