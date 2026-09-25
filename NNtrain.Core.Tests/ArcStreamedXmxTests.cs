using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcStreamedXmxTests
{
    [Theory]
    [InlineData(65536, 512, 1536, false, true, false, true, false, true, 16384, false, 49.5)]
    [InlineData(65536, 512, 1536, false, false, true, false, false, true, 16384, false, 49.5)]
    [InlineData(1536, 512, 65536, true, false, true, false, false, false, 16384, false, 64.0)]
    [InlineData(512, 1536, 65536, true, false, true, false, false, false, 16384, false, 64.0)]
    [InlineData(512, 512, 65536, true, false, true, false, false, false, 16384, true, 32.0)]
    public void B64ShapesHaveBoundedPlansWithOriginalSplitPolicy(int m, int n, int k, bool ta, bool tb,
        bool accumulate, bool bias, bool relu, bool rows, int tile, bool parallel, double panelMiB)
    {
        var plan = ArcXmxStorageOperand.PlanStreamed(m, n, k, ta, tb, accumulate, bias, relu, true)
            ?? throw new Xunit.Sdk.XunitException("The B64 matrix has no streamed plan.");
        Assert.Equal(rows, plan.RowTiled);
        Assert.Equal(tile, plan.TileLength);
        Assert.Equal(parallel, plan.ParallelSlices);
        Assert.Equal((long)(panelMiB * 1048576), plan.PeakPanelBytes);
    }

    [Fact]
    public void UnsupportedShapesRejectBeforeAllocationAndPartialScratchUsesFullK()
    {
        Assert.Null(ArcXmxStorageOperand.PlanStreamed(int.MaxValue, int.MaxValue, int.MaxValue,
            false, false, false, false, false, true));
        Assert.Null(ArcXmxStorageOperand.PlanStreamed(128, 64, 64, true, true, true, false, false, true));
        Assert.Null(ArcXmxStorageOperand.PlanStreamed(128, 64, 64, true, false, false, false, false, true));
        Assert.Null(ArcXmxStorageOperand.PlanStreamed(128, 64, 64, true, false, true, true, false, true));
        Assert.Null(ArcXmxStorageOperand.PlanStreamed(128, 64, 64, true, false, true, false, true, true));
        Assert.Null(ArcXmxStorageOperand.PlanStreamed(128, 64, 64, false, false, false, false, false, true, 1));
        // A 16K chunk alone would pass the 64MiB split limit, but full K does not.
        var full = ArcXmxStorageOperand.PlanStreamed(1536, 512, 65536, true, false, true, false, false, true)!.Value;
        var chunk = ArcXmxStorageOperand.PlanStreamed(1536, 512, full.TileLength, true, false, true, false, false, true)!.Value;
        Assert.False(full.ParallelSlices);
        Assert.True(chunk.ParallelSlices);
    }

    public static IEnumerable<object[]> NumericalCases()
    {
        foreach (var dtype in new[] { TensorDType.Float32, TensorDType.BFloat16, TensorDType.Bfp8 })
        {
            // Row tails, transposed B, fused bias/ReLU, nonzero prior output,
            // physical storage offsets, and gate offsets are all independent.
            yield return [dtype, 525, 71, 67, false, false, false, true, true, 64L * 1024];
            yield return [dtype, 525, 71, 67, false, true, true, false, true, 64L * 1024];
            yield return [dtype, 137, 71, 4101, true, false, true, false, true, 1024L * 1024];
            yield return [dtype, 137, 71, 4101, true, false, true, false, false, 1024L * 1024];
        }
        // A non-transposed long-K row split selects the existing 8x32 tile.
        yield return [TensorDType.Bfp8, 525, 71, 4101, false, true, true, true, true, 3L * 1024 * 1024];
    }

    [Theory]
    [MemberData(nameof(NumericalCases))]
    public void StreamedStorageIsBitExactWithFullPanelsAndDoesNotTransferToHost(TensorDType dtype,
        int m, int n, int k, bool ta, bool tb, bool accumulate, bool epilogue, bool parallel, long budget)
    {
        RequireXmx();
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32, options: new() {
            StreamedXmxMatrices = true, DirectXmxMatrices = true, ParallelWeightGradients = parallel });
        var lane = Tensor.ArcLane;
        const int offset = 5;
        var left = Make(dtype, checked(offset + m * k + 7), 2);
        var right = Make(dtype, checked(offset + n * k + 7), 5);
        float[] referenceA = left.Data.ToArray().Skip(offset).Take(m * k).Select(TensorStorageCodec.RoundToBFloat16).ToArray();
        float[] referenceB = right.Data.ToArray().Skip(offset).Take(n * k).Select(TensorStorageCodec.RoundToBFloat16).ToArray();
        using var wholeA = left.ArcMatrixOperand(); using var wholeB = right.ArcMatrixOperand();
        using var a = wholeA.Slice(offset, m * k); using var b = wholeB.Slice(offset, n * k);
        using var decodedA = lane.Upload(referenceA); using var decodedB = lane.Upload(referenceB);
        using var bias = lane.Upload(Enumerable.Range(0, n).Select(i => (i % 7 - 3) * .0013f).ToArray());
        foreach (int gated in new[] { 0, 1, 2 })
        {
            int gateLength = gated == 2 ? n * k : m * k;
            float[] mask = Enumerable.Range(0, offset + gateLength + 7)
                .Select(i => i % 11 == 0 ? float.NaN : i % 7 == 0 ? -0f : i % 3 == 0 ? -1f : 1f).ToArray();
            using var gate = lane.Upload(mask);
            using var referenceGate = lane.Upload(mask.Skip(offset).Take(gateLength).ToArray());
            float[] initial = Enumerable.Range(0, m * n).Select(i => (i % 13 - 6) * .0037f).ToArray();
            using var actual = lane.Upload(initial); using var expected = lane.Upload(initial);
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
            for (int repeat = 0; repeat < 2; repeat++)
            {
                Assert.True(ArcXmxStorageOperand.TryGemmStreamed(lane, a, b, actual, m, n, k, ta, tb, accumulate,
                    epilogue ? bias : null, epilogue, gated == 0 ? null : gate, gated, offset, budget));
                ArcMuonMath.Gemm(lane, decodedA, decodedB, expected, m, n, k, ta, tb, 3, accumulate,
                    epilogue ? bias : null, epilogue, gated == 0 ? null : referenceGate, gated);
            }
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            float[] result = new float[m * n], reference = new float[m * n];
            lane.Read(actual, result); lane.Read(expected, reference);
            Assert.Equal(reference.Select(BitConverter.SingleToInt32Bits), result.Select(BitConverter.SingleToInt32Bits));
        }
        Assert.Contains(lane.KernelTimings.Keys, name => name.StartsWith("gemm_xmx_streamed_", StringComparison.Ordinal));
        Assert.DoesNotContain("decode_bfp8", lane.KernelTimings.Keys);
        Assert.DoesNotContain("decode_bf16", lane.KernelTimings.Keys);
    }

    [Fact]
    public void DisabledStreamedPathDoesNotTouchOutputOrAllocate()
    {
        RequireXmx();
        using var execution = Tensor.BeginArcExecution(options: new() { StreamedXmxMatrices = false });
        var lane = Tensor.ArcLane;
        using var left = lane.Upload(new float[128 * 64]); using var right = lane.Upload(new float[64 * 64]);
        using var a = new ArcXmxStorageOperand(lane, left, null, TensorDType.Float32, 128 * 64);
        using var b = new ArcXmxStorageOperand(lane, right, null, TensorDType.Float32, 64 * 64);
        using var output = lane.Upload(Enumerable.Repeat(.125f, 128 * 64).ToArray());
        long allocations = lane.AllocationCount, launches = lane.KernelLaunchCount;
        Assert.False(ArcXmxStorageOperand.TryGemmStreamed(lane, a, b, output, 128, 64, 64));
        Assert.Equal(allocations, lane.AllocationCount); Assert.Equal(launches, lane.KernelLaunchCount);
        float[] values = new float[128 * 64]; lane.Read(output, values);
        Assert.All(values, value => Assert.Equal(.125f, value));
    }

    [Theory]
    [InlineData(512, 512, 4096, true, "gemm_xmx_streamed_block_32x32_wg16")]
    [InlineData(4096, 512, 512, false, "gemm_xmx_streamed_block_16x64_wg16")]
    public void ExpandedStreamedTilesPreserveBitwiseGradientAndDeviceResidency(
        int m, int n, int k, bool transposeLeft, string expectedKernel)
    {
        RequireXmx();
        using var execution = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix16_32, options: new() {
            StreamedXmxMatrices = true, DirectXmxMatrices = true,
            ExpandedStreamedXmxTiles = true, ParallelWeightGradients = true });
        var lane = Tensor.ArcLane;
        float[] sourceA = Enumerable.Range(0, checked(m * k))
            .Select(i => TensorStorageCodec.RoundToBFloat16((i % 17 - 8) * .0019f)).ToArray();
        float[] sourceB = Enumerable.Range(0, checked(n * k))
            .Select(i => TensorStorageCodec.RoundToBFloat16((i % 13 - 6) * .0021f)).ToArray();
        using var decodedA = lane.Upload(sourceA);
        using var decodedB = lane.Upload(sourceB);
        using var left = new ArcXmxStorageOperand(lane, decodedA, null, TensorDType.Float32, sourceA.Length);
        using var right = new ArcXmxStorageOperand(lane, decodedB, null, TensorDType.Float32, sourceB.Length);
        using var actual = lane.Upload(new float[m * n]);
        using var expected = lane.Upload(new float[m * n]);
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        Assert.True(ArcXmxStorageOperand.TryGemmStreamed(lane, left, right, actual,
            m, n, k, ta: transposeLeft, accumulate: true));
        ArcMuonMath.Gemm(lane, decodedA, decodedB, expected,
            m, n, k, transposeLeft, false, 3, true);
        Assert.Equal(uploads, lane.H2DBytes);
        Assert.Equal(downloads, lane.D2HBytes);
        float[] result = new float[m * n], reference = new float[m * n];
        lane.Read(actual, result); lane.Read(expected, reference);
        Assert.Equal(reference.Select(BitConverter.SingleToInt32Bits), result.Select(BitConverter.SingleToInt32Bits));
        Assert.Contains(expectedKernel, lane.KernelTimings.Keys);
    }

    private static Tensor Make(TensorDType dtype, int length, int seed)
    {
        var tensor = new Tensor(Enumerable.Range(0, length)
            .Select(i => MathF.ScaleB((i % 17 - 8) * .0019f, (i + seed) % 9 - 4)).ToArray(), [length]);
        tensor.ConvertStorageInPlace(dtype, dtype == TensorDType.Bfp8 ? Bfp8QuantizationDescriptor.Block(32) : null);
        return tensor;
    }

    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }
}
