using NNtrain;
using NNtrain.Arc;
using Xunit;

public sealed class ArcXmxStorageOperandTests
{
    private static void RequireXmx()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(), "Intel Arc is required.");
        Assert.SkipWhen(!ArcDevices.Enumerate()[0].SupportsXmx || ArcDevices.Enumerate()[0].MinimumSubgroupSize != 16,
            "SG16 Intel XMX is required.");
    }

    [Theory]
    [InlineData(TensorDType.Float32, 0)]
    [InlineData(TensorDType.BFloat16, 0)]
    [InlineData(TensorDType.Bfp8, 0)]
    [InlineData(TensorDType.Bfp8, 32)]
    [InlineData(TensorDType.Bfp8, 128)]
    [InlineData(TensorDType.Bfp8, 0, true)]
    [InlineData(TensorDType.Bfp8, 32, true)]
    [InlineData(TensorDType.Bfp8, 128, true)]
    public void SlicedStoragePanelsMatchDecodedRoundingWithTailsAndTransposes(TensorDType dtype, int block, bool powerOfTwo = false)
    {
        RequireXmx();
        const int outer = 137, k = 67, offset = 5, length = outer * k;
        using var scope = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32, options: new() { DirectXmxMatrices = true, PowerOfTwoPackScales = powerOfTwo });
        var tensor = Make(dtype, block, offset + length + 7, 3);
        float[] expected = tensor.Data.ToArray().Skip(offset).Take(length).Select(TensorStorageCodec.RoundToBFloat16).ToArray();
        var lane = Tensor.ArcLane;
        using var decoded = lane.Upload(expected);
        using var whole = tensor.ArcMatrixOperand();
        using var slice = whole.Slice(offset, length);
        // A sliced lease must not depend on the disposable parent's borrow.
        whole.Dispose();
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        foreach (bool transpose in new[] { false, true })
        foreach (bool right in new[] { false, true })
        {
            int count = ((outer + (right ? 15 : 7)) / (right ? 16 : 8)) * ((k + 15) / 16) * 128;
            using var actual = right ? slice.PackB(outer, k, transpose) : slice.PackA(outer, k, transpose);
            using var reference = lane.AllocateBytes(count * (right ? 4 : 2));
            lane.Run(right ? "xmx_next_pack_b" : "xmx_next_pack_a", count, 0,
                decoded, decoded, reference, outer, k, transpose ? 1 : 0, 0);
            Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
            ushort[] actualBits = new ushort[count * (right ? 2 : 1)], referenceBits = new ushort[actualBits.Length];
            lane.ReadRaw(actual, actualBits); lane.ReadRaw(reference, referenceBits);
            Assert.Equal(referenceBits, actualBits);
            downloads = lane.D2HBytes;
        }
        Assert.DoesNotContain("decode_bfp8", lane.KernelTimings.Keys);
        Assert.DoesNotContain("decode_bf16", lane.KernelTimings.Keys);
    }

    [Theory]
    [InlineData(TensorDType.Float32, 0)]
    [InlineData(TensorDType.BFloat16, 0)]
    [InlineData(TensorDType.Bfp8, 32)]
    [InlineData(TensorDType.Bfp8, 128)]
    public void StorageGemmMatchesFp32DecodePathIncludingAccumulatedSplitK(TensorDType dtype, int block)
    {
        RequireXmx();
        using var scope = Tensor.BeginArcExecution(precision: TensorPrecisionMode.Mix8_32, options: new() { DirectXmxMatrices = true });
        var lane = Tensor.ArcLane;
        foreach (var (m, n, k, ta, tb) in new[] { (137, 131, 67, false, false), (137, 71, 65, false, true),
            (129, 65, 4101, true, false), (137, 71, 65, true, true) })
        {
            var left = Make(dtype, block, m * k, 2); var right = Make(dtype, block, n * k, 5);
            using var aDecoded = lane.Upload(left.Data.ToArray().Select(TensorStorageCodec.RoundToBFloat16).ToArray());
            using var bDecoded = lane.Upload(right.Data.ToArray().Select(TensorStorageCodec.RoundToBFloat16).ToArray());
            using var a = left.ArcMatrixOperand(); using var b = right.ArcMatrixOperand();
            using var bias = lane.Upload(Enumerable.Range(0, n).Select(i => (i % 7 - 3) / 64f).ToArray());
            foreach (int gateOperand in new[] { 0, 1, 2 })
            {
                using var actual = lane.Upload(Enumerable.Repeat(.125f, m * n).ToArray());
                using var expected = lane.Upload(Enumerable.Repeat(.125f, m * n).ToArray());
                var gate = gateOperand == 0 ? null : gateOperand == 1 ? aDecoded : bDecoded;
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
                for (int step = 0; step < 2; step++)
                {
                    Assert.True(ArcXmxStorageOperand.TryGemm(lane, a, b, actual, m, n, k, ta, tb, true,
                        gateOperand == 0 ? null : bias, gateOperand != 0, gate, gateOperand));
                    ArcMuonMath.Gemm(lane, aDecoded, bDecoded, expected, m, n, k, ta, tb, 3, true,
                        gateOperand == 0 ? null : bias, gateOperand != 0, gate, gateOperand);
                }
                Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
                float[] actualValues = new float[m * n], expectedValues = new float[m * n];
                lane.Read(actual, actualValues); lane.Read(expected, expectedValues);
                Assert.Equal(expectedValues, actualValues);
            }
        }
        Assert.DoesNotContain("decode_bfp8", lane.KernelTimings.Keys);
        Assert.DoesNotContain("decode_bf16", lane.KernelTimings.Keys);
    }

    [Fact]
    public void RawBf16PanelPreservesInfinitiesAndQuietsNaNs()
    {
        RequireXmx();
        using var lane = new ArcExecutionLane();
        ushort[] special = [0, 0x8000, 0x0001, 0x3f80, 0x7f80, 0xff80, 0x7f81, 0xff81, 0x7fff, 0xffff];
        ushort[] source = Enumerable.Range(0, 128).Select(i => special[i % special.Length]).ToArray();
        using var value = lane.UploadRaw(source);
        using var operand = new ArcXmxStorageOperand(lane, value, null, TensorDType.BFloat16, 128);
        using var panel = operand.PackA(8, 16, false);
        ushort[] actual = new ushort[128]; lane.ReadRaw(panel, actual);
        Assert.Equal(source.Select(v => (ushort)(v | ((v & 0x7fff) > 0x7f80 ? 0x40 : 0))).ToArray(), actual);
        Assert.True(value.IsAlive);
    }

    [Fact]
    public void ShapeBudgetRejectsOversizeWithoutAllocatingOrOverflowing()
    {
        RequireXmx();
        using var lane = new ArcExecutionLane(options: new() { DirectXmxMatrices = true });
        long allocations = lane.AllocationCount;
        Assert.True(ArcXmxStorageOperand.CanRun(lane, 16384, 1536, 512));
        Assert.False(ArcXmxStorageOperand.CanRun(lane, int.MaxValue, int.MaxValue, int.MaxValue));
        Assert.False(ArcXmxStorageOperand.CanRun(lane, 127, 512, 512));
        Assert.Equal(allocations, lane.AllocationCount);
    }

    private static Tensor Make(TensorDType dtype, int block, int length, int seed)
    {
        var tensor = new Tensor(Enumerable.Range(0, length).Select(i => MathF.Sin(i * .013f + seed) * .031f).ToArray(), [length]);
        tensor.ConvertStorageInPlace(dtype, dtype == TensorDType.Bfp8
            ? block == 0 ? Bfp8QuantizationDescriptor.TensorWide : Bfp8QuantizationDescriptor.Block(block) : null);
        return tensor;
    }
}
