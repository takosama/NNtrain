using System.Buffers.Binary;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2PackedDecodeTests
{
    [Theory]
    [InlineData(256, 1)]
    [InlineData(512, 7)]
    [InlineData(5120, 33)]
    [InlineData(17408, 9)]
    public void CpuPackingPreservesEveryRandomSourceAndPackedByte(int width, int columns)
    {
        int bytes = checked(width / 256 * columns * 82);
        Assert.Equal(bytes, Qwen35QuantizedModel.Iq2PackedDecodeBytes(width, columns));
        byte[] raw = RandomBytes(bytes, width + columns), original = raw.ToArray();
        byte[] packed = Qwen35QuantizedModel.PackIq2DecodeBytes(raw, width, columns);
        Assert.Equal(bytes, packed.Length);
        Assert.Equal(original, raw);
        Assert.Equal(original, Qwen35QuantizedModel.UnpackIq2DecodeBytes(packed, width, columns));
        // Arbitrary encoded planes must also have a unique inverse: no bits
        // may be dropped or canonicalized by either direction of conversion.
        byte[] arbitraryPacked = RandomBytes(bytes, width + columns + 1);
        byte[] unpacked = Qwen35QuantizedModel.UnpackIq2DecodeBytes(arbitraryPacked, width, columns);
        Assert.Equal(arbitraryPacked, Qwen35QuantizedModel.PackIq2DecodeBytes(unpacked, width, columns));
    }

    [Fact]
    public void CpuPackingPreservesAllHalfEncodingsIncludingNanPayloadsAndSignedZero()
    {
        const int columns = 65536;
        byte[] raw = RandomBytes(columns * 82, 8256);
        for (int bits = 0; bits < columns; bits++)
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(bits * 82, 2), (ushort)bits);
        byte[] packed = Qwen35QuantizedModel.PackIq2DecodeBytes(raw, 256, columns);
        for (int bits = 0; bits < columns; bits++)
            Assert.Equal((ushort)bits, BinaryPrimitives.ReadUInt16LittleEndian(packed.AsSpan(columns * 80 + bits * 2, 2)));
        Assert.Equal(raw, Qwen35QuantizedModel.UnpackIq2DecodeBytes(packed, 256, columns));
    }

    [Fact]
    public void CpuPackingMatchesTheDocumentedThreePlanes()
    {
        byte[] raw = Enumerable.Range(0, 82).Select(value => (byte)value).ToArray();
        byte[] packed = Qwen35QuantizedModel.PackIq2DecodeBytes(raw, 256, 1);
        byte[] codes = Enumerable.Range(2, 32).SelectMany(value => new[] { (byte)value, (byte)(value + 32) }).ToArray();
        byte[] extras = [0xa2, 0x44, 0xb3, 0x44, 0xc4, 0x44, 0xd5, 0x44,
            0xe6, 0x44, 0xf7, 0x44, 0x08, 0x54, 0x19, 0x54];
        Assert.Equal(codes, packed[..64]);
        Assert.Equal(extras, packed[64..80]);
        Assert.Equal(new byte[] { 0, 1 }, packed[80..]);
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(-256, 1)]
    [InlineData(1, 1)]
    [InlineData(255, 1)]
    [InlineData(257, 1)]
    [InlineData(256, 0)]
    [InlineData(256, -1)]
    public void CpuPackingRejectsInvalidShapes(int width, int columns)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.Iq2PackedDecodeBytes(width, columns));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.PackIq2DecodeBytes([], width, columns));
        Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.UnpackIq2DecodeBytes([], width, columns));
    }

    [Theory]
    [InlineData(256, int.MaxValue)]
    [InlineData(2147483392, 256)]
    public void CpuPackingRejectsByteCountOverflowBeforeAllocation(int width, int columns)
    {
        Assert.Throws<OverflowException>(() => Qwen35QuantizedModel.Iq2PackedDecodeBytes(width, columns));
        Assert.Throws<OverflowException>(() => Qwen35QuantizedModel.PackIq2DecodeBytes([], width, columns));
        Assert.Throws<OverflowException>(() => Qwen35QuantizedModel.UnpackIq2DecodeBytes([], width, columns));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(81)]
    [InlineData(83)]
    [InlineData(164)]
    public void CpuPackingRequiresTheExactTensorByteCount(int length)
    {
        byte[] wrong = new byte[length];
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.PackIq2DecodeBytes(wrong, 256, 1));
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.UnpackIq2DecodeBytes(wrong, 256, 1));
    }

    [Theory]
    [InlineData(256, 1)]
    [InlineData(512, 7)]
    [InlineData(5120, 33)]
    public void GpuPackingMatchesCpuBytesRetainsSourceAndUsesOneSameSizeAllocation(int width, int columns)
    {
        RequireDevice();
        using var lane = new ArcExecutionLane(0, Options());
        byte[] raw = RandomBytes(Qwen35QuantizedModel.Iq2PackedDecodeBytes(width, columns), width + columns);
        // Packing must preserve NaN/Inf/zero coefficient payloads as raw bits.
        ushort[] halves = [0x7e01, 0xfeff, 0x7c00, 0xfc00, 0, 0x8000, 1, 0x8001];
        for (int block = 0; block < raw.Length / 82; block++)
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(block * 82, 2), halves[block % halves.Length]);
        using ArcBuffer encoded = lane.UploadRaw(raw);
        lane.Synchronize();
        long live = lane.AllocatedBytes, requested = lane.RequestedBytes, allocations = lane.AllocationCount;
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes, launches = lane.KernelLaunchCount;
        using (ArcBuffer packed = Qwen35QuantizedModel.PackIq2Decode(lane, encoded, width, columns))
        {
            Assert.Equal(raw.Length, packed.ByteLength);
            Assert.Equal(live + raw.Length, lane.AllocatedBytes);
            Assert.Equal(requested + raw.Length, lane.RequestedBytes);
            Assert.Equal(allocations + 1, lane.AllocationCount);
            Assert.Equal(launches + 1, lane.KernelLaunchCount);
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            byte[] actual = ReadBytes(lane, packed, raw.Length);
            Assert.Equal(Qwen35QuantizedModel.PackIq2DecodeBytes(raw, width, columns), actual);
            Assert.Equal(raw, Qwen35QuantizedModel.UnpackIq2DecodeBytes(actual, width, columns));
            Assert.Equal(raw, ReadBytes(lane, encoded, raw.Length));
            long unpackLive = lane.AllocatedBytes, unpackRequests = lane.RequestedBytes, unpackAllocations = lane.AllocationCount;
            long unpackUploads = lane.H2DBytes, unpackDownloads = lane.D2HBytes, unpackLaunches = lane.KernelLaunchCount;
            using (ArcBuffer restored = Qwen35QuantizedModel.UnpackIq2Decode(lane, packed, width, columns))
            {
                Assert.Equal(raw.Length, restored.ByteLength);
                Assert.Equal(unpackLive + raw.Length, lane.AllocatedBytes);
                Assert.Equal(unpackRequests + raw.Length, lane.RequestedBytes);
                Assert.Equal(unpackAllocations + 1, lane.AllocationCount);
                Assert.Equal(unpackLaunches + 1, lane.KernelLaunchCount);
                Assert.Equal(unpackUploads, lane.H2DBytes); Assert.Equal(unpackDownloads, lane.D2HBytes);
                Assert.Equal(raw, ReadBytes(lane, restored, raw.Length));
                Assert.Equal(actual, ReadBytes(lane, packed, raw.Length));
            }
            lane.Synchronize();
            Assert.Equal(unpackLive, lane.AllocatedBytes);
            using ArcBuffer unpackGuarded = lane.UploadRaw(Enumerable.Repeat((byte)0x5d, raw.Length + 13).ToArray());
            lane.Run("q35l_iq2_s_unpack82", (long)(raw.Length / 82 + 2) * 8, 0, packed, unpackGuarded, raw.Length / 82);
            byte[] restoredGuarded = ReadBytes(lane, unpackGuarded, raw.Length + 13);
            Assert.Equal(raw, restoredGuarded[..raw.Length]);
            Assert.All(restoredGuarded[raw.Length..], value => Assert.Equal((byte)0x5d, value));
        }
        lane.Synchronize();
        Assert.Equal(live, lane.AllocatedBytes);

        // Exercise the kernel's extra-work-item guard with an odd block count
        // and a byte sentinel directly after the last coefficient plane.
        using ArcBuffer guarded = lane.UploadRaw(Enumerable.Repeat((byte)0xa7, raw.Length + 13).ToArray());
        lane.Run("q35l_iq2_s_pack82", (long)(raw.Length / 82 + 2) * 16, 0, encoded, guarded, raw.Length / 82);
        byte[] guardedBytes = ReadBytes(lane, guarded, raw.Length + 13);
        Assert.Equal(Qwen35QuantizedModel.PackIq2DecodeBytes(raw, width, columns), guardedBytes[..raw.Length]);
        Assert.All(guardedBytes[raw.Length..], value => Assert.Equal((byte)0xa7, value));
    }

    [Theory]
    [InlineData(false, 32)]
    [InlineData(true, 32)]
    [InlineData(false, 64)]
    [InlineData(true, 64)]
    [InlineData(false, 128)]
    [InlineData(true, 128)]
    public void PackedSingleAndPairPreserveRawOutputBitsForRowsTailsAndScaleExtremes(bool nativeHalf, int workgroup)
    {
        RequireDevice();
        using var lane = new ArcExecutionLane(0, Options() with
        {
            Qwen35NativeHalfScale = nativeHalf, Qwen35ProjectionWorkgroupSize = workgroup
        });
        Assert.True(Qwen35QuantizedModel.SupportsIq2PackedDecode(lane));
        foreach (var shape in new[] { (M: 1, K: 256, N: 1), (M: 1, K: 512, N: 7),
            (M: 3, K: 512, N: 7), (M: 3, K: 512, N: 8),
            (M: 1, K: 5120, N: 33), (M: 3, K: 5120, N: 9), (M: 1, K: 17408, N: 5) })
        {
            TestContext.Current.CancellationToken.ThrowIfCancellationRequested();
            int count = shape.M * shape.N;
            var random = new Random(shape.K + shape.N + 719);
            float[] input = Enumerable.Range(0, shape.M * shape.K)
                .Select(i => i % 17 == 0 ? -0f : (random.NextSingle() * 2 - 1) * .003f).ToArray();
            float[] biases = Enumerable.Range(0, shape.N).Select(i => (i % 7 - 3) * .03f).ToArray();
            byte[] raw = FiniteWeights(shape.K, shape.N);
            using ArcBuffer x = lane.Upload(input), encoded = lane.UploadRaw(raw), bias = lane.Upload(biases);
            using ArcBuffer packed = Qwen35QuantizedModel.PackIq2Decode(lane, encoded, shape.K, shape.N);
            using ArcBuffer previous = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            float[]? firstBaseline = null;
            foreach (bool paired in new[] { false, true })
            {
                lane.Write(previous, Enumerable.Repeat(float.NaN, count + 5).ToArray());
                lane.Write(candidate, Enumerable.Repeat(float.NaN, count + 5).ToArray());
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes, requested = lane.RequestedBytes;
                long allocations = lane.AllocationCount, launches = lane.KernelLaunchCount;
                RunRaw(lane, paired, x, encoded, bias, previous, shape.M, shape.K, shape.N);
                Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, packed, bias, candidate, shape.M, shape.K, shape.N, paired);
                Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(requested, lane.RequestedBytes); Assert.Equal(allocations, lane.AllocationCount);
                Assert.Equal(launches + 2, lane.KernelLaunchCount);
                float[] expected = ReadFloats(lane, previous, count + 5), actual = ReadFloats(lane, candidate, count + 5);
                Assert.All(expected[..count], value => Assert.True(float.IsFinite(value)));
                AssertBits(expected[..count], actual[..count]);
                if (firstBaseline is not null) AssertBits(firstBaseline, expected[..count]);
                else firstBaseline = expected[..count];
                for (int i = count; i < count + 5; i++)
                    Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]), $"Output guard {i} was overwritten.");
            }
            AssertBits(input, ReadFloats(lane, x, input.Length));
            AssertBits(biases, ReadFloats(lane, bias, biases.Length));
            Assert.Equal(raw, ReadBytes(lane, encoded, raw.Length));
            Assert.Equal(Qwen35QuantizedModel.PackIq2DecodeBytes(raw, shape.K, shape.N), ReadBytes(lane, packed, raw.Length));
        }
        Assert.Contains("q35l_iq2_s_packed82_sg16", lane.KernelTimings.Keys);
        Assert.Contains("q35l_iq2_s_packed82_sg16_pair", lane.KernelTimings.Keys);
    }

    [Theory]
    [InlineData(false, 32)]
    [InlineData(true, 32)]
    [InlineData(true, 128)]
    public void PackedSingleAndPairLoraPreserveRawBitsWithOddRankAndRowTails(bool nativeHalf, int workgroup)
    {
        RequireDevice();
        using var lane = new ArcExecutionLane(0, Options() with
        {
            Qwen35NativeHalfScale = nativeHalf, Qwen35ProjectionWorkgroupSize = workgroup
        });
        const int rank = 7;
        const float scale = .3f;
        foreach (var shape in new[] { (M: 1, K: 512, N: 1), (M: 3, K: 512, N: 7),
            (M: 1, K: 5120, N: 33), (M: 3, K: 5120, N: 9) })
        {
            int count = shape.M * shape.N;
            var random = new Random(shape.K + shape.N + 379);
            float[] Values(int length) => Enumerable.Range(0, length).Select(_ => random.NextSingle() * .2f - .1f).ToArray();
            float[] input = Values(shape.M * shape.K), biases = Values(shape.N);
            float[] zValues = Values(shape.M * rank), adapterValues = Values(shape.N * rank);
            byte[] raw = FiniteWeights(shape.K, shape.N);
            for (int block = 0; block < raw.Length / 82; block++)
                BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(block * 82, 2), BitConverter.HalfToUInt16Bits((Half).002f));
            using ArcBuffer x = lane.Upload(input), encoded = lane.UploadRaw(raw), bias = lane.Upload(biases);
            using ArcBuffer z = lane.Upload(zValues), adapter = lane.Upload(adapterValues);
            using ArcBuffer packed = Qwen35QuantizedModel.PackIq2Decode(lane, encoded, shape.K, shape.N);
            using ArcBuffer previous = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            using ArcBuffer candidate = lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());
            RunRaw(lane, false, x, encoded, bias, previous, shape.M, shape.K, shape.N);
            float[] withoutLora = ReadFloats(lane, previous, count + 5);
            foreach (bool paired in new[] { false, true })
            {
                lane.Write(previous, Enumerable.Repeat(float.NaN, count + 5).ToArray());
                lane.Write(candidate, Enumerable.Repeat(float.NaN, count + 5).ToArray());
                string suffix = paired ? "_pair_lora" : "_lora";
                int group = paired ? 32 : workgroup;
                long subgroups = paired ? (long)shape.M * ((shape.N + 1) / 2) : count;
                long global = (subgroups + group / 16 - 1) / (group / 16) * group;
                long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocations = lane.AllocationCount;
                lane.Run("q35l_iq2_s_sg16" + suffix, global, group,
                    x, encoded, bias, previous, shape.M, shape.K, shape.N, z, adapter, rank, scale);
                lane.Run("q35l_iq2_s_packed82_sg16" + suffix, global, group,
                    x, packed, bias, candidate, shape.M, shape.K, shape.N, z, adapter, rank, scale);
                Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
                Assert.Equal(allocations, lane.AllocationCount);
                float[] expected = ReadFloats(lane, previous, count + 5), actual = ReadFloats(lane, candidate, count + 5);
                Assert.All(expected[..count], value => Assert.True(float.IsFinite(value)));
                AssertBits(expected[..count], actual[..count]);
                Assert.Contains(Enumerable.Range(0, count), i => BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(withoutLora[i]));
                for (int i = count; i < count + 5; i++)
                    Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]), $"LoRA output guard {i} was overwritten.");
            }
            AssertBits(input, ReadFloats(lane, x, input.Length));
            AssertBits(biases, ReadFloats(lane, bias, biases.Length));
            AssertBits(zValues, ReadFloats(lane, z, zValues.Length));
            AssertBits(adapterValues, ReadFloats(lane, adapter, adapterValues.Length));
            Assert.Equal(raw, ReadBytes(lane, encoded, raw.Length));
            Assert.Equal(Qwen35QuantizedModel.PackIq2DecodeBytes(raw, shape.K, shape.N), ReadBytes(lane, packed, raw.Length));
        }
        Assert.Contains("q35l_iq2_s_packed82_sg16_lora", lane.KernelTimings.Keys);
        Assert.Contains("q35l_iq2_s_packed82_sg16_pair_lora", lane.KernelTimings.Keys);
    }

    [Fact]
    public void GpuHelpersRejectInvalidShapesBeforeAllocationOrDispatch()
    {
        RequireDevice();
        using var lane = new ArcExecutionLane(0, Options());
        const int width = 512, columns = 7, rows = 3;
        int bytes = Qwen35QuantizedModel.Iq2PackedDecodeBytes(width, columns);
        using ArcBuffer x = lane.Allocate(rows * width), packed = lane.AllocateBytes(bytes);
        using ArcBuffer bias = lane.Allocate(columns), output = lane.Allocate(rows * columns);
        using ArcBuffer shortX = lane.Allocate(rows * width - 1), shortBias = lane.Allocate(columns - 1);
        using ArcBuffer shortOutput = lane.Allocate(rows * columns - 1);
        using ArcBuffer shortWeight = lane.AllocateBytes(bytes - 1), longWeight = lane.AllocateBytes(bytes + 1);
        long requests = lane.RequestedBytes, allocations = lane.AllocationCount, launches = lane.KernelLaunchCount;
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.PackIq2Decode(lane, shortWeight, width, columns));
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.PackIq2Decode(lane, longWeight, width, columns));
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.UnpackIq2Decode(lane, shortWeight, width, columns));
        Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.UnpackIq2Decode(lane, longWeight, width, columns));
        foreach (bool paired in new[] { false, true })
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, packed, bias, output, 0, width, columns, paired));
            Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, packed, bias, output, rows, width + 1, columns, paired));
            Assert.Throws<ArgumentOutOfRangeException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, packed, bias, output, rows, width, 0, paired));
            Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, shortX, packed, bias, output, rows, width, columns, paired));
            Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, packed, shortBias, output, rows, width, columns, paired));
            Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, packed, bias, shortOutput, rows, width, columns, paired));
            Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, shortWeight, bias, output, rows, width, columns, paired));
            Assert.Throws<ArgumentException>(() => Qwen35QuantizedModel.RunIq2PackedDecode(lane, x, longWeight, bias, output, rows, width, columns, paired));
        }
        Assert.Equal(requests, lane.RequestedBytes); Assert.Equal(allocations, lane.AllocationCount);
        Assert.Equal(launches, lane.KernelLaunchCount);
    }

    private static ArcExecutionOptions Options() => new()
    {
        Qwen35InferenceKernelsOnly = true, CacheProgramBinary = true,
        CacheKernelArguments = true, CollectKernelTimings = true,
        BufferPoolBytes = 0, DeferredReleaseBytes = 0
    };

    private static void RequireDevice()
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups"), "Intel SG16 is required.");
    }

    private static void RunRaw(ArcExecutionLane lane, bool paired, ArcBuffer x, ArcBuffer weights,
        ArcBuffer bias, ArcBuffer output, int rows, int width, int columns)
    {
        int group = paired ? 32 : lane.Options.Qwen35ProjectionWorkgroupSize;
        long groups = paired ? (long)rows * ((columns + 1) / 2) : (long)rows * columns;
        long global = (groups + group / 16 - 1) / (group / 16) * group;
        lane.Run(paired ? "q35l_iq2_s_sg16_pair" : "q35l_iq2_s_sg16", global, group,
            x, weights, bias, output, rows, width, columns);
    }

    private static byte[] RandomBytes(int length, int seed)
    {
        byte[] bytes = new byte[length]; new Random(seed).NextBytes(bytes); return bytes;
    }

    private static byte[] FiniteWeights(int width, int columns)
    {
        byte[] bytes = RandomBytes(Qwen35QuantizedModel.Iq2PackedDecodeBytes(width, columns), width + columns + 881);
        ushort[] scales = [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff];
        for (int block = 0; block < bytes.Length / 82; block++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(block * 82, 2), scales[block % scales.Length]);
            for (int group = 0; group < 8; group++)
                bytes[block * 82 + 74 + group] = (byte)(((block * 8 + group) % 16) | (((block * 8 + group + 8) % 16) << 4));
        }
        return bytes;
    }

    private static byte[] ReadBytes(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        byte[] bytes = new byte[count]; lane.ReadRaw(buffer, bytes); return bytes;
    }

    private static float[] ReadFloats(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        float[] values = new float[count]; lane.Read(buffer, values); return values;
    }

    private static void AssertBits(float[] expected, float[] actual)
        => Assert.Equal(expected.Select(BitConverter.SingleToInt32Bits), actual.Select(BitConverter.SingleToInt32Bits));
}
