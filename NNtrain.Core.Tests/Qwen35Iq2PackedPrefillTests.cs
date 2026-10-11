using System.Buffers.Binary;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35Iq2PackedPrefillTests
{
    [Theory]
    [InlineData(2, false)]
    [InlineData(2, true)]
    [InlineData(4, false)]
    [InlineData(4, true)]
    [InlineData(8, false)]
    [InlineData(8, true)]
    [InlineData(16, false)]
    [InlineData(16, true)]
    public void PackedRowTilesRetainRawBitsAcrossCompleteAndPartialTiles(int tileRows, bool nativeHalf)
    {
        RequireDevice(requireHalf: false);
        using var lane = new ArcExecutionLane(0, Options(nativeHalf));
        const int width = 512;
        foreach (var shape in new[] { (Rows: 1, Columns: 1), (Rows: tileRows, Columns: 7),
            (Rows: tileRows + 1, Columns: 17), (Rows: tileRows * 2 + 3, Columns: 65) })
        {
            TestContext.Current.CancellationToken.ThrowIfCancellationRequested();
            int count = shape.Rows * shape.Columns;
            float[] input = Inputs(shape.Rows, width, shape.Columns), biases = Biases(shape.Columns);
            byte[] raw = Weights(width, shape.Columns, extremeScales: true);
            using ArcBuffer x = lane.Upload(input), encoded = lane.UploadRaw(raw), bias = lane.Upload(biases);
            using ArcBuffer packed = Qwen35QuantizedModel.PackIq2Decode(lane, encoded, width, shape.Columns);
            using ArcBuffer single = GuardedOutput(lane, count), previous = GuardedOutput(lane, count), actual = GuardedOutput(lane, count);
            string originalKernel = $"q35l_prefill_linear_iq2_s_rows{tileRows}";
            string packedKernel = originalKernel + "_packed82";
            long subgroups = (long)((shape.Rows + tileRows - 1) / tileRows) * shape.Columns;
            long global = (subgroups + 1) / 2 * 32;
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocations = lane.AllocationCount, requests = lane.RequestedBytes;
            lane.Run("q35l_iq2_s_sg16", ((long)count + 1) / 2 * 32, 32,
                x, encoded, bias, single, shape.Rows, width, shape.Columns);
            lane.Run(originalKernel, global, 32, x, encoded, bias, previous, shape.Rows, width, shape.Columns);
            lane.Run(packedKernel, global, 32, x, packed, bias, actual, shape.Rows, width, shape.Columns);
            AssertResidentDispatch(lane, uploads, downloads, allocations, requests);
            float[] expected = AssertOutputs(lane, previous, actual, count);
            AssertOutputs(lane, single, previous, count);
            Assert.True(expected.Take(count).Distinct().Count() > 1 || count == 1);
            AssertInputsUnchanged(lane, x, input, bias, biases, encoded, raw, packed, width, shape.Columns);
            Assert.Contains(packedKernel, lane.KernelTimings.Keys);
        }
    }

    [Theory]
    [InlineData(false, false, false)]
    [InlineData(false, true, false)]
    [InlineData(true, false, false)]
    [InlineData(true, true, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, true)]
    [InlineData(true, false, true)]
    [InlineData(true, true, true)]
    public void PackedBslmAndK32rRetainRawBitsAndDeviceRangeFallback(bool rowMajor, bool nativeHalf, bool rangeFallback)
    {
        RequireDevice(requireHalf: true);
        using var lane = new ArcExecutionLane(0, Options(nativeHalf));
        const int width = 512, guards = 5;
        foreach (var shape in new[] { (Rows: 1, Columns: 1), (Rows: 17, Columns: 65), (Rows: 129, Columns: 67) })
        {
            TestContext.Current.CancellationToken.ThrowIfCancellationRequested();
            int count = shape.Rows * shape.Columns, panelElements = (shape.Rows + 7) / 8 * 8 * width;
            float[] input = Inputs(shape.Rows, width, shape.Columns), biases = Biases(shape.Columns);
            if (rangeFallback)
            {
                // A late-row offender must select the global device fallback
                // for every output tile, including the first and partial last.
                input[^1] = 100_000f;
                input[8] = -100_000f;
            }
            byte[] raw = Weights(width, shape.Columns, extremeScales: false);
            using ArcBuffer x = lane.Upload(input), encoded = lane.UploadRaw(raw), bias = lane.Upload(biases);
            using ArcBuffer packed = Qwen35QuantizedModel.PackIq2Decode(lane, encoded, width, shape.Columns);
            ushort[] initialPanel = Enumerable.Repeat((ushort)0xa55a, panelElements + guards).ToArray();
            using ArcBuffer high = lane.UploadRaw(initialPanel), low = lane.UploadRaw(initialPanel), status = lane.Allocate(1);
            using ArcBuffer previous = GuardedOutput(lane, count), actual = GuardedOutput(lane, count);
            using ArcBuffer fallbackReference = GuardedOutput(lane, count);
            string packKernel = "q35l_prefill_xmx_pack_input_f16x2" + (rowMajor ? "_rowmajor" : "");
            string originalKernel = "q35l_prefill_xmx_iq2_s_gguf_bslm" + (rowMajor ? "_k32r" : "");
            string packedKernel = originalKernel + "_packed82";
            lane.Run("q35a_zero", 1, 0, status, 1);
            lane.Run(packKernel, panelElements + 16L, 0, x, high, low, status, shape.Rows, width);
            // Retain a panel snapshot outside the projection sequence, then
            // ensure both consumers leave the shared input panel unchanged.
            ushort[] highBefore = ReadUshorts(lane, high, initialPanel.Length);
            ushort[] lowBefore = ReadUshorts(lane, low, initialPanel.Length);
            long uploads = lane.H2DBytes, downloads = lane.D2HBytes, allocations = lane.AllocationCount, requests = lane.RequestedBytes;
            long globalX = ((long)shape.Columns + 63) / 64 * 16;
            long globalY = ((long)shape.Rows + 127) / 128 * 16;
            lane.Run2D(originalKernel, globalX, globalY, 16, 16,
                high, low, encoded, bias, previous, shape.Rows, width, shape.Columns, x, status);
            lane.Run2D(packedKernel, globalX, globalY, 16, 16,
                high, low, packed, bias, actual, shape.Rows, width, shape.Columns, x, status);
            if (rangeFallback)
                lane.Run("q35l_iq2_s_sg16", ((long)count + 1) / 2 * 32, 32,
                    x, encoded, bias, fallbackReference, shape.Rows, width, shape.Columns);
            AssertResidentDispatch(lane, uploads, downloads, allocations, requests);
            AssertOutputs(lane, previous, actual, count);
            if (rangeFallback) AssertOutputs(lane, fallbackReference, actual, count);
            int[] flags = new int[1]; lane.ReadRaw(status, flags);
            Assert.Equal(rangeFallback ? 1 : 0, flags[0]);
            Assert.Equal(highBefore, ReadUshorts(lane, high, initialPanel.Length));
            Assert.Equal(lowBefore, ReadUshorts(lane, low, initialPanel.Length));
            Assert.All(highBefore[panelElements..], value => Assert.Equal((ushort)0xa55a, value));
            Assert.All(lowBefore[panelElements..], value => Assert.Equal((ushort)0xa55a, value));
            AssertInputsUnchanged(lane, x, input, bias, biases, encoded, raw, packed, width, shape.Columns);
            Assert.Contains(packedKernel, lane.KernelTimings.Keys);
        }
    }

    private static ArcExecutionOptions Options(bool nativeHalf) => new()
    {
        Qwen35InferenceKernelsOnly = true, Qwen35GgufBslmPrefill = true,
        Qwen35NativeHalfScale = nativeHalf, CacheProgramBinary = true,
        CacheKernelArguments = true, CollectKernelTimings = true,
        BufferPoolBytes = 0, DeferredReleaseBytes = 0
    };

    private static void RequireDevice(bool requireHalf)
    {
        ArcDeviceInfo? device = ArcDevices.Enumerate().FirstOrDefault();
        Assert.SkipWhen(device is null || !device.SupportsXmx || device.MinimumSubgroupSize != 16
            || !device.Extensions.Split(' ').Contains("cl_intel_subgroups")
            || requireHalf && !device.Extensions.Split(' ').Contains("cl_khr_fp16"), "Intel SG16/F16 XMX is required.");
    }

    private static float[] Inputs(int rows, int width, int columns)
    {
        var random = new Random(rows + width + columns + 571);
        return Enumerable.Range(0, rows * width)
            .Select(i => i % 17 == 0 ? -0f : (random.NextSingle() * 2 - 1) * .03f).ToArray();
    }

    private static float[] Biases(int columns)
        => Enumerable.Range(0, columns).Select(i => (i % 7 - 3) * .019f).ToArray();

    private static byte[] Weights(int width, int columns, bool extremeScales)
    {
        byte[] raw = new byte[width / 256 * columns * 82];
        new Random(width + columns + 983).NextBytes(raw);
        ushort[] scales = extremeScales
            ? [0, 0x8000, 1, 0x8001, 0x03ff, 0x83ff, 0x0400, 0x8400, 0x3c00, 0xbc00, 0x7bff, 0xfbff]
            : [0x1400, 0x1800, 0x1000, 0x9400];
        for (int block = 0; block < raw.Length / 82; block++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(raw.AsSpan(block * 82, 2), scales[block % scales.Length]);
            for (int group = 0; group < 8; group++)
                raw[block * 82 + 74 + group] = (byte)(((block * 8 + group) % 16) | (((block * 8 + group + 8) % 16) << 4));
        }
        return raw;
    }

    private static ArcBuffer GuardedOutput(ArcExecutionLane lane, int count)
        => lane.Upload(Enumerable.Repeat(float.NaN, count + 5).ToArray());

    private static float[] AssertOutputs(ArcExecutionLane lane, ArcBuffer expectedBuffer, ArcBuffer actualBuffer, int count)
    {
        float[] expected = new float[count + 5], actual = new float[count + 5];
        lane.Read(expectedBuffer, expected); lane.Read(actualBuffer, actual);
        Assert.All(expected[..count], value => Assert.True(float.IsFinite(value)));
        Assert.Equal(expected[..count].Select(BitConverter.SingleToInt32Bits), actual[..count].Select(BitConverter.SingleToInt32Bits));
        for (int i = count; i < count + 5; i++)
            Assert.True(float.IsNaN(expected[i]) && float.IsNaN(actual[i]), $"Output guard {i} was overwritten.");
        return expected;
    }

    private static void AssertResidentDispatch(ArcExecutionLane lane, long uploads, long downloads, long allocations, long requests)
    {
        Assert.Equal(uploads, lane.H2DBytes); Assert.Equal(downloads, lane.D2HBytes);
        Assert.Equal(allocations, lane.AllocationCount); Assert.Equal(requests, lane.RequestedBytes);
    }

    private static void AssertInputsUnchanged(ArcExecutionLane lane, ArcBuffer x, float[] input, ArcBuffer bias, float[] biases,
        ArcBuffer encoded, byte[] raw, ArcBuffer packed, int width, int columns)
    {
        float[] afterX = new float[input.Length], afterBias = new float[biases.Length];
        byte[] afterRaw = new byte[raw.Length], afterPacked = new byte[raw.Length];
        lane.Read(x, afterX); lane.Read(bias, afterBias); lane.ReadRaw(encoded, afterRaw); lane.ReadRaw(packed, afterPacked);
        Assert.Equal(input.Select(BitConverter.SingleToInt32Bits), afterX.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(biases.Select(BitConverter.SingleToInt32Bits), afterBias.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal(raw, afterRaw);
        Assert.Equal(Qwen35QuantizedModel.PackIq2DecodeBytes(raw, width, columns), afterPacked);
    }

    private static ushort[] ReadUshorts(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        ushort[] values = new ushort[count]; lane.ReadRaw(buffer, values); return values;
    }
}
