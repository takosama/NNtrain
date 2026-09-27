using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuLinearTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType, "coop64")]
    [InlineData(Qwen2Gguf.Q6KType, "coop64")]
    [InlineData(Qwen2Gguf.Q4KType, "sg16")]
    [InlineData(Qwen2Gguf.Q6KType, "sg16")]
    public void CooperativeAndSubgroupGemvMatchDecodedCpuAndOriginalKernel(uint type, string variant)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        bool subgroup = variant.StartsWith("sg16", StringComparison.Ordinal);
        Assert.SkipWhen(subgroup && !(lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16), "SG16 kernel compilation requires the Arc SG16 capability.");

        // Odd output counts exercise the final half-empty WG32. Long input
        // widths cover real model dot products without allocating a full model.
        foreach (var shape in new[]
        {
            (Rows: 1, Input: 256, Output: 5, Subnormal: false),
            (Rows: 3, Input: 512, Output: 7, Subnormal: false),
            (Rows: 1, Input: 5120, Output: 33, Subnormal: false),
            (Rows: 1, Input: 17408, Output: 17, Subnormal: false),
            (Rows: 1, Input: 512, Output: 7, Subnormal: true)
        })
        {
            byte[] encoded = Payload(type, shape.Input * shape.Output, shape.Subnormal);
            float[] decoded = type == Qwen2Gguf.Q4KType
                ? GgufQ4K.Dequantize(encoded, shape.Input * shape.Output)
                : GgufQ6K.Dequantize(encoded, shape.Input * shape.Output);
            var random = new Random(3191 + shape.Input);
            float[] input = Enumerable.Range(0, shape.Rows * shape.Input)
                .Select(_ => ((float)random.NextDouble() * 2f - 1f) * (shape.Subnormal ? 10000f : 1f)).ToArray();
            float[] bias = Enumerable.Range(0, shape.Output).Select(i => (i % 9 - 4) * 0.03f).ToArray();
            int count = shape.Rows * shape.Output;
            // Two guard elements also catch a subgroup tail writing too far.
            float[] sentinel = Enumerable.Repeat(float.NaN, count + 2).ToArray();
            using ArcBuffer x = lane.Upload(input), w = lane.UploadRaw(encoded), b = lane.Upload(bias);
            using ArcBuffer actualBuffer = lane.Upload(sentinel), referenceBuffer = lane.Allocate(count);
            Assert.Equal(encoded.Length, w.ByteLength);
            string quant = type == Qwen2Gguf.Q4KType ? "q4_k" : "q6_k";
            string kernel = $"q35l_{quant}_{variant}";
            long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes, allocated = lane.AllocatedBytes;
            if (subgroup)
                lane.Run(kernel, ((long)count + 1) / 2 * 32, 32,
                    x, w, b, actualBuffer, shape.Rows, shape.Input, shape.Output);
            else
                lane.Run2D(kernel, (long)shape.Output * 64, shape.Rows, 64, 1,
                    x, w, b, actualBuffer, shape.Rows, shape.Input, shape.Output);
            Assert.Equal(uploaded, lane.H2DBytes);
            Assert.Equal(downloaded, lane.D2HBytes);
            Assert.Equal(allocated, lane.AllocatedBytes);
            lane.Run($"qwen_linear_{quant}", count, 0,
                x, w, b, referenceBuffer, shape.Rows, shape.Input, shape.Output);
            var actual = new float[count + 2];
            var original = new float[count];
            lane.Read(actualBuffer, actual);
            lane.Read(referenceBuffer, original);
            Assert.True(float.IsNaN(actual[count]) && float.IsNaN(actual[count + 1]));
            for (int row = 0; row < shape.Rows; row++)
            for (int output = 0; output < shape.Output; output++)
            {
                double expected = bias[output], absoluteSum = Math.Abs(expected);
                for (int k = 0; k < shape.Input; k++)
                {
                    double product = (double)input[row * shape.Input + k] * decoded[output * shape.Input + k];
                    expected += product;
                    absoluteSum += Math.Abs(product);
                }
                int index = row * shape.Output + output;
                Assert.True(float.IsFinite(actual[index]) && float.IsFinite(original[index]));
                // Reduction order differs. Bound error relative to product magnitudes
                // so cancellation cannot disguise a wrong quantization layout.
                double tolerance = 2e-6 + absoluteSum * 3e-6;
                Assert.InRange(Math.Abs(actual[index] - expected), 0, tolerance);
                Assert.InRange(Math.Abs(original[index] - expected), 0, tolerance);
                Assert.InRange(Math.Abs(actual[index] - original[index]), 0, tolerance * 2);
            }
            Assert.Contains(kernel, lane.KernelTimings.Keys);
        }
    }

    private static byte[] Payload(uint type, int elements, bool subnormal)
    {
        int blockBytes = type == Qwen2Gguf.Q4KType ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        byte[] payload = new byte[elements / 256 * blockBytes];
        var random = new Random(297 + (int)type);
        random.NextBytes(payload);
        for (int block = 0; block < elements / 256; block++)
        {
            int offset = block * blockBytes;
            ushort scale = subnormal
                ? (ushort)((block % 4) switch { 0 => 0x0001, 1 => 0x03ff, 2 => 0x8001, _ => 0x83ff })
                : BitConverter.HalfToUInt16Bits((Half)((block % 7 + 1) / 8192f));
            if (type == Qwen2Gguf.Q4KType)
            {
                WriteHalfBits(payload, offset, scale);
                WriteHalfBits(payload, offset + 2, subnormal ? (ushort)0x0002
                    : BitConverter.HalfToUInt16Bits((Half)((block % 3 + 1) / 16384f)));
            }
            else WriteHalfBits(payload, offset + 208, scale);
        }
        return payload;
    }

    private static void WriteHalfBits(byte[] payload, int offset, ushort bits)
    {
        payload[offset] = (byte)bits;
        payload[offset + 1] = (byte)(bits >> 8);
    }
}
