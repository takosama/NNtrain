using NNtrain;
using NNtrain.Arc;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenQuantizedLinearFastTests
{
    [Theory]
    [InlineData(Qwen2Gguf.Q4KType)]
    [InlineData(Qwen2Gguf.Q6KType)]
    public void CooperativeKernelMatchesCpuAndReferenceAcrossDecodeAndPrefill(uint ggmlType)
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        QuantizedCase[] cases =
        [
            MakeSubnormalCase(ggmlType),
            MakeRandomCase(ggmlType, rows: 1, inputWidth: 256, outputWidth: 17),
            MakeRandomCase(ggmlType, rows: 1, inputWidth: 2048, outputWidth: 7),
            MakeRandomCase(ggmlType, rows: 1, inputWidth: 11008, outputWidth: 3),
            MakeRandomCase(ggmlType, rows: 3, inputWidth: 768, outputWidth: 7),
            MakeRandomCase(ggmlType, rows: 5, inputWidth: 512, outputWidth: 9),
        ];

        float[][] reference = Run(cases, fast: false);
        float[][] cooperative = Run(cases, fast: true);
        float[][] subgroup = Run(cases, fast: true, subgroup: true);
        float[][] subgroupPrefill = Run(cases, fast: true, subgroup: true,
            subgroupPrefill: true);
        for (int test = 0; test < cases.Length; test++)
        {
            float tolerance = cases[test].IsSubnormalOnly ? 1e-7f : 0.003f;
            for (int i = 0; i < cases[test].Expected.Length; i++)
            {
                float expected = cases[test].Expected[i];
                float bound = tolerance + MathF.Abs(expected) * 0.0003f;
                AssertClose(expected, reference[test][i], bound);
                AssertClose(expected, cooperative[test][i], bound);
                AssertClose(reference[test][i], cooperative[test][i], bound);
                AssertClose(expected, subgroup[test][i], bound);
                AssertClose(cooperative[test][i], subgroup[test][i], bound);
                AssertClose(expected, subgroupPrefill[test][i], bound);
                AssertClose(cooperative[test][i], subgroupPrefill[test][i], bound);
            }
        }
    }

    private static float[][] Run(IReadOnlyList<QuantizedCase> cases,
        bool fast, bool subgroup = false, bool subgroupPrefill = false)
    {
        using var execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32,
            new ArcExecutionOptions
            {
                QwenQuantizedLinearFast = fast,
                QwenQuantizedLinearSubgroup = subgroup,
                QwenQuantizedSubgroupPrefill = subgroupPrefill,
            });
        using var noGrad = AutogradContext.NoGrad();
        var observed = new float[cases.Count][];
        for (int test = 0; test < cases.Count; test++)
        {
            QuantizedCase value = cases[test];
            var input = new Tensor(value.Input, [value.Rows, value.InputWidth]);
            var bias = new Tensor(value.Bias, [value.OutputWidth]);
            using var matrix = new ArcQuantizedMatrix(value.Encoded, value.GgmlType,
                value.OutputWidth, value.InputWidth);
            using (Tensor.BeginArcInferenceFrame())
                observed[test] = matrix.Forward(input, bias).Data.ToArray();
            Assert.Equal(value.Rows * value.OutputWidth, observed[test].Length);
        }

        ArcExecutionLane lane = Tensor.ArcLane;
        lane.Synchronize();
        lane.CheckNumericStatus();
        ArcDeviceInfo device = ArcDevices.Enumerate()[0];
        bool expectSubgroup = subgroup
            && device.SupportsXmx && device.MinimumSubgroupSize == 16
            && device.Extensions.Split(' ').Contains("cl_intel_subgroups");
        string suffix = fast ? "_fast_r1" : string.Empty;
        string q4 = "qwen_linear_q4_k" + suffix;
        string q6 = "qwen_linear_q6_k" + suffix;
        string subgroupKernel = cases[0].GgmlType == Qwen2Gguf.Q4KType
            ? "qwen_linear_q4_k_sg16_r1" : "qwen_linear_q6_k_sg16_r1";
        Assert.True(lane.KernelTimings.ContainsKey(expectSubgroup
            ? subgroupKernel
            : cases[0].GgmlType == Qwen2Gguf.Q4KType ? q4 : q6));
        if (fast)
        {
            string prefill = cases[0].GgmlType == Qwen2Gguf.Q4KType
                ? "qwen_linear_q4_k_fast_r4" : "qwen_linear_q6_k_fast_r4";
            if (expectSubgroup && subgroupPrefill)
                Assert.False(lane.KernelTimings.ContainsKey(prefill));
            else
                Assert.True(lane.KernelTimings.ContainsKey(prefill));
        }
        return observed;
    }

    private static QuantizedCase MakeSubnormalCase(uint ggmlType)
    {
        int blockBytes = ggmlType == Qwen2Gguf.Q4KType
            ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        var encoded = new byte[blockBytes];
        if (ggmlType == Qwen2Gguf.Q4KType)
        {
            encoded[0] = 1;   // FP16 0x0001 = 2^-24.
            encoded[4] = 63;  // First group scale.
            encoded[16] = 15; // First low-nibble weight.
        }
        else
        {
            encoded[208] = 1;
            encoded[192] = 127;
            encoded[0] = 15;
            encoded[128] = 3; // Quantized first value = 63, signed = 31.
        }
        var input = new float[256];
        input[0] = 1f;
        float[] expected = CpuLinear(ggmlType, encoded, input, [0f], 1, 256, 1);
        Assert.True(expected[0] > 0f);
        return new QuantizedCase(ggmlType, 1, 256, 1, encoded, input,
            [0f], expected, IsSubnormalOnly: true);
    }

    private static QuantizedCase MakeRandomCase(
        uint ggmlType, int rows, int inputWidth, int outputWidth)
    {
        int blockBytes = ggmlType == Qwen2Gguf.Q4KType
            ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        int blocksPerRow = inputWidth / 256;
        var encoded = new byte[outputWidth * blocksPerRow * blockBytes];
        var random = new Random(checked(313 + (int)ggmlType * 19
            + rows * 29 + inputWidth + outputWidth));
        for (int blockIndex = 0; blockIndex < outputWidth * blocksPerRow; blockIndex++)
        {
            Span<byte> block = encoded.AsSpan(blockIndex * blockBytes, blockBytes);
            random.NextBytes(block);
            ushort d = (ushort)((blockIndex % 7) switch
            {
                0 => 0x0001,
                1 => 0x03ff,
                _ => 0x2000,
            });
            if (ggmlType == Qwen2Gguf.Q4KType)
            {
                block[0] = (byte)d;
                block[1] = (byte)(d >> 8);
                block[2] = 0;
                block[3] = 0x18; // FP16 dmin = 2^-9.
            }
            else
            {
                block[208] = (byte)d;
                block[209] = (byte)(d >> 8);
                for (int scale = 0; scale < 16; scale++)
                    block[192 + scale] = unchecked((byte)(random.Next(31) - 15));
            }
        }
        var input = new float[rows * inputWidth];
        for (int i = 0; i < input.Length; i++)
            input[i] = MathF.Sin(i * 0.091f + rows * 0.23f) * 0.08f;
        float[] bias = Enumerable.Range(0, outputWidth)
            .Select(i => (i % 5 - 2) * 0.017f).ToArray();
        float[] expected = CpuLinear(ggmlType, encoded, input, bias,
            rows, inputWidth, outputWidth);
        return new QuantizedCase(ggmlType, rows, inputWidth, outputWidth,
            encoded, input, bias, expected, IsSubnormalOnly: false);
    }

    private static float[] CpuLinear(uint ggmlType, byte[] encoded,
        float[] input, float[] bias, int rows, int inputWidth, int outputWidth)
    {
        int blockBytes = ggmlType == Qwen2Gguf.Q4KType
            ? GgufQ4K.BlockBytes : GgufQ6K.BlockBytes;
        int bytesPerRow = inputWidth / 256 * blockBytes;
        var expected = new float[rows * outputWidth];
        for (int output = 0; output < outputWidth; output++)
        {
            ReadOnlySpan<byte> packed = encoded.AsSpan(output * bytesPerRow, bytesPerRow);
            float[] weights = ggmlType == Qwen2Gguf.Q4KType
                ? GgufQ4K.Dequantize(packed, inputWidth)
                : GgufQ6K.Dequantize(packed, inputWidth);
            for (int row = 0; row < rows; row++)
            {
                float sum = bias[output];
                for (int k = 0; k < inputWidth; k++)
                    sum = MathF.FusedMultiplyAdd(input[row * inputWidth + k],
                        weights[k], sum);
                expected[row * outputWidth + output] = sum;
            }
        }
        return expected;
    }

    private static void AssertClose(float expected, float actual, float bound)
    {
        Assert.True(float.IsFinite(actual));
        Assert.InRange(MathF.Abs(expected - actual), 0f, bound);
    }

    private sealed record QuantizedCase(
        uint GgmlType, int Rows, int InputWidth, int OutputWidth,
        byte[] Encoded, float[] Input, float[] Bias, float[] Expected,
        bool IsSubnormalOnly);
}
