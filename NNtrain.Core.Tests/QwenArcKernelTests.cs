using NNtrain;
using NNtrain.Arc;
using NNtrain.Runtime.Execution;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class QwenArcKernelTests
{
    [Theory]
    [InlineData(TensorDType.Float16)]
    [InlineData(TensorDType.BFloat16)]
    public void CpuQwenPrimitivesPreserveHalfAndBfloat16Storage(TensorDType dtype)
    {
        using var cpu = TensorExecutionContext.Push(new TorchDevice(TensorDevice.Cpu));
        var x = new Tensor([1.1f, -0.3f, 0.7f, 1.9f], [2, 2], dtype: dtype);
        var weight = new Tensor([0.8f, -1.2f], [2], dtype: dtype);
        var up = new Tensor([1.3f, -0.7f, 0.9f, 0.2f], [2, 2], dtype: dtype);

        Tensor norm = x.RmsNormLastDim(weight);
        Tensor silu = x.SiluMultiply(up);

        Assert.Equal(dtype, norm.DType);
        Assert.Equal(dtype, silu.DType);
        for (int row = 0; row < 2; row++)
        {
            float first = x.Data[row * 2], second = x.Data[row * 2 + 1];
            float inv = 1f / MathF.Sqrt((first * first + second * second) / 2f + 1e-6f);
            AssertClose(first * inv * weight.Data[0], norm.Data[row * 2], 0.015f);
            AssertClose(second * inv * weight.Data[1], norm.Data[row * 2 + 1], 0.015f);
        }
        for (int i = 0; i < x.Numel; i++)
        {
            float gate = x.Data[i];
            AssertClose(gate / (1f + MathF.Exp(-gate)) * up.Data[i], silu.Data[i], 0.015f);
        }
    }

    [Fact]
    public void GroupedQueryAttentionMatchesCpuReferenceAndAvoidsQuadraticNoGradStorage()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32);
        using var noGrad = AutogradContext.NoGrad();

        const int sequence = 5, heads = 4, kvHeads = 2, headWidth = 4;
        var query = new Tensor(Values(sequence * heads * headWidth, 3),
            [1, sequence, heads * headWidth]);
        var key = new Tensor(Values(sequence * kvHeads * headWidth, 11),
            [1, sequence, kvHeads * headWidth]);
        var value = new Tensor(Values(sequence * kvHeads * headWidth, 29),
            [1, sequence, kvHeads * headWidth]);
        float[] expected = GqaReference(query.Data, key.Data, value.Data,
            sequence, heads, kvHeads, headWidth, 10_000f);
        float[] actual;
        using (Tensor.BeginArcInferenceFrame())
            actual = query.QwenGroupedQueryAttention(key, value, heads, kvHeads,
                ropeTheta: 10_000f).Data.ToArray();
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            AssertClose(expected[i], actual[i], 0.002f);

        // At S=512 the old saved probability matrix alone used 4 MiB.
        // The no-grad path should need only the row-local scores and linear
        // input/output buffers, well below one additional MiB.
        const int longSequence = 512, longHeadWidth = 8;
        var longQuery = new Tensor(Values(longSequence * heads * longHeadWidth, 17),
            [1, longSequence, heads * longHeadWidth]);
        var longKey = new Tensor(Values(longSequence * kvHeads * longHeadWidth, 19),
            [1, longSequence, kvHeads * longHeadWidth]);
        var longValue = new Tensor(Values(longSequence * kvHeads * longHeadWidth, 23),
            [1, longSequence, kvHeads * longHeadWidth]);
        ArcExecutionLane lane = Tensor.ArcLane;
        long before = lane.PeakAllocatedBytes;
        using (Tensor.BeginArcInferenceFrame())
        {
            Tensor output = longQuery.QwenGroupedQueryAttention(longKey, longValue,
                heads, kvHeads);
            Assert.Equal(longSequence * heads * longHeadWidth, output.Numel);
            lane.Synchronize();
            Assert.InRange(lane.PeakAllocatedBytes - before, 0L, 1_048_576L);
        }
        lane.CheckNumericStatus();

        // The current row-local implementation has a fixed local-memory bound.
        const int tooLong = 4097;
        var qTooLong = new Tensor(new float[tooLong * 2], [1, tooLong, 2]);
        var kTooLong = new Tensor(new float[tooLong * 2], [1, tooLong, 2]);
        var vTooLong = new Tensor(new float[tooLong * 2], [1, tooLong, 2]);
        Assert.Throws<NotSupportedException>(() =>
            qTooLong.QwenGroupedQueryAttention(kTooLong, vTooLong, 1, 1));
    }

    [Fact]
    public void QuantizedQ4AndQ6KArcKernelsMatchCpuForSubnormalScales()
    {
        Assert.SkipWhen(!Tensor.IsArcAvailable(0), "Intel Arc GPU is required.");
        using var execution = Tensor.BeginArcExecution(0, TensorPrecisionMode.Float32);
        using var noGrad = AutogradContext.NoGrad();
        var inputValues = new float[256];
        inputValues[0] = 1f;
        var input = new Tensor(inputValues, [1, 256]);
        var bias = new Tensor([0f], [1]);

        var q4 = new byte[GgufQ4K.BlockBytes];
        q4[0] = 0x01; // Smallest positive half subnormal: 2^-24.
        q4[4] = 63;   // First six-bit scale.
        q4[16] = 15;  // First low-nibble quantized value.
        using (var matrix = new ArcQuantizedMatrix(q4, Qwen2Gguf.Q4KType, 1, 256))
        using (Tensor.BeginArcInferenceFrame())
        {
            float expected = GgufQ4K.Dequantize(q4, 256)[0];
            float actual = matrix.Forward(input, bias).Data[0];
            Assert.True(expected > 0f);
            AssertClose(expected, actual, 1e-7f);
        }

        var q6 = new byte[GgufQ6K.BlockBytes];
        q6[208] = 0xff;
        q6[209] = 0x03; // Largest positive half subnormal.
        q6[0] = 0x0f;
        q6[128] = 0x03;
        q6[192] = 127; // First signed scale; decoded quantized value is 31.
        using (var matrix = new ArcQuantizedMatrix(q6, Qwen2Gguf.Q6KType, 1, 256))
        using (Tensor.BeginArcInferenceFrame())
        {
            float expected = GgufQ6K.Dequantize(q6, 256)[0];
            float actual = matrix.Forward(input, bias).Data[0];
            Assert.True(expected > 0f);
            AssertClose(expected, actual, 2e-5f);
        }
        Tensor.ArcLane.CheckNumericStatus();
    }

    private static float[] Values(int count, int seed)
    {
        var result = new float[count];
        for (int i = 0; i < count; i++)
            result[i] = MathF.Sin(i * 0.137f + seed * 0.41f) * 0.43f;
        return result;
    }

    private static float[] GqaReference(
        IReadOnlyList<float> query, IReadOnlyList<float> key,
        IReadOnlyList<float> value, int sequence, int heads,
        int kvHeads, int headWidth, float theta)
    {
        var output = new float[sequence * heads * headWidth];
        int kvWidth = kvHeads * headWidth;
        int qWidth = heads * headWidth;
        for (int qpos = 0; qpos < sequence; qpos++)
        for (int head = 0; head < heads; head++)
        {
            int kvHead = head / (heads / kvHeads);
            var scores = new float[qpos + 1];
            int qBase = qpos * qWidth + head * headWidth;
            float max = float.NegativeInfinity;
            for (int kpos = 0; kpos <= qpos; kpos++)
            {
                int kBase = kpos * kvWidth + kvHead * headWidth;
                float dot = 0f;
                for (int pair = 0; pair < headWidth / 2; pair++)
                {
                    float angleQ = qpos * MathF.Pow(theta, -2f * pair / headWidth);
                    float angleK = kpos * MathF.Pow(theta, -2f * pair / headWidth);
                    float cq = MathF.Cos(angleQ), sq = MathF.Sin(angleQ);
                    float ck = MathF.Cos(angleK), sk = MathF.Sin(angleK);
                    float q1 = query[qBase + pair] * cq
                        - query[qBase + pair + headWidth / 2] * sq;
                    float q2 = query[qBase + pair + headWidth / 2] * cq
                        + query[qBase + pair] * sq;
                    float k1 = key[kBase + pair] * ck
                        - key[kBase + pair + headWidth / 2] * sk;
                    float k2 = key[kBase + pair + headWidth / 2] * ck
                        + key[kBase + pair] * sk;
                    dot += q1 * k1 + q2 * k2;
                }
                scores[kpos] = dot / MathF.Sqrt(headWidth);
                max = MathF.Max(max, scores[kpos]);
            }
            float sum = 0f;
            for (int kpos = 0; kpos < scores.Length; kpos++)
            {
                scores[kpos] = MathF.Exp(scores[kpos] - max);
                sum += scores[kpos];
            }
            for (int channel = 0; channel < headWidth; channel++)
            {
                float weighted = 0f;
                for (int kpos = 0; kpos < scores.Length; kpos++)
                {
                    int vi = kpos * kvWidth + kvHead * headWidth + channel;
                    weighted += scores[kpos] / sum * value[vi];
                }
                output[qBase + channel] = weighted;
            }
        }
        return output;
    }

    private static void AssertClose(float expected, float actual, float tolerance)
    {
        Assert.True(float.IsFinite(actual));
        Assert.InRange(MathF.Abs(expected - actual), 0f, tolerance);
    }
}
