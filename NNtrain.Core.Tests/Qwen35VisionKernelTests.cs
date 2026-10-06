using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35VisionKernelTests
{
    [Fact]
    public void F16LinearUsesGgufOutputRowsAndGelu()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = VisionLane();
        using ArcBuffer input = lane.Upload([1f, 2f, -3f, 4f, 5f, 6f]);
        // Shape [input=3, output=2]. GGUF places the input channel first.
        using ArcBuffer weight = HalfBuffer(lane, [1f, 2f, 3f, -1f, 0.5f, 2f]);
        using ArcBuffer bias = lane.Upload([0.25f, -0.5f]);
        using ArcBuffer output = lane.Allocate(4);
        lane.Run2D("q35v_linear_f16", 16, 16, 16, 16,
            input, weight, bias, output, 2, 3, 2, 1);
        var values = new float[4];
        lane.Read(output, values);
        float[] expected =
        [
            Gelu(1f * 1f + 2f * 2f - 3f * 3f + 0.25f),
            Gelu(1f * -1f + 2f * 0.5f - 3f * 2f - 0.5f),
            Gelu(4f * 1f + 5f * 2f + 6f * 3f + 0.25f),
            Gelu(4f * -1f + 5f * 0.5f + 6f * 2f - 0.5f)
        ];
        for (int i = 0; i < values.Length; i++)
            Assert.InRange(Math.Abs(values[i] - expected[i]), 0, 2e-5f);
    }

    [Fact]
    public void LearnedPositionsInterpolateWithAlignCornersInGroupedOrder()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = VisionLane();
        using ArcBuffer hidden = lane.Upload(new float[8]);
        using ArcBuffer position = lane.Upload([1f, 2f, 3f, 4f]);
        lane.Run("q35v_add_position", 8, 0,
            hidden, position, 8, 1, 4, 2, 2, 2);
        var values = new float[8];
        lane.Read(hidden, values);
        float[] expected = [1f, 2f, 5f / 3f, 8f / 3f,
            7f / 3f, 10f / 3f, 3f, 4f];
        for (int i = 0; i < values.Length; i++)
            Assert.InRange(Math.Abs(values[i] - expected[i]), 0, 2e-6f);
    }

    [Fact]
    public void VisionRopeRotatesHeightAndWidthHalvesAndLeavesValueUntouched()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = VisionLane();
        const int rows = 4, width = 8, headWidth = 8;
        var source = new float[rows * 3 * width];
        for (int i = 0; i < source.Length; i++) source[i] = i * 0.01f + 0.1f;
        using ArcBuffer qkv = lane.Upload(source);
        lane.Run("q35v_rope", rows * headWidth / 2, 0,
            qkv, rows, width, 1, headWidth, 2, 2, 10_000f);
        var actual = new float[source.Length];
        lane.Read(qkv, actual);
        for (int row = 0; row < rows; row++)
        {
            int y = row / 2, x = row % 2;
            for (int part = 0; part < 3; part++)
            for (int channel = 0; channel < headWidth; channel++)
            {
                float expected;
                if (part == 2) expected = source[row * 3 * width + part * width + channel];
                else
                {
                    int pair = channel % (headWidth / 2);
                    int frequency = pair % (headWidth / 4);
                    float angle = (pair < headWidth / 4 ? y : x)
                        * MathF.Pow(10_000f, -2f * frequency / (headWidth / 2));
                    int first = row * 3 * width + part * width + pair;
                    float a = source[first], b = source[first + headWidth / 2];
                    expected = channel < headWidth / 2
                        ? a * MathF.Cos(angle) - b * MathF.Sin(angle)
                        : a * MathF.Sin(angle) + b * MathF.Cos(angle);
                }
                Assert.InRange(Math.Abs(actual[row * 3 * width + part * width + channel] - expected),
                    0, 2e-6f);
            }
        }
    }

    [Fact]
    public void PatchEmbeddingAddsBothTemporalWeightPlanes()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = VisionLane();
        using ArcBuffer patches = lane.Upload([1f, -2f, 3f, 4f]);
        using ArcBuffer weight0 = HalfBuffer(lane, [1f, 2f, 3f, 4f,
            0.5f, -0.5f, 1f, -1f]);
        using ArcBuffer weight1 = HalfBuffer(lane, [0.5f, -1f, 2f, 0f,
            1f, 1f, -2f, 2f]);
        using ArcBuffer bias = lane.Upload([0.25f, -0.25f]);
        using ArcBuffer output = lane.Allocate(2);
        lane.Run("q35v_patch_embed", 2, 0,
            patches, weight0, weight1, bias, output, 1, 4, 2);
        var actual = new float[2];
        lane.Read(output, actual);
        Assert.InRange(Math.Abs(actual[0] - 30.75f), 0, 1e-5f);
        Assert.InRange(Math.Abs(actual[1] - 1.25f), 0, 1e-5f);
    }

    [Fact]
    public void NormalLayerNormMatchesCenteredCpuReference()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = VisionLane();
        float[] source = [1f, 2f, 4f, 7f, -3f, -2f, 0f, 4f];
        float[] scale = [0.5f, 1f, -1f, 2f];
        float[] bias = [1f, 0f, 0.25f, -0.5f];
        using ArcBuffer input = lane.Upload(source);
        using ArcBuffer gamma = lane.Upload(scale);
        using ArcBuffer beta = lane.Upload(bias);
        using ArcBuffer output = lane.Allocate(source.Length);
        lane.Run("q35v_layer_norm", 2 * 256, 256,
            input, gamma, beta, output, 4, 1e-6f);
        var actual = new float[source.Length];
        lane.Read(output, actual);
        for (int row = 0; row < 2; row++)
        {
            float mean = source.Skip(row * 4).Take(4).Average();
            float variance = source.Skip(row * 4).Take(4)
                .Average(value => (value - mean) * (value - mean));
            for (int col = 0; col < 4; col++)
            {
                float expected = (source[row * 4 + col] - mean)
                    / MathF.Sqrt(variance + 1e-6f) * scale[col] + bias[col];
                Assert.InRange(Math.Abs(actual[row * 4 + col] - expected), 0, 2e-6f);
            }
        }
    }

    [Fact]
    public void BidirectionalAttentionMatchesCpuSoftmaxReference()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = VisionLane();
        const int rows = 2, heads = 1, headWidth = 4, width = heads * headWidth;
        float[] source =
        [
            1f, 0f, 0f, 1f,   1f, 0f, 0f, 1f,   2f, 3f, 4f, 5f,
            0f, 1f, 1f, 0f,   0f, 1f, 1f, 0f,   7f, 11f, 13f, 17f
        ];
        using ArcBuffer qkv = lane.Upload(source);
        using ArcBuffer scores = lane.Allocate(rows * heads * rows);
        using ArcBuffer output = lane.Allocate(rows * width);
        lane.Run("q35v_attention_scores", rows * heads * rows, 0,
            qkv, scores, rows, heads, headWidth);
        lane.Run("q35v_attention_softmax", rows * heads * 256, 256,
            scores, rows);
        lane.Run("q35v_attention_context", rows * width, 0,
            qkv, scores, output, rows, heads, headWidth);
        var actual = new float[rows * width];
        lane.Read(output, actual);
        for (int query = 0; query < rows; query++)
        {
            float[] logits = new float[rows];
            for (int key = 0; key < rows; key++)
            for (int feature = 0; feature < width; feature++)
                logits[key] += source[query * 3 * width + feature]
                    * source[key * 3 * width + width + feature] / MathF.Sqrt(headWidth);
            float[] exponentials = logits.Select(value => MathF.Exp(value - logits.Max())).ToArray();
            float sum = exponentials.Sum();
            for (int feature = 0; feature < width; feature++)
            {
                float expected = 0;
                for (int key = 0; key < rows; key++)
                    expected += exponentials[key] / sum
                        * source[key * 3 * width + 2 * width + feature];
                Assert.InRange(Math.Abs(actual[query * width + feature] - expected), 0, 2e-6f);
            }
        }
    }

    private static ArcExecutionLane VisionLane() => new(options: new ArcExecutionOptions
    {
        Qwen35InferenceKernelsOnly = true,
        CacheProgramBinary = false
    });

    private static ArcBuffer HalfBuffer(ArcExecutionLane lane, float[] values)
    {
        ushort[] half = values.Select(value =>
            BitConverter.HalfToUInt16Bits((Half)value)).ToArray();
        ArcBuffer buffer = lane.AllocateBytes(checked(half.Length * sizeof(ushort)));
        lane.WriteRaw(buffer, half);
        return buffer;
    }

    private static float Gelu(float x)
        => 0.5f * x * (1f + MathF.Tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}
