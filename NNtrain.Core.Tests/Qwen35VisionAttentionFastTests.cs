using NNtrain.Arc;
using static NNtrain.Arc.ArcExecutionLane;
using Xunit;

namespace NNtrain.Core.Tests;

public sealed class Qwen35VisionAttentionFastTests(ITestOutputHelper output)
{
    [Fact]
    public void FullImageAttentionPreservesScalarBits()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = new ArcExecutionLane(options: new ArcExecutionOptions
        {
            Qwen35InferenceKernelsOnly = true,
            CacheProgramBinary = true
        });
        const int rows = 2304, heads = 16, headWidth = 72, width = heads * headWidth;
        var source = new float[rows * 3 * width];
        var random = new Random(731042);
        for (int i = 0; i < source.Length; i++) source[i] = (random.NextSingle() - .5f) * 3f;
        using ArcBuffer qkv = lane.Upload(source);
        using ArcBuffer scalarScores = lane.Allocate(rows * heads * rows);
        using ArcBuffer tiledScores = lane.Allocate(rows * heads * rows);
        using ArcBuffer scalarContext = lane.Allocate(rows * width);
        using ArcBuffer tiledContext = lane.Allocate(rows * width);
        lane.Run("q35v_attention_scores", (long)rows * heads * rows, 128,
            qkv, scalarScores, rows, heads, headWidth);
        lane.Run2D("q35v_attention_scores_tiled", RoundTile(rows), heads * RoundTile(rows), 16, 16,
            qkv, tiledScores, rows, heads, headWidth);
        CompareBits(scalarScores, tiledScores, rows * heads * rows, "scores");
        lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, scalarScores, rows);
        lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, tiledScores, rows);
        CompareBits(scalarScores, tiledScores, rows * heads * rows, "probabilities");
        lane.Run("q35v_attention_context", (long)rows * width, 128,
            qkv, scalarScores, scalarContext, rows, heads, headWidth);
        lane.Run2D("q35v_attention_context_tiled", RoundTile(headWidth), heads * RoundTile(rows), 16, 16,
            qkv, tiledScores, tiledContext, rows, heads, headWidth);
        CompareBits(scalarContext, tiledContext, rows * width, "context");

        // Bound host comparison memory to two 4 MiB chunks. Full image score
        // arrays are 324 MiB each; copying both in full is unnecessary.
        void CompareBits(ArcBuffer reference, ArcBuffer actual, int count, string stage)
        {
            const int chunk = 1024 * 1024;
            var expectedValues = new float[Math.Min(chunk, count)];
            var actualValues = new float[expectedValues.Length];
            int changed = 0;
            float maxAbsoluteError = 0;
            for (int offset = 0; offset < count; offset += chunk)
            {
                int length = Math.Min(chunk, count - offset);
                if (expectedValues.Length != length)
                {
                    expectedValues = new float[length];
                    actualValues = new float[length];
                }
                lane.ReadFloatRange(reference, offset, expectedValues);
                lane.ReadFloatRange(actual, offset, actualValues);
                for (int i = 0; i < length; i++)
                {
                    if (BitConverter.SingleToInt32Bits(expectedValues[i])
                        == BitConverter.SingleToInt32Bits(actualValues[i])) continue;
                    changed++;
                    maxAbsoluteError = Math.Max(maxAbsoluteError, Math.Abs(expectedValues[i] - actualValues[i]));
                }
            }
            output.WriteLine($"Full image {stage}: changed={changed}, max_abs={maxAbsoluteError:G9}");
            Assert.Equal(0, changed);
        }
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(3, 2, 7)]
    [InlineData(19, 3, 72)]
    [InlineData(32, 2, 72)]
    public void TiledScoresAndContextMatchScalarAndCpu(int rows, int heads, int headWidth)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Arc OpenCL GPU required.");
        using var lane = new ArcExecutionLane(options: new ArcExecutionOptions
        {
            Qwen35InferenceKernelsOnly = true,
            CacheProgramBinary = false
        });
        int width = heads * headWidth;
        var source = new float[rows * 3 * width];
        for (int i = 0; i < source.Length; i++)
            source[i] = 0.4f * MathF.Sin(i * 0.037f) + 0.2f * MathF.Cos(i * 0.11f);
        using ArcBuffer qkv = lane.Upload(source);
        using ArcBuffer scalarScores = lane.Allocate(rows * heads * rows);
        using ArcBuffer tiledScores = lane.Allocate(rows * heads * rows);
        using ArcBuffer scalarContext = lane.Allocate(rows * width);
        using ArcBuffer tiledContext = lane.Allocate(rows * width);
        lane.Run("q35v_attention_scores", (long)rows * heads * rows, 0,
            qkv, scalarScores, rows, heads, headWidth);
        lane.Run2D("q35v_attention_scores_tiled", RoundTile(rows), heads * RoundTile(rows), 16, 16,
            qkv, tiledScores, rows, heads, headWidth);
        var oldScores = new float[rows * heads * rows];
        var newScores = new float[oldScores.Length];
        lane.Read(scalarScores, oldScores);
        lane.Read(tiledScores, newScores);
        Assert.Equal(oldScores.Select(BitConverter.SingleToInt32Bits), newScores.Select(BitConverter.SingleToInt32Bits));

        var expectedProbabilities = new float[rows * heads * rows];
        for (int query = 0; query < rows; query++)
        for (int head = 0; head < heads; head++)
        {
            int offset = (query * heads + head) * rows;
            float maximum = float.NegativeInfinity;
            for (int key = 0; key < rows; key++)
            {
                float score = 0;
                for (int feature = 0; feature < headWidth; feature++)
                    score = MathF.FusedMultiplyAdd(source[query * 3 * width + head * headWidth + feature],
                        source[key * 3 * width + width + head * headWidth + feature], score);
                score *= 1f / MathF.Sqrt(headWidth);
                Assert.InRange(Math.Abs(newScores[offset + key] - score), 0, 2e-5f);
                expectedProbabilities[offset + key] = score;
                maximum = Math.Max(maximum, score);
            }
            float sum = 0;
            for (int key = 0; key < rows; key++)
            {
                expectedProbabilities[offset + key] = MathF.Exp(expectedProbabilities[offset + key] - maximum);
                sum += expectedProbabilities[offset + key];
            }
            for (int key = 0; key < rows; key++) expectedProbabilities[offset + key] /= sum;
        }

        lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, scalarScores, rows);
        lane.Run("q35v_attention_softmax", (long)rows * heads * 256, 256, tiledScores, rows);
        lane.Run("q35v_attention_context", (long)rows * width, 0,
            qkv, scalarScores, scalarContext, rows, heads, headWidth);
        lane.Run2D("q35v_attention_context_tiled", RoundTile(headWidth), heads * RoundTile(rows), 16, 16,
            qkv, tiledScores, tiledContext, rows, heads, headWidth);
        var oldContext = new float[rows * width];
        var newContext = new float[oldContext.Length];
        lane.Read(scalarContext, oldContext);
        lane.Read(tiledContext, newContext);
        Assert.Equal(oldContext.Select(BitConverter.SingleToInt32Bits), newContext.Select(BitConverter.SingleToInt32Bits));

        for (int query = 0; query < rows; query++)
        for (int head = 0; head < heads; head++)
        for (int feature = 0; feature < headWidth; feature++)
        {
            float expected = 0;
            for (int key = 0; key < rows; key++)
                expected = MathF.FusedMultiplyAdd(expectedProbabilities[(query * heads + head) * rows + key],
                    source[key * 3 * width + 2 * width + head * headWidth + feature], expected);
            Assert.InRange(Math.Abs(newContext[query * width + head * headWidth + feature] - expected), 0, 3e-5f);
        }
    }

    private static int RoundTile(int count) => checked((count + 15) / 16 * 16);
}
