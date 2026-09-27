using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuAttentionTests
{
    [Fact]
    public void NormActivationAndResidualMatchCpuWithoutHostTransfers()
    {
        using ArcExecutionLane lane = CreateLane();
        const int rows = 3, width = 273;
        float[] input = Values(rows * width, 1), weights = Values(width, 3);
        float[] up = Values(input.Length, 7);
        // Include values that overflow a naive exp(-x) sigmoid implementation.
        input[0] = -100f; input[1] = 100f;
        using ArcBuffer x = lane.Upload(input), w = lane.Upload(weights), u = lane.Upload(up);
        long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        using ArcBuffer normalized = Qwen35Gpu.RmsNorm(lane, x, w, rows, width, 1e-5f);
        using ArcBuffer activated = Qwen35Gpu.SiluMultiply(lane, x, u, input.Length);
        Qwen35Gpu.AddInPlace(lane, activated, normalized, input.Length);
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded, lane.D2HBytes);
        float[] actual = Read(lane, activated, input.Length);
        float[] expectedNorm = Qwen35Math.RmsNorm(input, weights, 1e-5f, width);
        for (int i = 0; i < actual.Length; ++i)
            AssertClose(expectedNorm[i] + Qwen35Math.Silu(input[i]) * up[i], actual[i], 2e-5f);
        Assert.Equal(input, Read(lane, x, input.Length));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(8)]
    [InlineData(12)]
    public void AttentionPreservesMultiTokenGroupedCachesAndMatchesDoubleReference(int ropeDimensions)
    {
        using ArcExecutionLane lane = CreateLane();
        const int heads = 4, kvHeads = 2, width = 12, steps = 5;
        const float epsilon = 1e-5f, theta = 10000f;
        float[] qWeights = Values(width, 13).Select(x => x + 1).ToArray();
        float[] kWeights = Values(width, 23).Select(x => x + 1).ToArray();
        using ArcBuffer qNorm = lane.Upload(qWeights), kNorm = lane.Upload(kWeights);
        using ArcBuffer keysCache = lane.Allocate(steps * kvHeads * width);
        using ArcBuffer valuesCache = lane.Allocate(steps * kvHeads * width);
        var keys = new List<double[]>();
        var values = new List<double[]>();
        for (int position = 0; position < steps; ++position)
        {
            float[] qAndGate = Values(2 * heads * width, position + 37);
            float[] key = Values(kvHeads * width, position + 51);
            float[] value = Values(kvHeads * width, position + 73);
            // Different gates for every head expose incorrectly splitting all Qs from all gates.
            qAndGate[width] = -80f; qAndGate[3 * width + 1] = 80f;
            using ArcBuffer q = lane.Upload(qAndGate), k = lane.Upload(key), v = lane.Upload(value);
            long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
            using ArcBuffer result = Qwen35Gpu.AttentionStep(lane, q, k, v, qNorm, kNorm,
                keysCache, valuesCache, position, heads, kvHeads, width, ropeDimensions, theta, epsilon);
            Assert.Equal(uploaded, lane.H2DBytes);
            Assert.Equal(downloaded, lane.D2HBytes);

            double[] query = NormalizeAndRotate(
                Enumerable.Range(0, heads).SelectMany(h => qAndGate.Skip(2 * h * width).Take(width))
                    .Select(x => (double)x).ToArray(), qWeights, width, ropeDimensions, position, theta, epsilon);
            keys.Add(NormalizeAndRotate(key.Select(x => (double)x).ToArray(), kWeights,
                width, ropeDimensions, position, theta, epsilon));
            values.Add(value.Select(x => (double)x).ToArray());
            double[] expected = AttentionReference(query, qAndGate, keys, values, heads, kvHeads, width);
            AssertClose(expected, Read(lane, result, heads * width), 2e-5);
            AssertClose(keys.SelectMany(x => x).ToArray(), Read(lane, keysCache, (position + 1) * kvHeads * width), 2e-5);
            Assert.Equal(values.SelectMany(x => x).Select(x => (float)x).ToArray(),
                Read(lane, valuesCache, (position + 1) * kvHeads * width));
            Assert.Equal(key, Read(lane, k, key.Length));
        }
    }

    [Fact]
    public void AttentionSupportsMoreThan4096CachedTokensAndStableSoftmax()
    {
        using ArcExecutionLane lane = CreateLane();
        const int steps = 4097, heads = 2, kvHeads = 1, width = 4;
        // All values are identical, so their weighted average is exact even with
        // highly separated scores. This also catches any fixed 4096 scratch limit.
        float[] cachedKeys = Values(steps * width, 5).Select(x => x * 300f).ToArray();
        float[] cachedValues = Enumerable.Range(0, steps).SelectMany(_ => new[] { 2f, -3f, 4f, -5f }).ToArray();
        float[] qAndGate = [1, 2, 3, 4, 0, 0, 0, 0, 4, 3, 2, 1, 0, 0, 0, 0];
        using ArcBuffer q = lane.Upload(qAndGate), k = lane.Upload(new float[] { 3, 1, 4, 2 });
        using ArcBuffer v = lane.Upload(new float[] { 2, -3, 4, -5 });
        using ArcBuffer norm = lane.Upload(Enumerable.Repeat(1f, width).ToArray());
        using ArcBuffer keys = lane.Upload(cachedKeys), values = lane.Upload(cachedValues);
        long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        using ArcBuffer output = Qwen35Gpu.AttentionStep(lane, q, k, v, norm, norm, keys, values,
            steps - 1, heads, kvHeads, width, 2, 10000f, 1e-5f);
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded, lane.D2HBytes);
        float[] expected = [1, -1.5f, 2, -2.5f, 1, -1.5f, 2, -2.5f];
        AssertClose(expected.Select(x => (double)x).ToArray(), Read(lane, output, heads * width), 2e-5);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3001)]
    public void ArgMaxSelectsFirstTieAndReadsOnlyTokenAndStatus(int count)
    {
        using ArcExecutionLane lane = CreateLane();
        float[] logits = Enumerable.Repeat(-20f, count).ToArray();
        int expected = count == 1 ? 0 : 19;
        logits[expected] = -1;
        logits[count - 1] = -1;
        using ArcBuffer input = lane.Upload(logits);
        long uploaded = lane.H2DBytes, downloaded = lane.D2HBytes;
        Assert.Equal(expected, Qwen35Gpu.ArgMax(lane, input, count));
        Assert.Equal(uploaded, lane.H2DBytes);
        Assert.Equal(downloaded + 2 * sizeof(int), lane.D2HBytes);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void ArgMaxRejectsAnyNonFiniteLogitEvenAwayFromMaximum(float invalid)
    {
        using ArcExecutionLane lane = CreateLane();
        float[] logits = new float[3001];
        logits[2] = 4;
        logits[2999] = invalid;
        using ArcBuffer input = lane.Upload(logits);
        Assert.Throws<ArithmeticException>(() => Qwen35Gpu.ArgMax(lane, input, logits.Length));
    }

    private static ArcExecutionLane CreateLane()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        return new ArcExecutionLane(0, new ArcExecutionOptions
        {
            BufferPoolBytes = 4 * 1024 * 1024,
            Qwen35InferenceKernelsOnly = true
        });
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var result = new float[count];
        lane.Read(buffer, result);
        return result;
    }

    // Double precision mathematical reference, independent of the OpenCL reduction/layout.
    private static double[] NormalizeAndRotate(double[] data, float[] weight, int width,
        int ropeDimensions, int position, float theta, float epsilon)
    {
        var result = new List<double>();
        foreach (double[] row in data.Chunk(width))
        {
            double denominator = Math.Sqrt(row.Select(x => x * x).Average() + epsilon);
            double[] normalized = row.Select((x, j) => x / denominator * weight[j]).ToArray();
            int split = ropeDimensions / 2;
            for (int j = 0; j < split; ++j)
            {
                var rotation = System.Numerics.Complex.FromPolarCoordinates(1,
                    position / Math.Pow(theta, 2.0 * j / ropeDimensions));
                var pair = new System.Numerics.Complex(normalized[j], normalized[j + split]) * rotation;
                normalized[j] = pair.Real;
                normalized[j + split] = pair.Imaginary;
            }
            result.AddRange(normalized);
        }
        return result.ToArray();
    }

    private static double[] AttentionReference(double[] query, float[] qAndGate,
        List<double[]> keys, List<double[]> values, int heads, int kvHeads, int width)
    {
        return Enumerable.Range(0, heads).SelectMany(head =>
        {
            int kv = head * kvHeads / heads;
            double[] scores = keys.Select(key => Enumerable.Range(0, width)
                .Sum(j => query[head * width + j] * key[kv * width + j]) / Math.Sqrt(width)).ToArray();
            double maximum = scores.Max();
            double[] probabilities = scores.Select(score => Math.Exp(score - maximum)).ToArray();
            double sum = probabilities.Sum();
            return Enumerable.Range(0, width).Select(j =>
                values.Select((value, time) => value[kv * width + j] * probabilities[time] / sum).Sum()
                / (1 + Math.Exp(-qAndGate[(2 * head + 1) * width + j])));
        }).ToArray();
    }

    private static void AssertClose(double[] expected, float[] actual, double tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) AssertClose(expected[i], actual[i], tolerance);
    }

    private static void AssertClose(double expected, float actual, double tolerance)
        => Assert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}.");
}
