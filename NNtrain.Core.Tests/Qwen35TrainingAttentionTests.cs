using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35TrainingAttentionTests
{
    [Theory]
    [InlineData(0, false)]
    [InlineData(4, false)]
    [InlineData(6, false)]
    [InlineData(0, true)]
    [InlineData(4, true)]
    [InlineData(6, true)]
    [InlineData(4, false, 129)]
    [InlineData(4, true, 129)]
    [InlineData(4, false, 1)]
    [InlineData(4, true, 1)]
    public void FullSequenceForwardMatchesIncrementalAttentionAndKeepsInputsOwnedByCaller(int ropeDimensions, bool packedScores, int sequence = 5)
    {
        using ArcExecutionLane lane = CreateLane();
        const int heads = 4, kvHeads = 2, width = 6;
        Qwen35GgufDescriptor d = Descriptor(sequence, heads, kvHeads, width, ropeDimensions);
        float[] queries = Values(sequence * 2 * heads * width, 13);
        float[] keys = Values(sequence * kvHeads * width, 17);
        float[] values = Values(keys.Length, 19);
        float[] qWeights = Values(width, 23).Select(x => 1 + x).ToArray();
        float[] kWeights = Values(width, 29).Select(x => 1 + x).ToArray();
        queries[width] = -80; queries[3 * width + 1] = 80;
        using ArcBuffer q = lane.Upload(queries), k = lane.Upload(keys), v = lane.Upload(values);
        using ArcBuffer qNorm = lane.Upload(qWeights), kNorm = lane.Upload(kWeights);
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        float[] actual;
        using (var attention = new Qwen35TrainingAttention(lane, q, k, v, qNorm, kNorm, d, sequence, packedScores))
        {
            Assert.Equal(uploads, lane.H2DBytes);
            Assert.Equal(downloads, lane.D2HBytes);
            actual = Read(lane, attention.Output, sequence * heads * width);
        }
        Assert.True(q.IsAlive && k.IsAlive && v.IsAlive && qNorm.IsAlive && kNorm.IsAlive);
        Assert.Equal(queries, Read(lane, q, queries.Length));
        Assert.Equal(keys, Read(lane, k, keys.Length));
        Assert.Equal(values, Read(lane, v, values.Length));
        using ArcBuffer keyCache = lane.Allocate(keys.Length), valueCache = lane.Allocate(values.Length);
        for (int t = 0; t < sequence; ++t)
        {
            using ArcBuffer qt = lane.Upload(queries.AsSpan(t * 2 * heads * width, 2 * heads * width).ToArray());
            using ArcBuffer kt = lane.Upload(keys.AsSpan(t * kvHeads * width, kvHeads * width).ToArray());
            using ArcBuffer vt = lane.Upload(values.AsSpan(t * kvHeads * width, kvHeads * width).ToArray());
            using ArcBuffer step = Qwen35Gpu.AttentionStep(lane, qt, kt, vt, qNorm, kNorm,
                keyCache, valueCache, t, heads, kvHeads, width, ropeDimensions, d.RopeTheta, d.RmsEpsilon);
            float[] expected = Read(lane, step, heads * width);
            for (int i = 0; i < expected.Length; ++i)
                Close(expected[i], actual[t * expected.Length + i], 2e-5);
        }
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(2, false)]
    [InlineData(4, false)]
    [InlineData(0, true)]
    [InlineData(2, true)]
    [InlineData(4, true)]
    public void BackwardMatchesIndependentFiniteDifferencesAndAccumulates(int ropeDimensions, bool packedScores)
    {
        using ArcExecutionLane lane = CreateLane();
        const int sequence = 3, heads = 4, kvHeads = 2, width = 4;
        Qwen35GgufDescriptor d = Descriptor(sequence, heads, kvHeads, width, ropeDimensions);
        float[] queries = Values(sequence * 2 * heads * width, 31);
        float[] keys = Values(sequence * kvHeads * width, 37);
        float[] values = Values(keys.Length, 41);
        float[] qWeights = Values(width, 43).Select(x => 1 + x).ToArray();
        float[] kWeights = Values(width, 47).Select(x => 1 + x).ToArray();
        float[] gradient = Values(sequence * heads * width, 53);
        using ArcBuffer q = lane.Upload(queries), k = lane.Upload(keys), v = lane.Upload(values);
        using ArcBuffer qNorm = lane.Upload(qWeights), kNorm = lane.Upload(kWeights), dy = lane.Upload(gradient);
        using ArcBuffer dq = lane.Upload(Enumerable.Repeat(.25f, queries.Length).ToArray());
        using ArcBuffer dk = lane.Upload(Enumerable.Repeat(-.125f, keys.Length).ToArray());
        using ArcBuffer dv = lane.Upload(Enumerable.Repeat(.5f, values.Length).ToArray());
        long uploads = lane.H2DBytes, downloads = lane.D2HBytes;
        using var attention = new Qwen35TrainingAttention(lane, q, k, v, qNorm, kNorm, d, sequence, packedScores);
        attention.Backward(dy, dq, dk, dv);
        Assert.Equal(uploads, lane.H2DBytes);
        Assert.Equal(downloads, lane.D2HBytes);
        double[] qDouble = queries.Select(x => (double)x).ToArray();
        double[] kDouble = keys.Select(x => (double)x).ToArray();
        double[] vDouble = values.Select(x => (double)x).ToArray();
        double Loss() => Reference(qDouble, kDouble, vDouble, qWeights, kWeights, d, sequence)
            .Zip(gradient, (output, upstream) => output * upstream).Sum();
        double[] expectedQ = FiniteDifferences(qDouble, Loss);
        double[] expectedK = FiniteDifferences(kDouble, Loss);
        double[] expectedV = FiniteDifferences(vDouble, Loss);
        AssertGradient(expectedQ, Read(lane, dq, queries.Length), .25, 1);
        AssertGradient(expectedK, Read(lane, dk, keys.Length), -.125, 1);
        AssertGradient(expectedV, Read(lane, dv, values.Length), .5, 1);
        attention.Backward(dy, dq, dk, dv);
        AssertGradient(expectedQ, Read(lane, dq, queries.Length), .25, 2);
        AssertGradient(expectedK, Read(lane, dk, keys.Length), -.125, 2);
        AssertGradient(expectedV, Read(lane, dv, values.Length), .5, 2);
        Assert.Equal(qWeights, Read(lane, qNorm, width));
        Assert.Equal(kWeights, Read(lane, kNorm, width));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void BackwardOfFirstPositionDoesNotReachFutureInputs(bool packedScores)
    {
        using ArcExecutionLane lane = CreateLane();
        const int sequence = 3, heads = 2, kvHeads = 1, width = 4;
        Qwen35GgufDescriptor d = Descriptor(sequence, heads, kvHeads, width, 2);
        using ArcBuffer q = lane.Upload(Values(sequence * 2 * heads * width, 61));
        using ArcBuffer k = lane.Upload(Values(sequence * kvHeads * width, 67));
        using ArcBuffer v = lane.Upload(Values(sequence * kvHeads * width, 71));
        using ArcBuffer norm = lane.Upload(Enumerable.Repeat(1f, width).ToArray());
        float[] upstream = new float[sequence * heads * width];
        Array.Fill(upstream, 1f, 0, heads * width);
        using ArcBuffer dy = lane.Upload(upstream);
        using ArcBuffer dq = lane.Upload(new float[sequence * 2 * heads * width]);
        using ArcBuffer dk = lane.Upload(new float[sequence * kvHeads * width]);
        using ArcBuffer dv = lane.Upload(new float[sequence * kvHeads * width]);
        using var attention = new Qwen35TrainingAttention(lane, q, k, v, norm, norm, d, sequence, packedScores);
        attention.Backward(dy, dq, dk, dv);
        Assert.All(Read(lane, dq, sequence * 2 * heads * width).Skip(2 * heads * width), x => Assert.Equal(0f, x));
        // At position zero softmax has one item, so all Q/K score gradients vanish.
        Assert.All(Read(lane, dk, sequence * kvHeads * width), x => Assert.Equal(0f, x));
        Assert.All(Read(lane, dv, sequence * kvHeads * width).Skip(kvHeads * width), x => Assert.Equal(0f, x));
    }

    [Theory]
    [InlineData(1, false, false)]
    [InlineData(63, false, false)]
    [InlineData(64, false, false)]
    [InlineData(65, false, false)]
    [InlineData(129, false, false)]
    [InlineData(1, true, false)]
    [InlineData(63, true, false)]
    [InlineData(64, true, false)]
    [InlineData(65, true, false)]
    [InlineData(129, true, false)]
    [InlineData(1, true, true)]
    [InlineData(63, true, true)]
    [InlineData(64, true, true)]
    [InlineData(65, true, true)]
    [InlineData(129, true, true)]
    [InlineData(65, true, true, 136)]
    public void StreamedTilesMatchPackedForwardAndAllInputGradients(
        int sequence, bool fusedRows, bool fusedOutput, int width = 4)
    {
        using ArcExecutionLane lane = CreateLane();
        const int heads = 4, kvHeads = 2;
        Qwen35GgufDescriptor d = Descriptor(sequence, heads, kvHeads, width, 2);
        float[] queries = Values(sequence * 2 * heads * width, 73);
        float[] keys = Values(sequence * kvHeads * width, 79);
        float[] values = Values(keys.Length, 83);
        float[] upstream = Values(sequence * heads * width, 89);
        using ArcBuffer q = lane.Upload(queries), k = lane.Upload(keys), v = lane.Upload(values);
        using ArcBuffer norm = lane.Upload(Enumerable.Repeat(1f, width).ToArray());
        using ArcBuffer dy = lane.Upload(upstream);

        (float[] Output, float[] Q, float[] K, float[] V) Compute(
            int tileRows, bool fuseRows, bool fuseOutput)
        {
            using ArcBuffer dq = lane.Upload(Enumerable.Repeat(.25f, queries.Length).ToArray());
            using ArcBuffer dk = lane.Upload(Enumerable.Repeat(-.125f, keys.Length).ToArray());
            using ArcBuffer dv = lane.Upload(Enumerable.Repeat(.5f, values.Length).ToArray());
            using var attention = new Qwen35TrainingAttention(lane, q, k, v, norm, norm,
                d, sequence, packedScores: true, streamedTileRows: tileRows,
                fusedRows: fuseRows, fusedOutput: fuseOutput);
            float[] output = Read(lane, attention.Output, sequence * heads * width);
            attention.Backward(dy, dq, dk, dv);
            return (output, Read(lane, dq, queries.Length), Read(lane, dk, keys.Length),
                Read(lane, dv, values.Length));
        }

        var packed = Compute(0, false, false);
        var streamed = Compute(64, fusedRows, fusedOutput);
        void Match(float[] expected, float[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++) Close(expected[i], actual[i], 1e-5);
        }
        Match(packed.Output, streamed.Output);
        Match(packed.Q, streamed.Q);
        Match(packed.K, streamed.K);
        Match(packed.V, streamed.V);
    }

    [Theory]
    [InlineData(1, false)]
    [InlineData(3, false)]
    [InlineData(64, false)]
    [InlineData(65, false)]
    [InlineData(129, false)]
    [InlineData(1, true)]
    [InlineData(3, true)]
    [InlineData(64, true)]
    [InlineData(65, true)]
    [InlineData(129, true)]
    public void RowFusedMatchesPackedForwardAndAllInputGradients(int sequence, bool fusedOutput)
    {
        using ArcExecutionLane lane = CreateLane();
        const int heads = 4, kvHeads = 2, width = 8;
        Qwen35GgufDescriptor d = Descriptor(sequence, heads, kvHeads, width, 4);
        float[] queries = Values(sequence * 2 * heads * width, 97);
        float[] keys = Values(sequence * kvHeads * width, 101);
        float[] values = Values(keys.Length, 103);
        float[] upstream = Values(sequence * heads * width, 107);
        using ArcBuffer q = lane.Upload(queries), k = lane.Upload(keys), v = lane.Upload(values);
        using ArcBuffer norm = lane.Upload(Enumerable.Repeat(1f, width).ToArray());
        using ArcBuffer dy = lane.Upload(upstream);

        (float[] Output, float[] Q, float[] K, float[] V) Compute(bool fused, bool fuseOutput)
        {
            using ArcBuffer dq = lane.Upload(Enumerable.Repeat(.25f, queries.Length).ToArray());
            using ArcBuffer dk = lane.Upload(Enumerable.Repeat(-.125f, keys.Length).ToArray());
            using ArcBuffer dv = lane.Upload(Enumerable.Repeat(.5f, values.Length).ToArray());
            using var attention = new Qwen35TrainingAttention(lane, q, k, v, norm, norm,
                d, sequence, packedScores: true, fusedRows: fused,
                fusedOutput: fuseOutput);
            float[] output = Read(lane, attention.Output, sequence * heads * width);
            attention.Backward(dy, dq, dk, dv);
            return (output, Read(lane, dq, queries.Length), Read(lane, dk, keys.Length),
                Read(lane, dv, values.Length));
        }

        var packed = Compute(false, false);
        var fused = Compute(true, fusedOutput);
        void Match(float[] expected, float[] actual)
        {
            Assert.Equal(expected.Length, actual.Length);
            for (int i = 0; i < expected.Length; i++) Close(expected[i], actual[i], 1e-5);
        }
        Match(packed.Output, fused.Output);
        Match(packed.Q, fused.Q);
        Match(packed.K, fused.K);
        Match(packed.V, fused.V);
    }

    private static double[] FiniteDifferences(double[] input, Func<double> loss)
    {
        const double epsilon = 1e-5;
        var gradient = new double[input.Length];
        for (int i = 0; i < input.Length; ++i)
        {
            double original = input[i];
            input[i] = original + epsilon;
            double plus = loss();
            input[i] = original - epsilon;
            double minus = loss();
            input[i] = original;
            gradient[i] = (plus - minus) / (2 * epsilon);
        }
        return gradient;
    }

    // Independent double-precision reference used only to derive finite differences.
    private static double[] Reference(double[] qAndGate, double[] key, double[] value,
        float[] qWeights, float[] kWeights, Qwen35GgufDescriptor d, int sequence)
    {
        int heads = d.HeadCount, kvHeads = d.KvHeadCount, width = d.HeadWidth;
        double[] Normalize(double[] source, float[] weights, int count, int stride)
        {
            var normalized = new double[sequence * count * width];
            for (int t = 0; t < sequence; ++t)
                for (int h = 0; h < count; ++h)
                {
                    int row = t * count + h;
                    double sum = Enumerable.Range(0, width).Sum(j => Math.Pow(source[row * stride + j], 2));
                    double inverse = 1 / Math.Sqrt(sum / width + d.RmsEpsilon);
                    for (int j = 0; j < width; ++j)
                        normalized[row * width + j] = source[row * stride + j] * weights[j] * inverse;
                    int pairs = d.RopeDimensionCount / 2;
                    for (int j = 0; j < pairs; ++j)
                    {
                        double angle = t / Math.Pow(d.RopeTheta, 2.0 * j / d.RopeDimensionCount);
                        int first = row * width + j, second = first + pairs;
                        double a = normalized[first], b = normalized[second];
                        normalized[first] = a * Math.Cos(angle) - b * Math.Sin(angle);
                        normalized[second] = a * Math.Sin(angle) + b * Math.Cos(angle);
                    }
                }
            return normalized;
        }
        double[] query = Normalize(qAndGate, qWeights, heads, 2 * width);
        double[] keys = Normalize(key, kWeights, kvHeads, width);
        var output = new double[sequence * heads * width];
        for (int t = 0; t < sequence; ++t)
            for (int h = 0; h < heads; ++h)
            {
                int row = t * heads + h, kv = h / (heads / kvHeads);
                double[] scores = Enumerable.Range(0, t + 1).Select(s => Enumerable.Range(0, width)
                    .Sum(j => query[row * width + j] * keys[(s * kvHeads + kv) * width + j]) / Math.Sqrt(width)).ToArray();
                double maximum = scores.Max();
                double[] probability = scores.Select(x => Math.Exp(x - maximum)).ToArray();
                double total = probability.Sum();
                for (int j = 0; j < width; ++j)
                    output[row * width + j] = Enumerable.Range(0, t + 1)
                        .Sum(s => probability[s] / total * value[(s * kvHeads + kv) * width + j])
                        / (1 + Math.Exp(-qAndGate[(2 * row + 1) * width + j]));
            }
        return output;
    }

    private static Qwen35GgufDescriptor Descriptor(int sequence, int heads, int kvHeads, int width, int rope)
        => new(8, 1, heads * width, heads, kvHeads, width, sequence, heads * width * 2,
            .01f, 10000f, rope, 1, 1, width, 4, 1, []);

    private static ArcExecutionLane CreateLane()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        return new ArcExecutionLane(0, new ArcExecutionOptions
        {
            BufferPoolBytes = 4 * 1024 * 1024,
            Qwen35InferenceKernelsOnly = true,
            Qwen35TrainingKernels = true
        });
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => (float)(random.NextDouble() * 1.6 - .8)).ToArray();
    }

    private static float[] Read(ArcExecutionLane lane, ArcBuffer buffer, int count)
    {
        var values = new float[count];
        lane.Read(buffer, values);
        return values;
    }

    private static void AssertGradient(double[] expected, float[] actual, double offset, int multiplier)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; ++i) Close(offset + multiplier * expected[i], actual[i], 3e-4);
    }

    private static void Close(double expected, float actual, double tolerance)
        => Assert.True(float.IsFinite(actual) && Math.Abs(expected - actual) <= tolerance,
            $"Expected {expected:R}, actual {actual:R}, tolerance {tolerance:R}.");
}
