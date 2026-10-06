using System.Diagnostics;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuAttentionBatchTests(ITestOutputHelper output)
{
    [Theory]
    [InlineData(1, 0, 0, false, false, 128)]
    [InlineData(4, 5, 64, false, false, 128)]
    [InlineData(16, 0, 64, true, false, 128)]
    [InlineData(16, 131, 64, true, true, 128)]
    [InlineData(33, 5, 64, false, true, 128)]
    [InlineData(16, 593, 64, true, true, 256)]
    public void ChunkedCausalAttentionPreservesScalarOutputsAndCaches(
        int rows, int startPosition, int ropeDimensions, bool spatial, bool fastNorm, int width)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU required.");
        using var lane = new ArcExecutionLane(0, new ArcExecutionOptions
        {
            BufferPoolBytes = 8 * 1024 * 1024,
            Qwen35InferenceKernelsOnly = true,
            Qwen35FastRmsNorm = fastNorm,
            XmxMatrices = fastNorm
        });
        const int heads = 4, kvHeads = 2;
        const float epsilon = 1e-6f, theta = 10_000_000f;
        int querySize = heads * width, kvSize = kvHeads * width;
        int sequence = startPosition + rows;
        float[] qValues = Values(rows * 2 * querySize, 11);
        float[] kValues = Values(rows * kvSize, 17), vValues = Values(rows * kvSize, 21);
        float[] oldKeys = Values(sequence * kvSize, 71), oldValues = Values(sequence * kvSize, 81);
        using ArcBuffer qNorm = lane.Upload(Values(width, 13));
        using ArcBuffer kNorm = lane.Upload(Values(width, 23));
        using ArcBuffer scalarKeys = lane.Upload(oldKeys), scalarValues = lane.Upload(oldValues);
        using ArcBuffer batchKeys = lane.Upload(oldKeys), batchValues = lane.Upload(oldValues);
        var expected = new float[rows * querySize];
        Qwen35Position[] positions = Enumerable.Range(0, rows).Select(row => spatial
            ? new Qwen35Position(startPosition + 2, startPosition + row / 4, startPosition + row % 4)
            : Qwen35Position.Scalar(startPosition + row + 7)).ToArray();
        for (int row = 0; row < rows; row++)
        {
            using ArcBuffer q = lane.Upload(qValues.AsSpan(row * 2 * querySize, 2 * querySize).ToArray());
            using ArcBuffer k = lane.Upload(kValues.AsSpan(row * kvSize, kvSize).ToArray());
            using ArcBuffer v = lane.Upload(vValues.AsSpan(row * kvSize, kvSize).ToArray());
            using ArcBuffer attention = Qwen35Gpu.AttentionStep(lane, q, k, v, qNorm, kNorm,
                scalarKeys, scalarValues, startPosition + row, heads, kvHeads, width,
                ropeDimensions, theta, epsilon, positions[row], [11, 11, 10]);
            var rowValues = new float[querySize];
            lane.Read(attention, rowValues);
            rowValues.CopyTo(expected, row * querySize);
        }
        using ArcBuffer batchQ = lane.Upload(qValues), batchK = lane.Upload(kValues), batchV = lane.Upload(vValues);
        using ArcBuffer result = Qwen35Gpu.AttentionRows(lane, batchQ, batchK, batchV, qNorm, kNorm,
            batchKeys, batchValues, startPosition, rows, heads, kvHeads, width, ropeDimensions,
            theta, epsilon, positions, [11, 11, 10]);
        var actual = new float[expected.Length];
        lane.Read(result, actual);
        Assert.Equal(expected, actual);
        var expectedCache = new float[sequence * kvSize];
        var actualCache = new float[expectedCache.Length];
        lane.Read(scalarKeys, expectedCache);
        lane.Read(batchKeys, actualCache);
        Assert.Equal(expectedCache, actualCache);
        lane.Read(scalarValues, expectedCache);
        lane.Read(batchValues, actualCache);
        Assert.Equal(expectedCache, actualCache);
    }

    private static float[] Values(int length, int seed)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, length).Select(_ => random.NextSingle() * 4f - 2f).ToArray();
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void RealisticAttentionChunkReportsScalarAndBatchTimings(int device)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count <= device, "Intel Arc GPU required.");
        using var lane = new ArcExecutionLane(device, new ArcExecutionOptions
        {
            BufferPoolBytes = 64 * 1024 * 1024,
            Qwen35InferenceKernelsOnly = true,
            Qwen35FastRmsNorm = true,
            XmxMatrices = true
        });
        const int heads = 24, kvHeads = 4, width = 256, rows = 16, start = 583;
        int querySize = heads * width, kvSize = kvHeads * width;
        using ArcBuffer q = lane.Upload(Values(rows * querySize * 2, 101));
        using ArcBuffer k = lane.Upload(Values(rows * kvSize, 103));
        using ArcBuffer v = lane.Upload(Values(rows * kvSize, 105));
        using ArcBuffer qNorm = lane.Upload(Values(width, 107)), kNorm = lane.Upload(Values(width, 109));
        using ArcBuffer keys = lane.Upload(Values((start + rows) * kvSize, 111));
        using ArcBuffer values = lane.Upload(Values((start + rows) * kvSize, 113));
        using ArcBuffer rowQuery = lane.Allocate(2 * querySize), rowKey = lane.Allocate(kvSize), rowValue = lane.Allocate(kvSize);
        using ArcBuffer scalarOutput = lane.Allocate(rows * querySize);
        Qwen35Position[] positions = Enumerable.Range(0, rows)
            .Select(i => new Qwen35Position(12, 12 + i / 24, 12 + i % 24)).ToArray();
        void Scalar()
        {
            for (int row = 0; row < rows; row++)
            {
                lane.CopyBytes(q, rowQuery, row * 2 * querySize * sizeof(float), 0, 2 * querySize * sizeof(float));
                lane.CopyBytes(k, rowKey, row * kvSize * sizeof(float), 0, kvSize * sizeof(float));
                lane.CopyBytes(v, rowValue, row * kvSize * sizeof(float), 0, kvSize * sizeof(float));
                using ArcBuffer attention = Qwen35Gpu.AttentionStep(lane, rowQuery, rowKey, rowValue,
                    qNorm, kNorm, keys, values, start + row, heads, kvHeads, width, 64, 10_000_000f,
                    1e-6f, positions[row], [11, 11, 10]);
                lane.CopyBytes(attention, scalarOutput, 0, row * querySize * sizeof(float), querySize * sizeof(float));
            }
        }
        void Batch()
        {
            using ArcBuffer attention = Qwen35Gpu.AttentionRows(lane, q, k, v, qNorm, kNorm,
                keys, values, start, rows, heads, kvHeads, width, 64, 10_000_000f,
                1e-6f, positions, [11, 11, 10]);
        }
        Scalar(); Batch(); lane.Synchronize();
        var timings = new List<(double Scalar, double Batch)>();
        for (int iteration = 0; iteration < 7; iteration++)
        {
            long begin = Stopwatch.GetTimestamp();
            Scalar(); lane.Synchronize();
            double scalarMilliseconds = Stopwatch.GetElapsedTime(begin).TotalMilliseconds;
            begin = Stopwatch.GetTimestamp();
            Batch(); lane.Synchronize();
            timings.Add((scalarMilliseconds, Stopwatch.GetElapsedTime(begin).TotalMilliseconds));
        }
        double scalarMedian = timings.Select(t => t.Scalar).Order().ElementAt(3);
        double batchMedian = timings.Select(t => t.Batch).Order().ElementAt(3);
        output.WriteLine($"Device {device} {lane.Device.Name}: scalar {scalarMedian:F3} ms, batch {batchMedian:F3} ms, speedup {scalarMedian / batchMedian:F3}x");
    }
}
