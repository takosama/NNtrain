using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferenceArgmaxTests
{
    [Fact]
    public void ParallelArgmaxMatchesReferenceAndCpuAcrossPartitions()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var reference = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        using var parallel = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35ParallelArgmax = true
        });
        var random = new Random(726);
        // Include partial work-groups, partition boundaries, the actual model's
        // vocabulary and more than 128 partitions for the second-stage loop.
        foreach (int count in new[] { 1, 127, 128, 129, 4095, 4096, 4097, 248320, 524289 })
        {
            float[] values = Enumerable.Range(0, count)
                .Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
            int expected = 0;
            for (int i = 1; i < count; i++) if (values[i] > values[expected]) expected = i;
            AssertBoth(reference, parallel, values, expected);
        }
        Assert.Contains("q35a_argmax_partition", parallel.KernelTimings.Keys);
        Assert.Contains("q35a_argmax_reduce", parallel.KernelTimings.Keys);
        Assert.Contains("q35a_argmax", parallel.KernelTimings.Keys);
    }

    [Fact]
    public void ParallelArgmaxPreservesLowestTokenTieIncludingSignedZero()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var reference = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        using var parallel = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35ParallelArgmax = true
        });
        float[] values = Enumerable.Repeat(-3f, 12301).ToArray();
        foreach (int i in new[] { 127, 128, 4095, 4096, 8192, values.Length - 1 }) values[i] = 7f;
        AssertBoth(reference, parallel, values, 127);
        Array.Fill(values, float.MinValue);
        AssertBoth(reference, parallel, values, 0);
        Array.Fill(values, -1f);
        values[4095] = -0f;
        values[4096] = 0f;
        values[^1] = 0f;
        AssertBoth(reference, parallel, values, 4095);
    }

    [Fact]
    public void ParallelArgmaxRejectsNonfiniteLogitsInEveryPartition()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var reference = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        using var parallel = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35ParallelArgmax = true
        });
        float[] values = new float[12291];
        foreach (int index in new[] { 0, 4095, 4096, 8192, values.Length - 1 })
        foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            values[index] = invalid;
            using ArcBuffer baseline = reference.Upload(values), candidate = parallel.Upload(values);
            Assert.Throws<ArithmeticException>(() => Qwen35Gpu.ArgMax(reference, baseline, values.Length));
            Assert.Throws<ArithmeticException>(() => Qwen35Gpu.ArgMax(parallel, candidate, values.Length));
            values[index] = 0f;
        }
        Array.Fill(values, float.NaN);
        using ArcBuffer allInvalid = parallel.Upload(values);
        Assert.Throws<ArithmeticException>(() => Qwen35Gpu.ArgMax(parallel, allInvalid, values.Length));
    }

    private static void AssertBoth(ArcExecutionLane reference, ArcExecutionLane parallel,
        float[] values, int expected)
    {
        using ArcBuffer baseline = reference.Upload(values), candidate = parallel.Upload(values);
        Assert.Equal(expected, Qwen35Gpu.ArgMax(reference, baseline, values.Length));
        Assert.Equal(expected, Qwen35Gpu.ArgMax(parallel, candidate, values.Length));
    }
}
