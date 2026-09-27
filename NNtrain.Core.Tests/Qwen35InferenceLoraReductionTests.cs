using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35InferenceLoraReductionTests
{
    [Theory]
    [InlineData(16)]
    [InlineData(128)]
    [InlineData(256)]
    [InlineData(512)]
    [InlineData(1024)]
    public void InferenceReductionMatchesCpuAndSerialForRealAndRaggedWidths(int reductionSize)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35CooperativeLora = true,
            Qwen35LoraReductionSize = reductionSize
        });
        var random = new Random(713);
        foreach (int width in new[] { 513, 5120, 17408 })
        foreach (int rank in new[] { 3, 8, 16 })
        foreach (int rows in new[] { 1, 2 })
        {
            float[] input = Enumerable.Range(0, rows * width)
                .Select(_ => (float)(random.NextDouble() * 2 - 1)).ToArray();
            float[] a = Enumerable.Range(0, rank * width)
                .Select(_ => (float)((random.NextDouble() * 2 - 1) / Math.Sqrt(width))).ToArray();
            using var matrix = new Qwen35LoraMatrix(lane, width, 7,
                new() { Rank = rank, Alpha = rank * 2 }, random);
            lane.Write(matrix.A, a);
            using ArcBuffer x = lane.Upload(input), reference = lane.Allocate(rows * rank);
            using ArcBuffer candidate = matrix.ProjectA(x, rows);
            lane.Run("q35l_lora_a", rows * rank, 0, x, matrix.A, reference, rows, width, rank);
            var actual = new float[rows * rank];
            var serial = new float[rows * rank];
            lane.Read(candidate, actual);
            lane.Read(reference, serial);
            for (int row = 0; row < rows; row++)
            for (int r = 0; r < rank; r++)
            {
                double expected = 0, absoluteSum = 0;
                for (int j = 0; j < width; j++)
                {
                    double product = (double)input[row * width + j] * a[r * width + j];
                    expected += product;
                    absoluteSum += Math.Abs(product);
                }
                int index = row * rank + r;
                // FP32 accumulation order changes. Scale the error bound by
                // product magnitudes so cancellation cannot hide layout bugs.
                double tolerance = 2e-7 + absoluteSum * 2e-6;
                Assert.True(float.IsFinite(actual[index]) && float.IsFinite(serial[index]));
                Assert.InRange(Math.Abs(actual[index] - expected), 0, tolerance);
                Assert.InRange(Math.Abs(serial[index] - expected), 0, tolerance);
                Assert.InRange(Math.Abs(actual[index] - serial[index]), 0, tolerance * 2);
            }
        }
        bool subgroup = lane.Options.XmxMatrices && lane.Device.SupportsXmx
            && lane.Device.MinimumSubgroupSize == 16
            && lane.Device.Extensions.Split(' ').Contains("cl_intel_subgroups");
        string expectedKernel = reductionSize == 16
            ? subgroup ? "q35l_lora_a_sg16" : "q35l_lora_a_coop"
            : reductionSize == 128 ? "q35l_lora_a_coop" : $"q35l_lora_a_coop{reductionSize}";
        Assert.Contains(expectedKernel, lane.KernelTimings.Keys);
    }

    [Fact]
    public void SubgroupOptionFallsBackToOriginalCooperativeKernelWhenDisabled()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc GPU is required.");
        using var lane = new ArcExecutionLane(0, new()
        {
            Qwen35InferenceKernelsOnly = true, Qwen35CooperativeLora = true,
            Qwen35LoraReductionSize = 16, XmxMatrices = false
        });
        using var matrix = new Qwen35LoraMatrix(lane, 513, 7,
            new() { Rank = 3, Alpha = 6 }, new Random(6));
        using ArcBuffer input = lane.Upload(Enumerable.Repeat(0.25f, 513).ToArray());
        using ArcBuffer output = matrix.ProjectA(input, 1);
        lane.Read(output, new float[3]);
        Assert.Contains("q35l_lora_a_coop", lane.KernelTimings.Keys);
        Assert.DoesNotContain("q35l_lora_a_sg16", lane.KernelTimings.Keys);
    }
}
