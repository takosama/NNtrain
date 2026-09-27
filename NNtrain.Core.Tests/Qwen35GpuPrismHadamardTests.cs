using System.Numerics;
using NNtrain.Arc;
using Xunit;
using static NNtrain.Arc.ArcExecutionLane;

namespace NNtrain.Core.Tests;

public sealed class Qwen35GpuPrismHadamardTests
{
    [Theory]
    [InlineData(1024, 1, false)]
    [InlineData(5120, 2, false)]
    [InlineData(6144, 2, true)]
    [InlineData(17408, 1, false)]
    public void ForwardAndInverseMatchExplicitSylvesterMatrix(int width, int rows, bool grouped)
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count == 0, "Intel Arc is required.");
        using var lane = new ArcExecutionLane(0, new() { Qwen35InferenceKernelsOnly = true });
        float[] signs = Enumerable.Range(0, width).Select(i => (i * 37 % 11) < 5 ? -1f : 1f).ToArray();
        float[] input = Enumerable.Range(0, width * rows).Select(i => MathF.Sin(i * .019f) * .125f + (i % 9 - 4) * .007f).ToArray();
        using ArcBuffer x = lane.Upload(input), s = lane.Upload(signs);
        using ArcBuffer forward = lane.Upload(Enumerable.Repeat(float.NaN, input.Length + 2).ToArray());
        lane.Run("q35l_prism_hadamard", (long)width * rows / 1024 * 256, 256,
            x, s, forward, width, rows, 0, grouped ? 16 : 0, grouped ? 48 : 0, 128);
        var actual = new float[input.Length + 2]; lane.Read(forward, actual);
        Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
        for (int row = 0; row < rows; row++)
        for (int block = 0; block < width; block += 1024)
        for (int output = 0; output < 1024; output++)
        {
            double sum = 0;
            for (int col = 0; col < 1024; col++)
            {
                int j = block + col;
                int source = grouped ? j % 128 + 128 * (j / 128 / 3 + 16 * (j / 128 % 3)) : j;
                int h = (BitOperations.PopCount((uint)(output & col)) & 1) == 0 ? 1 : -1;
                sum += input[row * width + source] * signs[j] * h;
            }
            Assert.InRange(Math.Abs(actual[row * width + block + output] - sum / 32), 0, 1e-6);
        }
        // H is symmetric; signs are applied AFTER the inverse. For grouped
        // activations the round trip returns the explicitly permuted input.
        using ArcBuffer inverse = lane.Upload(Enumerable.Repeat(float.NaN, input.Length + 2).ToArray());
        lane.Run("q35l_prism_hadamard", (long)width * rows / 1024 * 256, 256,
            forward, s, inverse, width, rows, 1, 0, 0, 128);
        lane.Read(inverse, actual);
        Assert.True(float.IsNaN(actual[^1]) && float.IsNaN(actual[^2]));
        for (int row = 0; row < rows; row++)
        for (int j = 0; j < width; j++)
        {
            int source = grouped ? j % 128 + 128 * (j / 128 / 3 + 16 * (j / 128 % 3)) : j;
            Assert.InRange(Math.Abs(actual[row * width + j] - input[row * width + source]), 0, 1e-6);
        }
    }
}
