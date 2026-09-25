using NNtrain.Arc;
using Xunit;

public sealed class ArcRowBiasKernelTests
{
    [Fact]
    public void AddsFp32BiasOncePerRowAfterPartialSumsOnBothArcDevices()
    {
        Assert.SkipWhen(ArcDevices.Enumerate().Count < 2,
            "Two Intel Arc OpenCL GPUs are required.");

        const int rows = 7;
        const int columns = 5;
        const int length = rows * columns;
        float[] left = Enumerable.Range(0, length)
            .Select(index => (index - 17) * 0.25f).ToArray();
        float[] right = Enumerable.Range(0, length)
            .Select(index => (index % 7 - 3) * 0.5f).ToArray();
        float[] bias = [0.125f, -0.25f, 0.5f, -1f, 2f];
        float[] expected = Enumerable.Range(0, length)
            .Select(index => (left[index] + right[index]) + bias[index % columns])
            .ToArray();

        foreach (int deviceIndex in new[] { 0, 1 })
        {
            using var lane = new ArcExecutionLane(deviceIndex);
            using var leftBuffer = lane.Upload(left);
            using var rightBuffer = lane.Upload(right);
            using var biasBuffer = lane.Upload(bias);
            using var partialSum = lane.Allocate(length);
            using var output = lane.Allocate(length);

            lane.Run("add", length, 0,
                leftBuffer, rightBuffer, partialSum, length);
            lane.Run("add_row_bias", length, 0,
                partialSum, biasBuffer, output, length, columns);

            var actual = new float[length];
            lane.Read(output, actual);
            Assert.Equal(expected, actual);
        }
    }
}
